"""
detect_and_label.py
────────────────────────────────────────────────────────────────────────────────
Plays a video through YOLOv8s frame-by-frame (every Nth frame).
For each qualifying detection (height >= MIN_HEIGHT_PX) a labelling window
pops up. YOLO determines the vehicle type; the operator only decides direction:

    1  →  _away
    2  →  _towards
    s  →  skip (no file saved)
    b  →  go back to previous detection
    q  →  quit

The final label is  <yolo_class>_away  or  <yolo_class>_towards.
Saved as YOLO format: <class_id> 0.5 0.5 1.0 1.0

Output:
    <output>/
        images/   <stem>_f<frame>_d<det>.jpg
        labels/   <stem>_f<frame>_d<det>.txt

Usage:
    python detect_and_label.py --video highway.mp4
    python detect_and_label.py --video highway.mp4 --output my_dataset --frame-skip 3
    python detect_and_label.py --video highway.mp4 --skip-labelled   # resume
────────────────────────────────────────────────────────────────────────────────
"""

import argparse
import torch
import cv2
import sys
import tkinter as tk
from pathlib import Path
from PIL import Image, ImageTk, ImageDraw
from ultralytics import YOLO

# ── Tunables ──────────────────────────────────────────────────────────────────
MIN_HEIGHT_PX   = 100
FRAME_SKIP      = 5
CONF_THRESHOLD  = 0.40
JPEG_QUALITY    = 95
CROP_PADDING_PC = 0.05

# COCO indices → base vehicle name
TARGET_CLASSES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Final labels = all combinations of vehicle × direction.
# class_id = index into this list (written to the .txt file).
LABELS: list[str] = [
    "car_away",           # 0
    "car_towards",        # 1
    "truck_away",         # 2
    "truck_towards",      # 3
    "bus_away",           # 4
    "bus_towards",        # 5
    "motorcycle_away",    # 6
    "motorcycle_towards", # 7
    "bicycle_away",       # 8
    "bicycle_towards",    # 9
]

# (yolo_cls_id, direction) → LABELS index
LABEL_MAP: dict[tuple[int, str], int] = {
    (2, "away"):     0,
    (2, "towards"):  1,
    (7, "away"):     2,
    (7, "towards"):  3,
    (5, "away"):     4,
    (5, "towards"):  5,
    (3, "away"):     6,
    (3, "towards"):  7,
    (1, "away"):     8,
    (1, "towards"):  9,
}

CLASS_COLOURS = {
    2: "#00d4ff",   # car
    7: "#ff6b35",   # truck
    5: "#a855f7",   # bus
    3: "#22c55e",   # motorcycle
    1: "#facc15",   # bicycle
}
DEFAULT_COLOUR = "#ffffff"

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_output_dirs(root: Path) -> tuple[Path, Path]:
    img_dir = root / "images"
    lbl_dir = root / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    return img_dir, lbl_dir


def already_saved(stem: str, img_dir: Path, lbl_dir: Path) -> bool:
    return (img_dir / f"{stem}.jpg").exists() and (lbl_dir / f"{stem}.txt").exists()


def delete_pair(stem: str, img_dir: Path, lbl_dir: Path):
    for p in [img_dir / f"{stem}.jpg", lbl_dir / f"{stem}.txt"]:
        if p.exists():
            p.unlink()


def save_pair(stem: str, crop_bgr, class_id: int, img_dir: Path, lbl_dir: Path):
    cv2.imwrite(str(img_dir / f"{stem}.jpg"), crop_bgr,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    (lbl_dir / f"{stem}.txt").write_text(f"{class_id} 0.5 0.5 1.0 1.0\n")


def draw_overlay(frame_bgr, boxes_meta: list[dict]):
    out = frame_bgr.copy()
    for m in boxes_meta:
        x1, y1, x2, y2 = m["xyxy"]
        hex_c = CLASS_COLOURS.get(m["cls_id"], DEFAULT_COLOUR)
        r, g, b = int(hex_c[1:3], 16), int(hex_c[3:5], 16), int(hex_c[5:7], 16)
        cv2.rectangle(out, (x1, y1), (x2, y2), (b, g, r), 2)
        cv2.putText(out, f"{TARGET_CLASSES.get(m['cls_id'], '?')} {m['conf']:.2f}",
                    (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (b, g, r), 1, cv2.LINE_AA)
    return out


# ── Labeller GUI ──────────────────────────────────────────────────────────────

class Labeller:
    """
    Blocking modal window.
    .ask() returns: "away" | "towards" | None (skip) | "back" | "quit"
    """

    IMG_W = 900
    IMG_H = 520

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Vehicle Labeller")
        self.root.configure(bg="#0d0d0d")
        self.root.resizable(False, False)
        self._result = None
        self.quit_requested = False
        self._tk_img = None
        self._build_ui()
        self.root.withdraw()

    # ── Build UI ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        # top bar
        top = tk.Frame(self.root, bg="#0d0d0d")
        top.pack(fill=tk.X)

        tk.Label(top, text="VEHICLE LABELLER", bg="#0d0d0d", fg="#00d4ff",
                 font=("Courier New", 11, "bold"), anchor="w",
                 padx=14, pady=8).pack(side=tk.LEFT)

        self.counter_lbl = tk.Label(top, text="", bg="#0d0d0d", fg="#444444",
                                    font=("Courier New", 10), padx=14)
        self.counter_lbl.pack(side=tk.RIGHT)

        # canvas (image display)
        self.canvas = tk.Canvas(self.root, bg="#111111", highlightthickness=0,
                                width=self.IMG_W, height=self.IMG_H)
        self.canvas.pack()

        # detection info strip
        self.info_lbl = tk.Label(self.root, text="", bg="#111111", fg="#777777",
                                 font=("Courier New", 9), pady=5)
        self.info_lbl.pack(fill=tk.X)

        tk.Frame(self.root, bg="#222222", height=1).pack(fill=tk.X)

        # ── The two big direction buttons ─────────────────────────────────────
        choice = tk.Frame(self.root, bg="#0d0d0d")
        choice.pack(fill=tk.X, padx=16, pady=14)

        btn_cfg = dict(
            font=("Courier New", 15, "bold"),
            relief=tk.FLAT, bd=0,
            padx=0, pady=16,
            highlightthickness=2,
            cursor="hand2",
        )

        self.btn_away = tk.Button(
            choice,
            text="[ 1 ]   ←  AWAY",
            bg="#0d1a26", fg="#00d4ff",
            activebackground="#00d4ff", activeforeground="#000000",
            highlightbackground="#1a3a4a",
            command=lambda: self._select("away"),
            **btn_cfg,
        )
        self.btn_away.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 6))

        self.btn_towards = tk.Button(
            choice,
            text="TOWARDS  →   [ 2 ]",
            bg="#0d260d", fg="#22c55e",
            activebackground="#22c55e", activeforeground="#000000",
            highlightbackground="#1a4a1a",
            command=lambda: self._select("towards"),
            **btn_cfg,
        )
        self.btn_towards.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(6, 0))

        # footer controls
        foot = tk.Frame(self.root, bg="#0d0d0d")
        foot.pack(fill=tk.X, padx=16, pady=(0, 12))

        foot_cfg = dict(font=("Courier New", 9), relief=tk.FLAT,
                        padx=10, pady=5, cursor="hand2")

        tk.Button(foot, text="[b]  Back", bg="#1a1a1a", fg="#888888",
                  activebackground="#2a2a2a", activeforeground="#cccccc",
                  command=self._back, **foot_cfg).pack(side=tk.LEFT)

        tk.Button(foot, text="[s]  Skip", bg="#1a1a1a", fg="#666666",
                  activebackground="#2a2a2a", activeforeground="#999999",
                  command=self._skip, **foot_cfg).pack(side=tk.LEFT, padx=6)

        self.status_lbl = tk.Label(foot, text="", bg="#0d0d0d", fg="#00d4ff",
                                   font=("Courier New", 10, "bold"))
        self.status_lbl.pack(side=tk.LEFT, padx=10)

        tk.Button(foot, text="[q]  Quit & save", bg="#1a1a1a", fg="#cc3333",
                  activebackground="#2a2a2a", activeforeground="#ff4444",
                  command=self._quit, **foot_cfg).pack(side=tk.RIGHT)

        # key bindings
        self.root.bind("1", lambda _: self._select("away"))
        self.root.bind("2", lambda _: self._select("towards"))
        self.root.bind("s", lambda _: self._skip())
        self.root.bind("b", lambda _: self._back())
        self.root.bind("q", lambda _: self._quit())
        self.root.protocol("WM_DELETE_WINDOW", self._quit)

    # ── Actions ───────────────────────────────────────────────────────────────

    def _select(self, direction: str):
        self._flash(direction)
        self._result = direction
        self.status_lbl.config(text=f"✓  <vehicle>_{direction}")
        self.root.after(200, self._unblock)

    def _flash(self, direction: str):
        """Briefly highlight the chosen button."""
        if direction == "away":
            self.btn_away.config(bg="#00d4ff", fg="#000000")
            self.btn_towards.config(bg="#0d260d", fg="#22c55e")
        else:
            self.btn_towards.config(bg="#22c55e", fg="#000000")
            self.btn_away.config(bg="#0d1a26", fg="#00d4ff")

    def _reset_buttons(self):
        self.btn_away.config(bg="#0d1a26", fg="#00d4ff")
        self.btn_towards.config(bg="#0d260d", fg="#22c55e")

    def _skip(self):
        self._result = None
        self.status_lbl.config(text="Skipped.")
        self._unblock()

    def _back(self):
        self._result = "back"
        self.status_lbl.config(text="Going back…")
        self._unblock()

    def _quit(self):
        self.quit_requested = True
        self._result = None
        self._unblock()

    def _unblock(self):
        if hasattr(self, "_var"):
            self._var.set(1)

    # ── Public API ────────────────────────────────────────────────────────────

    def ask(self, crop_bgr, meta: dict,
            frame_overlay_bgr=None, counter_text: str = "") -> str | None:
        self._result = None
        self._reset_buttons()
        self.status_lbl.config(text="")
        self.counter_lbl.config(text=counter_text)

        cls_name = TARGET_CLASSES.get(meta["cls_id"], "unknown")
        bh = meta["xyxy"][3] - meta["xyxy"][1]
        bw = meta["xyxy"][2] - meta["xyxy"][0]
        hex_c = CLASS_COLOURS.get(meta["cls_id"], DEFAULT_COLOUR)
        self.info_lbl.config(
            text=f"  YOLO: {cls_name}  |  conf {meta['conf']:.2f}"
                 f"  |  box {bw}×{bh}px  |  frame {meta['frame_idx']}",
            fg=hex_c,
        )

        self._display_image(crop_bgr, frame_overlay_bgr)
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()
        self._var = tk.IntVar(value=0)
        self.root.wait_variable(self._var)
        return self._result

    # ── Image rendering ───────────────────────────────────────────────────────

    def _display_image(self, crop_bgr, context_bgr=None):
        def to_pil(bgr):
            return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        W, H = self.IMG_W, self.IMG_H

        if context_bgr is not None:
            ctx_w = int(W * 0.62)
            crp_w = W - ctx_w - 2

            ctx_pil = to_pil(context_bgr)
            ctx_pil.thumbnail((ctx_w, H), Image.LANCZOS)
            crp_pil = to_pil(crop_bgr)
            crp_pil.thumbnail((crp_w, H), Image.LANCZOS)

            img = Image.new("RGB", (W, H), (17, 17, 17))
            img.paste(ctx_pil, ((ctx_w - ctx_pil.width) // 2,
                                (H - ctx_pil.height) // 2))
            ImageDraw.Draw(img).rectangle([ctx_w, 0, ctx_w + 1, H], fill=(40, 40, 40))
            img.paste(crp_pil, (ctx_w + 2 + (crp_w - crp_pil.width) // 2,
                                (H - crp_pil.height) // 2))
        else:
            crp_pil = to_pil(crop_bgr)
            crp_pil.thumbnail((W, H), Image.LANCZOS)
            img = Image.new("RGB", (W, H), (17, 17, 17))
            img.paste(crp_pil, ((W - crp_pil.width) // 2,
                                (H - crp_pil.height) // 2))

        self._tk_img = ImageTk.PhotoImage(img)
        self.canvas.config(width=W, height=H)
        self.canvas.create_image(0, 0, anchor="nw", image=self._tk_img)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(args):
    video_path  = Path(args.video)
    output_root = Path(args.output)
    img_dir, lbl_dir = make_output_dirs(output_root)
    video_stem  = video_path.stem

    print(f"Loading YOLO model: {args.model}")
    model = YOLO(args.model)
    if torch.cuda.is_available():
        model.to("cuda")
    elif torch.backends.mps.is_available():
        model.to("mps")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    print(f"Video : {video_path.name}  ({total_frames} frames @ {fps:.1f} fps)")
    print(f"Output: {output_root.resolve()}\n")

    # ── Pass 1: scan entire video, collect all qualifying detections ───────────
    # Storing everything upfront makes "back" trivial — just decrement a pointer.
    print("Scanning video for detections…")
    all_detections: list[dict] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % args.frame_skip != 0:
            frame_idx += 1
            continue

        results = model(
            frame,
            classes=list(TARGET_CLASSES.keys()),
            conf=args.conf,
            verbose=False,
        )[0]

        qualifying = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            if (y2 - y1) < args.min_height:
                continue
            qualifying.append({
                "cls_id"   : cls_id,
                "conf"     : float(box.conf[0]),
                "xyxy"     : (x1, y1, x2, y2),
                "frame_idx": frame_idx,
            })

        if qualifying:
            H_f, W_f = frame.shape[:2]
            overlay   = draw_overlay(frame, qualifying)

            for det_idx, meta in enumerate(qualifying):
                x1, y1, x2, y2 = meta["xyxy"]
                bw, bh = x2 - x1, y2 - y1
                pad_x  = max(4, int(bw * CROP_PADDING_PC))
                pad_y  = max(4, int(bh * CROP_PADDING_PC))
                crop   = frame[max(0, y1 - pad_y):min(H_f, y2 + pad_y),
                               max(0, x1 - pad_x):min(W_f, x2 + pad_x)]
                if crop.size == 0:
                    continue

                # Context: highlight *this* detection with a thick border
                ctx = overlay.copy()
                hex_c = CLASS_COLOURS.get(meta["cls_id"], DEFAULT_COLOUR)
                r, g, b = (int(hex_c[i:i+2], 16) for i in (1, 3, 5))
                cv2.rectangle(ctx, (x1, y1), (x2, y2), (b, g, r), 4)

                stem = f"{video_stem}_f{frame_idx:06d}_d{det_idx:02d}"
                all_detections.append({
                    "meta"   : meta,
                    "crop"   : crop.copy(),
                    "ctx"    : ctx,
                    "stem"   : stem,
                })

        frame_idx += 1

    cap.release()
    total_dets = len(all_detections)
    print(f"Found {total_dets} qualifying detections.\n")

    if total_dets == 0:
        print("Nothing to label. Exiting.")
        return

    # ── Pass 2: interactive labelling ─────────────────────────────────────────
    labeller = Labeller()
    saved    = 0
    skipped  = 0
    i        = 0

    while i < total_dets:
        entry    = all_detections[i]
        meta     = entry["meta"]
        stem     = entry["stem"]
        cls_id   = meta["cls_id"]
        cls_name = TARGET_CLASSES.get(cls_id, "unknown")

        # Resume mode: silently skip already-labelled detections
        if args.skip_labelled and already_saved(stem, img_dir, lbl_dir):
            i += 1
            continue

        counter_txt = (
            f"det {i + 1} / {total_dets}   ·   "
            f"frame {meta['frame_idx']}   ·   "
            f"saved {saved}"
        )

        direction = labeller.ask(entry["crop"], meta, entry["ctx"], counter_txt)

        if labeller.quit_requested:
            print(f"\nQuit by operator.  Saved {saved}, skipped {skipped}.")
            return

        # ── Back: undo previous save and revisit it ───────────────────────────
        if direction == "back":
            if i > 0:
                i -= 1
                prev_stem = all_detections[i]["stem"]
                if already_saved(prev_stem, img_dir, lbl_dir):
                    delete_pair(prev_stem, img_dir, lbl_dir)
                    saved = max(0, saved - 1)
                    print(f"  ← Back  (undid {prev_stem})")
                else:
                    print(f"  ← Back  (to {prev_stem})")
            else:
                print("  ← Already at the first detection.")
            continue   # re-show detection at current i

        # ── Skip ──────────────────────────────────────────────────────────────
        if direction is None:
            skipped += 1
            i += 1
            continue

        # ── Label ─────────────────────────────────────────────────────────────
        label_id = LABEL_MAP.get((cls_id, direction))
        if label_id is None:
            print(f"  [WARN] No mapping for ({cls_name}, {direction}), skipping.")
            skipped += 1
            i += 1
            continue

        save_pair(stem, entry["crop"], label_id, img_dir, lbl_dir)
        saved += 1
        print(f"  Saved  {stem}.jpg  →  class {label_id}  ({cls_name}_{direction})")
        i += 1

    print(f"\nFinished.  Saved: {saved}  |  Skipped: {skipped}")
    print(f"Images : {img_dir.resolve()}")
    print(f"Labels : {lbl_dir.resolve()}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Detect vehicles in a video and label each crop as _away or _towards."
    )
    parser.add_argument("--video",  required=True, help="Path to input video file.")
    parser.add_argument("--output", default="dataset",
                        help="Output root directory (default: dataset/).")
    parser.add_argument("--model",  default="yolov8s.pt",
                        help="YOLO weights (default: yolov8s.pt, auto-downloaded).")
    parser.add_argument("--frame-skip", type=int,   default=FRAME_SKIP,
                        help=f"Process every Nth frame (default: {FRAME_SKIP}).")
    parser.add_argument("--min-height", type=int,   default=MIN_HEIGHT_PX,
                        help=f"Min bbox height in px (default: {MIN_HEIGHT_PX}).")
    parser.add_argument("--conf",       type=float, default=CONF_THRESHOLD,
                        help=f"YOLO confidence threshold (default: {CONF_THRESHOLD}).")
    parser.add_argument("--skip-labelled", action="store_true",
                        help="Skip detections already saved to disk (resume mode).")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()