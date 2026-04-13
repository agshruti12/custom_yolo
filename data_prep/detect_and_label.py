"""
detect_and_label.py
────────────────────────────────────────────────────────────────────────────────
Plays a video through YOLOv8s frame-by-frame (every Nth frame).
For each qualifying detection (height >= MIN_HEIGHT_PX) a labelling window
pops up so the operator can assign a class with a single keypress.
The crop (JPG) and its YOLO label (TXT) are saved with matching filenames.

Output:
    <output>/
        images/   <stem>_f<frame>_d<det>.jpg
        labels/   <stem>_f<frame>_d<det>.txt   ← <class_id> 0.5 0.5 1.0 1.0

Usage:
    python detect_and_label.py --video highway.mp4
    python detect_and_label.py --video highway.mp4 --output my_dataset --frame-skip 3
    python detect_and_label.py --video highway.mp4 --skip-labelled   # resume session

Keys (labeller window):
    1–9 / 0   label with that class (auto-advances to next detection)
    s         skip this detection (no file saved)
    q         quit and save progress
────────────────────────────────────────────────────────────────────────────────
"""

import argparse

import torch
import cv2
import os
import sys
import tkinter as tk
from pathlib import Path
from PIL import Image, ImageTk, ImageDraw, ImageFont
from ultralytics import YOLO

# ── Tunables ─────────────────────────────────────────────────────────────────
MIN_HEIGHT_PX   = 100
FRAME_SKIP      = 5
CONF_THRESHOLD  = 0.40
JPEG_QUALITY    = 95
CROP_PADDING_PC = 0.05          # 5 % padding around the bbox

# COCO class indices → friendly name
TARGET_CLASSES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# ── Label taxonomy ────────────────────────────────────────────────────────────
# Edit freely.  Index = class_id written to the .txt file.
LABELS: list[str] = [
    "car_towards",        # 0
    "car_away",           # 1
    "truck_towards",      # 3
    "truck_away",         # 4
    "bus_towards",        # 6
    "bus_away",           # 7
    "motorcycle_towards", # 9
    "motorcycle_away",    # 10
    "bicycle_towards",    # 12
    "bicycle_away",       # 13
]

# ── Colour palette per YOLO class ─────────────────────────────────────────────
CLASS_COLOURS = {
    2: "#00d4ff",   # car
    7: "#ff6b35",   # truck
    5: "#a855f7",   # bus
    3: "#22c55e",   # motorcycle
    1: "#facc15",   # bicycle
}
DEFAULT_COLOUR = "#ffffff"

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_output_dirs(root: Path) -> tuple[Path, Path]:
    img_dir = root / "images"
    lbl_dir = root / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    return img_dir, lbl_dir


def already_saved(stem: str, img_dir: Path, lbl_dir: Path) -> bool:
    return (img_dir / f"{stem}.jpg").exists() and (lbl_dir / f"{stem}.txt").exists()


def save_pair(
    stem: str,
    crop_bgr,
    class_id: int,
    img_dir: Path,
    lbl_dir: Path,
):
    """Write crop JPG + YOLO label TXT with matching stem."""
    cv2.imwrite(str(img_dir / f"{stem}.jpg"), crop_bgr,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    (lbl_dir / f"{stem}.txt").write_text(f"{class_id} 0.5 0.5 1.0 1.0\n")


def draw_overlay(frame_bgr, boxes_meta: list[dict]) -> "np.ndarray":
    """Draw bounding boxes + labels on a copy of frame_bgr (for the video view)."""
    out = frame_bgr.copy()
    for m in boxes_meta:
        x1, y1, x2, y2 = m["xyxy"]
        colour_hex = CLASS_COLOURS.get(m["cls_id"], DEFAULT_COLOUR)
        r, g, b = int(colour_hex[1:3], 16), int(colour_hex[3:5], 16), int(colour_hex[5:7], 16)
        cv2.rectangle(out, (x1, y1), (x2, y2), (b, g, r), 2)
        label_text = f"{TARGET_CLASSES.get(m['cls_id'], '?')} {m['conf']:.2f}"
        cv2.putText(out, label_text, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (b, g, r), 1, cv2.LINE_AA)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Labeller GUI (one window, reused across all detections)
# ─────────────────────────────────────────────────────────────────────────────

class Labeller:
    """
    Blocking modal labeller.
    Call  labeller.ask(crop_bgr, meta)  → returns class_id (int) or None (skip/quit).
    After a quit the  .quit_requested  flag is True.
    """

    MAX_W = 640
    MAX_H = 560

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Vehicle Labeller")
        self.root.configure(bg="#0d0d0d")
        self.root.resizable(False, False)
        self._result: int | None = None
        self.quit_requested = False
        self._tk_img = None
        self._build_ui()
        self.root.withdraw()          # hide until first ask()

    # ── Build ─────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── top bar ──
        top = tk.Frame(self.root, bg="#0d0d0d")
        top.pack(fill=tk.X, padx=0, pady=0)

        self.title_lbl = tk.Label(
            top, text="VEHICLE LABELLER", bg="#0d0d0d", fg="#00d4ff",
            font=("Courier New", 11, "bold"), anchor="w", padx=14, pady=8,
        )
        self.title_lbl.pack(side=tk.LEFT)

        self.counter_lbl = tk.Label(
            top, text="", bg="#0d0d0d", fg="#555555",
            font=("Courier New", 10), anchor="e", padx=14,
        )
        self.counter_lbl.pack(side=tk.RIGHT)

        # ── image area ──
        self.canvas = tk.Canvas(
            self.root, bg="#111111", highlightthickness=0,
            width=self.MAX_W, height=self.MAX_H,
        )
        self.canvas.pack(padx=0, pady=0)

        # ── detection info strip ──
        self.info_lbl = tk.Label(
            self.root, text="", bg="#111111", fg="#888888",
            font=("Courier New", 9), pady=4,
        )
        self.info_lbl.pack(fill=tk.X)

        # ── divider ──
        tk.Frame(self.root, bg="#1e1e1e", height=1).pack(fill=tk.X)

        # ── label buttons (2-column grid) ──
        btn_frame = tk.Frame(self.root, bg="#0d0d0d")
        btn_frame.pack(fill=tk.X, padx=12, pady=10)

        self._buttons: list[tk.Button] = []
        cols = 2
        for i, lbl in enumerate(LABELS):
            key_char = str(i + 1) if i < 9 else ("0" if i == 9 else "·")
            colour   = self._label_colour(lbl)
            btn = tk.Button(
                btn_frame,
                text=f"  [{key_char}]  {lbl}",
                bg="#1a1a1a", fg=colour,
                activebackground="#2a2a2a", activeforeground=colour,
                font=("Courier New", 10, "bold"),
                anchor="w", relief=tk.FLAT,
                padx=10, pady=5,
                bd=0, highlightthickness=1,
                highlightbackground="#2a2a2a",
                command=lambda idx=i: self._select(idx),
            )
            btn.grid(
                in_=btn_frame,
                row=i // cols, column=i % cols,
                sticky="ew", padx=4, pady=2,
            )
            self._buttons.append(btn)

        for c in range(cols):
            btn_frame.columnconfigure(c, weight=1)

        # ── bottom controls ──
        foot = tk.Frame(self.root, bg="#0d0d0d")
        foot.pack(fill=tk.X, padx=12, pady=(0, 10))

        tk.Button(
            foot, text="[s]  Skip detection", bg="#1a1a1a", fg="#666666",
            activebackground="#2a2a2a", activeforeground="#999999",
            font=("Courier New", 10), relief=tk.FLAT, padx=10, pady=5,
            command=self._skip,
        ).pack(side=tk.LEFT, padx=(0, 6))

        tk.Button(
            foot, text="[q]  Quit & save", bg="#1a1a1a", fg="#cc3333",
            activebackground="#2a2a2a", activeforeground="#ff4444",
            font=("Courier New", 10), relief=tk.FLAT, padx=10, pady=5,
            command=self._quit,
        ).pack(side=tk.RIGHT)

        self.status_lbl = tk.Label(
            foot, text="", bg="#0d0d0d", fg="#00d4ff",
            font=("Courier New", 10, "bold"),
        )
        self.status_lbl.pack(side=tk.LEFT, padx=10)

        # ── key bindings ──
        self.root.bind("s", lambda _: self._skip())
        self.root.bind("q", lambda _: self._quit())
        for i in range(min(9, len(LABELS))):
            self.root.bind(str(i + 1), lambda _, idx=i: self._select(idx))
        if len(LABELS) >= 10:
            self.root.bind("0", lambda _: self._select(9))

        self.root.protocol("WM_DELETE_WINDOW", self._quit)

    @staticmethod
    def _label_colour(lbl: str) -> str:
        if "car"        in lbl: return "#00d4ff"
        if "truck"      in lbl: return "#ff6b35"
        if "bus"        in lbl: return "#a855f7"
        if "motorcycle" in lbl: return "#22c55e"
        if "bicycle"    in lbl: return "#facc15"
        return "#ffffff"

    # ── Public API ────────────────────────────────────────────────────────────

    def ask(
        self,
        crop_bgr,
        meta: dict,
        frame_overlay_bgr=None,
        counter_text: str = "",
    ) -> int | None:
        """
        Show the crop (+ optional frame context) and block until the operator
        presses a key.  Returns the class_id or None (skip).
        Sets self.quit_requested = True on quit.
        """
        self._result = None
        self._display_image(crop_bgr, frame_overlay_bgr)
        self._reset_buttons()

        cls_name = TARGET_CLASSES.get(meta["cls_id"], "unknown")
        bh = meta["xyxy"][3] - meta["xyxy"][1]
        bw = meta["xyxy"][2] - meta["xyxy"][0]
        self.info_lbl.config(
            text=f"  YOLO: {cls_name}  |  conf {meta['conf']:.2f}"
                 f"  |  box {bw}×{bh}px  |  frame {meta['frame_idx']}"
        )
        self.counter_lbl.config(text=counter_text)
        self.status_lbl.config(text="")

        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()
        self.root.wait_variable(self._done_var())   # blocks
        return self._result

    # ── Internal ──────────────────────────────────────────────────────────────

    def _done_var(self):
        """Create a fresh IntVar used as a blocking sentinel."""
        self._var = tk.IntVar(value=0)
        return self._var

    def _unblock(self):
        if hasattr(self, "_var"):
            self._var.set(1)

    def _select(self, class_id: int):
        self._result = class_id
        self._highlight(class_id)
        self.status_lbl.config(text=f"✓  {LABELS[class_id]}")
        self.root.after(220, self._unblock)   # brief flash before advancing

    def _skip(self):
        self._result = None
        self.status_lbl.config(text="Skipped.")
        self._unblock()

    def _quit(self):
        self.quit_requested = True
        self._result = None
        self._unblock()

    def _highlight(self, class_id: int):
        colour = self._label_colour(LABELS[class_id])
        for i, btn in enumerate(self._buttons):
            if i == class_id:
                btn.config(bg=colour, fg="#000000")
            else:
                btn.config(bg="#1a1a1a", fg=self._label_colour(LABELS[i]))

    def _reset_buttons(self):
        for i, btn in enumerate(self._buttons):
            btn.config(bg="#1a1a1a", fg=self._label_colour(LABELS[i]))

    def _display_image(self, crop_bgr, context_bgr=None):
        """
        If context_bgr is provided, show [context | crop] side-by-side,
        otherwise fill the canvas with the crop.
        """
        def bgr_to_pil(img_bgr):
            return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

        W, H = self.MAX_W, self.MAX_H

        if context_bgr is not None:
            # left panel: context frame  (60 % width)
            ctx_w = int(W * 0.60)
            ctx_h = H
            ctx_pil = bgr_to_pil(context_bgr)
            ctx_pil.thumbnail((ctx_w, ctx_h), Image.LANCZOS)
            # right panel: crop  (40 % width)
            crp_w = W - ctx_w - 2   # 2px divider
            crp_h = H
            crp_pil = bgr_to_pil(crop_bgr)
            crp_pil.thumbnail((crp_w, crp_h), Image.LANCZOS)

            canvas_img = Image.new("RGB", (W, H), (17, 17, 17))
            # paste context (centred vertically)
            cx_off = (ctx_w - ctx_pil.width)  // 2
            cy_off = (ctx_h - ctx_pil.height) // 2
            canvas_img.paste(ctx_pil, (cx_off, cy_off))
            # divider
            draw = ImageDraw.Draw(canvas_img)
            draw.rectangle([ctx_w, 0, ctx_w + 1, H], fill=(40, 40, 40))
            # paste crop (centred vertically)
            rx_off = ctx_w + 2 + (crp_w - crp_pil.width)  // 2
            ry_off = (crp_h - crp_pil.height) // 2
            canvas_img.paste(crp_pil, (rx_off, ry_off))
        else:
            crp_pil = bgr_to_pil(crop_bgr)
            crp_pil.thumbnail((W, H), Image.LANCZOS)
            canvas_img = Image.new("RGB", (W, H), (17, 17, 17))
            ox = (W - crp_pil.width) // 2
            oy = (H - crp_pil.height) // 2
            canvas_img.paste(crp_pil, (ox, oy))

        self._tk_img = ImageTk.PhotoImage(canvas_img)
        self.canvas.config(width=W, height=H)
        self.canvas.create_image(0, 0, anchor="nw", image=self._tk_img)


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    video_path  = Path(args.video)
    output_root = Path(args.output)
    img_dir, lbl_dir = make_output_dirs(output_root)
    video_stem  = video_path.stem

    print(f"Loading YOLO model: {args.model}")
    model = YOLO(args.model)
    
    # use cuda or mps
    if torch.cuda.is_available():
        model.to("cuda")
    elif torch.backends.mps.is_available():
        model.to("mps")


    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25
    print(f"Video : {video_path.name}  ({total_frames} frames @ {fps:.1f} fps)")
    print(f"Output: {output_root.resolve()}\n")

    labeller  = Labeller()
    saved     = 0
    skipped   = 0
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % args.frame_skip != 0:
                frame_idx += 1
                continue

            # ── YOLO inference ────────────────────────────────────────────────
            results = model(
                frame,
                classes=list(TARGET_CLASSES.keys()),
                conf=args.conf,
                verbose=False,
            )[0]

            # Build metadata list for qualifying detections
            detections = []
            for box in results.boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                bh = y2 - y1
                if bh < args.min_height:
                    continue
                detections.append({
                    "cls_id"   : cls_id,
                    "conf"     : float(box.conf[0]),
                    "xyxy"     : (x1, y1, x2, y2),
                    "frame_idx": frame_idx,
                })

            if not detections:
                frame_idx += 1
                continue

            # Draw all boxes on a context copy of the frame
            overlay = draw_overlay(frame, detections)
            H_f, W_f = frame.shape[:2]

            for det_idx, meta in enumerate(detections):
                stem = f"{video_stem}_f{frame_idx:06d}_d{det_idx:02d}"

                # Skip already-labelled if resuming
                if args.skip_labelled and already_saved(stem, img_dir, lbl_dir):
                    continue

                x1, y1, x2, y2 = meta["xyxy"]
                bw = x2 - x1
                bh = y2 - y1
                pad_x = max(4, int(bw * CROP_PADDING_PC))
                pad_y = max(4, int(bh * CROP_PADDING_PC))
                cx1 = max(0, x1 - pad_x)
                cy1 = max(0, y1 - pad_y)
                cx2 = min(W_f, x2 + pad_x)
                cy2 = min(H_f, y2 + pad_y)
                crop = frame[cy1:cy2, cx1:cx2]
                if crop.size == 0:
                    continue

                # Highlight the *current* detection on the context overlay
                ctx = overlay.copy()
                r_hex = CLASS_COLOURS.get(meta["cls_id"], "#ffffff")
                r, g, b = (int(r_hex[i:i+2], 16) for i in (1, 3, 5))
                cv2.rectangle(ctx, (x1, y1), (x2, y2), (b, g, r), 4)

                counter_txt = (
                    f"det {det_idx + 1}/{len(detections)}  ·  "
                    f"frame {frame_idx}/{total_frames}  ·  "
                    f"saved {saved}"
                )
                class_id = labeller.ask(crop, meta, ctx, counter_txt)

                if labeller.quit_requested:
                    print(f"\nQuit by operator.  Saved {saved}, skipped {skipped}.")
                    cap.release()
                    return

                if class_id is None:
                    skipped += 1
                    continue

                save_pair(stem, crop, class_id, img_dir, lbl_dir)
                saved += 1
                print(f"  Saved  {stem}.jpg  →  class {class_id} ({LABELS[class_id]})")

            frame_idx += 1

    finally:
        cap.release()

    print(f"\nFinished.  Saved: {saved}  |  Skipped: {skipped}")
    print(f"Images : {img_dir.resolve()}")
    print(f"Labels : {lbl_dir.resolve()}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Stream YOLOv8s detections from a video and label them on-the-fly."
    )
    parser.add_argument("--video",  required=True, help="Path to input video file.")
    parser.add_argument("--output", default="dataset",
                        help="Output root directory (default: dataset/).")
    parser.add_argument("--model",  default="yolov8s.pt",
                        help="YOLO weights (default: yolov8s.pt, auto-downloaded).")
    parser.add_argument("--frame-skip", type=int,   default=FRAME_SKIP,
                        help=f"Label every Nth frame (default: {FRAME_SKIP}).")
    parser.add_argument("--min-height", type=int,   default=MIN_HEIGHT_PX,
                        help=f"Min bbox height in px (default: {MIN_HEIGHT_PX}).")
    parser.add_argument("--conf",       type=float, default=CONF_THRESHOLD,
                        help=f"YOLO confidence threshold (default: {CONF_THRESHOLD}).")
    parser.add_argument("--skip-labelled", action="store_true",
                        help="Skip detections that already have a saved label (resume mode).")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
