"""Qualitative before/after-TensorRT segmentation comparison (JRTIP revision, R3.3).

Runs the same model in PyTorch FP32 and in TensorRT (FP16, and INT8 if an engine
exists) on a few leak-free test images, and renders a grid so reviewers can see
that TensorRT optimization does not visibly degrade the masks:

    rows    = selected test images (one per interesting class)
    columns = [Original | PyTorch FP32 | TRT FP16 | (TRT INT8 if available)]

Must run ON a device that holds the .engine files (e.g. the RTX workstation or a
Jetson). Weights and engines are resolved from results/rtx5090/... by default.

Usage:
    python scripts/gen_qual_trt_figure.py                 # yolo26 medium pretrained
    python scripts/gen_qual_trt_figure.py --size nano --approach scratch
    python scripts/gen_qual_trt_figure.py --images a.jpg,b.jpg,c.jpg

Output:
    fig_qual_trt.pdf   (copy to the paper repo as paper/figures/fig_qual.pdf)
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils import PROJECT_ROOT, get_weights_path

TEST_IMAGES = os.path.join(PROJECT_ROOT, "data", "test_clean", "images")
FONT = "DejaVu Sans"
MASK_COLOR = np.array([0.16, 0.62, 0.45])   # green overlay
ALPHA = 0.45


def overlay(ax, img_rgb, result, title):
    ax.imshow(img_rgb)
    if result is not None and result.masks is not None:
        canvas = np.zeros((*img_rgb.shape[:2], 4), dtype=float)
        for m in result.masks.data.cpu().numpy():
            mm = m.astype(bool)
            # resize mask to image if needed
            if mm.shape != img_rgb.shape[:2]:
                mm = np.array(Image.fromarray(mm.astype(np.uint8) * 255)
                              .resize((img_rgb.shape[1], img_rgb.shape[0]))) > 127
            canvas[mm, :3] = MASK_COLOR
            canvas[mm, 3] = ALPHA
        ax.imshow(canvas)
    ax.set_title(title, fontsize=8, fontfamily=FONT)
    ax.axis("off")


def load_engine_or_none(pt_path, precision):
    engine = os.path.splitext(pt_path)[0] + f"_{precision}.engine"
    return engine if os.path.exists(engine) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="yolo26")
    ap.add_argument("--size", default="medium")
    ap.add_argument("--approach", default="pretrained")
    ap.add_argument("--images", default=None,
                    help="Comma-separated image filenames (default: first 3 test images)")
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "fig_qual_trt.pdf"))
    args = ap.parse_args()

    from ultralytics import YOLO

    pt_path = get_weights_path("core_comparison", args.arch, "segment",
                               args.size, args.approach)
    if not os.path.exists(pt_path):
        sys.exit(f"weights not found: {pt_path}")

    # Columns: always PyTorch FP32 + TRT FP16; add INT8 if the engine exists.
    fp16 = load_engine_or_none(pt_path, "fp16")
    int8 = load_engine_or_none(pt_path, "int8")
    cols = [("Original", None, None),
            ("PyTorch FP32", pt_path, "pytorch")]
    if fp16:
        cols.append(("TRT FP16", fp16, "trt"))
    if int8:
        cols.append(("TRT INT8", int8, "trt"))
    print(f"columns: {[c[0] for c in cols]}")

    if args.images:
        imgs = args.images.split(",")
    else:
        imgs = sorted(os.listdir(TEST_IMAGES))[:3]
    img_paths = [os.path.join(TEST_IMAGES, f) for f in imgs]

    # Pre-load models once
    models = {}
    for _, path, kind in cols:
        if path and path not in models:
            models[path] = YOLO(path, task="segment")

    nrow, ncol = len(img_paths), len(cols)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.3 * ncol, 2.3 * nrow))
    if nrow == 1:
        axes = axes[None, :]

    for r, ipath in enumerate(img_paths):
        img_rgb = np.array(Image.open(ipath).convert("RGB"))
        for c, (title, path, kind) in enumerate(cols):
            ax = axes[r, c]
            if path is None:
                overlay(ax, img_rgb, None, title if r == 0 else "")
            else:
                res = models[path].predict(ipath, imgsz=640, verbose=False)[0]
                overlay(ax, img_rgb, res, title if r == 0 else "")

    fig.suptitle(f"{args.arch} {args.size} {args.approach}: segmentation before/after "
                 f"TensorRT optimization", fontsize=9, fontfamily=FONT)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.out, format="pdf", bbox_inches="tight", dpi=300)
    fig.savefig(args.out.replace(".pdf", "_preview.png"), format="png",
                bbox_inches="tight", dpi=150)
    print(f"saved {args.out}")
    print("Copy to paper/figures/fig_qual.pdf")


if __name__ == "__main__":
    main()
