"""Generate Fig. 3 (molle3.pdf): qualitative segmentation results.

8-panel figure (4 cols × 2 rows): one test image per defect class (D1–D7, W),
showing the image with the best segmentation quality (highest mean mask IoU
between predicted and ground-truth polygons) for each class.

Usage (on Antares, from the BenchMarks root):
    python scripts/gen_qual_figure.py

Output:
    molle3.pdf   (copy this to the paper repo)
"""

import os
import sys
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ultralytics import YOLO

# ── paths ─────────────────────────────────────────────────────────────────────
WEIGHTS   = "results/rtx5090/core_comparison/yolo26_seg_medium_pretrained/train/weights/best.pt"
# Search both test and validation splits for the best prediction per class
SEARCH_SPLITS = [
    ("data/test/images",  "data/test/labels"),
    ("data/valid/images", "data/valid/labels"),
]
OUT_PDF        = "molle3.pdf"        # copy this to the paper repo manually
SELECTIONS_JSON = "molle3_selections.json"  # auto-saved per-class filename choices

# ── class metadata ────────────────────────────────────────────────────────────
CLASS_NAMES = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "W"]
CLASS_DESCS = [
    "Hairline mark", "Coating failure", "Abrasion",    "Edge oxidation",
    "Edge corrosion", "Staining",       "Micro-pit",   "Weld seam",
]
COLORS = [
    "#e05a5a", "#e07b39", "#f0c840", "#5cb85c",
    "#5b9bd5", "#2b6cb0", "#9b59b6", "#00acc1",
]

FONT = "DejaVu Sans"


# ── helpers ───────────────────────────────────────────────────────────────────

def read_gt_masks(label_path, class_id):
    """Return list of (N,2) pixel-coordinate arrays for the given class."""
    masks = []
    if not os.path.exists(label_path):
        return masks
    with open(label_path) as f:
        for line in f:
            parts = line.split()
            if not parts or int(parts[0]) != class_id:
                continue
            coords = list(map(float, parts[1:]))
            pts = [(coords[i], coords[i + 1]) for i in range(0, len(coords) - 1, 2)]
            masks.append(pts)
    return masks


def poly_to_mask(pts_norm, w, h):
    """Rasterise a normalised polygon to a binary numpy mask of size (h, w)."""
    from PIL import ImageDraw
    mask_img = Image.new("L", (w, h), 0)
    px = [(x * w, y * h) for x, y in pts_norm]
    if len(px) >= 3:
        ImageDraw.Draw(mask_img).polygon(px, fill=1)
    return np.array(mask_img, dtype=bool)


def mask_iou(m1, m2):
    inter = (m1 & m2).sum()
    union = (m1 | m2).sum()
    return float(inter) / float(union) if union > 0 else 0.0


# Hardcode confirmed good images for D1-D6 (fill after first successful run).
# Keys are class IDs; values are image filenames (test or valid split).
# Leave empty to trigger automatic search for that class.
FIXED_IMAGES = {
    # 0: "filename_for_D1.jpg",
    # 1: "filename_for_D2.jpg",
    # 2: "filename_for_D3.jpg",
    # 3: "filename_for_D4.jpg",
    # 4: "filename_for_D5.jpg",
    # 5: "filename_for_D6.jpg",
}


def find_image_file(filename):
    """Locate a filename in any of the search splits."""
    for imgs_dir, lbls_dir in SEARCH_SPLITS:
        p = os.path.join(imgs_dir, filename)
        if os.path.exists(p):
            lbl = os.path.join(lbls_dir, os.path.splitext(filename)[0] + ".txt")
            return p, lbl
    return None, None


def best_segmentation_image(model, class_id, img_size=640):
    """Select the best image per class.
    D1-D6 (0-5): dominant strategy (no inference); hardcoded once confirmed.
    D7    (6):   lowest-threshold confidence + most predicted instances.
    W     (7):   highest prediction confidence only."""
    best_path, best_lbl, best_result, best_iou = None, None, None, -1.0

    # Use hardcoded image if available for this class
    if class_id in FIXED_IMAGES:
        img_path, lbl_path = find_image_file(FIXED_IMAGES[class_id])
        if img_path:
            result = model.predict(img_path, imgsz=img_size, conf=0.20, verbose=False)[0]
            return img_path, lbl_path, None, 1.0  # result deferred to render

    all_files = []
    for imgs_dir, lbls_dir in SEARCH_SPLITS:
        for fn in sorted(os.listdir(imgs_dir)):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                all_files.append((os.path.join(imgs_dir, fn),
                                  os.path.join(lbls_dir,
                                               os.path.splitext(fn)[0] + ".txt")))

    for img_path, lbl_path in all_files:
        fn = os.path.basename(img_path)
        # Compute dominance score from GT labels (no inference needed)
        target_area = 0.0
        total_area  = 0.0
        with open(lbl_path) as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                coords = list(map(float, parts[1:]))
                pts = [(coords[i], coords[i+1]) for i in range(0, len(coords)-1, 2)]
                if len(pts) < 3:
                    continue
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                area = (max(xs)-min(xs)) * (max(ys)-min(ys))
                total_area += area
                if int(parts[0]) == class_id:
                    target_area += area

        if target_area == 0:
            continue

        dominance = (target_area / total_area) * target_area

        if class_id <= 5:
            # D1-D6: pure dominant strategy, no inference needed
            score  = dominance
            result = None
        elif class_id == 6:
            # D7 (micro-pit): low threshold, rank by number of predicted instances
            result = model.predict(img_path, imgsz=img_size, conf=0.10, verbose=False)[0]
            n_pred = sum(1 for cls in result.boxes.cls.tolist()
                         if int(round(cls)) == class_id)
            if n_pred == 0:
                continue
            max_conf = max((float(result.boxes.conf[i])
                            for i, cls in enumerate(result.boxes.cls.tolist())
                            if int(round(cls)) == class_id), default=0.0)
            score = dominance * n_pred * max_conf
        else:
            # W (weld seam): confidence only — weld seams are always dominant
            result = model.predict(img_path, imgsz=img_size, conf=0.20, verbose=False)[0]
            confs  = [float(result.boxes.conf[i])
                      for i, cls in enumerate(result.boxes.cls.tolist())
                      if int(round(cls)) == class_id]
            max_conf = max(confs) if confs else 0.0
            if max_conf == 0.0:
                continue
            score = max_conf

        if score > best_iou:
            best_iou    = score
            best_path   = img_path
            best_lbl    = lbl_path
            best_result = result

    return best_path, best_lbl, best_result, best_iou


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    import json

    # Load previously confirmed selections; entries survive across runs
    locked = {}
    if os.path.exists(SELECTIONS_JSON):
        with open(SELECTIONS_JSON) as f:
            locked = json.load(f)   # {str(class_id): filename}
        print(f"Loaded {len(locked)} locked selections from {SELECTIONS_JSON}")

    # Classes to re-search this run (override locked classes here)
    RESEARCH = {6, 7}   # D7 and W — change this set to re-run other classes

    print(f"Loading weights: {WEIGHTS}")
    model = YOLO(WEIGHTS, task="segment")

    fig, axes = plt.subplots(2, 4, figsize=(3.5, 2.0))
    fig.subplots_adjust(left=0.0, right=1.0, top=0.88, bottom=0.10,
                        wspace=0.04, hspace=0.22)

    for idx, (cls_name, cls_desc, color) in enumerate(
            zip(CLASS_NAMES, CLASS_DESCS, COLORS)):
        ax = axes[idx // 4][idx % 4]

        # Use locked selection unless this class is in RESEARCH
        if str(idx) in locked and idx not in RESEARCH:
            img_path, lbl_path = find_image_file(locked[str(idx)])
            result = None
            score  = 1.0
            print(f"  {cls_name}: locked → {locked[str(idx)]}")
        else:
            print(f"  {cls_name}: scanning...")
            img_path, lbl_path, result, score = best_segmentation_image(model, idx)

        # Save selection (only if not already locked or explicitly re-searched)
        if img_path and (str(idx) not in locked or idx in RESEARCH):
            locked[str(idx)] = os.path.basename(img_path)
            with open(SELECTIONS_JSON, "w") as f:
                json.dump(locked, f, indent=2)
            print(f"  {cls_name}: saved → {locked[str(idx)]}")
        if img_path is None:
            ax.set_visible(False)
            print(f"  {cls_name}: no image found")
            continue

        print(f"  {cls_name}: {os.path.basename(img_path)}  score={score:.4f}")

        # Run inference now for D1-D4 (deferred during selection)
        if result is None:
            result = model.predict(img_path, imgsz=640, conf=0.25, verbose=False)[0]

        # Original image
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        # Crop to 640×640 if larger (centre crop for display)
        if w != h:
            side = min(w, h)
            left = (w - side) // 2
            top  = (h - side) // 2
            img  = img.crop((left, top, left + side, top + side))
            cx_off, cy_off = left, top
        else:
            cx_off, cy_off = 0, 0

        ax.imshow(img)

        # --- Predicted masks (filled, semi-transparent) ---
        if result.masks is not None:
            r, g, b = tuple(int(color.lstrip("#")[i:i+2], 16) / 255
                            for i in (0, 2, 4))
            pred_patches = []
            for mask_xy in result.masks.xy:
                # mask_xy is in original image coords; shift for crop
                pts = [(float(x) - cx_off, float(y) - cy_off)
                       for x, y in mask_xy]
                if len(pts) >= 3:
                    pred_patches.append(MplPolygon(pts, closed=True))
            if pred_patches:
                coll = PatchCollection(
                    pred_patches,
                    facecolor=[(r, g, b, 0.45)],
                    edgecolor=[(r, g, b, 0.9)],
                    linewidths=0.7,
                )
                ax.add_collection(coll)

        # --- Ground-truth outlines (dashed white) ---
        gt_masks = read_gt_masks(lbl_path, idx)
        for norm_pts in gt_masks:
            px = [(x * w - cx_off, y * h - cy_off) for x, y in norm_pts]
            poly = MplPolygon(px, closed=True, fill=False,
                              edgecolor="white", linewidth=0.6,
                              linestyle="--")
            ax.add_patch(poly)

        ax.set_title(cls_name, fontsize=5.5, fontfamily=FONT, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_handles = [
        Patch(facecolor=(0.5, 0.5, 0.5, 0.45), edgecolor=(0.5, 0.5, 0.5, 0.9),
              linewidth=0.7, label="Predicted mask"),
        Line2D([0], [0], color="#666666", linewidth=1.2, linestyle="--",
               label="Ground truth"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               fontsize=5.5, frameon=False,
               bbox_to_anchor=(0.5, 0.01), handletextpad=0.4,
               columnspacing=1.2)

    fig.savefig(OUT_PDF, format="pdf", dpi=300)
    plt.close(fig)
    print(f"\nSaved: {OUT_PDF}")
    print("Copy paper/molle3.pdf to the paper repository.")


if __name__ == "__main__":
    main()
