"""Extract optimal confidence threshold from trained models.

Runs model.val() on each trained model and extracts the confidence
threshold that maximizes the F1 score from the F1-confidence curve.

Usage:
    python scripts/extract_conf.py

Outputs a JSON file with per-model optimal confidence values.
"""

import glob
import json
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from ultralytics import YOLO


def extract_optimal_conf():
    """Run validation on each trained model and extract optimal conf threshold."""
    results_dir = os.path.join(PROJECT_ROOT, "results")
    output_path = os.path.join(PROJECT_ROOT, "conf_thresholds.json")

    pattern = os.path.join(results_dir, "*", "*", "*", "train", "weights", "best.pt")
    weight_files = sorted(glob.glob(pattern))

    print(f"Found {len(weight_files)} trained models")

    conf_data = {}
    for wf in weight_files:
        # Extract model info from path
        # .../results/{device}/{experiment}/{model_folder}/train/weights/best.pt
        parts = wf.replace("\\", "/").split("/")
        idx = parts.index("results")
        device = parts[idx + 1]
        experiment = parts[idx + 2]
        model_folder = parts[idx + 3]

        key = f"{device}/{experiment}/{model_folder}"
        print(f"\nProcessing {key}...")

        try:
            model = YOLO(wf)

            # Get data.yaml path from args.yaml
            args_path = os.path.join(os.path.dirname(os.path.dirname(wf)), "args.yaml")
            data_yaml = None
            if os.path.exists(args_path):
                with open(args_path, "r") as f:
                    for line in f:
                        if line.startswith("data:"):
                            data_yaml = line.split(":", 1)[1].strip()
                            break

            if not data_yaml:
                print(f"  No data.yaml found, skipping")
                continue

            # Run validation
            metrics = model.val(data=data_yaml, verbose=False)

            # Extract F1 curve and optimal confidence
            # Box metrics
            box_f1 = None
            box_conf = None
            if hasattr(metrics, 'box'):
                box = metrics.box
                if hasattr(box, 'f1') and hasattr(box, 'conf'):
                    # f1 is array of shape (num_classes, num_conf_thresholds)
                    # conf is array of confidence thresholds
                    import numpy as np
                    mean_f1 = box.f1.mean(0)  # average across classes
                    best_idx = mean_f1.argmax()
                    box_f1 = float(mean_f1[best_idx])
                    box_conf = float(box.conf[best_idx]) if hasattr(box, 'conf') else None

            # Mask/segment metrics
            mask_f1 = None
            mask_conf = None
            if hasattr(metrics, 'seg'):
                seg = metrics.seg
                if hasattr(seg, 'f1') and hasattr(seg, 'conf'):
                    import numpy as np
                    mean_f1 = seg.f1.mean(0)
                    best_idx = mean_f1.argmax()
                    mask_f1 = float(mean_f1[best_idx])
                    mask_conf = float(seg.conf[best_idx]) if hasattr(seg, 'conf') else None

            conf_data[key] = {
                "box_f1": round(box_f1, 4) if box_f1 else None,
                "box_conf": round(box_conf, 4) if box_conf else None,
                "mask_f1": round(mask_f1, 4) if mask_f1 else None,
                "mask_conf": round(mask_conf, 4) if mask_conf else None,
            }
            print(f"  Box: F1={box_f1:.4f} @ conf={box_conf:.4f}" if box_f1 else "  Box: N/A")
            print(f"  Mask: F1={mask_f1:.4f} @ conf={mask_conf:.4f}" if mask_f1 else "  Mask: N/A")

            # Extract per-class metrics
            per_class_data = None
            precision_mean = 0.0
            recall_mean = 0.0
            if hasattr(metrics, "box") and hasattr(metrics.box, "ap_class_index"):
                class_names = metrics.names if hasattr(metrics, "names") else {}
                box = metrics.box
                p_cls        = box.p[:-1] if hasattr(box, "p") else []
                r_cls        = box.r[:-1] if hasattr(box, "r") else []
                map50_cls    = box.ap50   if hasattr(box, "ap50") else []
                map50_95_cls = box.ap     if hasattr(box, "ap") else []
                ap_class_index = box.ap_class_index if hasattr(box, "ap_class_index") else range(len(map50_cls))
                per_class_data = {}
                for i, cls_id in enumerate(ap_class_index):
                    name = class_names.get(int(cls_id), f"class_{cls_id}")
                    per_class_data[name] = {
                        "precision": float(p_cls[i]),
                        "recall": float(r_cls[i]),
                        "map50": float(map50_cls[i]),
                        "map50_95": float(map50_95_cls[i]),
                    }
                precision_mean = float(sum(p_cls) / len(p_cls)) if len(p_cls) > 0 else 0.0
                recall_mean = float(sum(r_cls) / len(r_cls)) if len(r_cls) > 0 else 0.0
                print(f"  Per-class: {len(per_class_data)} classes")

            # Update report.txt with missing metrics
            report_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(wf))),
                "report.txt"
            )
            if os.path.exists(report_path):
                with open(report_path, "r") as f:
                    content = f.read()

                updated = False
                # Update P/R mean if still zero
                if "P (mean):  0.0000" in content:
                    content = content.replace(
                        "P (mean):  0.0000",
                        f"P (mean):  {precision_mean:.4f}"
                    )
                    content = content.replace(
                        "R (mean):  0.0000",
                        f"R (mean):  {recall_mean:.4f}"
                    )
                    updated = True

                # Add conf/F1 if missing
                if "Best conf:" not in content:
                    lines = content.rstrip().split("\n")
                    if box_conf is not None:
                        lines.append(f"Best conf: {box_conf:.4f}")
                    if box_f1 is not None:
                        lines.append(f"Best F1:   {box_f1:.4f}")
                    content = "\n".join(lines) + "\n"
                    updated = True

                # Add per-class section if missing
                if per_class_data and "Per-class Accuracy" not in content:
                    lines = content.rstrip().split("\n")
                    lines.append("-" * 50)
                    lines.append("--- Per-class Accuracy ---")
                    lines.append(f"{'Class':10s}  {'P':>8s}  {'R':>8s}  {'mAP50':>8s}  {'mAP50-95':>8s}")
                    for cls_name, cm in per_class_data.items():
                        lines.append(
                            f"{cls_name:10s}  {cm['precision']:8.4f}  "
                            f"{cm['recall']:8.4f}  "
                            f"{cm['map50']:8.4f}  "
                            f"{cm['map50_95']:8.4f}"
                        )
                    content = "\n".join(lines) + "\n"
                    updated = True

                if updated:
                    with open(report_path, "w") as f:
                        f.write(content)
                    print(f"  Updated {report_path}")

        except Exception as e:
            print(f"  Error: {e}")
            continue

    # Save results
    with open(output_path, "w") as f:
        json.dump(conf_data, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    extract_optimal_conf()
