"""Export PyTorch weights to TensorRT engine.

Must be run on the target device (TensorRT engines are GPU-architecture specific).

Usage:
    python scripts/export.py --weights path/to/best.pt --precision fp16
    python scripts/export.py --weights path/to/best.pt --precision int8 --data data/data.yaml
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO
from scripts.utils import get_data_yaml_path


def export_model(weights_path, precision, imgsz=640):
    """Export a PyTorch model to TensorRT format.

    Args:
        weights_path: Path to the .pt weights file.
        precision: 'fp16' or 'int8'.
        imgsz: Input image size for the engine.

    Returns:
        Path to the exported .engine file.
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    print(f"Exporting {weights_path} to TensorRT {precision.upper()} (imgsz={imgsz})")

    model = YOLO(weights_path)

    export_args = {
        "format": "engine",
        "imgsz": imgsz,
    }

    if precision == "fp16":
        export_args["half"] = True
    elif precision == "int8":
        export_args["int8"] = True
        export_args["data"] = get_data_yaml_path()

    engine_path = model.export(**export_args)
    print(f"Export complete: {engine_path}")
    return engine_path


def main():
    parser = argparse.ArgumentParser(description="Export YOLO model to TensorRT")
    parser.add_argument("--weights", required=True, help="Path to .pt weights")
    parser.add_argument("--precision", required=True, choices=["fp16", "int8"],
                        help="TensorRT precision")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    args = parser.parse_args()

    export_model(args.weights, args.precision, args.imgsz)


if __name__ == "__main__":
    main()
