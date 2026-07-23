"""Export PyTorch weights to TensorRT engine.

Must be run on the target device (TensorRT engines are GPU-architecture specific).

Usage:
    python scripts/export.py --weights path/to/best.pt --precision fp16
    python scripts/export.py --weights path/to/best.pt --precision int8 --data data/data_calib.yaml

INT8 calibration note (JRTIP revision, R4.1/R4.4): pass --data data/data_calib.yaml
so calibration draws from the train-derived subset (data/calib_train), never from
the evaluation split. --verbose captures the TensorRT build log (per-layer
precisions) next to the engine file for the INT8-execution audit.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO
from scripts.utils import get_data_yaml_path


def export_model(weights_path, precision, imgsz=640, data_yaml=None, verbose=False):
    """Export a PyTorch model to TensorRT format.

    Args:
        weights_path: Path to the .pt weights file.
        precision: 'fp16' or 'int8'.
        imgsz: Input image size for the engine.
        data_yaml: Data yaml for INT8 calibration (default: data/data.yaml).
        verbose: Capture the TensorRT build log (per-layer precisions) to
            <engine>.buildlog.txt.

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
        "verbose": verbose,
    }

    if precision == "fp16":
        export_args["half"] = True
    elif precision == "int8":
        export_args["int8"] = True
        export_args["data"] = data_yaml or get_data_yaml_path()
        print(f"INT8 calibration data: {export_args['data']}")

    if verbose:
        # Route TensorRT logger output to a file for per-layer precision audit.
        import logging
        log_path = os.path.splitext(weights_path)[0] + f".{precision}.buildlog.txt"
        handler = logging.FileHandler(log_path, mode="w")
        handler.setLevel(logging.DEBUG)
        logging.getLogger().addHandler(handler)
        print(f"Build log: {log_path}")

    engine_path = model.export(**export_args)

    # Rename to a precision-distinct filename. Ultralytics always writes
    # best.engine regardless of precision, which caused the original campaign
    # to silently reuse the FP16 engine for INT8 runs (export skipped as
    # "already exported"). best_fp16.engine / best_int8.engine are unambiguous.
    distinct_path = os.path.splitext(weights_path)[0] + f"_{precision}.engine"
    if os.path.abspath(engine_path) != os.path.abspath(distinct_path):
        if os.path.exists(distinct_path):
            os.remove(distinct_path)
        os.replace(engine_path, distinct_path)
        engine_path = distinct_path

    print(f"Export complete: {engine_path}")
    return engine_path


def main():
    parser = argparse.ArgumentParser(description="Export YOLO model to TensorRT")
    parser.add_argument("--weights", required=True, help="Path to .pt weights")
    parser.add_argument("--precision", required=True, choices=["fp16", "int8"],
                        help="TensorRT precision")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--data", default=None,
                        help="Data yaml for INT8 calibration "
                             "(use data/data_calib.yaml for the revision)")
    parser.add_argument("--verbose", action="store_true",
                        help="Capture TensorRT build log for per-layer precision audit")
    args = parser.parse_args()

    export_model(args.weights, args.precision, args.imgsz,
                 data_yaml=args.data, verbose=args.verbose)


if __name__ == "__main__":
    main()
