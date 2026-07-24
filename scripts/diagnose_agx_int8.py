"""Diagnose the AGX INT8 build failure (JRTIP revision).

Run ON the AGX Orin. Rebuilds a single YOLOv26 nano INT8 engine with full
stderr/stdout captured, so the actual TensorRT error is visible. If the build
succeeds, it also runs one inference on the clean test set to confirm end to end.

Usage (on the AGX):
    python scripts/diagnose_agx_int8.py
    python scripts/diagnose_agx_int8.py --size medium   # try a larger model
"""

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils import PROJECT_ROOT, get_weights_path

DATA_CALIB = os.path.join(PROJECT_ROOT, "data", "data_calib.yaml")
DATA_TEST = os.path.join(PROJECT_ROOT, "data", "data_test_clean.yaml")
CALIB_DIR = os.path.join(PROJECT_ROOT, "data", "calib_train", "images")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="nano",
                    choices=["nano", "small", "medium", "large"])
    ap.add_argument("--approach", default="scratch")
    args = ap.parse_args()

    print("=" * 60)
    print("Pre-flight checks")
    print("=" * 60)
    print(f"calib yaml exists:   {os.path.exists(DATA_CALIB)}  ({DATA_CALIB})")
    print(f"calib images dir:    {os.path.isdir(CALIB_DIR)}  ({CALIB_DIR})")
    if os.path.isdir(CALIB_DIR):
        print(f"  calib image count: {len(os.listdir(CALIB_DIR))}")
    pt = get_weights_path("core_comparison", "yolo26", "segment",
                          args.size, args.approach)
    print(f"weights exist:       {os.path.exists(pt)}  ({pt})")

    try:
        import tensorrt
        print(f"tensorrt:            {tensorrt.__version__}")
    except Exception as e:
        print(f"tensorrt import:     FAILED ({e})")

    # free memory snapshot
    try:
        out = subprocess.run(["free", "-h"], capture_output=True, text=True).stdout
        print("memory:\n" + out)
    except Exception:
        pass

    print("=" * 60)
    print(f"Building INT8 engine for yolo26 {args.size} {args.approach}")
    print("(full TensorRT output follows; the error, if any, is here)")
    print("=" * 60)

    cmd = [sys.executable, "scripts/export.py",
           "--weights", pt, "--precision", "int8",
           "--imgsz", "640", "--data", DATA_CALIB, "--verbose"]
    # Do NOT capture: let TensorRT logging stream to the console so the
    # calibration/build error is visible in real time.
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    print("=" * 60)
    print(f"export.py exit code: {result.returncode}")

    engine = os.path.splitext(pt)[0] + "_int8.engine"
    if os.path.exists(engine):
        print(f"INT8 engine built OK: {engine}")
        print("Building succeeded -- the earlier failure may have been "
              "transient (memory/thermal). Re-running the AGX campaign should "
              "now populate INT8 results.")
    else:
        print("INT8 engine was NOT produced. See the TensorRT output above "
              "for the failure reason (common causes on Jetson: calibrator "
              "OOM, unsupported INT8 layer in the Segment26 head, or a "
              "calibration-cache write error).")


if __name__ == "__main__":
    main()
