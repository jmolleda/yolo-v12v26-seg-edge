"""Measure board power (watts) for YOLOv26 pretrained on Jetson Orin Nano.

Runs predict() in a loop for MEASURE_SECONDS while tegrastats samples power,
then patches the watts and fps_per_watt columns in benchmark_results.csv.

Run as root (tegrastats requires it on fresh JetPack):
    sudo python scripts/measure_power_nano.py
"""

import csv
import glob
import os
import re
import statistics
import subprocess
import sys
import threading
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NANO_CSV      = os.path.join(PROJECT_ROOT, "results", "jetson_nano", "benchmark_results.csv")
VAL_IMAGES    = os.path.join(PROJECT_ROOT, "data", "valid", "images")
MEASURE_SECS  = 120   # seconds of sustained inference per configuration
WARMUP_RUNS   = 3


# ── configs ───────────────────────────────────────────────────────────────────

CONFIGS = [
    # (size,     format,      precision)
    ("nano",   "pytorch",   "fp32"),
    ("small",  "pytorch",   "fp32"),
    ("medium", "pytorch",   "fp32"),
    ("large",  "pytorch",   "fp32"),
    ("nano",   "tensorrt",  "fp16"),
    ("small",  "tensorrt",  "fp16"),
    ("medium", "tensorrt",  "fp16"),
    ("large",  "tensorrt",  "fp16"),
    ("nano",   "tensorrt",  "int8"),
    ("small",  "tensorrt",  "int8"),
    ("medium", "tensorrt",  "int8"),
    ("large",  "tensorrt",  "int8"),
]


# ── TegrastatsReader (same as infer.py) ───────────────────────────────────────

class TegrastatsReader:
    # Orin Nano: VDD_IN 4581mW/4581mW  (current/average — take first value)
    # AGX Orin:  VIN_SYS_5V0 4581mW | fallback VDD_GPU_SOC + VDD_CPU_CV
    _PATTERN = re.compile(
        r'VDD_IN\s+(\d+)mW'
        r'|VIN_SYS_5V0\s+(\d+)mW'
        r'|VDD_GPU_SOC\s+(\d+)mW'
        r'|VDD_CPU_CV\s+(\d+)mW'
    )

    def __init__(self, interval_ms=500):
        self._interval_ms = interval_ms
        self._samples = []
        self._stop = threading.Event()
        self._thread = None
        self._proc = None

    def start(self):
        self._stop.clear()
        self._samples = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._proc:
            self._proc.terminate()
        if self._thread:
            self._thread.join(timeout=5)

    def mean_watts(self):
        return statistics.mean(self._samples) / 1000.0 if self._samples else None

    def n_samples(self):
        return len(self._samples)

    def _run(self):
        try:
            self._proc = subprocess.Popen(
                ["tegrastats", "--interval", str(self._interval_ms)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, bufsize=1
            )
            for line in self._proc.stdout:
                if self._stop.is_set():
                    break
                vdd_in = vin = gpu_soc = cpu_cv = None
                for m in self._PATTERN.finditer(line):
                    if m.group(1) is not None:
                        vdd_in = int(m.group(1))
                    elif m.group(2) is not None:
                        vin = int(m.group(2))
                    elif m.group(3) is not None:
                        gpu_soc = int(m.group(3))
                    elif m.group(4) is not None:
                        cpu_cv = int(m.group(4))
                if vdd_in is not None:
                    self._samples.append(vdd_in)
                elif vin is not None:
                    self._samples.append(vin)
                elif gpu_soc is not None or cpu_cv is not None:
                    self._samples.append((gpu_soc or 0) + (cpu_cv or 0))
        except (FileNotFoundError, OSError):
            pass


# ── weights lookup ────────────────────────────────────────────────────────────

def find_weights(size, fmt):
    """Return path to best.pt or best.engine for yolo26 pretrained core_comparison."""
    folder = os.path.join(PROJECT_ROOT, "results", "rtx5090",
                          "core_comparison", f"yolo26_seg_{size}_pretrained")
    train_dirs = sorted(glob.glob(os.path.join(folder, "train*")))
    if not train_dirs:
        raise FileNotFoundError(f"No train* dir in {folder}")
    weights_dir = os.path.join(train_dirs[-1], "weights")
    ext = ".engine" if fmt == "tensorrt" else ".pt"
    path = os.path.join(weights_dir, f"best{ext}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Weights not found: {path}")
    return path


# ── val image ─────────────────────────────────────────────────────────────────

def find_val_image():
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        found = glob.glob(os.path.join(VAL_IMAGES, ext))
        if found:
            return found[0]
    raise FileNotFoundError(f"No images in {VAL_IMAGES}")


# ── measure one config ────────────────────────────────────────────────────────

def measure(size, fmt, precision, val_image):
    from ultralytics import YOLO

    weights = find_weights(size, fmt)
    task = "segment"

    print(f"\n{'='*60}")
    print(f"  {size:8s} | {fmt:10s} | {precision}")
    print(f"  weights: {weights}")
    print(f"{'='*60}")

    model = YOLO(weights, task=task)

    predict_kwargs = dict(source=val_image, imgsz=640, verbose=False, save=False)

    print(f"  Warm-up ({WARMUP_RUNS} runs)...")
    for _ in range(WARMUP_RUNS):
        model.predict(**predict_kwargs)

    print(f"  Measuring power for {MEASURE_SECS}s...")
    reader = TegrastatsReader(interval_ms=500)
    reader.start()

    t_end = time.time() + MEASURE_SECS
    n_infer = 0
    while time.time() < t_end:
        model.predict(**predict_kwargs)
        n_infer += 1

    reader.stop()
    watts = reader.mean_watts()
    n_samples = reader.n_samples()

    if watts is None:
        print("  WARNING: no tegrastats samples collected — is tegrastats available?")
        return None

    print(f"  Done. {n_infer} inferences, {n_samples} power samples, mean = {watts:.2f} W")
    return watts


# ── CSV patch ─────────────────────────────────────────────────────────────────

def patch_csv(results):
    """Write watts and fps_per_watt into matching rows of benchmark_results.csv.

    results: dict of (size, fmt, precision) -> watts
    """
    with open(NANO_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    fieldnames = list(rows[0].keys()) if rows else []

    n_patched = 0
    for row in rows:
        if (row["experiment"] != "core_comparison"
                or row["architecture"] != "yolo26"
                or row["approach"] != "pretrained"
                or row["imgsz"] != "640"
                or row["batch"] != "1"):
            continue

        # Determine size label from model_size (MB string → nano/small/medium/large)
        mb = float(row["model_size"].replace(" MB", ""))
        if mb < 12:
            size = "nano"
        elif mb < 40:
            size = "small"
        elif mb < 58:
            size = "medium"
        else:
            size = "large"

        key = (size, row["format"], row["precision"])
        if key not in results:
            continue

        watts = results[key]
        fps = float(row["fps"]) if row["fps"] else None
        row["watts"] = f"{watts:.2f}"
        row["fps_per_watt"] = f"{fps / watts:.2f}" if fps and watts else ""
        n_patched += 1

    with open(NANO_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nPatched {n_patched} rows in {NANO_CSV}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    val_image = find_val_image()
    print(f"Val image: {val_image}")

    results = {}
    for size, fmt, precision in CONFIGS:
        try:
            watts = measure(size, fmt, precision, val_image)
            if watts is not None:
                results[(size, fmt, precision)] = watts
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
        except Exception as e:
            print(f"  ERROR ({size} {fmt} {precision}): {e}")

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"{'Size':<8} {'Format':<12} {'Prec':<6} {'Watts':>8}")
    for (size, fmt, precision), watts in results.items():
        print(f"{size:<8} {fmt:<12} {precision:<6} {watts:>8.2f}")

    if results:
        patch_csv(results)
    else:
        print("No results to write.")


if __name__ == "__main__":
    main()
