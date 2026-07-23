"""Capture the device environment for the JRTIP revision (R4.5, R2.2, R4.3).

Records into results_revision/{device}/environment.txt:
  - Python / PyTorch / CUDA / TensorRT / Ultralytics versions
  - GPU name and driver
  - Jetson only: nvpmodel mode, jetson_clocks status, thermal zones,
    30 s idle-power sample via tegrastats
  - RTX only: SHA-256 checksums of all trained best.pt checkpoints

Usage:
    python scripts/capture_env.py --device rtx5090
    python scripts/capture_env.py --device jetson_agx
    python scripts/capture_env.py --device jetson_nano
"""

import argparse
import glob
import hashlib
import os
import platform
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils import PROJECT_ROOT


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=60).stdout.strip()
    except Exception as e:
        return f"<error: {e}>"


def versions():
    out = [f"platform: {platform.platform()}",
           f"python: {sys.version.split()[0]}"]
    try:
        import torch
        out.append(f"torch: {torch.__version__}")
        out.append(f"torch.cuda: {torch.version.cuda}")
        if torch.cuda.is_available():
            out.append(f"gpu: {torch.cuda.get_device_name(0)}")
    except ImportError:
        out.append("torch: not installed")
    try:
        import tensorrt
        out.append(f"tensorrt: {tensorrt.__version__}")
    except ImportError:
        out.append("tensorrt: not importable")
    try:
        import ultralytics
        out.append(f"ultralytics: {ultralytics.__version__}")
    except ImportError:
        out.append("ultralytics: not installed")
    out.append(f"nvidia-smi driver: {sh('nvidia-smi --query-gpu=driver_version --format=csv,noheader')}")
    return out


def jetson_state():
    """All reads are unprivileged: nvpmodel -q (query only) and sysfs."""
    out = ["", "--- Jetson state ---"]
    out.append(f"nvpmodel: {sh('nvpmodel -q 2>/dev/null')}")
    out.append("cpu governors/frequencies:")
    for cpu in sorted(glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq/")):
        gov = sh(f"cat {cpu}scaling_governor")
        cur = sh(f"cat {cpu}scaling_cur_freq")
        mx = sh(f"cat {cpu}scaling_max_freq")
        out.append(f"  {cpu.split('/')[-3]}: governor={gov} cur={cur} max={mx}")
    out.append("gpu devfreq:")
    for df in sorted(glob.glob("/sys/class/devfreq/*/")):
        gov = sh(f"cat {df}governor")
        cur = sh(f"cat {df}cur_freq")
        mx = sh(f"cat {df}max_freq")
        out.append(f"  {os.path.basename(df.rstrip('/'))}: governor={gov} cur={cur} max={mx}")
    out.append("thermal zones:")
    for tz in sorted(glob.glob("/sys/devices/virtual/thermal/thermal_zone*/")):
        ttype = sh(f"cat {tz}type")
        temp = sh(f"cat {tz}temp")
        out.append(f"  {ttype}: {temp}")
    out.append("fan pwm:")
    for fan in sorted(glob.glob("/sys/class/hwmon/hwmon*/pwm1")):
        out.append(f"  {fan}: {sh(f'cat {fan}')}")
    return out


def jetson_idle_power(seconds=30):
    out = ["", f"--- Idle power ({seconds}s tegrastats sample) ---"]
    try:
        proc = subprocess.Popen(["tegrastats", "--interval", "500"],
                                stdout=subprocess.PIPE, text=True, bufsize=1)
        samples = []
        t0 = time.time()
        for line in proc.stdout:
            samples.append(line.strip())
            if time.time() - t0 > seconds:
                break
        proc.terminate()
        out.append(f"samples: {len(samples)}")
        out.extend(samples[:5])
        out.append("...")
        out.extend(samples[-2:])
    except FileNotFoundError:
        out.append("tegrastats not found")
    return out


def checkpoint_checksums():
    out = ["", "--- Checkpoint SHA-256 (trained best.pt) ---"]
    pattern = os.path.join(PROJECT_ROOT, "results", "rtx5090", "*", "*", "train*",
                           "weights", "best.pt")
    for path in sorted(glob.glob(pattern)):
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        rel = os.path.relpath(path, PROJECT_ROOT)
        out.append(f"{h.hexdigest()}  {rel}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True,
                        choices=["rtx5090", "jetson_agx", "jetson_nano"])
    args = parser.parse_args()

    lines = [f"device: {args.device}", ""]
    lines += versions()
    if args.device.startswith("jetson"):
        lines += jetson_state()
        lines += jetson_idle_power()
    else:
        lines += checkpoint_checksums()

    out_dir = os.path.join(PROJECT_ROOT, "results_revision", args.device)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "environment.txt")
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
