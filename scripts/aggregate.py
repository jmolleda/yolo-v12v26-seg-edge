"""Aggregate benchmark results into a single CSV summary.

Scans all report files under results/ and produces:
  - results/benchmark_results.csv

Usage:
    python scripts/aggregate.py
    python scripts/aggregate.py --device rtx5090
    python scripts/aggregate.py --results-dir /path/to/results
"""

import argparse
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.utils import PROJECT_ROOT, RESULTS_DIR_NAME

CSV_COLUMNS = [
    "experiment", "architecture", "model_size", "approach",
    "format", "precision", "task", "imgsz", "batch", "device", "machine",
    "preprocess_ms", "inference_ms", "postprocess_ms", "total_ms", "fps",
    "map50", "map50_95", "p_mean", "r_mean",
    "watts", "fps_per_watt",
    "duration", "start_time", "end_time",
]

# Mapping from report field labels to CSV column names
FIELD_MAP = {
    "Machine": "machine",
    "Device": "device",
    "Experiment": "experiment",
    "Architecture": "architecture",
    "Model size": "model_size",
    "Task": "task",
    "Approach": "approach",
    "Format": "format",
    "Precision": "precision",
    "Input size": "imgsz",
    "Batch size": "batch",
    "Start time": "start_time",
    "End time": "end_time",
    "Duration": "duration",
    "Preprocess": "preprocess_ms",
    "Inference": "inference_ms",
    "Postprocess": "postprocess_ms",
    "Total": "total_ms",
    "FPS": "fps",
    "mAP50-95": "map50_95",
    "mAP50": "map50",
    "P (mean)": "p_mean",
    "R (mean)": "r_mean",
    "Power": "watts",
    "FPS/Watt": "fps_per_watt",
}


def parse_report(filepath):
    """Parse a benchmark report.txt into a dict of values."""
    data = {}
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if ":" not in line or line.startswith("=") or line.startswith("-"):
                continue
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()

            # Remove units (ms/img, W, etc.)
            value = re.sub(r"\s*(ms/img|ms|W|FPS)$", "", value)

            col = FIELD_MAP.get(key)
            if col:
                try:
                    value = float(value)
                    if value == int(value):
                        value = int(value)
                except ValueError:
                    pass
                data[col] = value
    return data


def find_reports(results_dir, device_filter=None):
    """Find all report files under results_dir.

    Args:
        results_dir: Root results directory.
        device_filter: Optional device name to filter ('rtx5090', 'jetson_agx', etc.)

    Returns:
        List of parsed report dicts.
    """
    reports = []
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return reports

    for root, dirs, files in os.walk(results_dir):
        for fname in files:
            if fname.startswith("report") and fname.endswith(".txt"):
                filepath = os.path.join(root, fname)
                data = parse_report(filepath)
                if data and (device_filter is None or data.get("device") == device_filter):
                    reports.append(data)

    return reports


def write_csv(reports, output_path):
    """Write reports to a CSV file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for report in sorted(reports, key=lambda r: (
            r.get("experiment", ""),
            r.get("architecture", ""),
            r.get("model_size", ""),
            r.get("format", ""),
        )):
            writer.writerow(report)
    print(f"CSV saved: {output_path} ({len(reports)} rows)")


def main():
    parser = argparse.ArgumentParser(description="Aggregate benchmark results")
    parser.add_argument("--results-dir", default=os.path.join(PROJECT_ROOT, RESULTS_DIR_NAME),
                        help="Results directory to scan")
    parser.add_argument("--device", default=None,
                        choices=["rtx5090", "jetson_agx", "jetson_nano"],
                        help="Filter by device")
    parser.add_argument("--output", default=None,
                        help="Output file path (without extension)")
    args = parser.parse_args()

    reports = find_reports(args.results_dir, args.device)

    if not reports:
        print("No reports found.")
        return

    print(f"Found {len(reports)} reports")

    output_base = args.output or os.path.join(args.results_dir, "benchmark_results")
    write_csv(reports, output_base + ".csv")


if __name__ == "__main__":
    main()
