"""Generic inference benchmark script.

Runs validation with warm-up and multiple measurement passes for stable timing.

Usage:
    python scripts/infer.py --weights best.pt --format pytorch --imgsz 640 --batch 1 \
        --arch yolo26 --size nano --task segment --approach scratch \
        --experiment core_comparison --device rtx5090

    python scripts/infer.py --weights best.engine --format tensorrt --precision fp16 \
        --imgsz 640 --batch 1 --arch yolo26 --size nano --task segment \
        --approach scratch --experiment core_comparison --device jetson_agx
"""

import argparse
import datetime
import os
import sys
import statistics

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO
from scripts.utils import (
    get_data_yaml_path,
    get_results_dir,
    get_machine_name,
    save_report,
    format_duration,
)

# Warm-up and measurement config (defaults, overridable via CLI)
DEFAULT_WARMUP_RUNS = 5
DEFAULT_MEASURE_RUNS = 10


def measure_power_jetson():
    """Measure current power consumption on Jetson via jtop.

    Returns:
        Power in watts, or None if not on a Jetson / jtop unavailable.
    """
    try:
        from jtop import jtop
        with jtop() as jetson:
            if jetson.ok():
                power = jetson.power
                # jtop reports power in milliwatts for total
                total_mw = power.get("tot", {}).get("power", 0)
                return total_mw / 1000.0 if total_mw else None
    except (ImportError, Exception):
        return None


def run_inference(weights_path, fmt, precision, imgsz, batch, architecture,
                  model_size, task, approach, experiment_name, device_name,
                  warmup_runs=DEFAULT_WARMUP_RUNS, measure_runs=DEFAULT_MEASURE_RUNS):
    """Run inference benchmark with warm-up and repeated measurements.

    Args:
        weights_path: Path to .pt or .engine weights.
        fmt: 'pytorch' or 'tensorrt'.
        precision: 'fp32', 'fp16', or 'int8'.
        imgsz: Input image size.
        batch: Batch size.
        architecture: 'yolo26' or 'yolo12'.
        model_size: 'nano', 'small', 'medium', 'large'.
        task: 'segment' or 'detect'.
        approach: 'scratch' or 'pretrained'.
        experiment_name: Experiment name for results organization.
        device_name: 'rtx5090', 'jetson_agx', 'jetson_nano'.
    """
    start_time = datetime.datetime.now()
    machine_name = get_machine_name()
    data_yaml = get_data_yaml_path()
    fmt_precision = precision  # Save before variable is reused for accuracy metric

    run_label = f"{architecture} {model_size} | {fmt} {precision} | {task} | {approach}"
    print("=" * 60)
    print(f"INFERENCE: {run_label}")
    print(f"Weights: {weights_path}")
    print(f"Input: {imgsz}, Batch: {batch}, Device: {device_name}")
    print(f"Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    model = YOLO(weights_path)
    model_file_size_mb = os.path.getsize(weights_path) / (1024 * 1024)

    # Reset peak memory tracking
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    val_kwargs = {
        "data": data_yaml,
        "imgsz": imgsz,
        "batch": batch,
        "split": "val",
        "verbose": True,
    }

    # Warm-up runs
    print(f"\nWarm-up: {warmup_runs} runs...")
    for i in range(warmup_runs):
        model.val(**val_kwargs)
        print(f"  Warm-up {i + 1}/{warmup_runs} done")

    # Measurement runs
    print(f"\nMeasuring: {measure_runs} runs...")
    all_pre, all_inf, all_post = [], [], []
    map50_values, map50_95_values = [], []
    precision_values, recall_values = [], []
    per_class_data = None

    # Measure power during inference (Jetsons only)
    watts = measure_power_jetson() if device_name.startswith("jetson") else None

    for i in range(measure_runs):
        val_results = model.val(**val_kwargs)
        speed = val_results.speed
        all_pre.append(speed.get("preprocess", 0.0))
        all_inf.append(speed.get("inference", 0.0))
        all_post.append(speed.get("postprocess", 0.0))

        if hasattr(val_results, "box"):
            map50_values.append(float(val_results.box.map50))
            map50_95_values.append(float(val_results.box.map))
            precision_values.append(float(val_results.box.mp))
            recall_values.append(float(val_results.box.mr))

        print(f"  Run {i + 1}/{measure_runs}: "
              f"inf={speed.get('inference', 0.0):.2f}ms")

    # Capture peak GPU memory
    gpu_mem_peak_mb = None
    if torch.cuda.is_available():
        gpu_mem_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # Extract per-class metrics from the last run
    try:
        box = val_results.box
        class_names = val_results.names if hasattr(val_results, "names") else {}
        per_class_data = {}
        for i, cls_id in enumerate(box.ap_class_index):
            name = class_names.get(int(cls_id), f"class_{cls_id}")
            p, r, ap50, ap = box.class_result(i)
            per_class_data[name] = {
                "precision": float(p),
                "recall":    float(r),
                "map50":     float(ap50),
                "map50_95":  float(ap),
            }
    except Exception as e:
        print(f"Warning: could not extract per-class metrics: {e}")

    # Compute averages
    t_pre = statistics.mean(all_pre)
    t_inf = statistics.mean(all_inf)
    t_post = statistics.mean(all_post)
    t_total_ms = t_pre + t_inf + t_post
    fps = 1000.0 / t_total_ms if t_total_ms > 0 else 0.0

    # Compute latency percentiles on total per-run time
    all_total = [p + i + o for p, i, o in zip(all_pre, all_inf, all_post)]
    all_total_sorted = sorted(all_total)
    n = len(all_total_sorted)
    t_median = statistics.median(all_total_sorted)
    t_stdev = statistics.stdev(all_total_sorted) if n >= 2 else 0.0
    t_p95 = all_total_sorted[int(n * 0.95)] if n >= 20 else all_total_sorted[-1]
    t_p99 = all_total_sorted[int(n * 0.99)] if n >= 100 else all_total_sorted[-1]

    map50 = statistics.mean(map50_values) if map50_values else 0.0
    map50_95 = statistics.mean(map50_95_values) if map50_95_values else 0.0
    precision_acc = statistics.mean(precision_values) if precision_values else 0.0
    recall = statistics.mean(recall_values) if recall_values else 0.0

    fps_per_watt = fps / watts if watts and watts > 0 else None

    end_time = datetime.datetime.now()
    duration = end_time - start_time

    # Results directory
    results_dir = get_results_dir(
        experiment_name, architecture, task, model_size, approach, device_name
    )
    os.makedirs(results_dir, exist_ok=True)

    # Determine report filename based on format/precision/imgsz/batch
    report_name = f"report_{fmt}_{precision}_img{imgsz}_b{batch}.txt"
    report_path = os.path.join(results_dir, report_name)

    report_data = {
        "machine": machine_name,
        "device": device_name,
        "experiment": experiment_name,
        "architecture": architecture,
        "model_size": model_size,
        "task": task,
        "approach": approach,
        "format": fmt,
        "format_precision": fmt_precision,
        "imgsz": imgsz,
        "batch": batch,
        "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration": format_duration(duration),
        "preprocess_ms": t_pre,
        "inference_ms": t_inf,
        "postprocess_ms": t_post,
        "total_ms": t_total_ms,
        "fps": fps,
        "median_ms": t_median,
        "stdev_ms": t_stdev,
        "p95_ms": t_p95,
        "p99_ms": t_p99,
        "measure_runs": measure_runs,
        "map50": map50,
        "map50_95": map50_95,
        "precision": precision_acc,
        "recall": recall,
        "watts": watts,
        "fps_per_watt": fps_per_watt,
        "per_class": per_class_data,
        "model_file_size_mb": model_file_size_mb,
        "gpu_mem_peak_mb": gpu_mem_peak_mb,
    }
    save_report(report_path, report_data)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"RESULTS: {run_label}")
    print(f"  Preprocess:  {t_pre:.2f} ms/img")
    print(f"  Inference:   {t_inf:.2f} ms/img")
    print(f"  Postprocess: {t_post:.2f} ms/img")
    print(f"  Total:       {t_total_ms:.2f} ms/img (mean, n={measure_runs})")
    print(f"  Median:      {t_median:.2f} ms/img")
    print(f"  Std dev:     {t_stdev:.2f} ms")
    print(f"  p95:         {t_p95:.2f} ms/img")
    print(f"  p99:         {t_p99:.2f} ms/img")
    print(f"  FPS:         {fps:.2f}")
    print(f"  Model size:  {model_file_size_mb:.1f} MB")
    if gpu_mem_peak_mb is not None:
        print(f"  GPU mem:     {gpu_mem_peak_mb:.1f} MB (peak)")
    print(f"  mAP50:       {map50:.4f}")
    print(f"  mAP50-95:    {map50_95:.4f}")
    print(f"  Precision:   {precision_acc:.4f}")
    print(f"  Recall:      {recall:.4f}")
    if watts:
        print(f"  Power:       {watts:.2f} W")
        print(f"  FPS/Watt:    {fps_per_watt:.2f}")
    if per_class_data:
        print(f"\n  --- Per-class Accuracy ---")
        for cls_name, cls_metrics in per_class_data.items():
            print(f"  {cls_name:8s}  P={cls_metrics['precision']:.4f}  "
                  f"R={cls_metrics['recall']:.4f}  "
                  f"mAP50={cls_metrics['map50']:.4f}  "
                  f"mAP50-95={cls_metrics['map50_95']:.4f}")
    print(f"  Duration:    {format_duration(duration)}")
    print(f"  Report:      {report_path}")
    print(f"{'=' * 60}")

    return report_data


def main():
    parser = argparse.ArgumentParser(description="Run YOLO inference benchmark")
    parser.add_argument("--weights", required=True, help="Path to .pt or .engine weights")
    parser.add_argument("--format", required=True, choices=["pytorch", "tensorrt"],
                        dest="fmt", help="Model format")
    parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "int8"],
                        help="Precision")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--arch", required=True, choices=["yolo26", "yolo12"],
                        help="Architecture")
    parser.add_argument("--size", required=True,
                        choices=["nano", "small", "medium", "large"],
                        help="Model size")
    parser.add_argument("--task", required=True, choices=["segment", "detect"],
                        help="Task")
    parser.add_argument("--approach", required=True,
                        choices=["scratch", "pretrained",
                                 "scratch_balanced", "pretrained_balanced"],
                        help="Training approach")
    parser.add_argument("--experiment", default="core_comparison",
                        help="Experiment name")
    parser.add_argument("--device", required=True,
                        choices=["rtx5090", "jetson_agx", "jetson_nano"],
                        help="Device name")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP_RUNS,
                        help=f"Number of warm-up runs (default: {DEFAULT_WARMUP_RUNS})")
    parser.add_argument("--runs", type=int, default=DEFAULT_MEASURE_RUNS,
                        help=f"Number of measurement runs (default: {DEFAULT_MEASURE_RUNS})")
    args = parser.parse_args()

    run_inference(
        weights_path=args.weights,
        fmt=args.fmt,
        precision=args.precision,
        imgsz=args.imgsz,
        batch=args.batch,
        architecture=args.arch,
        model_size=args.size,
        task=args.task,
        approach=args.approach,
        experiment_name=args.experiment,
        device_name=args.device,
        warmup_runs=args.warmup,
        measure_runs=args.runs,
    )


if __name__ == "__main__":
    main()
