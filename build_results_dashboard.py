"""
Build a self-contained HTML dashboard for ALL YOLO benchmark results.
Scans run_rtx5090/, run_jetson_agx/, run_jetson_nano/ for results.
Produces results_dashboard.html with interactive charts and tables.

Usage:
    python build_results_dashboard.py
"""

import csv
import json
import os
import glob
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_HTML = os.path.join(BASE_DIR, "docs", "results_dashboard.html")

# Auto-discover device folders inside results/
# Layout: results/{device}/{experiment}/{model}/
DEVICE_DIRS = {}
top_results = os.path.join(BASE_DIR, "results")
if os.path.isdir(top_results):
    for sub in sorted(os.listdir(top_results)):
        sub_path = os.path.join(top_results, sub)
        if os.path.isdir(sub_path):
            DEVICE_DIRS[sub] = sub_path


def parse_model_name(folder_name):
    """Extract arch, task_key, size, approach from folder like yolo26_seg_nano_scratch."""
    parts = folder_name.split("_")
    arch = parts[0]
    task_key = parts[1]  # seg or det
    size = parts[2]
    approach = "_".join(parts[3:])  # scratch, pretrained, scratch_balanced, pretrained_balanced
    return arch, task_key, size, approach


def read_training_results():
    """Read all training results.csv files across all devices.

    Only includes models whose report.txt exists — meaning training fully completed.
    results.csv is updated every epoch so it exists for in-progress runs too,
    but report.txt is only written at the end.
    """
    models = []
    csv_files = []

    for device, device_path in DEVICE_DIRS.items():
        # Match train/ and train2/, train3/ etc. — Ultralytics increments the folder
        # name on rerun. Pick the latest one per model folder.
        pattern = os.path.join(device_path, "*", "*", "train*", "results.csv")
        seen_models = {}
        for f in sorted(glob.glob(pattern)):
            model_dir = os.path.dirname(os.path.dirname(f))
            train_dir = os.path.basename(os.path.dirname(f))
            # Keep the highest-numbered train folder (train < train2 < train3 lexically)
            if model_dir not in seen_models or train_dir > os.path.basename(os.path.dirname(seen_models[model_dir])):
                seen_models[model_dir] = f
        for f in seen_models.values():
            csv_files.append((device, f))
    print(f"Found {len(csv_files)} training result files")

    for device, csv_path in csv_files:
        model_folder = os.path.basename(os.path.dirname(os.path.dirname(csv_path)))
        experiment = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(csv_path))))
        arch, task_key, size, approach = parse_model_name(model_folder)

        # Only show completed runs — report.txt is the definitive completion signal
        report_path = os.path.join(os.path.dirname(os.path.dirname(csv_path)), "report.txt")
        if not os.path.exists(report_path):
            print(f"  Skipping incomplete run: {model_folder} ({device})")
            continue

        rows = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            reader.fieldnames = [h.strip() for h in reader.fieldnames]
            for row in reader:
                cleaned = {}
                for k, v in row.items():
                    k = k.strip()
                    try:
                        cleaned[k] = float(v)
                    except (ValueError, TypeError):
                        cleaned[k] = v
                rows.append(cleaned)

        if not rows:
            continue

        epochs_data = []
        for r in rows:
            epochs_data.append({
                "epoch": int(r.get("epoch", 0)),
                "time": r.get("time", 0),
                "mAP50_B": r.get("metrics/mAP50(B)", 0),
                "mAP50_M": r.get("metrics/mAP50(M)", 0),
                "mAP50_95_B": r.get("metrics/mAP50-95(B)", 0),
                "mAP50_95_M": r.get("metrics/mAP50-95(M)", 0),
                "precision_B": r.get("metrics/precision(B)", 0),
                "recall_B": r.get("metrics/recall(B)", 0),
                "precision_M": r.get("metrics/precision(M)", 0),
                "recall_M": r.get("metrics/recall(M)", 0),
                "train_box_loss": r.get("train/box_loss", 0),
                "train_seg_loss": r.get("train/seg_loss", 0),
                "train_cls_loss": r.get("train/cls_loss", 0),
                "train_dfl_loss": r.get("train/dfl_loss", 0),
                "val_box_loss": r.get("val/box_loss", 0),
                "val_seg_loss": r.get("val/seg_loss", 0),
                "val_cls_loss": r.get("val/cls_loss", 0),
                "val_dfl_loss": r.get("val/dfl_loss", 0),
                "lr": r.get("lr/pg0", 0),
            })

        best_idx = max(range(len(epochs_data)), key=lambda i: epochs_data[i]["mAP50_95_B"])
        best = epochs_data[best_idx]

        first_time_s = rows[0].get("time", 0)
        last_time_s = rows[-1].get("time", 0)
        total_time_s = last_time_s - first_time_s
        hours = int(total_time_s // 3600)
        minutes = int((total_time_s % 3600) // 60)
        time_str = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"

        models.append({
            "name": model_folder,
            "device": device,
            "experiment": experiment,
            "arch": arch,
            "task": task_key,
            "size": size,
            "approach": approach,
            "total_epochs": len(rows),
            "training_time_s": total_time_s,
            "training_time": time_str,
            "best_epoch": best["epoch"],
            "best_mAP50_B": round(best["mAP50_B"], 5),
            "best_mAP50_M": round(best["mAP50_M"], 5),
            "best_mAP50_95_B": round(best["mAP50_95_B"], 5),
            "best_mAP50_95_M": round(best["mAP50_95_M"], 5),
            "best_precision_B": round(best["precision_B"], 5),
            "best_recall_B": round(best["recall_B"], 5),
            "epochs": epochs_data,
        })

    models.sort(key=lambda m: m["best_mAP50_95_B"], reverse=True)
    return models


def read_inference_reports():
    """Read all inference report_*.txt files across all devices."""
    reports = []
    report_files = []
    for device, device_path in DEVICE_DIRS.items():
        pattern = os.path.join(device_path, "*", "*", "report_*.txt")
        for f in sorted(glob.glob(pattern)):
            report_files.append((device, f))
    print(f"Found {len(report_files)} inference report files")

    for device, rpath in report_files:
        model_folder = os.path.basename(os.path.dirname(rpath))
        experiment = os.path.basename(os.path.dirname(os.path.dirname(rpath)))
        arch, task_key, size, approach = parse_model_name(model_folder)

        data = {
            "name": model_folder,
            "device": device,
            "experiment": experiment,
            "arch": arch,
            "task": task_key,
            "size": size,
            "approach": approach,
            "file": os.path.basename(rpath),
        }

        with open(rpath, "r") as f:
            content = f.read()

        # Parse key fields
        field_patterns = {
            "format": r"Format:\s*(.+)",
            "format_precision": r"Precision:\s*(.+)",
            "imgsz": r"Input size:\s*(\d+)",
            "batch": r"Batch size:\s*(\S+)",
            "duration": r"Duration:\s*(.+)",
            "preprocess_ms": r"Preprocess:\s*([\d.]+)",
            "inference_ms": r"Inference:\s*([\d.]+)",
            "postprocess_ms": r"Postprocess:\s*([\d.]+)",
            "total_ms": r"Total:\s*([\d.]+)",
            "fps": r"FPS:\s*([\d.]+)",
            "median_ms": r"Median:\s*([\d.]+)",
            "stdev_ms": r"Std dev:\s*([\d.]+)",
            "p95_ms": r"p95:\s*([\d.]+)",
            "p99_ms": r"p99:\s*([\d.]+)",
            "map50": r"mAP50:\s*([\d.]+)",
            "map50_95": r"mAP50-95:\s*([\d.]+)",
            "p_mean": r"P \(mean\):\s*([\d.]+)",
            "r_mean": r"R \(mean\):\s*([\d.]+)",
            "model_size_mb": r"Model size:\s*([\d.]+)",
            "gpu_mem_mb": r"GPU mem peak:\s*([\d.]+)",
        }

        for key, pat in field_patterns.items():
            m = re.search(pat, content)
            if m:
                val = m.group(1).strip()
                try:
                    val = float(val)
                except ValueError:
                    pass
                data[key] = val

        reports.append(data)

    return reports


def read_train_reports():
    """Read training report.txt files (one per training run) across all devices.

    report.txt is only written by the orchestrator after training fully completes,
    so its existence is the definitive signal that a run is done.
    """
    reports = []
    report_files = []

    for device, device_path in DEVICE_DIRS.items():
        pattern = os.path.join(device_path, "*", "*", "report.txt")
        for f in sorted(glob.glob(pattern)):
            report_files.append((device, f))
    print(f"Found {len(report_files)} training report files")

    for device, rpath in report_files:
        model_folder = os.path.basename(os.path.dirname(rpath))
        experiment = os.path.basename(os.path.dirname(os.path.dirname(rpath)))
        arch, task_key, size, approach = parse_model_name(model_folder)

        data = {
            "name": model_folder,
            "device": device,
            "experiment": experiment,
            "arch": arch,
            "task": task_key,
            "size": size,
            "approach": approach,
        }

        with open(rpath, "r") as f:
            content = f.read()

        field_patterns = {
            "batch": r"Batch size:\s*(\S+)",
            "duration": r"Duration:\s*(.+)",
            "fps": r"FPS:\s*([\d.]+)",
            "map50": r"mAP50:\s*([\d.]+)",
            "map50_95": r"mAP50-95:\s*([\d.]+)",
            "best_conf": r"Best conf:\s*([\d.]+)",
            "best_f1": r"Best F1:\s*([\d.]+)",
        }

        for key, pat in field_patterns.items():
            m = re.search(pat, content)
            if m:
                val = m.group(1).strip()
                try:
                    val = float(val)
                except ValueError:
                    pass
                data[key] = val

        # Parse per-class accuracy section
        per_class = {}
        pc_match = re.search(r'--- Per-class Accuracy ---\n(.+?)(?:\n-{10,}|\Z)', content, re.DOTALL)
        if pc_match:
            lines = pc_match.group(1).strip().split('\n')
            for line in lines[1:]:  # skip header
                parts = line.split()
                if len(parts) >= 5:
                    per_class[parts[0]] = {
                        "precision": float(parts[1]),
                        "recall": float(parts[2]),
                        "map50": float(parts[3]),
                        "map50_95": float(parts[4]),
                    }
        if per_class:
            data["per_class"] = per_class

        reports.append(data)

    return reports


def read_hardware_metrics():
    """Parse logs/{device}_stdout.log and train folders for hardware/model metrics.

    Extracts per-model: parameters, GFLOPs, layers, GPU memory usage,
    AutoBatch memory, model file size from weights/best.pt.
    Returns dict keyed by model config string (e.g. 'YOLO26n-seg').
    """
    metrics = {}

    # Collect stdout log paths: logs/{device}_stdout.log
    bench_logs = []
    logs_dir = os.path.join(BASE_DIR, "logs")
    if os.path.isdir(logs_dir):
        for fname in sorted(os.listdir(logs_dir)):
            if fname.endswith("_stdout.log"):
                device = fname.replace("_stdout.log", "")
                bench_logs.append((device, os.path.join(logs_dir, fname)))

    for device, log_path in bench_logs:

        # Strip ANSI escape codes for easier parsing
        with open(log_path, "r", errors="replace") as f:
            content = f.read()
        content = re.sub(r'\x1b\[[0-9;]*m', '', content)

        # Parse model summaries: "YOLO26n-seg summary: 309 layers, 3,056,240 parameters, ... 10.2 GFLOPs"
        for m in re.finditer(
            r'(YOLO\w+-?\w*)\s+summary:\s*(\d+)\s+layers,\s+([\d,]+)\s+parameters,\s+[\d,]+\s+gradients,\s+([\d.]+)\s+GFLOPs',
            content
        ):
            model_key = m.group(1)
            layers = int(m.group(2))
            params = int(m.group(3).replace(",", ""))
            gflops = float(m.group(4))

            if model_key not in metrics:
                metrics[model_key] = {
                    "model_key": model_key,
                    "device": device,
                    "layers": layers,
                    "params": params,
                    "gflops": gflops,
                    "gpu_name": None,
                    "gpu_total_gb": None,
                    "autobatch_mem_gb": None,
                    "autobatch_pct": None,
                    "train_gpu_mem_gb": None,
                    "model_file_mb": None,
                }

        # Parse GPU info: "CUDA:0 (NVIDIA GeForce RTX 5090, 32120MiB)"
        gpu_match = re.search(r'CUDA:0 \(([^,]+),\s*(\d+)MiB\)', content)
        if gpu_match:
            gpu_name = gpu_match.group(1)
            gpu_mib = int(gpu_match.group(2))
            for k in metrics:
                if metrics[k]["device"] == device:
                    metrics[k]["gpu_name"] = gpu_name
                    metrics[k]["gpu_total_gb"] = round(gpu_mib / 1024, 2)

        # Parse AutoBatch: "Using batch-size 40 for CUDA:0 19.02G/31.37G (61%)"
        # These appear in order matching the summary lines
        autobatch_matches = list(re.finditer(
            r'AutoBatch:.*Using batch-size \d+ for CUDA:0 ([\d.]+)G/([\d.]+)G \((\d+)%\)',
            content
        ))
        summary_matches = list(re.finditer(
            r'(YOLO\w+-?\w*)\s+summary:', content
        ))
        for sm, ab in zip(summary_matches, autobatch_matches):
            model_key = sm.group(1)
            if model_key in metrics:
                metrics[model_key]["autobatch_mem_gb"] = float(ab.group(1))
                metrics[model_key]["autobatch_pct"] = int(ab.group(3))

        # Parse per-epoch GPU memory (first epoch line after each summary)
        # Pattern: "1/1000      8.67G"
        for sm in summary_matches:
            model_key = sm.group(1)
            # Find the first epoch line after this summary
            after = content[sm.end():]
            epoch_mem = re.search(r'\d+/\d+\s+([\d.]+)G\s+[\d.]+\s+[\d.]+\s+[\d.]+', after)
            if epoch_mem and model_key in metrics:
                metrics[model_key]["train_gpu_mem_gb"] = float(epoch_mem.group(1))

    # Get model file sizes from weights/best.pt.
    # Use train*/ glob and pick the latest folder in case Ultralytics incremented it.
    for device_name, device_path in DEVICE_DIRS.items():
        seen_model_dirs = {}
        for weight_path in glob.glob(os.path.join(device_path, "*", "*", "train*", "weights", "best.pt")):
            model_dir = os.path.dirname(os.path.dirname(os.path.dirname(weight_path)))
            train_dir_name = os.path.basename(os.path.dirname(os.path.dirname(weight_path)))
            if model_dir not in seen_model_dirs or train_dir_name > os.path.basename(os.path.dirname(os.path.dirname(seen_model_dirs[model_dir]))):
                seen_model_dirs[model_dir] = weight_path
        for weight_path in seen_model_dirs.values():
            # Extract model config from args.yaml
            train_dir = os.path.dirname(os.path.dirname(weight_path))
            args_path = os.path.join(train_dir, "args.yaml")
            if os.path.exists(args_path):
                with open(args_path, "r") as f:
                    for line in f:
                        if line.startswith("model:"):
                            model_yaml = line.split(":")[1].strip()
                            # Convert "yolo26n-seg.yaml" -> "YOLO26n-seg"
                            model_key = model_yaml.replace(".yaml", "").replace("yolo", "YOLO")
                            if model_key in metrics:
                                size_mb = os.path.getsize(weight_path) / (1024 * 1024)
                                metrics[model_key]["model_file_mb"] = round(size_mb, 1)
                            break

    # Merge with cached metrics (preserves data from previous runs)
    cache_path = os.path.join(BASE_DIR, "hw_metrics_cache.json")
    cached = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            for item in json.load(f):
                cached[item["model_key"]] = item

    # Merge: cached non-None values are preserved; new data only fills gaps
    for key, val in metrics.items():
        if key not in cached:
            cached[key] = val
        else:
            for field, fval in val.items():
                if cached[key].get(field) is None and fval is not None:
                    cached[key][field] = fval

    # Save updated cache
    with open(cache_path, "w") as f:
        json.dump(list(cached.values()), f, indent=2)

    result = list(cached.values())
    print(f"Found hardware metrics for {len(result)} model configs")
    return result


def build_html(training_models, inference_reports, train_reports, hw_metrics):
    train_json = json.dumps(training_models)
    infer_json = json.dumps(inference_reports)
    train_reports_json = json.dumps(train_reports)
    hw_json = json.dumps(hw_metrics)

    # Count stats
    n_train = len(training_models)
    n_infer = len(inference_reports)
    devices = sorted(set(m["device"] for m in training_models + inference_reports + train_reports))
    n_devices = len(devices)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>YOLO v12 v26 Segmentation Edge — Results Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #0f172a; color: #e2e8f0; padding: 20px; min-height: 100vh;
  }}
  h1 {{
    text-align: center; font-size: 2rem; margin-bottom: 4px;
    background: linear-gradient(135deg, #38bdf8, #818cf8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }}
  .subtitle {{ text-align: center; color: #94a3b8; margin-bottom: 24px; font-size: 0.95rem; }}
  .tabs {{
    display: flex; gap: 4px; margin-bottom: 20px; justify-content: center; flex-wrap: wrap;
  }}
  .tab-btn {{
    padding: 8px 20px; border-radius: 8px 8px 0 0; border: 1px solid #334155;
    border-bottom: none; background: #1e293b; color: #94a3b8; cursor: pointer;
    font-size: 0.9rem; font-weight: 600; transition: all 0.15s;
  }}
  .tab-btn:hover {{ background: #263349; }}
  .tab-btn.active {{ background: #334155; color: #38bdf8; border-color: #38bdf8; }}
  .tab-content {{ display: none; }}
  .tab-content.active {{ display: block; }}
  .filters {{
    display: flex; justify-content: center; gap: 12px; margin-bottom: 20px; flex-wrap: wrap;
  }}
  .filter-group {{
    display: flex; align-items: center; gap: 6px;
    background: #1e293b; padding: 8px 16px; border-radius: 8px; border: 1px solid #334155;
  }}
  .filter-group label {{
    color: #94a3b8; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;
  }}
  .filter-btn {{
    padding: 4px 10px; border-radius: 6px; border: 1px solid #475569;
    background: #334155; color: #cbd5e1; cursor: pointer; font-size: 0.8rem; transition: all 0.15s;
  }}
  .filter-btn:hover {{ background: #475569; }}
  .filter-btn.active {{ background: #3b82f6; border-color: #3b82f6; color: #fff; }}
  .card {{
    background: #1e293b; border-radius: 12px; border: 1px solid #334155;
    padding: 20px; margin-bottom: 20px;
  }}
  .card h2 {{ font-size: 1.1rem; color: #94a3b8; margin-bottom: 16px; font-weight: 600; }}
  .card h3 {{ font-size: 0.95rem; color: #64748b; margin-bottom: 12px; }}
  .stats-row {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 12px; margin-bottom: 20px;
  }}
  .stat-card {{
    background: #0f172a; border-radius: 8px; padding: 16px; text-align: center;
    border: 1px solid #334155;
  }}
  .stat-card .value {{ font-size: 1.8rem; font-weight: 700; color: #38bdf8; }}
  .stat-card .label {{ font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 4px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
  th {{
    background: #334155; color: #94a3b8; padding: 8px 8px; text-align: left;
    cursor: pointer; user-select: none; white-space: nowrap;
    font-weight: 600; text-transform: uppercase; font-size: 0.7rem; letter-spacing: 0.5px;
  }}
  th:hover {{ background: #475569; }}
  td {{ padding: 7px 8px; border-bottom: 1px solid #0f172a; white-space: nowrap; }}
  tr {{ background: #1e293b; transition: background 0.1s; }}
  tr:hover {{ background: #263349; }}
  tr.best-row {{ background: #1e3a2f; }}
  .table-wrap {{ overflow-x: auto; }}
  .arch-badge {{
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-size: 0.75rem; font-weight: 600;
  }}
  .arch-yolo26 {{ background: #1e3a5f; color: #38bdf8; }}
  .arch-yolo12 {{ background: #3b1f5e; color: #c084fc; }}
  .size-badge {{
    display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem;
    font-weight: 600; background: #1e293b; border: 1px solid #475569; color: #cbd5e1;
  }}
  .approach-scratch {{ color: #fb923c; }}
  .approach-pretrained {{ color: #34d399; }}
  .approach-scratch_balanced {{ color: #f472b6; }}
  .approach-pretrained_balanced {{ color: #2dd4bf; }}
  .device-badge {{
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-size: 0.75rem; font-weight: 600;
  }}
  .device-rtx5090 {{ background: #1e3a2f; color: #4ade80; }}
  .device-jetson_agx {{ background: #3b2f1e; color: #fbbf24; }}
  .device-jetson_nano {{ background: #3b1e2f; color: #fb7185; }}
  .charts-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }}
  .chart-card {{
    background: #1e293b; border-radius: 12px; border: 1px solid #334155; padding: 20px;
  }}
  .chart-card h3 {{ font-size: 0.9rem; color: #94a3b8; margin-bottom: 12px; font-weight: 600; }}
  .chart-card canvas {{ width: 100% !important; }}
  .metric-highlight {{ font-weight: 700; color: #34d399; }}
  .metric-warn {{ font-weight: 700; color: #fb923c; }}
  @media (max-width: 1000px) {{ .charts-grid {{ grid-template-columns: 1fr; }} }}
  .timestamp {{ text-align: center; color: #475569; font-size: 0.8rem; margin-top: 20px; }}
</style>
</head>
<body>

<h1>YOLO v12 v26 Segmentation Edge &mdash; Benchmark Results</h1>
<p class="subtitle">{n_devices} device(s) &bull; {n_train} trained &bull; {n_infer} inference runs</p>

<div class="tabs">
  <button class="tab-btn active" data-tab="overview">Overview</button>
  <button class="tab-btn" data-tab="training">Training Curves</button>
  <button class="tab-btn" data-tab="convergence">Convergence</button>
  <button class="tab-btn" data-tab="inference">Inference</button>
  <button class="tab-btn" data-tab="comparison">Head-to-Head</button>
  <button class="tab-btn" data-tab="perclass">Per-Class</button>
  <button class="tab-btn" data-tab="resources">Resources</button>
</div>

<!-- ============ OVERVIEW TAB ============ -->
<div class="tab-content active" id="tab-overview">
  <div class="stats-row" id="overviewStats"></div>
  <div class="card">
    <h2>Training Summary</h2>
    <div class="table-wrap">
      <table id="trainSummaryTable">
        <thead><tr>
          <th data-col="device">Device</th>
          <th data-col="experiment">Exp</th>
          <th data-col="arch">Arch</th>
          <th data-col="size">Size</th>
          <th data-col="task">Task</th>
          <th data-col="approach">Approach</th>
          <th data-col="batch">Batch</th>
          <th data-col="total_epochs">Epochs</th>
          <th data-col="training_time_s">Time</th>
          <th data-col="best_epoch">Best Ep.</th>
          <th data-col="best_mAP50_B">mAP50(B)</th>
          <th data-col="best_mAP50_M">mAP50(M)</th>
          <th data-col="best_mAP50_95_B">mAP50-95(B)</th>
          <th data-col="best_precision_B">Precision</th>
          <th data-col="best_recall_B">Recall</th>
          <th>F1</th>
          <th>Conf</th>
        </tr></thead>
        <tbody id="trainTableBody"></tbody>
      </table>
    </div>
  </div>
  <div class="charts-grid">
    <div class="chart-card"><h3>mAP50-95 by Model</h3><canvas id="chartOverviewBar"></canvas></div>
    <div class="chart-card"><h3>Training Time vs Accuracy</h3><canvas id="chartTimeVsAcc"></canvas></div>
    <div class="chart-card"><h3>Best F1 Score by Model</h3><canvas id="chartF1Bar"></canvas></div>
    <div class="chart-card"><h3>Optimal Confidence Threshold</h3><canvas id="chartConfBar"></canvas></div>
  </div>
</div>

<!-- ============ TRAINING CURVES TAB ============ -->
<div class="tab-content" id="tab-training">
  <div class="filters" id="trainingFilters"></div>
  <div class="charts-grid">
    <div class="chart-card"><h3>mAP50 (Box) &mdash; Training Curves</h3><canvas id="chartCurvesB"></canvas></div>
    <div class="chart-card"><h3>mAP50-95 (Box) &mdash; Training Curves</h3><canvas id="chartCurves95B"></canvas></div>
    <div class="chart-card"><h3>mAP50 (Mask) &mdash; Training Curves</h3><canvas id="chartCurvesM"></canvas></div>
    <div class="chart-card"><h3>mAP50-95 (Mask) &mdash; Training Curves</h3><canvas id="chartCurves95M"></canvas></div>
    <div class="chart-card"><h3>Training Loss (Box + Seg + Cls)</h3><canvas id="chartTrainLoss"></canvas></div>
    <div class="chart-card"><h3>Validation Loss (Box + Seg + Cls)</h3><canvas id="chartValLoss"></canvas></div>
    <div class="chart-card"><h3>Precision (Box)</h3><canvas id="chartPrecCurves"></canvas></div>
    <div class="chart-card"><h3>Recall (Box)</h3><canvas id="chartRecCurves"></canvas></div>
  </div>
</div>

<!-- ============ CONVERGENCE TAB ============ -->
<div class="tab-content" id="tab-convergence">
  <div class="card">
    <h2>Convergence Analysis</h2>
    <p style="color:#64748b;margin-bottom:16px">Best epoch, early stopping, and efficiency metrics</p>
    <div class="table-wrap">
      <table id="convTable">
        <thead><tr>
          <th>Device</th><th>Model</th><th>Arch</th><th>Size</th><th>Approach</th>
          <th>Total Ep.</th><th>Best Ep.</th><th>Patience Used</th>
          <th>Best mAP50-95</th><th>Time</th><th>sec/epoch</th>
        </tr></thead>
        <tbody id="convTableBody"></tbody>
      </table>
    </div>
  </div>
  <div class="charts-grid">
    <div class="chart-card"><h3>Best Epoch vs Model Size</h3><canvas id="chartConvEpoch"></canvas></div>
    <div class="chart-card"><h3>Training Efficiency (mAP50-95 / hour)</h3><canvas id="chartEfficiency"></canvas></div>
  </div>
</div>

<!-- ============ INFERENCE TAB ============ -->
<div class="tab-content" id="tab-inference">
  <div class="card">
    <h2>Inference Results</h2>
    <div class="table-wrap">
      <table id="inferTable">
        <thead><tr>
          <th>Device</th><th>Exp</th><th>Arch</th><th>Size</th><th>Approach</th>
          <th>Format</th><th>Prec</th><th>ImgSz</th><th>Batch</th>
          <th>Inf (ms)</th><th>Total (ms)</th><th>FPS</th>
          <th>Median</th><th>p95</th>
          <th>mAP50</th><th>mAP50-95</th>
          <th>Model MB</th><th>GPU MB</th>
        </tr></thead>
        <tbody id="inferTableBody"></tbody>
      </table>
    </div>
  </div>
  <div class="charts-grid">
    <div class="chart-card"><h3>FPS by Model</h3><canvas id="chartInferFPS"></canvas></div>
    <div class="chart-card"><h3>Latency Breakdown (ms)</h3><canvas id="chartLatency"></canvas></div>
  </div>
</div>

<!-- ============ HEAD-TO-HEAD TAB ============ -->
<div class="tab-content" id="tab-comparison">
  <div class="charts-grid">
    <div class="chart-card"><h3>YOLOv26 vs YOLOv12 &mdash; mAP50-95 by Size</h3><canvas id="chartH2H_map"></canvas></div>
    <div class="chart-card"><h3>YOLOv26 vs YOLOv12 &mdash; Best Epoch</h3><canvas id="chartH2H_epoch"></canvas></div>
    <div class="chart-card"><h3>Scratch vs Pretrained &mdash; mAP50-95</h3><canvas id="chartH2H_approach"></canvas></div>
    <div class="chart-card"><h3>AutoBatch Size by Model</h3><canvas id="chartH2H_batch"></canvas></div>
  </div>
</div>

<!-- ============ PER-CLASS TAB ============ -->
<div class="tab-content" id="tab-perclass">
  <div class="card">
    <h2>Per-Class Accuracy</h2>
    <p>Per-class mAP50 and mAP50-95 for each trained model. Sorted by class frequency (most to least common).</p>
    <div id="perclassFilters" style="margin-bottom: 12px;"></div>
    <div class="table-wrap">
      <table id="perclassTable">
        <thead><tr>
          <th>Model</th><th>Device</th>
          <th id="pcClassHeaders"></th>
        </tr></thead>
        <tbody id="perclassTableBody"></tbody>
      </table>
    </div>
  </div>
  <div class="charts-grid">
    <div class="chart-card"><h3>mAP50 by Class</h3><canvas id="chartPerClassMap50"></canvas></div>
    <div class="chart-card"><h3>mAP50-95 by Class</h3><canvas id="chartPerClassMap5095"></canvas></div>
  </div>
</div>

<!-- ============ RESOURCES TAB ============ -->
<div class="tab-content" id="tab-resources">
  <div class="card">
    <h2>Hardware &amp; Model Resources</h2>
    <p style="color:#64748b;margin-bottom:16px">Model complexity, GPU memory usage during training, and file sizes</p>
    <div class="table-wrap">
      <table id="hwTable">
        <thead><tr>
          <th>Model</th><th>Device</th><th>Layers</th><th>Params</th><th>GFLOPs</th>
          <th>GPU</th><th>Train GPU Mem</th><th>AutoBatch Mem</th><th>AutoBatch %</th>
          <th>Weights (MB)</th>
        </tr></thead>
        <tbody id="hwTableBody"></tbody>
      </table>
    </div>
  </div>
  <div class="charts-grid">
    <div class="chart-card"><h3>Parameters by Model</h3><canvas id="chartHwParams"></canvas></div>
    <div class="chart-card"><h3>GFLOPs by Model</h3><canvas id="chartHwGflops"></canvas></div>
    <div class="chart-card"><h3>Training GPU Memory (GB)</h3><canvas id="chartHwGpuMem"></canvas></div>
    <div class="chart-card"><h3>Model File Size (MB)</h3><canvas id="chartHwFileSize"></canvas></div>
  </div>
</div>

<p class="timestamp">Generated: <span id="genTime"></span></p>

<script>
const TRAIN_DATA = {train_json};
const INFER_DATA = {infer_json};
const TRAIN_REPORTS = {train_reports_json};
const HW_DATA = {hw_json};

const COLORS = [
  '#38bdf8','#f472b6','#34d399','#fb923c','#a78bfa',
  '#facc15','#f87171','#2dd4bf','#e879f9','#60a5fa',
  '#4ade80','#fbbf24','#c084fc','#22d3ee','#fb7185'
];
const SIZE_ORDER = {{'nano':0,'small':1,'medium':2,'large':3}};

let charts = {{}};
let trainFilters = {{ device: 'all', arch: 'all', size: 'all', approach: 'all', experiment: 'all' }};

document.getElementById('genTime').textContent = new Date().toLocaleString();

// ---- TABS ----
document.querySelectorAll('.tab-btn').forEach(btn => {{
  btn.addEventListener('click', () => {{
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
    // Resize charts in newly visible tab without flicker
    const activeTab = document.getElementById('tab-' + btn.dataset.tab);
    activeTab.style.visibility = 'hidden';
    requestAnimationFrame(() => {{
      activeTab.querySelectorAll('canvas').forEach(canvas => {{
        const chart = Chart.getChart(canvas);
        if (chart) chart.resize();
      }});
      activeTab.style.visibility = '';
    }});
  }});
}});

function getColor(i) {{ return COLORS[i % COLORS.length]; }}
function shortName(m) {{
  const ap = m.approach.includes('balanced') ? (m.approach.includes('pretrained') ? '(PB)' : '(SB)') :
             m.approach === 'pretrained' ? '(P)' : '';
  return m.arch.replace('yolo','Y') + '-' + m.size[0].toUpperCase() + ap;
}}
function chartDefaults() {{
  Chart.defaults.color = '#94a3b8';
  Chart.defaults.borderColor = '#334155';
  Chart.defaults.font.family = "'Segoe UI', system-ui, sans-serif";
}}
function destroyCharts() {{ Object.values(charts).forEach(c => c.destroy()); charts = {{}}; }}
function fmtPct(v) {{ return v != null ? (v * 100).toFixed(2) + '%' : '-'; }}
function fmtMs(v) {{ return v != null ? v.toFixed(2) : '-'; }}
function fmtNum(v, d) {{ return v != null ? Number(v).toFixed(d || 1) : '-'; }}

function filteredTrain() {{
  return TRAIN_DATA.filter(m => {{
    if (trainFilters.device !== 'all' && m.device !== trainFilters.device) return false;
    if (trainFilters.arch !== 'all' && m.arch !== trainFilters.arch) return false;
    if (trainFilters.size !== 'all' && m.size !== trainFilters.size) return false;
    if (trainFilters.approach !== 'all' && m.approach !== trainFilters.approach) return false;
    if (trainFilters.experiment !== 'all' && m.experiment !== trainFilters.experiment) return false;
    return true;
  }});
}}

// ---- OVERVIEW ----
function renderOverview() {{
  const models = TRAIN_DATA;
  const bestModel = models.length > 0 ? models[0] : null;
  const totalTime = models.reduce((s, m) => s + m.training_time_s, 0);
  const hours = (totalTime / 3600).toFixed(1);

  const devices = [...new Set([...TRAIN_DATA.map(m=>m.device), ...INFER_DATA.map(m=>m.device)])];
  document.getElementById('overviewStats').innerHTML = `
    <div class="stat-card"><div class="value">${{devices.length}}</div><div class="label">Devices</div></div>
    <div class="stat-card"><div class="value">${{models.length}}</div><div class="label">Models Trained</div></div>
    <div class="stat-card"><div class="value">${{INFER_DATA.length}}</div><div class="label">Inference Runs</div></div>
    <div class="stat-card"><div class="value">${{hours}}h</div><div class="label">Total Training</div></div>
    <div class="stat-card"><div class="value">${{bestModel ? fmtPct(bestModel.best_mAP50_95_B) : '-'}}</div><div class="label">Best mAP50-95</div></div>
    <div class="stat-card"><div class="value">${{bestModel ? shortName(bestModel) : '-'}}</div><div class="label">Top Model</div></div>
  `;

  // Training summary table
  const bestMap = models.length ? Math.max(...models.map(m => m.best_mAP50_95_B)) : 0;
  const tbody = document.getElementById('trainTableBody');

  // Get batch, conf, f1 from train reports
  const batchMap = {{}};
  const confMap = {{}};
  const f1Map = {{}};
  TRAIN_REPORTS.forEach(r => {{
    const key = r.name + '_' + r.experiment + '_' + r.device;
    batchMap[key] = r.batch;
    confMap[key] = r.best_conf;
    f1Map[key] = r.best_f1;
  }});

  tbody.innerHTML = models.map(m => {{
    const isBest = m.best_mAP50_95_B === bestMap;
    const rkey = m.name + '_' + m.experiment + '_' + m.device;
    const batch = batchMap[rkey] || '-';
    const conf = confMap[rkey];
    const f1 = f1Map[rkey];
    return `<tr class="${{isBest ? 'best-row' : ''}}">
      <td><span class="device-badge device-${{m.device}}">${{m.device}}</span></td>
      <td>${{m.experiment}}</td>
      <td><span class="arch-badge arch-${{m.arch}}">${{m.arch.toUpperCase()}}</span></td>
      <td><span class="size-badge">${{m.size}}</span></td>
      <td>${{m.task}}</td>
      <td class="approach-${{m.approach}}">${{m.approach}}</td>
      <td>${{batch}}</td>
      <td>${{m.total_epochs}}</td>
      <td>${{m.training_time}}</td>
      <td>${{m.best_epoch}}</td>
      <td>${{fmtPct(m.best_mAP50_B)}}</td>
      <td>${{fmtPct(m.best_mAP50_M)}}</td>
      <td class="${{isBest ? 'metric-highlight' : ''}}">${{fmtPct(m.best_mAP50_95_B)}}</td>
      <td>${{fmtPct(m.best_precision_B)}}</td>
      <td>${{fmtPct(m.best_recall_B)}}</td>
      <td>${{f1 != null ? fmtPct(f1) : '-'}}</td>
      <td>${{conf != null ? conf.toFixed(2) : '-'}}</td>
    </tr>`;
  }}).join('');

  // Bar chart
  chartDefaults();
  const labels = models.map(m => shortName(m));
  if (charts.overviewBar) charts.overviewBar.destroy();
  charts.overviewBar = new Chart(document.getElementById('chartOverviewBar'), {{
    type: 'bar',
    data: {{
      labels,
      datasets: [{{
        label: 'mAP50-95 (Box)',
        data: models.map(m => m.best_mAP50_95_B * 100),
        backgroundColor: models.map((m, i) => getColor(i) + 'cc'),
        borderColor: models.map((m, i) => getColor(i)),
        borderWidth: 1, borderRadius: 4
      }}, {{
        label: 'mAP50-95 (Mask)',
        data: models.map(m => m.best_mAP50_95_M * 100),
        backgroundColor: models.map((m, i) => getColor(i) + '44'),
        borderColor: models.map((m, i) => getColor(i)),
        borderWidth: 1, borderRadius: 4
      }}]
    }},
    options: {{
      responsive: true,
      plugins: {{ legend: {{ position: 'top' }} }},
      scales: {{
        y: {{ beginAtZero: true, title: {{ display: true, text: 'mAP50-95 (%)' }}, grid: {{ color: '#1e293b' }} }},
        x: {{ grid: {{ display: false }} }}
      }}
    }}
  }});

  // Scatter: time vs accuracy
  if (charts.timeVsAcc) charts.timeVsAcc.destroy();
  charts.timeVsAcc = new Chart(document.getElementById('chartTimeVsAcc'), {{
    type: 'scatter',
    data: {{
      datasets: models.map((m, i) => ({{
        label: shortName(m),
        data: [{{ x: m.training_time_s / 3600, y: m.best_mAP50_95_B * 100 }}],
        backgroundColor: getColor(i),
        borderColor: getColor(i),
        pointRadius: 8, pointHoverRadius: 12
      }}))
    }},
    options: {{
      responsive: true,
      plugins: {{ legend: {{ display: false }} }},
      scales: {{
        x: {{ title: {{ display: true, text: 'Training Time (hours)' }}, grid: {{ color: '#1e293b' }} }},
        y: {{ title: {{ display: true, text: 'mAP50-95 (%)' }}, grid: {{ color: '#1e293b' }} }}
      }}
    }}
  }});

  // F1 bar chart
  const f1Labels = models.map(m => shortName(m));
  const f1Values = models.map(m => {{
    const rkey = m.name + '_' + m.experiment + '_' + m.device;
    const v = f1Map[rkey];
    return v != null ? v * 100 : null;
  }});
  if (charts.f1Bar) charts.f1Bar.destroy();
  charts.f1Bar = new Chart(document.getElementById('chartF1Bar'), {{
    type: 'bar',
    data: {{
      labels: f1Labels,
      datasets: [{{ label: 'F1 (%)', data: f1Values, backgroundColor: f1Labels.map((_, i) => getColor(i)) }}]
    }},
    options: {{
      responsive: true,
      plugins: {{ legend: {{ display: false }} }},
      scales: {{ y: {{ beginAtZero: false, grid: {{ color: '#1e293b' }} }} }}
    }}
  }});

  // Confidence threshold bar chart
  const confLabels = models.map(m => shortName(m));
  const confValues = models.map(m => {{
    const rkey = m.name + '_' + m.experiment + '_' + m.device;
    return confMap[rkey] != null ? confMap[rkey] : null;
  }});
  if (charts.confBar) charts.confBar.destroy();
  charts.confBar = new Chart(document.getElementById('chartConfBar'), {{
    type: 'bar',
    data: {{
      labels: confLabels,
      datasets: [{{ label: 'Conf threshold', data: confValues, backgroundColor: confLabels.map((_, i) => getColor(i)) }}]
    }},
    options: {{
      responsive: true,
      plugins: {{ legend: {{ display: false }} }},
      scales: {{ y: {{ beginAtZero: false, max: 1.0, grid: {{ color: '#1e293b' }} }} }}
    }}
  }});
}}

// ---- TRAINING CURVES ----
function buildTrainingFilters() {{
  const devs = [...new Set(TRAIN_DATA.map(m => m.device))].sort();
  const exps = [...new Set(TRAIN_DATA.map(m => m.experiment))].sort();
  const archs = [...new Set(TRAIN_DATA.map(m => m.arch))].sort();
  const sizes = ['nano','small','medium','large'].filter(s => TRAIN_DATA.some(m => m.size === s));
  const approaches = [...new Set(TRAIN_DATA.map(m => m.approach))].sort();

  let html = '';
  const groups = [
    ['device', 'Device', devs],
    ['experiment', 'Exp', exps],
    ['arch', 'Arch', archs],
    ['size', 'Size', sizes],
    ['approach', 'Approach', approaches]
  ];
  for (const [key, label, vals] of groups) {{
    html += `<div class="filter-group"><label>${{label}}:</label>`;
    html += `<button class="filter-btn active" data-filter="${{key}}" data-value="all">All</button>`;
    for (const v of vals) {{
      html += `<button class="filter-btn" data-filter="${{key}}" data-value="${{v}}">${{v}}</button>`;
    }}
    html += '</div>';
  }}
  document.getElementById('trainingFilters').innerHTML = html;

  document.querySelectorAll('#trainingFilters .filter-btn').forEach(btn => {{
    btn.addEventListener('click', () => {{
      const ft = btn.dataset.filter;
      trainFilters[ft] = btn.dataset.value;
      document.querySelectorAll(`#trainingFilters .filter-btn[data-filter="${{ft}}"]`).forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      renderTrainingCurves();
    }});
  }});
}}

function makeCurveChart(canvasId, data, yField, yLabel) {{
  const key = canvasId;
  if (charts[key]) charts[key].destroy();
  charts[key] = new Chart(document.getElementById(canvasId), {{
    type: 'line',
    data: {{
      datasets: data.map((m, i) => ({{
        label: shortName(m),
        data: m.epochs.map(e => ({{ x: e.epoch, y: typeof yField === 'function' ? yField(e) : e[yField] * 100 }})),
        borderColor: getColor(i),
        backgroundColor: getColor(i) + '22',
        borderWidth: 1.5, pointRadius: 0, tension: 0.3
      }}))
    }},
    options: {{
      responsive: true,
      interaction: {{ mode: 'nearest', axis: 'x', intersect: false }},
      plugins: {{ legend: {{ position: 'top', labels: {{ boxWidth: 12, font: {{ size: 10 }} }} }} }},
      scales: {{
        x: {{ type: 'linear', title: {{ display: true, text: 'Epoch' }}, grid: {{ color: '#1e293b' }} }},
        y: {{ title: {{ display: true, text: yLabel }}, grid: {{ color: '#1e293b' }} }}
      }}
    }}
  }});
}}

function renderTrainingCurves() {{
  chartDefaults();
  const data = filteredTrain();
  makeCurveChart('chartCurvesB', data, 'mAP50_B', 'mAP50 Box (%)');
  makeCurveChart('chartCurves95B', data, 'mAP50_95_B', 'mAP50-95 Box (%)');
  makeCurveChart('chartCurvesM', data, 'mAP50_M', 'mAP50 Mask (%)');
  makeCurveChart('chartCurves95M', data, 'mAP50_95_M', 'mAP50-95 Mask (%)');
  makeCurveChart('chartTrainLoss', data,
    e => e.train_box_loss + e.train_seg_loss + e.train_cls_loss, 'Train Loss');
  makeCurveChart('chartValLoss', data,
    e => e.val_box_loss + e.val_seg_loss + e.val_cls_loss, 'Val Loss');
  makeCurveChart('chartPrecCurves', data, 'precision_B', 'Precision Box (%)');
  makeCurveChart('chartRecCurves', data, 'recall_B', 'Recall Box (%)');
}}

// ---- CONVERGENCE ----
function renderConvergence() {{
  chartDefaults();
  const models = TRAIN_DATA;
  const tbody = document.getElementById('convTableBody');
  tbody.innerHTML = models.map(m => {{
    const patience = m.total_epochs - m.best_epoch;
    const secPerEpoch = m.training_time_s / m.total_epochs;
    return `<tr>
      <td><span class="device-badge device-${{m.device}}">${{m.device}}</span></td>
      <td>${{m.name.replace(/_/g,' ')}}</td>
      <td><span class="arch-badge arch-${{m.arch}}">${{m.arch.toUpperCase()}}</span></td>
      <td><span class="size-badge">${{m.size}}</span></td>
      <td class="approach-${{m.approach}}">${{m.approach}}</td>
      <td>${{m.total_epochs}}</td>
      <td>${{m.best_epoch}}</td>
      <td>${{patience}}</td>
      <td class="metric-highlight">${{fmtPct(m.best_mAP50_95_B)}}</td>
      <td>${{m.training_time}}</td>
      <td>${{secPerEpoch.toFixed(1)}}</td>
    </tr>`;
  }}).join('');

  // Best epoch chart
  const sizes = ['nano','small','medium','large'];
  const archGroups = [...new Set(models.map(m => m.arch + '_' + m.approach))].sort();
  if (charts.convEpoch) charts.convEpoch.destroy();
  charts.convEpoch = new Chart(document.getElementById('chartConvEpoch'), {{
    type: 'bar',
    data: {{
      labels: sizes,
      datasets: archGroups.map((ag, i) => {{
        const [arch, ...apParts] = ag.split('_');
        const approach = apParts.join('_');
        return {{
          label: ag.replace('_', ' '),
          data: sizes.map(s => {{
            const m = models.find(m => m.arch === arch && m.size === s && m.approach === approach);
            return m ? m.best_epoch : null;
          }}),
          backgroundColor: getColor(i) + 'aa',
          borderColor: getColor(i),
          borderWidth: 1, borderRadius: 4
        }};
      }})
    }},
    options: {{
      responsive: true,
      plugins: {{ legend: {{ position: 'top' }} }},
      scales: {{
        y: {{ title: {{ display: true, text: 'Best Epoch' }}, grid: {{ color: '#1e293b' }} }},
        x: {{ grid: {{ display: false }} }}
      }}
    }}
  }});

  // Efficiency chart (mAP per hour)
  if (charts.efficiency) charts.efficiency.destroy();
  const labels = models.map(m => shortName(m));
  charts.efficiency = new Chart(document.getElementById('chartEfficiency'), {{
    type: 'bar',
    data: {{
      labels,
      datasets: [{{
        label: 'mAP50-95 / hour',
        data: models.map(m => m.training_time_s > 0 ? (m.best_mAP50_95_B * 100) / (m.training_time_s / 3600) : 0),
        backgroundColor: models.map((m, i) => getColor(i) + 'cc'),
        borderColor: models.map((m, i) => getColor(i)),
        borderWidth: 1, borderRadius: 4
      }}]
    }},
    options: {{
      responsive: true,
      plugins: {{ legend: {{ display: false }} }},
      scales: {{
        y: {{ title: {{ display: true, text: 'mAP50-95 (%) per hour' }}, grid: {{ color: '#1e293b' }} }},
        x: {{ grid: {{ display: false }} }}
      }}
    }}
  }});
}}

// ---- INFERENCE ----
function renderInference() {{
  chartDefaults();
  const data = INFER_DATA;
  const tbody = document.getElementById('inferTableBody');
  tbody.innerHTML = data.map(r => {{
    return `<tr>
      <td><span class="device-badge device-${{r.device}}">${{r.device}}</span></td>
      <td>${{r.experiment}}</td>
      <td><span class="arch-badge arch-${{r.arch}}">${{r.arch.toUpperCase()}}</span></td>
      <td><span class="size-badge">${{r.size}}</span></td>
      <td class="approach-${{r.approach}}">${{r.approach}}</td>
      <td>${{r.format || '-'}}</td>
      <td>${{r.format_precision || '-'}}</td>
      <td>${{r.imgsz || '-'}}</td>
      <td>${{r.batch || '-'}}</td>
      <td>${{fmtMs(r.inference_ms)}}</td>
      <td>${{fmtMs(r.total_ms)}}</td>
      <td>${{fmtNum(r.fps, 1)}}</td>
      <td>${{fmtMs(r.median_ms)}}</td>
      <td>${{fmtMs(r.p95_ms)}}</td>
      <td>${{fmtPct(r.map50)}}</td>
      <td>${{fmtPct(r.map50_95)}}</td>
      <td>${{fmtNum(r.model_size_mb, 1)}}</td>
      <td>${{fmtNum(r.gpu_mem_mb, 0)}}</td>
    </tr>`;
  }}).join('');

  if (data.length === 0) return;

  // FPS bar
  const labels = data.map(r => shortName(r));
  if (charts.inferFPS) charts.inferFPS.destroy();
  charts.inferFPS = new Chart(document.getElementById('chartInferFPS'), {{
    type: 'bar',
    data: {{
      labels,
      datasets: [{{
        label: 'FPS',
        data: data.map(r => r.fps || 0),
        backgroundColor: data.map((r, i) => getColor(i) + 'cc'),
        borderColor: data.map((r, i) => getColor(i)),
        borderWidth: 1, borderRadius: 4
      }}]
    }},
    options: {{
      responsive: true, plugins: {{ legend: {{ display: false }} }},
      scales: {{
        y: {{ title: {{ display: true, text: 'FPS' }}, grid: {{ color: '#1e293b' }} }},
        x: {{ grid: {{ display: false }} }}
      }}
    }}
  }});

  // Latency stacked bar
  if (charts.latency) charts.latency.destroy();
  charts.latency = new Chart(document.getElementById('chartLatency'), {{
    type: 'bar',
    data: {{
      labels,
      datasets: [
        {{ label: 'Preprocess', data: data.map(r => r.preprocess_ms || 0), backgroundColor: '#38bdf8aa', borderColor: '#38bdf8', borderWidth: 1 }},
        {{ label: 'Inference', data: data.map(r => r.inference_ms || 0), backgroundColor: '#f472b6aa', borderColor: '#f472b6', borderWidth: 1 }},
        {{ label: 'Postprocess', data: data.map(r => r.postprocess_ms || 0), backgroundColor: '#34d399aa', borderColor: '#34d399', borderWidth: 1 }}
      ]
    }},
    options: {{
      responsive: true, plugins: {{ legend: {{ position: 'top' }} }},
      scales: {{
        y: {{ stacked: true, title: {{ display: true, text: 'ms/image' }}, grid: {{ color: '#1e293b' }} }},
        x: {{ stacked: true, grid: {{ display: false }} }}
      }}
    }}
  }});
}}

// ---- HEAD-TO-HEAD ----
function renderComparison() {{
  chartDefaults();
  const models = TRAIN_DATA;
  const sizes = ['nano','small','medium','large'];

  function groupedBar(canvasId, field, yLabel, groups) {{
    if (charts[canvasId]) charts[canvasId].destroy();
    charts[canvasId] = new Chart(document.getElementById(canvasId), {{
      type: 'bar',
      data: {{
        labels: sizes,
        datasets: groups.map((g, i) => ({{
          label: g.label,
          data: sizes.map(s => {{
            const m = models.find(m => g.match(m) && m.size === s);
            return m ? (typeof field === 'function' ? field(m) : m[field] * 100) : null;
          }}),
          backgroundColor: getColor(i) + 'bb',
          borderColor: getColor(i),
          borderWidth: 1, borderRadius: 4
        }}))
      }},
      options: {{
        responsive: true, plugins: {{ legend: {{ position: 'top' }} }},
        scales: {{
          y: {{ title: {{ display: true, text: yLabel }}, grid: {{ color: '#1e293b' }} }},
          x: {{ grid: {{ display: false }} }}
        }}
      }}
    }});
  }}

  // v26 vs v12 mAP
  const archApproaches = [...new Set(models.map(m => m.arch + '|' + m.approach))].sort();
  groupedBar('chartH2H_map', 'best_mAP50_95_B', 'mAP50-95 (%)',
    archApproaches.map(aa => {{
      const [arch, approach] = aa.split('|');
      return {{ label: arch + ' ' + approach, match: m => m.arch === arch && m.approach === approach }};
    }})
  );

  // v26 vs v12 best epoch
  groupedBar('chartH2H_epoch', m => m.best_epoch, 'Best Epoch',
    archApproaches.map(aa => {{
      const [arch, approach] = aa.split('|');
      return {{ label: arch + ' ' + approach, match: m => m.arch === arch && m.approach === approach }};
    }})
  );

  // Scratch vs pretrained
  const approaches = [...new Set(models.map(m => m.approach))].sort();
  groupedBar('chartH2H_approach', 'best_mAP50_95_B', 'mAP50-95 (%)',
    approaches.map(a => ({{ label: a, match: m => m.approach === a }}))
  );

  // AutoBatch from train reports
  const batchData = {{}};
  TRAIN_REPORTS.forEach(r => {{
    const key = r.arch + '|' + r.approach;
    if (!batchData[key]) batchData[key] = {{}};
    batchData[key][r.size] = r.batch;
  }});
  const batchGroups = Object.keys(batchData).sort();
  if (charts.chartH2H_batch) charts.chartH2H_batch.destroy();
  charts.chartH2H_batch = new Chart(document.getElementById('chartH2H_batch'), {{
    type: 'bar',
    data: {{
      labels: sizes,
      datasets: batchGroups.map((g, i) => ({{
        label: g.replace('|', ' '),
        data: sizes.map(s => batchData[g][s] || null),
        backgroundColor: getColor(i) + 'bb',
        borderColor: getColor(i),
        borderWidth: 1, borderRadius: 4
      }}))
    }},
    options: {{
      responsive: true, plugins: {{ legend: {{ position: 'top' }} }},
      scales: {{
        y: {{ title: {{ display: true, text: 'Batch Size (AutoBatch)' }}, grid: {{ color: '#1e293b' }} }},
        x: {{ grid: {{ display: false }} }}
      }}
    }}
  }});
}}

// ---- PER-CLASS ----
function renderPerClass() {{
  chartDefaults();
  // Collect models that have per_class data
  const modelsWithPC = TRAIN_REPORTS.filter(r => r.per_class);
  if (modelsWithPC.length === 0) {{
    document.getElementById('perclassTableBody').innerHTML =
      '<tr><td colspan="10" style="text-align:center;color:#94a3b8;">No per-class data available yet. Run extract_conf.py after benchmark completes.</td></tr>';
    return;
  }}

  // Class order by frequency (most to least common in the dataset)
  const classOrder = ['Solda','IV-2','IV-1B','IV-4','IV-1A','IV-3','IV-6','IV-5'];
  // Get actual classes from data
  const allClasses = new Set();
  modelsWithPC.forEach(r => Object.keys(r.per_class).forEach(c => allClasses.add(c)));
  const classes = classOrder.filter(c => allClasses.has(c));
  if (classes.length === 0) {{
    // Use whatever classes exist
    allClasses.forEach(c => classes.push(c));
  }}

  // Build table header
  const headerRow = document.getElementById('perclassTable').querySelector('thead tr');
  headerRow.innerHTML = '<th>Model</th><th>Device</th>' +
    classes.map(c => `<th colspan="2">${{c}}</th>`).join('');
  // Sub-header
  const subHeader = document.createElement('tr');
  subHeader.innerHTML = '<th></th><th></th>' +
    classes.map(() => '<th style="font-size:0.7rem;color:#94a3b8;">mAP50</th><th style="font-size:0.7rem;color:#94a3b8;">50-95</th>').join('');
  headerRow.after(subHeader);

  // Build table body
  const tbody = document.getElementById('perclassTableBody');
  tbody.innerHTML = modelsWithPC.map(r => {{
    const label = r.arch.replace('yolo','Y').toUpperCase() + '-' + r.size[0].toUpperCase() +
      (r.approach.includes('balanced') ? '(B)' : r.approach === 'pretrained' ? '(P)' : '');
    const cells = classes.map(c => {{
      const cm = r.per_class[c];
      if (!cm) return '<td>-</td><td>-</td>';
      return `<td>${{(cm.map50 * 100).toFixed(1)}}</td><td>${{(cm.map50_95 * 100).toFixed(1)}}</td>`;
    }}).join('');
    return `<tr>
      <td><span class="arch-badge arch-${{r.arch}}">${{label}}</span></td>
      <td><span class="device-badge device-${{r.device}}">${{r.device}}</span></td>
      ${{cells}}
    </tr>`;
  }}).join('');

  // Grouped bar chart: mAP50 by class
  const modelLabels = modelsWithPC.map(r => {{
    return r.arch.replace('yolo','Y').toUpperCase() + '-' + r.size[0].toUpperCase() +
      (r.approach.includes('balanced') ? '(B)' : r.approach === 'pretrained' ? '(P)' : '');
  }});

  if (charts.pcMap50) charts.pcMap50.destroy();
  charts.pcMap50 = new Chart(document.getElementById('chartPerClassMap50'), {{
    type: 'bar',
    data: {{
      labels: classes,
      datasets: modelsWithPC.map((r, i) => ({{
        label: modelLabels[i],
        data: classes.map(c => r.per_class[c] ? r.per_class[c].map50 * 100 : null),
        backgroundColor: getColor(i),
      }}))
    }},
    options: {{
      responsive: true,
      plugins: {{ legend: {{ position: 'top' }} }},
      scales: {{ y: {{ beginAtZero: true, title: {{ display: true, text: 'mAP50 (%)' }}, grid: {{ color: '#1e293b' }} }} }}
    }}
  }});

  if (charts.pcMap5095) charts.pcMap5095.destroy();
  charts.pcMap5095 = new Chart(document.getElementById('chartPerClassMap5095'), {{
    type: 'bar',
    data: {{
      labels: classes,
      datasets: modelsWithPC.map((r, i) => ({{
        label: modelLabels[i],
        data: classes.map(c => r.per_class[c] ? r.per_class[c].map50_95 * 100 : null),
        backgroundColor: getColor(i),
      }}))
    }},
    options: {{
      responsive: true,
      plugins: {{ legend: {{ position: 'top' }} }},
      scales: {{ y: {{ beginAtZero: true, title: {{ display: true, text: 'mAP50-95 (%)' }}, grid: {{ color: '#1e293b' }} }} }}
    }}
  }});
}}

// ---- RESOURCES ----
function renderResources() {{
  chartDefaults();
  const data = HW_DATA;
  const tbody = document.getElementById('hwTableBody');

  function fmtParams(p) {{
    if (p == null) return '-';
    if (p >= 1e6) return (p / 1e6).toFixed(1) + 'M';
    if (p >= 1e3) return (p / 1e3).toFixed(0) + 'K';
    return p.toString();
  }}

  tbody.innerHTML = data.map(h => {{
    return `<tr>
      <td><strong>${{h.model_key}}</strong></td>
      <td><span class="device-badge device-${{h.device}}">${{h.device}}</span></td>
      <td>${{h.layers || '-'}}</td>
      <td>${{fmtParams(h.params)}}</td>
      <td>${{h.gflops || '-'}}</td>
      <td>${{h.gpu_name || '-'}}</td>
      <td>${{h.train_gpu_mem_gb ? h.train_gpu_mem_gb.toFixed(2) + ' GB' : '-'}}</td>
      <td>${{h.autobatch_mem_gb ? h.autobatch_mem_gb.toFixed(1) + ' GB' : '-'}}</td>
      <td>${{h.autobatch_pct ? h.autobatch_pct + '%' : '-'}}</td>
      <td>${{h.model_file_mb ? h.model_file_mb.toFixed(1) : '-'}}</td>
    </tr>`;
  }}).join('');

  if (data.length === 0) return;
  const labels = data.map(h => h.model_key);

  // Params chart
  if (charts.hwParams) charts.hwParams.destroy();
  charts.hwParams = new Chart(document.getElementById('chartHwParams'), {{
    type: 'bar',
    data: {{
      labels,
      datasets: [{{
        label: 'Parameters (M)',
        data: data.map(h => h.params ? h.params / 1e6 : 0),
        backgroundColor: data.map((h, i) => getColor(i) + 'cc'),
        borderColor: data.map((h, i) => getColor(i)),
        borderWidth: 1, borderRadius: 4
      }}]
    }},
    options: {{
      responsive: true, plugins: {{ legend: {{ display: false }} }},
      scales: {{
        y: {{ title: {{ display: true, text: 'Parameters (millions)' }}, grid: {{ color: '#1e293b' }} }},
        x: {{ grid: {{ display: false }} }}
      }}
    }}
  }});

  // GFLOPs chart
  if (charts.hwGflops) charts.hwGflops.destroy();
  charts.hwGflops = new Chart(document.getElementById('chartHwGflops'), {{
    type: 'bar',
    data: {{
      labels,
      datasets: [{{
        label: 'GFLOPs',
        data: data.map(h => h.gflops || 0),
        backgroundColor: data.map((h, i) => getColor(i) + 'cc'),
        borderColor: data.map((h, i) => getColor(i)),
        borderWidth: 1, borderRadius: 4
      }}]
    }},
    options: {{
      responsive: true, plugins: {{ legend: {{ display: false }} }},
      scales: {{
        y: {{ title: {{ display: true, text: 'GFLOPs' }}, grid: {{ color: '#1e293b' }} }},
        x: {{ grid: {{ display: false }} }}
      }}
    }}
  }});

  // GPU memory chart
  if (charts.hwGpuMem) charts.hwGpuMem.destroy();
  charts.hwGpuMem = new Chart(document.getElementById('chartHwGpuMem'), {{
    type: 'bar',
    data: {{
      labels,
      datasets: [{{
        label: 'Training GPU Mem (GB)',
        data: data.map(h => h.train_gpu_mem_gb || 0),
        backgroundColor: '#f472b6aa',
        borderColor: '#f472b6',
        borderWidth: 1, borderRadius: 4
      }}, {{
        label: 'AutoBatch Allocated (GB)',
        data: data.map(h => h.autobatch_mem_gb || 0),
        backgroundColor: '#38bdf8aa',
        borderColor: '#38bdf8',
        borderWidth: 1, borderRadius: 4
      }}]
    }},
    options: {{
      responsive: true, plugins: {{ legend: {{ position: 'top' }} }},
      scales: {{
        y: {{ title: {{ display: true, text: 'GB' }}, grid: {{ color: '#1e293b' }} }},
        x: {{ grid: {{ display: false }} }}
      }}
    }}
  }});

  // File size chart
  if (charts.hwFileSize) charts.hwFileSize.destroy();
  charts.hwFileSize = new Chart(document.getElementById('chartHwFileSize'), {{
    type: 'bar',
    data: {{
      labels,
      datasets: [{{
        label: 'Weights (MB)',
        data: data.map(h => h.model_file_mb || 0),
        backgroundColor: data.map((h, i) => getColor(i) + 'cc'),
        borderColor: data.map((h, i) => getColor(i)),
        borderWidth: 1, borderRadius: 4
      }}]
    }},
    options: {{
      responsive: true, plugins: {{ legend: {{ display: false }} }},
      scales: {{
        y: {{ title: {{ display: true, text: 'MB' }}, grid: {{ color: '#1e293b' }} }},
        x: {{ grid: {{ display: false }} }}
      }}
    }}
  }});
}}

// ---- INIT ----
renderOverview();
buildTrainingFilters();
renderTrainingCurves();
renderConvergence();
renderInference();
renderComparison();
renderPerClass();
renderResources();
</script>
</body>
</html>"""

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Written {OUTPUT_HTML}")
    print(f"File size: {os.path.getsize(OUTPUT_HTML) / 1024:.1f} KB")



if __name__ == "__main__":
    training_models = read_training_results()
    inference_reports = read_inference_reports()
    train_reports = read_train_reports()

    hw_metrics = read_hardware_metrics()

    print(f"\nDevices found: {list(DEVICE_DIRS.keys())}")
    print(f"\nProcessed {len(training_models)} trained models:")
    for m in training_models:
        print(f"  [{m['device']:12s}] {m['experiment']:25s} {m['name']:40s} epochs={m['total_epochs']:3d} "
              f"mAP50-95(B)={m['best_mAP50_95_B']:.4f} time={m['training_time']}")

    print(f"\nProcessed {len(inference_reports)} inference reports")
    print(f"Processed {len(train_reports)} training reports")

    if hw_metrics:
        print(f"\nHardware metrics:")
        for h in hw_metrics:
            print(f"  {h['model_key']:20s} params={h['params']:>12,}  GFLOPs={h['gflops']:>6.1f}  "
                  f"GPU mem={h.get('train_gpu_mem_gb', 0) or 0:.1f}G  "
                  f"weights={h.get('model_file_mb', 0) or 0:.1f}MB")

    build_html(training_models, inference_reports, train_reports, hw_metrics)
    print("\nDone!")
