"""Shared utilities for benchmark scripts: reports, timing, config loading."""

import datetime
import os
import platform
import yaml


# Root directory of the benchmark project
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_experiments(device_name):
    """Load experiments config and return list of expanded runs for a device.

    Args:
        device_name: One of 'rtx5090', 'jetson_agx', 'jetson_nano'.

    Returns:
        List of dicts, each describing a single run with keys:
        experiment_id, experiment_name, format, precision, approach,
        task, imgsz, batch, architecture, model_size, action.
    """
    config_path = os.path.join(PROJECT_ROOT, "config", "experiments.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    runs = []
    for experiment in config["experiments"]:
        for run_template in experiment["runs"]:
            action = run_template["devices"].get(device_name, "skip")
            if action == "skip":
                continue
            for arch in run_template["architectures"]:
                for size in run_template["model_sizes"]:
                    runs.append({
                        "experiment_id": experiment["id"],
                        "experiment_name": experiment["name"],
                        "format": run_template["format"],
                        "precision": run_template["precision"],
                        "approach": run_template["approach"],
                        "task": run_template["task"],
                        "imgsz": run_template["imgsz"],
                        "batch": run_template["batch"],
                        "architecture": arch,
                        "model_size": size,
                        "action": action,
                    })
    return runs, config


def load_hyperparams():
    """Load shared training hyperparameters from hiperparametros.yaml."""
    path = os.path.join(PROJECT_ROOT, "hiperparametros.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_data_yaml_path():
    """Return absolute path to data/data.yaml."""
    return os.path.join(PROJECT_ROOT, "data", "data.yaml")


def get_model_config(architecture, task, model_size):
    """Return the Ultralytics built-in model config string.

    Args:
        architecture: 'yolo26' or 'yolo12'
        task: 'segment' or 'detect'
        model_size: 'nano', 'small', 'medium', 'large'

    Returns:
        Built-in model string like 'yolo26n-seg.yaml' or 'yolo12s.yaml'
    """
    size_key = {"nano": "n", "small": "s", "medium": "m", "large": "l"}[model_size]
    task_suffix = "-seg" if task == "segment" else ""
    return f"{architecture}{size_key}{task_suffix}.yaml"


def get_results_dir(experiment_name, architecture, task, model_size, approach, device_name):
    """Return the results directory path for a specific run.

    Structure: results/{device}/{experiment}/{arch}_{task}_{size}_{approach}/
    """
    task_key = "seg" if task == "segment" else "det"
    folder_name = f"{architecture}_{task_key}_{model_size}_{approach}"
    return os.path.join(PROJECT_ROOT, "results", device_name, experiment_name, folder_name)


def get_weights_path(experiment_name, architecture, task, model_size, approach):
    """Return path to trained weights (best.pt) from RTX 5090 training.

    Training always happens on RTX 5090. Other devices load these weights.
    """
    results_dir = get_results_dir(
        experiment_name, architecture, task, model_size, approach, "rtx5090"
    )
    # Ultralytics saves to {project}/{name}/weights/best.pt
    return os.path.join(results_dir, "train", "weights", "best.pt")


def get_machine_name():
    """Return the machine hostname."""
    return platform.node()



def save_report(filepath, report_data):
    """Save a benchmark report to a text file.

    Args:
        filepath: Output file path.
        report_data: Dict with report fields.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write("=== BENCHMARK REPORT ===\n")
        f.write(f"Machine: {report_data.get('machine', 'unknown')}\n")
        f.write(f"Device: {report_data.get('device', 'unknown')}\n")
        f.write(f"Experiment: {report_data.get('experiment', 'unknown')}\n")
        f.write(f"Architecture: {report_data.get('architecture', 'unknown')}\n")
        f.write(f"Model size: {report_data.get('model_size', 'unknown')}\n")
        f.write(f"Task: {report_data.get('task', 'unknown')}\n")
        f.write(f"Approach: {report_data.get('approach', 'unknown')}\n")
        f.write(f"Format: {report_data.get('format', 'unknown')}\n")
        f.write(f"Precision: {report_data.get('precision', 'unknown')}\n")
        f.write(f"Input size: {report_data.get('imgsz', 'unknown')}\n")
        f.write(f"Batch size: {report_data.get('batch', 'unknown')}\n")
        f.write(f"Start time: {report_data.get('start_time', 'unknown')}\n")
        f.write(f"End time: {report_data.get('end_time', 'unknown')}\n")
        f.write(f"Duration: {report_data.get('duration', 'unknown')}\n")
        f.write("-" * 50 + "\n")

        f.write("--- Performance ---\n")
        f.write(f"Preprocess:  {report_data.get('preprocess_ms', 0.0):.2f} ms/img\n")
        f.write(f"Inference:   {report_data.get('inference_ms', 0.0):.2f} ms/img\n")
        f.write(f"Postprocess: {report_data.get('postprocess_ms', 0.0):.2f} ms/img\n")
        f.write(f"Total:       {report_data.get('total_ms', 0.0):.2f} ms/img\n")
        f.write(f"FPS:         {report_data.get('fps', 0.0):.2f}\n")
        if report_data.get("median_ms") is not None:
            f.write(f"Median:      {report_data['median_ms']:.2f} ms/img\n")
            f.write(f"Std dev:     {report_data['stdev_ms']:.2f} ms\n")
            f.write(f"p95:         {report_data['p95_ms']:.2f} ms/img\n")
            f.write(f"p99:         {report_data['p99_ms']:.2f} ms/img\n")
            f.write(f"Runs:        {report_data.get('measure_runs', 'N/A')}\n")
        f.write("-" * 50 + "\n")

        f.write("--- Resources ---\n")
        f.write(f"Model size:  {report_data.get('model_file_size_mb', 0.0):.1f} MB\n")
        if report_data.get("gpu_mem_peak_mb") is not None:
            f.write(f"GPU mem:     {report_data['gpu_mem_peak_mb']:.1f} MB (peak)\n")
        f.write("-" * 50 + "\n")

        f.write("--- Accuracy ---\n")
        f.write(f"mAP50:     {report_data.get('map50', 0.0):.4f}\n")
        f.write(f"mAP50-95:  {report_data.get('map50_95', 0.0):.4f}\n")
        f.write(f"Precision: {report_data.get('precision', 0.0):.4f}\n")
        f.write(f"Recall:    {report_data.get('recall', 0.0):.4f}\n")
        f.write("-" * 50 + "\n")

        if report_data.get("per_class"):
            f.write("--- Per-class Accuracy ---\n")
            f.write(f"{'Class':10s}  {'P':>8s}  {'R':>8s}  {'mAP50':>8s}  {'mAP50-95':>8s}\n")
            for cls_name, cls_metrics in report_data["per_class"].items():
                f.write(f"{cls_name:10s}  {cls_metrics['precision']:8.4f}  "
                        f"{cls_metrics['recall']:8.4f}  "
                        f"{cls_metrics['map50']:8.4f}  "
                        f"{cls_metrics['map50_95']:8.4f}\n")
            f.write("-" * 50 + "\n")

        if report_data.get("watts") is not None:
            f.write("--- Power ---\n")
            f.write(f"Power:     {report_data.get('watts', 0.0):.2f} W\n")
            f.write(f"FPS/Watt:  {report_data.get('fps_per_watt', 0.0):.2f}\n")
            f.write("-" * 50 + "\n")

        if report_data.get("hyperparams"):
            f.write("--- Training Hyperparameters ---\n")
            for key, value in report_data["hyperparams"].items():
                f.write(f"{key}: {value}\n")


def format_duration(delta):
    """Format a timedelta as HH:MM:SS string."""
    return str(delta).split(".")[0]


def run_already_completed(results_dir, action):
    """Check if a run has already been completed (for resume capability).

    Args:
        results_dir: Path to the run's results directory.
        action: The action type ('train+infer', 'infer', 'export+infer').

    Returns:
        True if the run appears to be already completed.
    """
    report_path = os.path.join(results_dir, "report.txt")
    if action == "train+infer":
        weights_path = os.path.join(results_dir, "train", "weights", "best.pt")
        return os.path.exists(report_path) and os.path.exists(weights_path)
    return os.path.exists(report_path)
