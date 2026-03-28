"""Generic training script for YOLO benchmark.

Usage:
    python scripts/train.py --arch yolo26 --size nano --task segment --approach scratch
    python scripts/train.py --arch yolo12 --size large --task detect --approach pretrained
"""

import argparse
import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO
from scripts.utils import (
    PROJECT_ROOT,
    load_hyperparams,
    get_data_yaml_path,
    get_model_config,
    get_results_dir,
    get_machine_name,
    save_report,
    format_duration,
)


def train_model(architecture, model_size, task, approach, experiment_name="core_comparison",
                hyperparam_overrides=None):
    """Train a YOLO model with the given configuration.

    Args:
        architecture: 'yolo26' or 'yolo12'
        model_size: 'nano', 'small', 'medium', 'large'
        task: 'segment' or 'detect'
        approach: 'scratch', 'pretrained', 'scratch_balanced', or 'pretrained_balanced'
        experiment_name: Name of the experiment for results organization.
        hyperparam_overrides: Optional dict of hyperparameters to override (e.g. epochs, patience).

    Returns:
        Path to the trained weights (best.pt).
    """
    start_time = datetime.datetime.now()
    machine_name = get_machine_name()
    model_config = get_model_config(architecture, task, model_size)

    print("=" * 60)
    print(f"TRAINING: {architecture} {model_size} | {task} | {approach}")
    print(f"Model config: {model_config}")
    print(f"Machine: {machine_name}")
    print(f"Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Load hyperparameters
    hyperparams = load_hyperparams()
    train_params = {
        "data": get_data_yaml_path(),
        "task": task,
    }
    train_params.update(hyperparams)

    # Apply overrides (e.g. --quick-test reduces epochs/patience)
    if hyperparam_overrides:
        train_params.update(hyperparam_overrides)

    # Approach-specific config
    if "pretrained" in approach:
        train_params["pretrained"] = True
    else:
        train_params["pretrained"] = False

    # Weighted sampling for balanced approaches
    if "balanced" in approach:
        from scripts.weighted_sampler import apply_weighted_sampling
        apply_weighted_sampling(get_data_yaml_path())

    # Lower lr for models prone to NaN loss (attention-heavy or large capacity)
    if model_size == "large" or (architecture == "yolo12" and model_size == "medium"):
        train_params["lr0"] = 0.0005

    # Results directory
    results_dir = get_results_dir(
        experiment_name, architecture, task, model_size, approach, "rtx5090"
    )
    os.makedirs(results_dir, exist_ok=True)

    try:
        # Initialize and train
        model = YOLO(model_config)
        results = model.train(
            project=results_dir,
            name="train",
            **train_params,
        )

        # Capture the actual batch size used (AutoBatch resolves -1 to a real value)
        trainer = getattr(model, "trainer", None)
        if trainer is not None:
            actual_batch = getattr(trainer, "batch_size", None) or getattr(trainer.args, "batch", None)
        else:
            actual_batch = None

        # Restore default sampling if balanced was used
        if "balanced" in approach:
            from scripts.weighted_sampler import restore_default_sampling
            restore_default_sampling()

        save_dir = results.save_dir

        # Run validation to get speed metrics
        print("\n--- Extracting inference metrics ---")
        val_results = model.val(
            data=train_params["data"],
            imgsz=train_params["imgsz"],
            split="val",
            project=save_dir,
            name="val",
        )

        speed = val_results.speed
        t_pre = speed.get("preprocess", 0.0)
        t_inf = speed.get("inference", 0.0)
        t_post = speed.get("postprocess", 0.0)
        t_total_ms = t_pre + t_inf + t_post
        fps = 1000.0 / t_total_ms if t_total_ms > 0 else 0.0

        # Accuracy metrics
        map50 = float(val_results.box.map50) if hasattr(val_results, "box") else 0.0
        map50_95 = float(val_results.box.map) if hasattr(val_results, "box") else 0.0

        # Optimal confidence threshold (from F1-confidence curve)
        best_conf = None
        best_f1 = None
        try:
            box = val_results.box
            if hasattr(box, 'f1') and box.f1 is not None:
                import numpy as np
                mean_f1 = box.f1.mean(0)
                best_idx = int(mean_f1.argmax())
                best_f1 = float(mean_f1[best_idx])
                if hasattr(box, 'conf') and box.conf is not None:
                    best_conf = float(box.conf[best_idx])
        except Exception:
            pass

        # Per-class metrics
        per_class_data = None
        precision_mean = 0.0
        recall_mean = 0.0
        if hasattr(val_results, "box") and hasattr(val_results.box, "class_result"):
            class_names = val_results.names if hasattr(val_results, "names") else {}
            p_cls, r_cls, map50_cls, map50_95_cls = val_results.box.class_result
            per_class_data = {}
            for idx in range(len(p_cls)):
                name = class_names.get(idx, f"class_{idx}")
                per_class_data[name] = {
                    "precision": float(p_cls[idx]),
                    "recall": float(r_cls[idx]),
                    "map50": float(map50_cls[idx]),
                    "map50_95": float(map50_95_cls[idx]),
                }
            precision_mean = float(sum(p_cls) / len(p_cls)) if len(p_cls) > 0 else 0.0
            recall_mean = float(sum(r_cls) / len(r_cls)) if len(r_cls) > 0 else 0.0

        end_time = datetime.datetime.now()
        duration = end_time - start_time

        # Save report
        report_data = {
            "machine": machine_name,
            "device": "rtx5090",
            "experiment": experiment_name,
            "architecture": architecture,
            "model_size": model_size,
            "task": task,
            "approach": approach,
            "format": "pytorch",
            "format_precision": "fp32",
            "imgsz": train_params["imgsz"],
            "batch": actual_batch,
            "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration": format_duration(duration),
            "preprocess_ms": t_pre,
            "inference_ms": t_inf,
            "postprocess_ms": t_post,
            "total_ms": t_total_ms,
            "fps": fps,
            "map50": map50,
            "map50_95": map50_95,
            "precision": precision_mean,
            "recall": recall_mean,
            "best_conf": best_conf,
            "best_f1": best_f1,
            "per_class": per_class_data,
            "watts": None,
            "hyperparams": train_params,
        }
        report_path = os.path.join(results_dir, "report.txt")
        save_report(report_path, report_data)

        weights_path = os.path.join(save_dir, "weights", "best.pt")
        print(f"\nTraining complete. Batch size: {actual_batch} | Weights: {weights_path}")
        return {"weights_path": weights_path, "actual_batch": actual_batch}

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Train a YOLO model for benchmark")
    parser.add_argument("--arch", required=True, choices=["yolo26", "yolo12"],
                        help="Model architecture")
    parser.add_argument("--size", required=True, choices=["nano", "small", "medium", "large"],
                        help="Model size")
    parser.add_argument("--task", required=True, choices=["segment", "detect"],
                        help="Task type")
    parser.add_argument("--approach", required=True,
                        choices=["scratch", "pretrained",
                                 "scratch_balanced", "pretrained_balanced"],
                        help="Training approach")
    parser.add_argument("--experiment", default="core_comparison",
                        help="Experiment name for results organization")
    parser.add_argument("--quick-test", action="store_true",
                        help="Quick smoke test: 2 epochs, patience=2")
    args = parser.parse_args()

    overrides = {"epochs": 2, "patience": 2} if args.quick_test else None
    train_model(args.arch, args.size, args.task, args.approach, args.experiment,
                hyperparam_overrides=overrides)


if __name__ == "__main__":
    main()
