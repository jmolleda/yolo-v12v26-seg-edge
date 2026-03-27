"""Orchestrator for RTX 5090 benchmark runs.

Executes all training and inference runs designated for the RTX 5090.
Supports resume: skips runs that have already completed.

Usage:
    python run_rtx5090.py
    python run_rtx5090.py --dry-run
    python run_rtx5090.py --quick-test
"""

import argparse
import os

from scripts.utils import (
    PROJECT_ROOT,
    load_experiments,
    get_results_dir,
    get_weights_path,
    run_already_completed,
)
from scripts import utils as _utils_module
from scripts.benchmark_logger import BenchmarkLogger
from scripts.aggregate import find_reports, write_csv

DEVICE = "rtx5090"


def main():
    parser = argparse.ArgumentParser(description="RTX 5090 benchmark orchestrator")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print runs without executing")
    parser.add_argument("--quick-test", action="store_true",
                        help="Quick smoke test: 2 epochs, 0 warmup, 1 measurement run")
    args = parser.parse_args()

    # Quick-test overrides
    train_overrides = {}
    infer_overrides = {}
    if args.quick_test:
        train_overrides = {"epochs": 2, "patience": 2}
        infer_overrides = {"warmup_runs": 0, "measure_runs": 1}
        _utils_module.RESULTS_DIR_NAME = "results-quick-test"
        print("*** QUICK-TEST MODE: epochs=2, patience=2, warmup=0, measure=1 ***")
        print("*** Results → results-quick-test/ ***\n")

    runs, config = load_experiments(DEVICE)
    logger = BenchmarkLogger(DEVICE)

    # Separate training, export, and inference runs
    train_runs = [r for r in runs if "train" in r["action"]]
    export_runs = [r for r in runs if "export" in r["action"]]
    infer_runs = runs  # all runs include inference

    logger.log("info", f"Training: {len(train_runs)}, Export: {len(export_runs)}, Inference: {len(infer_runs)}")

    if args.dry_run:
        print("\n--- TRAINING RUNS ---")
        for i, r in enumerate(train_runs, 1):
            print(f"  {i:3d}. {r['architecture']} {r['model_size']} | "
                  f"{r['task']} | {r['approach']} | exp={r['experiment_name']}")
        print(f"\n--- EXPORT RUNS ---")
        for i, r in enumerate(export_runs, 1):
            print(f"  {i:3d}. {r['architecture']} {r['model_size']} | "
                  f"{r['precision']} | {r['task']} | {r['approach']}")
        print(f"\n--- INFERENCE RUNS ---")
        for i, r in enumerate(infer_runs, 1):
            print(f"  {i:3d}. {r['architecture']} {r['model_size']} | "
                  f"{r['format']} {r['precision']} | {r['task']} | {r['approach']} | "
                  f"img={r['imgsz']} b={r['batch']} | exp={r['experiment_name']}")
        return

    # Lazy imports (require ultralytics)
    from scripts.train import train_model
    from scripts.infer import run_inference
    from scripts.export import export_model

    # Register all runs with the logger
    logger.register_runs(runs)

    # Phase 1: Training
    logger.set_phase("training")

    for run in train_runs:
        run_id = BenchmarkLogger.make_run_id(run)
        results_dir = get_results_dir(
            run["experiment_name"], run["architecture"], run["task"],
            run["model_size"], run["approach"], DEVICE,
        )

        if run_already_completed(results_dir, "train+infer"):
            logger.skip_run(run_id, "already done")
            continue

        logger.start_run(run_id)
        try:
            train_result = train_model(
                architecture=run["architecture"],
                model_size=run["model_size"],
                task=run["task"],
                approach=run["approach"],
                experiment_name=run["experiment_name"],
                hyperparam_overrides=train_overrides if train_overrides else None,
            )
            logger.complete_run(run_id, {
                "batch": train_result.get("actual_batch"),
            })
        except Exception as e:
            logger.fail_run(run_id, str(e))

    # Phase 2: Export TensorRT models
    logger.set_phase("export")
    exported = set()

    for run in export_runs:
        run_id = BenchmarkLogger.make_run_id(run)

        if run["experiment_name"] in ("input_size", "batch_throughput"):
            weights_experiment = "core_comparison"
        elif run["experiment_name"] == "detection_vs_segmentation":
            weights_experiment = "detection_vs_segmentation"
        else:
            weights_experiment = run["experiment_name"]

        pt_path = get_weights_path(
            weights_experiment, run["architecture"], run["task"],
            run["model_size"], run["approach"],
        )
        engine_path = pt_path.replace(".pt", ".engine")
        export_key = (pt_path, run["precision"])

        if export_key in exported or os.path.exists(engine_path):
            logger.skip_run(run_id, "already exported")
            exported.add(export_key)
            continue

        if not os.path.exists(pt_path):
            logger.fail_run(run_id, f"no weights: {pt_path}")
            continue

        logger.start_run(run_id)
        try:
            export_model(pt_path, run["precision"], run["imgsz"])
            exported.add(export_key)
            logger.complete_run(run_id)
        except Exception as e:
            logger.fail_run(run_id, str(e))

    # Phase 3: Inference
    logger.set_phase("inference")

    for run in infer_runs:
        run_id = BenchmarkLogger.make_run_id(run)
        results_dir = get_results_dir(
            run["experiment_name"], run["architecture"], run["task"],
            run["model_size"], run["approach"], DEVICE,
        )

        # Determine which experiment produced the weights
        if run["experiment_name"] in ("input_size", "batch_throughput"):
            weights_experiment = "core_comparison"
        elif run["experiment_name"] == "detection_vs_segmentation":
            weights_experiment = "detection_vs_segmentation"
        else:
            weights_experiment = run["experiment_name"]

        pt_path = get_weights_path(
            weights_experiment, run["architecture"], run["task"],
            run["model_size"], run["approach"],
        )

        # Use engine file for TensorRT runs
        if run["format"] == "tensorrt":
            weights_path = pt_path.replace(".pt", ".engine")
        else:
            weights_path = pt_path

        # Check if already run
        report_name = (f"report_{run['format']}_{run['precision']}_"
                       f"img{run['imgsz']}_b{run['batch']}.txt")
        report_path = os.path.join(results_dir, report_name)

        if os.path.exists(report_path):
            logger.skip_run(run_id, "already done")
            continue

        if not os.path.exists(weights_path):
            logger.fail_run(run_id, f"no weights: {weights_path}")
            continue

        logger.start_run(run_id)
        try:
            result = run_inference(
                weights_path=weights_path,
                fmt=run["format"],
                precision=run["precision"],
                imgsz=run["imgsz"],
                batch=run["batch"],
                architecture=run["architecture"],
                model_size=run["model_size"],
                task=run["task"],
                approach=run["approach"],
                experiment_name=run["experiment_name"],
                device_name=DEVICE,
                **infer_overrides,
            )
            logger.complete_run(run_id, {
                "fps": result.get("fps", 0),
                "map50": result.get("map50", 0),
                "map50_95": result.get("map50_95", 0),
            })
        except Exception as e:
            logger.fail_run(run_id, str(e))

    # Phase 4: Aggregate results
    logger.set_phase("aggregation")

    results_dir = os.path.join(PROJECT_ROOT, _utils_module.RESULTS_DIR_NAME)
    reports = find_reports(results_dir, device_filter=DEVICE)
    if reports:
        output_base = os.path.join(results_dir, DEVICE, "benchmark_results")
        write_csv(reports, output_base + ".csv")

    logger.set_phase("complete")
    logger.log("info", "RTX 5090 benchmark finished")


if __name__ == "__main__":
    main()
