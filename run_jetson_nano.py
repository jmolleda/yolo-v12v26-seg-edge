"""Orchestrator for Jetson Orin Nano benchmark runs.

Handles TensorRT export and inference. Weights must be copied from RTX 5090 first.
Gracefully handles OOM errors (8GB shared memory) by skipping and continuing.
Supports resume: skips runs that have already completed.

Usage:
    python run_jetson_nano.py
    python run_jetson_nano.py --dry-run
"""

import argparse
import os

from scripts.utils import (
    PROJECT_ROOT,
    load_experiments,
    get_results_dir,
    get_weights_path,
)
from scripts.benchmark_logger import BenchmarkLogger
from scripts.aggregate import find_reports, write_csv

DEVICE = "jetson_nano"

# Combinations likely to OOM on Orin Nano (8GB shared RAM)
OOM_SKIP_RULES = [
    {"model_size": "large", "batch": 16},
    {"model_size": "large", "imgsz": 1280, "batch": 8},
    {"model_size": "medium", "batch": 16, "imgsz": 1280},
]


def should_skip_oom(run):
    """Check if a run is likely to OOM on the Orin Nano."""
    for rule in OOM_SKIP_RULES:
        if all(run.get(k) == v for k, v in rule.items()):
            return True
    return False


def resolve_weights(run):
    """Find the weights path for a run. Returns (pt_path, final_path, needs_export)."""
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

    if run["format"] == "tensorrt":
        engine_path = pt_path.replace(".pt", ".engine")
        return pt_path, engine_path, True
    return pt_path, pt_path, False


def main():
    parser = argparse.ArgumentParser(description="Jetson Orin Nano benchmark orchestrator")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print runs without executing")
    args = parser.parse_args()

    runs, config = load_experiments(DEVICE)
    logger = BenchmarkLogger(DEVICE)

    export_runs = [r for r in runs if "export" in r["action"]]
    all_infer_runs = runs

    logger.log("info", f"Export runs: {len(export_runs)}, Inference runs: {len(all_infer_runs)}")

    if args.dry_run:
        print("\n--- EXPORT RUNS ---")
        for i, r in enumerate(export_runs, 1):
            oom = " [OOM SKIP]" if should_skip_oom(r) else ""
            print(f"  {i:3d}. {r['architecture']} {r['model_size']} | "
                  f"{r['precision']} | {r['task']} | {r['approach']}{oom}")
        print(f"\n--- INFERENCE RUNS ---")
        for i, r in enumerate(all_infer_runs, 1):
            oom = " [OOM SKIP]" if should_skip_oom(r) else ""
            print(f"  {i:3d}. {r['architecture']} {r['model_size']} | "
                  f"{r['format']} {r['precision']} | {r['task']} | {r['approach']} | "
                  f"img={r['imgsz']} b={r['batch']} | exp={r['experiment_name']}{oom}")
        return

    # Lazy imports (require ultralytics)
    from scripts.export import export_model
    from scripts.infer import run_inference

    logger.register_runs(runs)

    # Phase 1: Export TensorRT models
    logger.set_phase("export")
    exported = set()

    for run in export_runs:
        run_id = BenchmarkLogger.make_run_id(run)
        pt_path, engine_path, _ = resolve_weights(run)
        export_key = (pt_path, run["precision"])

        if should_skip_oom(run):
            logger.skip_run(run_id, "OOM risk")
            continue

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
        except (RuntimeError, MemoryError) as e:
            if "out of memory" in str(e).lower() or isinstance(e, MemoryError):
                logger.skip_run(run_id, f"OOM: {e}")
            else:
                logger.fail_run(run_id, str(e))
        except Exception as e:
            logger.fail_run(run_id, str(e))

    # Phase 2: Inference
    logger.set_phase("inference")

    for run in all_infer_runs:
        run_id = BenchmarkLogger.make_run_id(run)

        if should_skip_oom(run):
            logger.skip_run(run_id, "OOM risk")
            continue

        results_dir = get_results_dir(
            run["experiment_name"], run["architecture"], run["task"],
            run["model_size"], run["approach"], DEVICE,
        )

        report_name = (f"report_{run['format']}_{run['precision']}_"
                       f"img{run['imgsz']}_b{run['batch']}.txt")
        report_path = os.path.join(results_dir, report_name)

        if os.path.exists(report_path):
            logger.skip_run(run_id, "already done")
            continue

        pt_path, engine_path, needs_export = resolve_weights(run)
        weights_path = engine_path if needs_export else pt_path

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
            )
            logger.complete_run(run_id, {
                "fps": result.get("fps", 0),
                "map50": result.get("map50", 0),
                "map50_95": result.get("map50_95", 0),
            })
        except (RuntimeError, MemoryError) as e:
            if "out of memory" in str(e).lower() or isinstance(e, MemoryError):
                logger.skip_run(run_id, f"OOM: {e}")
            else:
                logger.fail_run(run_id, str(e))
        except Exception as e:
            logger.fail_run(run_id, str(e))

    # Phase 3: Aggregate results
    logger.set_phase("aggregation")

    results_dir = os.path.join(PROJECT_ROOT, "results")
    reports = find_reports(results_dir, device_filter=DEVICE)
    if reports:
        output_base = os.path.join(results_dir, DEVICE, "benchmark_results")
        write_csv(reports, output_base + ".csv")

    logger.set_phase("complete")
    logger.log("info", "Jetson Nano benchmark finished")


if __name__ == "__main__":
    main()
