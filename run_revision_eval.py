"""JRTIP major-revision evaluation campaign orchestrator.

Re-evaluates the already-trained models on the LEAK-FREE TEST SUBSET
(data/test_clean, 146 images) instead of the validation split, and rebuilds
all TensorRT engines with precision-distinct filenames:

  - FP16: re-export with --verbose (build log for the per-layer audit)
  - INT8: re-export with train-drawn calibration (data/data_calib.yaml)
          -- the original campaign silently reused FP16 engines for INT8
          (best.engine name collision), so INT8 must be rebuilt everywhere.

Results go to results_revision/{device}/... (original results untouched).
Supports resume: a run is skipped if its report file already exists.

Usage (per device):
    python run_revision_eval.py --device rtx5090
    python run_revision_eval.py --device jetson_agx
    python run_revision_eval.py --device jetson_nano
    python run_revision_eval.py --device rtx5090 --dry-run
"""

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.utils import (
    PROJECT_ROOT,
    load_experiments,
    get_results_dir,
    get_weights_path,
)
from scripts import utils as _utils_module

RESULTS_DIR_NAME = "results_revision"
DATA_TEST_CLEAN = os.path.join(PROJECT_ROOT, "data", "data_test_clean.yaml")
DATA_CALIB = os.path.join(PROJECT_ROOT, "data", "data_calib.yaml")

# Experiments whose accuracy numbers appear in the paper and therefore need
# clean-test re-evaluation. batch_throughput / detection_vs_segmentation are
# throughput-only comparisons and keep their original measurements.
EXPERIMENTS = {"core_comparison", "input_size", "class_imbalance"}


def weights_experiment_for(run):
    if run["experiment_name"] in ("input_size", "batch_throughput"):
        return "core_comparison"
    return run["experiment_name"]


def resolve_paths(run):
    """Return (pt_path, weights_for_inference)."""
    pt_path = get_weights_path(
        weights_experiment_for(run), run["architecture"], run["task"],
        run["model_size"], run["approach"],
    )
    if run["format"] == "tensorrt":
        engine = os.path.splitext(pt_path)[0] + f"_{run['precision']}.engine"
        return pt_path, engine
    return pt_path, pt_path


def main():
    parser = argparse.ArgumentParser(description="JRTIP revision evaluation campaign")
    parser.add_argument("--device", required=True,
                        choices=["rtx5090", "jetson_agx", "jetson_nano"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--experiments", default=None,
                        help="Comma-separated override of experiment names")
    args = parser.parse_args()

    experiments = set(args.experiments.split(",")) if args.experiments else EXPERIMENTS

    _utils_module.RESULTS_DIR_NAME = RESULTS_DIR_NAME

    runs, _ = load_experiments(args.device)
    runs = [r for r in runs if r["experiment_name"] in experiments]

    # Export set: one (pt_path, precision) per TRT run
    exports = {}
    for r in runs:
        if r["format"] == "tensorrt":
            pt_path, engine = resolve_paths(r)
            exports[(pt_path, r["precision"])] = (r, pt_path, engine)

    print(f"Device: {args.device}")
    print(f"Experiments: {sorted(experiments)}")
    print(f"Inference runs: {len(runs)}, engine builds: {len(exports)}")
    print(f"Evaluation data: {DATA_TEST_CLEAN}")
    print(f"INT8 calibration: {DATA_CALIB}")
    print(f"Results dir: {RESULTS_DIR_NAME}/{args.device}/\n")

    if args.dry_run:
        print("--- ENGINE BUILDS ---")
        for (pt, prec), (r, _, engine) in sorted(exports.items()):
            status = "EXISTS" if os.path.exists(engine) else "build"
            print(f"  [{status}] {r['architecture']} {r['model_size']} {r['approach']} {prec} -> {os.path.basename(engine)}")
        print("\n--- INFERENCE RUNS ---")
        for r in runs:
            print(f"  {r['experiment_name']} | {r['architecture']} {r['model_size']} | "
                  f"{r['format']} {r['precision']} | {r['approach']} | img={r['imgsz']} b={r['batch']}")
        return

    from scripts.infer import run_inference

    # ── Phase 1: engine builds (precision-distinct, forced for INT8) ──
    failures = []
    for (pt_path, precision), (r, _, engine) in sorted(exports.items()):
        if os.path.exists(engine):
            print(f"[export] exists, skip: {os.path.basename(engine)}")
            continue
        if not os.path.exists(pt_path):
            print(f"[export] MISSING WEIGHTS: {pt_path}")
            failures.append(("export", pt_path, precision, "missing weights"))
            continue
        cmd = [sys.executable, "scripts/export.py",
               "--weights", pt_path,
               "--precision", precision,
               "--imgsz", str(r["imgsz"]),
               "--verbose"]
        if precision == "int8":
            cmd += ["--data", DATA_CALIB]
        print(f"[export] {r['architecture']} {r['model_size']} {r['approach']} {precision}")
        try:
            subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
        except subprocess.CalledProcessError as e:
            print(f"[export] FAILED: {e}")
            failures.append(("export", pt_path, precision, str(e)))

    # ── Phase 2: inference on the clean test subset ──
    for r in runs:
        pt_path, weights = resolve_paths(r)
        results_dir = get_results_dir(
            r["experiment_name"], r["architecture"], r["task"],
            r["model_size"], r["approach"], args.device,
        )
        report_name = (f"report_{r['format']}_{r['precision']}_"
                       f"img{r['imgsz']}_b{r['batch']}.txt")
        report_path = os.path.join(results_dir, report_name)
        label = (f"{r['experiment_name']} | {r['architecture']} {r['model_size']} | "
                 f"{r['format']} {r['precision']} | {r['approach']}")

        if os.path.exists(report_path):
            print(f"[infer] done, skip: {label}")
            continue
        if not os.path.exists(weights):
            print(f"[infer] MISSING: {weights} ({label})")
            failures.append(("infer", weights, r["precision"], "missing weights/engine"))
            continue

        print(f"[infer] {label}")
        try:
            run_inference(
                weights_path=weights,
                fmt=r["format"],
                precision=r["precision"],
                imgsz=r["imgsz"],
                batch=r["batch"],
                architecture=r["architecture"],
                model_size=r["model_size"],
                task=r["task"],
                approach=r["approach"],
                experiment_name=r["experiment_name"],
                device_name=args.device,
                data_yaml_override=DATA_TEST_CLEAN,
            )
        except Exception as e:
            print(f"[infer] FAILED: {label}: {e}")
            failures.append(("infer", weights, r["precision"], str(e)))

    # ── Summary ──
    print("\n" + "=" * 60)
    if failures:
        print(f"{len(failures)} failures:")
        for phase, path, prec, msg in failures:
            print(f"  [{phase}] {os.path.basename(path)} {prec}: {msg[:80]}")
    else:
        print("All runs completed.")


if __name__ == "__main__":
    main()
