# How to Run

## Prerequisites

```bash
pip install ultralytics pyyaml
```

The dataset must be placed in `data/` following the structure defined in `data/data.yaml`.

## Dry Run

Preview all runs without executing:

```bash
python run_rtx5090.py --dry-run
python run_jetson_agx.py --dry-run
python run_jetson_nano.py --dry-run
```

## Execution

### RTX 5090

Trains all models, then runs inference:

```bash
python run_rtx5090.py
```

### Jetson Devices

Three automated phases: (1) copy trained weights from RTX 5090, (2) export to TensorRT, (3) run inference.

```bash
# 1. Copy trained weights (.pt) from RTX 5090 to the Jetson
scp -r user@rtx-host:/path/to/results/rtx5090/ results/rtx5090/

# 2-3. The orchestrator handles TensorRT export + inference automatically
python run_jetson_agx.py
python run_jetson_nano.py
```

!!! warning "TensorRT engines are GPU-architecture specific"
    They cannot be built on the RTX 5090 and copied over. The orchestrator scripts
    automatically export each `.pt` model to a TensorRT `.engine` file on the target
    Jetson (FP16 and INT8 with calibration) before running inference.
    No manual export step is needed.

## Background Execution (SSH)

To keep the process running after closing an SSH session:

```bash
nohup python run_jetson_agx.py > /dev/null 2>&1 &
```

Or use `tmux` / `screen` for an interactive session that survives disconnects.

## Resume

All orchestrators support **automatic resume**. If a run's report file already exists, it is skipped. Simply re-run the same command to continue from where it stopped.

## OOM Protection (Jetson Orin Nano)

The Nano orchestrator proactively skips combinations likely to exceed 8 GB shared memory:

- Large model + batch 16
- Large model + imgsz 1280 + batch 8
- Medium model + batch 16 + imgsz 1280

Runtime OOM errors are caught and the run is marked as skipped rather than failed.
