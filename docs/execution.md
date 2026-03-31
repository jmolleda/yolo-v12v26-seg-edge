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

Trains all models, exports to TensorRT (FP16/INT8), then runs inference:

```bash
python run_rtx5090.py
```

#### Background Execution (SSH)

To keep the process running after closing an SSH session use `tmux`.
Run the benchmark and the dashboard autopush in **separate sessions** so restarting the benchmark doesn't kill autopush:

```bash
tmux new-session -s bench    "python run_rtx5090.py 2>&1 | tee -a logs/rtx5090_stdout.log"
tmux new-session -s autopush "bash scripts/autopush_dashboard.sh"
```

Reattach later:

```bash
tmux attach -t bench
tmux attach -t autopush
```

Stop:

```bash
tmux kill-session -t bench    2>/dev/null
tmux kill-session -t autopush 2>/dev/null
```

Or if you prefer a simpler approach:

```bash
nohup python run_rtx5090.py > logs/rtx5090_stdout.log 2>&1 &
```

Monitor progress:

```bash
tail -f logs/rtx5090_stdout.log  # Console output
tail -f logs/rtx5090.log         # Orchestrator log
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

#### Background Execution (SSH)

Run the benchmark and the dashboard autopush in **separate sessions** so restarting the benchmark doesn't kill autopush:

```bash
# Jetson AGX Orin
tmux new-session -s bench    "python run_jetson_agx.py 2>&1 | tee -a logs/jetson_agx_stdout.log"
tmux new-session -s autopush "bash scripts/autopush_dashboard.sh"

# Jetson Orin Nano
tmux new-session -s bench    "python run_jetson_nano.py 2>&1 | tee -a logs/jetson_nano_stdout.log"
tmux new-session -s autopush "bash scripts/autopush_dashboard.sh"
```

Reattach later:

```bash
tmux attach -t bench
tmux attach -t autopush
```

Stop:

```bash
tmux kill-session -t bench    2>/dev/null
tmux kill-session -t autopush 2>/dev/null
```

Or with nohup:

```bash
nohup python run_jetson_agx.py  > logs/jetson_agx_stdout.log  2>&1 &
nohup python run_jetson_nano.py > logs/jetson_nano_stdout.log 2>&1 &
```

Monitor progress:

```bash
tail -f logs/jetson_agx_stdout.log   # Console output
tail -f logs/jetson_agx.log          # Orchestrator log
```

!!! warning "TensorRT engines are GPU-architecture specific"
    They cannot be built on the RTX 5090 and copied over. The orchestrator scripts
    automatically export each `.pt` model to a TensorRT `.engine` file on the target
    Jetson (FP16 and INT8 with calibration) before running inference.
    No manual export step is needed.

## Quick Test (Smoke Test)

Verify the full pipeline works before launching the real benchmark:

```bash
python run_rtx5090.py --quick-test
python run_jetson_agx.py --quick-test
python run_jetson_nano.py --quick-test
```

| Parameter | Normal | Quick Test |
|-----------|--------|------------|
| Training epochs | 1000 | 2 |
| Early stopping patience | 50 | 2 |
| Inference warmup runs | 5 | 0 |
| Inference measurement runs | 10 | 1 |

This runs the exact same experiments and code paths but finishes in minutes instead of days. Results are saved to `results-quick-test/` (separate from the real `results/` directory), so there is no risk of mixing smoke-test data with real benchmark results.

## Resume

All orchestrators support **automatic resume**. If a run's report file already exists, it is skipped. Simply re-run the same command to continue from where it stopped.

## OOM Protection (Jetson Orin Nano)

The Nano orchestrator proactively skips combinations likely to exceed 8 GB shared memory:

- Large model + batch 16
- Large model + imgsz 1280 + batch 8
- Medium model + batch 16 + imgsz 1280

Runtime OOM errors are caught and the run is marked as skipped rather than failed.
