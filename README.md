# YOLO Benchmark Suite

Benchmark framework for comparing YOLO model performance across GPU and edge devices, developed as part of a Master's thesis (TFM) at Universidad de Oviedo.

The system evaluates **YOLOv26** and **YOLOv12** architectures on a weld inspection dataset (8 classes) across three hardware platforms, measuring inference speed, accuracy, and power efficiency.

## Devices

| Device | Role | Memory |
|--------|------|--------|
| **NVIDIA RTX 5090** | Training + inference (PyTorch FP32) | Dedicated GPU VRAM |
| **Jetson Orin AGX** | TensorRT export + inference | 64 GB shared |
| **Jetson Orin Nano** | TensorRT export + inference | 8 GB shared |

## Experimental Design

Four experiments with a total of **292 runs** across all devices:

### Experiment 1 — Core Comparison
- **Fixed:** batch=1, imgsz=640, task=segment
- **Varies:** format (PyTorch FP32, TensorRT FP16, TensorRT INT8), approach (scratch, pretrained), architecture, model size
- RTX 5090: train + infer (PyTorch) | Jetsons: export + infer (TensorRT)

### Experiment 2 — Input Size Impact
- **Fixed:** batch=1, format=PyTorch FP32, task=segment
- **Varies:** imgsz (320, 1280), approach (scratch, pretrained), architecture, model size
- All devices: inference only (reuses weights from Experiment 1)

### Experiment 3 — Batch Throughput
- **Fixed:** imgsz=640, format=PyTorch FP32, approach=scratch, task=segment, architecture=yolo26
- **Varies:** batch (4, 8, 16), model size
- All devices: inference only (reuses weights from Experiment 1)

### Experiment 4 — Detection vs Segmentation
- **Fixed:** batch=1, imgsz=640, format=PyTorch FP32
- **Varies:** approach (scratch, pretrained), architecture, model size
- RTX 5090: train + infer | Jetsons: inference only

### Model Sizes

| Size | Depth multiplier | Width multiplier |
|------|-----------------|-----------------|
| nano | 0.50 | 0.25 |
| small | 0.50 | 0.50 |
| medium | 0.50 | 1.00 |
| large | 1.00 | 1.00 |

## Project Structure

```
BenchMarks/
├── config/
│   └── experiments.yaml          # Declarative experiment definitions
├── scripts/
│   ├── utils.py                  # Config loading, path resolution, report saving
│   ├── train.py                  # Generic training (both architectures/tasks/approaches)
│   ├── infer.py                  # Inference benchmark with warm-up + measurement
│   ├── export.py                 # TensorRT export (FP16/INT8)
│   ├── aggregate.py              # Collect reports into CSV
│   └── benchmark_logger.py       # HTML dashboard + JSON status logging
├── run_rtx5090.py                # RTX 5090 orchestrator (train → infer → aggregate)
├── run_jetson_agx.py             # Jetson AGX orchestrator (export → infer → aggregate)
├── run_jetson_nano.py            # Jetson Nano orchestrator (export → infer → aggregate + OOM protection)
├── generar_tabla_benchmark.py    # Generate experiment plan Excel (Spanish)
├── hiperparametros.yaml          # Shared training hyperparameters
├── data/
│   └── data.yaml                 # Dataset config (8 weld inspection classes)
├── results/                      # Generated results (gitignored)
└── logs/                         # Dashboard HTML + JSON status (gitignored)
```

## Metrics

Each inference run measures:

- **Preprocess / Inference / Postprocess** timing (ms/image, averaged over 10 runs after 5 warm-up runs)
- **FPS** (frames per second)
- **mAP50** and **mAP50-95** (accuracy)
- **Power consumption** in watts (Jetson devices only, via `jtop`)
- **FPS/Watt** efficiency (Jetson devices only)

## How to Run

### Prerequisites

```bash
pip install ultralytics pyyaml
```

The dataset must be placed in `data/` following the structure defined in `data/data.yaml`.

### Dry Run (preview without executing)

```bash
python run_rtx5090.py --dry-run
python run_jetson_agx.py --dry-run
python run_jetson_nano.py --dry-run
```

### Execution

**RTX 5090** — trains all models, then runs inference:
```bash
python run_rtx5090.py
```

**Jetson devices** — three automated phases: (1) copy trained weights from RTX 5090, (2) export to TensorRT, (3) run inference:
```bash
# 1. Copy trained weights (.pt) from RTX 5090 to the Jetson
scp -r user@rtx-host:/path/to/results/rtx5090/ results/rtx5090/

# 2-3. The orchestrator handles TensorRT export + inference automatically
python run_jetson_agx.py
python run_jetson_nano.py
```

TensorRT engines are **GPU-architecture specific** — they cannot be built on the RTX 5090 and copied over. The orchestrator scripts automatically export each `.pt` model to a TensorRT `.engine` file on the target Jetson (FP16 and INT8 with calibration) before running inference. No manual export step is needed.

### Background Execution (SSH sessions)

To keep the process running after closing an SSH session:

```bash
nohup python run_jetson_agx.py > /dev/null 2>&1 &
```

Or use `tmux` / `screen` for an interactive session that survives disconnects.

### Resume

All orchestrators support **automatic resume**. If a run's report file already exists, it is skipped. Simply re-run the same command to continue from where it stopped.

## Monitoring

Each orchestrator generates a self-contained **HTML dashboard** in `logs/`:

```
logs/{device}_dashboard.html    # Auto-refreshes every 30 seconds
logs/{device}_status.json       # Machine-readable status
logs/{device}.log               # Plain-text rolling log
```

To monitor remotely, serve the logs directory over HTTP:

```bash
cd logs && python -m http.server 8080
```

Then open `http://<device-ip>:8080/jetson_agx_dashboard.html` in a browser.

The dashboard shows:
- Overall progress bar with run counts (completed / failed / skipped / pending)
- Phase indicator (export → inference → aggregation → complete)
- Filterable runs table with status, timing, and key metrics
- Recent log entries

## Training Configuration

Defined in `hiperparametros.yaml`:

| Parameter | Value |
|-----------|-------|
| Epochs | 500 |
| Image size | 640 |
| Batch size | 16 |
| Patience | 50 |
| Optimizer | AdamW |
| Learning rate | 0.001 (0.0005 for large models) |
| Cosine LR schedule | Yes |
| Augmentation | Mosaic 1.0, mixup 0.15, copy-paste 0.5 |

## OOM Protection (Jetson Orin Nano)

The Nano orchestrator proactively skips combinations likely to exceed 8 GB shared memory:
- Large model + batch 16
- Large model + imgsz 1280 + batch 8
- Medium model + batch 16 + imgsz 1280

Runtime OOM errors are caught and the run is marked as skipped rather than failed.

## Results Aggregation

After all runs complete, each orchestrator collects report files into a single CSV:

```
results/{device}/benchmark_results.csv
```

Reports are parsed from individual `report_*.txt` files in the results tree.
