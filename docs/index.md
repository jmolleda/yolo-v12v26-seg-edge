# YOLO Benchmark Suite

Benchmark framework for comparing YOLO model performance across GPU and edge devices, developed as part of a Master's thesis (TFM) at Universidad de Oviedo.

The system evaluates **YOLOv26** and **YOLOv12** architectures on a weld inspection dataset (8 classes) across three hardware platforms, measuring inference speed, accuracy, and power efficiency.

## Devices

| Device | Role | Memory |
|--------|------|--------|
| **NVIDIA RTX 5090** | Training + TensorRT export + inference (PyTorch FP32, TensorRT FP16/INT8) | 32 GB dedicated VRAM |
| **Jetson Orin AGX** | TensorRT export + inference | 64 GB shared |
| **Jetson Orin Nano** | TensorRT export + inference | 8 GB shared |

## Model Sizes

| Size | Depth multiplier | Width multiplier |
|------|-----------------|-----------------|
| nano | 0.50 | 0.25 |
| small | 0.50 | 0.50 |
| medium | 0.50 | 1.00 |
| large | 1.00 | 1.00 |

## Metrics

Each inference run measures:

- **Preprocess / Inference / Postprocess** timing (ms/image, averaged over N runs after warm-up; configurable via `--runs` and `--warmup`)
- **Latency statistics**: mean, median, std dev, p95, p99
- **FPS** (frames per second)
- **mAP50** and **mAP50-95** (accuracy)
- **Precision** and **Recall** (overall and per-class)
- **Model file size** (MB on disk)
- **GPU peak memory** usage (MB)
- **Power consumption** in watts (Jetson devices only, via `jtop`)
- **FPS/Watt** efficiency (Jetson devices only)

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
├── run_rtx5090.py                # RTX 5090 orchestrator (train → export → infer → aggregate)
├── run_jetson_agx.py             # Jetson AGX orchestrator (export → infer → aggregate)
├── run_jetson_nano.py            # Jetson Nano orchestrator (export → infer → aggregate + OOM protection)
├── hiperparametros.yaml          # Shared training hyperparameters
├── mkdocs.yml                    # MkDocs documentation config
├── data/
│   └── data.yaml                 # Dataset config (8 weld inspection classes)
├── docs/                         # Project documentation (MkDocs)
├── results/                      # Generated results (gitignored)
└── logs/                         # Dashboard HTML + JSON status (gitignored)
```

## Results Aggregation

After all runs complete, each orchestrator collects report files into a single CSV:

```
results/{device}/benchmark_results.csv
```

Reports are parsed from individual `report_*.txt` files in the results tree.
