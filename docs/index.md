# YOLO v12 v26 Segmentation Edge

Benchmark framework for comparing YOLO model performance across GPU and edge devices.

The system evaluates **YOLOv12** and **YOLOv26** architectures on a steel surface defects and welds dataset (8 classes) across three hardware platforms (RTX 5090, Jetson AGX Orin, Jetson Orin Nano), measuring inference speed, accuracy, and power efficiency.   

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
│   └── experiments.yaml              # Declarative experiment matrix
├── scripts/
│   ├── utils.py                      # Config loading, path resolution, report saving
│   ├── train.py                      # Generic YOLO training script
│   ├── infer.py                      # Inference benchmark (warm-up + measurement)
│   ├── export.py                     # TensorRT export (FP16 / INT8)
│   ├── aggregate.py                  # Collect all report*.txt into a summary CSV
│   ├── benchmark_logger.py           # JSON status + live HTML dashboard logging
│   ├── weighted_sampler.py           # Class-balanced sampling for balanced approaches
│   └── autopush_dashboard.sh         # Auto-rebuild and push dashboard to gh-pages
├── run_rtx5090.py                    # RTX 5090 orchestrator (train → export → infer → aggregate)
├── run_jetson_agx.py                 # Jetson AGX orchestrator (export → infer → aggregate)
├── run_jetson_nano.py                # Jetson Nano orchestrator (export → infer → aggregate + OOM protection)
├── build_results_dashboard.py        # Build self-contained HTML results dashboard
├── hyperparameters.yaml              # Shared training hyperparameters
├── hw_metrics_cache.json             # Cached hardware metrics (params, GFLOPs, GPU memory)
├── mkdocs.yml                        # MkDocs documentation config
├── data/
│   ├── data.yaml                     # Dataset config (8 weld inspection classes)
│   └── {train,valid,test}/           # Images, labels, polygons
├── docs/                             # Project documentation (MkDocs source)
├── logs/                             # Runtime logs — gitignored
│   ├── {device}_stdout.log           # Full stdout (tee'd from orchestrator)
│   ├── {device}.log                  # Structured orchestrator log
│   ├── {device}_status.json          # Live run status (phase, counters, per-run state)
│   └── {device}_dashboard.html       # Live local dashboard
└── results/                          # Benchmark outputs — gitignored
    └── {device}/{experiment}/{model}/
        ├── report.txt                # Training report (metrics + timing)
        ├── report_{fmt}_{prec}_img{sz}_b{bs}.txt  # Inference report
        ├── train/results.csv         # Per-epoch training curve
        └── train/weights/best.pt     # Best model weights
```

## Documentation

- [Execution](execution.md) — how to run the benchmark on each device
- [Orchestrator Pipeline](orchestrator.md) — phase-by-phase pseudocode for each orchestrator
- [Experiments](experiments.md) — experiment matrix definition
- [Hyperparameters](hyperparameters.md) — training hyperparameter reference
- [Monitoring](monitoring.md) — live dashboard and logging
- [Environment](environment.md) — software versions and hardware setup

## Results Aggregation

After all runs complete, each orchestrator collects report files into a single CSV:

```
results/{device}/benchmark_results.csv
```

Reports are parsed from individual `report_*.txt` files in the results tree.
