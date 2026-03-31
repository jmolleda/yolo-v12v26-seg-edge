# Orchestrator Pipeline

Each device has a dedicated orchestrator (`run_rtx5090.py`, `run_jetson_agx.py`, `run_jetson_nano.py`) that executes the full benchmark pipeline in sequential phases.

## RTX 5090

```
ORCHESTRATOR (run_rtx5090.py)
│
├── Load experiment matrix from config
│   └── Produces: train_runs, export_runs, infer_runs
│
├── PHASE 1 — Training
│   └── FOR each model (arch × size × task × approach):
│       ├── IF report.txt already exists → SKIP (resume)
│       ├── train_model()
│       │   ├── Train YOLO from scratch / pretrained weights
│       │   ├── Run one validation pass
│       │   └── Write report.txt  (metrics + timing)
│       └── Mark run as done/failed in status JSON
│
├── PHASE 2 — TensorRT Export
│   └── FOR each model × precision (FP16, INT8):
│       ├── IF .engine already exists → SKIP
│       ├── IF .pt weights missing → FAIL
│       └── export_model()  →  writes .engine file
│
├── PHASE 3 — Inference Benchmark
│   └── FOR each model × format × precision × imgsz × batch:
│       ├── IF report_{format}_{prec}_img{sz}_b{bs}.txt exists → SKIP
│       ├── IF weights missing → FAIL
│       └── run_inference()
│           ├── N warm-up passes  (discard)
│           ├── M measurement passes  (timed)
│           └── Write report_{...}.txt  (FPS, latency, mAP)
│
└── PHASE 4 — Aggregation
    ├── Collect all report*.txt across results/rtx5090/
    └── Write benchmark_results.csv  (single summary table)
```

## Jetson AGX Orin

Weights must be copied from the RTX 5090 first (`scp results/rtx5090/ ...`).
No training phase — export and inference only.

```
ORCHESTRATOR (run_jetson_agx.py)
│
├── Load experiment matrix from config
│   └── Produces: export_runs, infer_runs
│
├── PHASE 1 — TensorRT Export
│   └── FOR each model × precision (FP16, INT8):
│       ├── IF .engine already exists → SKIP
│       ├── IF .pt weights missing → FAIL
│       └── export_model()  →  writes .engine file
│           (built on Jetson hardware — not portable from RTX)
│
├── PHASE 2 — Inference Benchmark
│   └── FOR each model × format × precision × imgsz × batch:
│       ├── IF report_{...}.txt exists → SKIP
│       ├── IF weights missing → FAIL
│       └── run_inference()
│           ├── N warm-up passes  (discard)
│           ├── M measurement passes  (timed, + power via jtop)
│           └── Write report_{...}.txt  (FPS, latency, mAP, FPS/W)
│
└── PHASE 3 — Aggregation
    ├── Collect all report*.txt across results/jetson_agx/
    └── Write benchmark_results.csv
```

## Jetson Orin Nano

Same pipeline as the AGX, with an additional OOM protection layer due to the 8 GB shared memory constraint.

```
ORCHESTRATOR (run_jetson_nano.py)
│
├── Load experiment matrix from config
│   └── Produces: export_runs, infer_runs
│
├── PHASE 1 — TensorRT Export
│   └── FOR each model × precision (FP16, INT8):
│       ├── IF likely to OOM → SKIP proactively
│       │     (large+b16, large+img1280+b8, medium+img1280+b16)
│       ├── IF .engine already exists → SKIP
│       ├── IF .pt weights missing → FAIL
│       ├── export_model()  →  writes .engine file
│       └── ON OOM error → SKIP (caught, not failed)
│
├── PHASE 2 — Inference Benchmark
│   └── FOR each model × format × precision × imgsz × batch:
│       ├── IF likely to OOM → SKIP proactively
│       ├── IF report_{...}.txt exists → SKIP
│       ├── IF weights missing → FAIL
│       ├── run_inference()
│       │   ├── N warm-up passes  (discard)
│       │   ├── M measurement passes  (timed, + power via jtop)
│       │   └── Write report_{...}.txt  (FPS, latency, mAP, FPS/W)
│       └── ON OOM error → SKIP (caught, not failed)
│
└── PHASE 3 — Aggregation
    ├── Collect all report*.txt across results/jetson_nano/
    └── Write benchmark_results.csv
```

## Resume behaviour

Every phase is fully resumable. Completed runs are detected by the existence of their output file and skipped on re-run — no re-training or re-inference needed after an interruption.

| Phase | Skip condition |
|-------|---------------|
| Training | `report.txt` exists |
| Export | `.engine` file exists |
| Inference | `report_{format}_{prec}_img{sz}_b{bs}.txt` exists |

## Device differences

| Feature | RTX 5090 | Jetson AGX | Jetson Orin Nano |
|---------|----------|------------|-----------------|
| Training | Yes | No (uses RTX weights) | No (uses RTX weights) |
| TensorRT export | Yes (FP16/INT8) | Yes (FP16/INT8) | Yes (FP16/INT8) |
| Inference | Yes | Yes | Yes |
| OOM protection | No | No | Yes |

!!! warning "TensorRT engines are GPU-architecture specific"
    Engines built on the RTX 5090 cannot be used on Jetson devices and vice versa.
    Each orchestrator exports its own `.engine` files on the target hardware.
