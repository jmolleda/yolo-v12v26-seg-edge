# YOLO Benchmark Suite

Benchmark framework for comparing YOLO model performance across GPU and edge devices, developed as part of a Master's thesis (TFM) at Universidad de Oviedo.

The system evaluates **YOLOv26** and **YOLOv12** architectures on a weld inspection dataset (8 classes) across three hardware platforms (RTX 5090, Jetson Orin AGX, Jetson Orin Nano), measuring inference speed, accuracy, and power efficiency.

## Quick Start

```bash
pip install ultralytics pyyaml
python run_rtx5090.py --dry-run    # Preview runs
python run_rtx5090.py              # Execute benchmark
```

## Documentation

Full documentation is available in the `docs/` folder. To browse it locally:

```bash
pip install mkdocs mkdocs-material
mkdocs serve
```

Then open `http://localhost:8000` in your browser.

### Documentation contents

- [Home](docs/index.md) — Project overview, structure, and metrics
- [Experimental Design](docs/experiments.md) — Four experiments, 292 runs
- [How to Run](docs/execution.md) — Per-device execution, resume, OOM protection
- [Monitoring](docs/monitoring.md) — HTML dashboard, remote monitoring
- [Hyperparameters](docs/hyperparameters.md) — Training parameters with Ultralytics defaults
- [Environment](docs/environment.md) — Software versions per device
- [TODO](docs/todo.md) — Dataset imbalance analysis and proposals
