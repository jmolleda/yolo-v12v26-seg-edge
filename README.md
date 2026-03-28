# YOLO Benchmark Suite

Benchmark framework for comparing YOLO model performance across GPU and edge devices.

The system evaluates **YOLOv12** and **YOLOv26** architectures on a steel surface defects and welds dataset (8 classes) across three hardware platforms (RTX 5090, Jetson AGX Orin, Jetson Orin Nano), measuring inference speed, accuracy, and power efficiency.

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
- [Experimental Design](docs/experiments.md) — Five experiments, 552 runs
- [How to Run](docs/execution.md) — Per-device execution, resume, OOM protection
- [Monitoring](docs/monitoring.md) — HTML dashboard, remote monitoring
- [Hyperparameters](docs/hyperparameters.md) — Training parameters with Ultralytics defaults
- [Environment](docs/environment.md) — Software versions per device
- [Class Imbalance](docs/class_imbalance.md) — Dataset imbalance analysis and weighted sampling solution
