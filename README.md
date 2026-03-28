# YOLO v12 v26 Segmentation Edge

Benchmark framework for comparing YOLO model performance across GPU and edge devices.

The system evaluates **YOLOv12** and **YOLOv26** architectures on a steel surface defects and welds dataset (8 classes) across three hardware platforms (RTX 5090, Jetson AGX Orin, Jetson Orin Nano), measuring inference speed, accuracy, and power efficiency.

## 📊 Interactive results
**[Dashboard](https://jmolleda.github.io/yolo-v12v26-seg-edge/dashboard/)** (partial, benchmark still running)

## 🚀 Quick Start

```bash
pip install ultralytics pyyaml
python run_rtx5090.py --dry-run    # Preview runs
python run_rtx5090.py              # Execute benchmark
```

## 📖 Documentation

Full documentation is [available online](https://jmolleda.github.io/yolo-v12v26-seg-edge/). 

To browse it locally:

```bash
pip install mkdocs mkdocs-material
mkdocs serve
```

Then open `http://localhost:8000` in your browser.

