# Experimental Design

Four experiments with a total of **292 runs** across all devices.

## Experiment 1 — Core Comparison

- **Fixed:** batch=1, imgsz=640, task=segment
- **Varies:** format (PyTorch FP32, TensorRT FP16, TensorRT INT8), approach (scratch, pretrained), architecture, model size
- RTX 5090: train + infer (PyTorch) | Jetsons: export + infer (TensorRT)

## Experiment 2 — Input Size Impact

- **Fixed:** batch=1, format=PyTorch FP32, task=segment
- **Varies:** imgsz (320, 1280), approach (scratch, pretrained), architecture, model size
- All devices: inference only (reuses weights from Experiment 1)

## Experiment 3 — Batch Throughput

- **Fixed:** imgsz=640, format=PyTorch FP32, approach=scratch, task=segment, architecture=yolo26
- **Varies:** batch (4, 8, 16), model size
- All devices: inference only (reuses weights from Experiment 1)

## Experiment 4 — Detection vs Segmentation

- **Fixed:** batch=1, imgsz=640, format=PyTorch FP32
- **Varies:** approach (scratch, pretrained), architecture, model size
- RTX 5090: train + infer | Jetsons: inference only

## Run Distribution

| Device | Training | Inference | Total |
|--------|:--------:|:---------:|:-----:|
| RTX 5090 | 32 | 76 | 76 |
| Jetson Orin AGX | 0 | 108 | 108 |
| Jetson Orin Nano | 0 | 108 | 108 |
| **Total** | **32** | **292** | **292** |

!!! note "Weight reuse"
    Experiments 2 and 3 reuse trained weights from Experiment 1.
    Inference-only runs do not require retraining.
