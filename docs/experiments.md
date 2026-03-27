# Experimental Design

Five experiments with a total of **408 inference runs** across all devices.

## Experiment 1 — Core Comparison

- **Fixed:** batch=1, imgsz=640, task=segment
- **Varies:** format (PyTorch FP32, TensorRT FP16, TensorRT INT8), approach (scratch, pretrained), architecture, model size
- RTX 5090: train + infer (PyTorch), export + infer (TensorRT FP16/INT8) | Jetsons: export + infer (TensorRT)

## Experiment 2 — Input Size Impact

- **Fixed:** batch=1, format=PyTorch FP32, task=segment
- **Varies:** imgsz (320, 1280), approach (scratch, pretrained), architecture, model size
- All devices: inference only (reuses weights from Experiment 1)

## Experiment 3 — Batch Throughput

- **Fixed:** imgsz=640, format=PyTorch FP32, approach=scratch, task=segment
- **Varies:** batch (4, 8, 16), architecture, model size
- All devices: inference only (reuses weights from Experiment 1)

## Experiment 4 — Detection vs Segmentation

- **Fixed:** batch=1, imgsz=640, format=PyTorch FP32
- **Varies:** approach (scratch, pretrained), architecture, model size
- RTX 5090: train + infer | Jetsons: inference only

## Experiment 5 — Class Imbalance Impact

- **Fixed:** batch=1, imgsz=640, format=PyTorch FP32, task=segment
- **Varies:** approach (scratch_balanced, pretrained_balanced), architecture, model size (nano, small, medium, large)
- RTX 5090: train + infer | Jetsons: inference only
- Compares per-class mAP against unbalanced baselines from Experiment 1

!!! info "Weighted sampling"
    Images containing rare classes (IV-5, IV-6, IV-3) are sampled more frequently
    during training via a `WeightedRandomSampler`. The validation set is unchanged,
    ensuring mAP scores reflect true model performance.

## Run Distribution

| Device | Training | Export | Inference | Total |
|--------|:--------:|:------:|:---------:|:-----:|
| RTX 5090 | 48 | 32 | 136 | 216 |
| Jetson Orin AGX | 0 | 32 | 136 | 168 |
| Jetson Orin Nano | 0 | 32 | 136 | 168 |
| **Total** | **48** | **96** | **408** | **552** |

!!! note "Weight reuse"
    Experiments 2 and 3 reuse trained weights from Experiment 1.
    Inference-only runs do not require retraining.
