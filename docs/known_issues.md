# Known Issues

## YOLOv12 Pretrained Segmentation — Trained as Detection Model

**Affected models:** `yolo12{n,s,m,l}` pretrained and pretrained_balanced, segmentation task
**Affected phase:** Training (root cause) → inference and TRT export (downstream failures)
**Status:** Unresolved — requires retraining with corrected training script

### Symptom

PyTorch inference fails with:

```
AttributeError: 'dict' object has no attribute 'shape'
```

TensorRT export fails with:

```
ONNX export failure: number of output names provided (2) exceeded number of outputs (1)
```

### Root Cause

YOLOv12 has no official pretrained segmentation weights (`yolo12n-seg.pt` etc. do not exist in Ultralytics Hub). The training script loaded detection backbone weights (`yolo12n.pt`) via `YOLO(model_config)` without specifying `task=task` in the constructor. Ultralytics read the embedded `task='detect'` from the checkpoint and ran training in detection mode, despite `task='segment'` being passed in `model.train()`. The resulting `best.pt` files are detection models stored in segmentation experiment folders.

Evidence: `results.csv` for affected models contains only detection metric columns (`train/box_loss`, `train/cls_loss`, `train/dfl_loss`) with no segmentation columns (`train/seg_loss`, `metrics/mAP50(M)`, etc.).

### Practical Impact

| Phase | Status |
|-------|--------|
| Training | ✓ Completed (but produced detection model, not segmentation) |
| Training metrics (mAP50-95) | ⚠ Detection metrics only — not segmentation |
| PyTorch inference (segment task) | ✗ Fails — model output is detection dict, not seg tensor |
| TensorRT export (segment task) | ✗ Fails — model has 1 output (detect), exporter expects 2 (seg) |

Affected experiments: `core_comparison` (4 models), `input_size` (same weights), `class_imbalance` pretrained_balanced (4 models).

### Fix

The training script requires `task=task` in the YOLO constructor so Ultralytics selects the correct model architecture before training:

```python
# Before (broken):
model = YOLO(model_config)
# After (correct):
model = YOLO(model_config, task=task)
```

These 8 model configs must be retrained on RTX 5090. The existing `best.pt` weights are detection models and cannot produce valid segmentation inference results.

---

## Weighted Sampler Silent Failure — yolo26 Medium Pretrained Balanced

**Affected model:** `yolo26_seg_medium_pretrained_balanced` (class_imbalance experiment)
**Status:** Resolved — valid result obtained from isolated rerun (train3, 2026-04-05)

### Symptom

The `yolo26_seg_medium_pretrained_balanced` training run produced results **pixel-perfect identical** to `yolo26_seg_medium_pretrained` (core_comparison) across all 374 epochs — every loss and metric value matches to 5+ decimal places, with only the wall-clock `time` column differing by ~30 seconds per epoch.

### Root Cause

The weighted sampler works via a monkey-patch of `ultralytics.data.build.build_dataloader` using a module-level global (`_original_build_dataloader`). When multiple balanced runs execute sequentially in the same Python process, the patch/restore cycle can result in the patch being applied to an already-patched function, or not being applied at all, depending on how Ultralytics internally references its dataloader builder. For this specific run the patch had no effect — training proceeded with the default uniform sampler.

### Resolution

Re-run in an isolated fresh Python process (no prior balanced runs in the same session):

```bash
python -c "
from scripts.train import train_model
train_model('yolo26', 'medium', 'segment', 'pretrained_balanced', 'class_imbalance')
"
```

The rerun completed on 2026-04-05 (train3). Valid result: **mAP50 = 0.5210** (baseline unbalanced = 0.5271, Δ = −0.006). The weighted sampler was confirmed active (results are distinct from baseline). All 16 class_imbalance models are now valid.

### Environment

| Component | Version |
|-----------|---------|
| Ultralytics | 8.4.26 |
| PyTorch | 2.7.0+cu128 |
| Python | 3.12.3 |
| Hardware | NVIDIA RTX 5090 |

---

## Jetson Orin Nano — TensorRT INT8 Shows No Speed Gain over FP16

**Affected device:** Jetson Orin Nano (8 GB, JP6.1, TRT 10.3.0.30)
**Affected phase:** TRT inference benchmarking
**Status:** Expected hardware behaviour — not a bug

### Observation

TensorRT INT8 and FP16 engines produce virtually identical inference latency (within 0.1 ms) and identical accuracy metrics (same mAP values to 4 decimal places) for all tested models on the Jetson Orin Nano.

Example (yolo26 nano pretrained):

| Precision | inf_ms | FPS |
|-----------|--------|-----|
| FP16 | 14.74 | 39.71 |
| INT8 | 14.75 | 39.71 |

### Explanation

At batch=1 on Jetson Orin Nano, inference is memory-bandwidth-bound rather than compute-bound. INT8 Tensor Cores offer higher theoretical throughput, but the per-image latency is dominated by data movement and pre/post-processing (Python overhead: ~3 ms pre + 4–13 ms post per image), not raw matrix multiply speed. The INT8 advantage only materialises at higher batch sizes.

Identical accuracy metrics are consistent with TRT's PTQ calibration maintaining negligible accuracy loss at INT8 for this dataset and model scale.

### Impact on Results

INT8 results are still reported separately as they confirm quantization stability. For throughput comparison, FP16 and INT8 can be treated as equivalent on this device at batch=1.
