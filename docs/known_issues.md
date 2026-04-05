# Known Issues

## YOLOv12 Pretrained Segmentation — Inference Failure

**Affected models:** `yolo12{n,s,m,l}` pretrained, segmentation task
**Affected phase:** Inference benchmarking (PyTorch FP32)
**Status:** Unresolved — Ultralytics bug

### Symptom

```
AttributeError: 'dict' object has no attribute 'shape'
```

Occurs inside Ultralytics 8.4.26 at `ultralytics/models/yolo/segment/val.py`, line 104:

```python
imgsz = [4 * x for x in proto.shape[2:]]  # get image size from proto
```

The segmentation validator expects `proto` to be a tensor, but receives a dict when running inference on a yolo12 model loaded from a saved `.pt` file with a detection backbone used as segmentation pretrained weights.

### Root Cause

YOLOv12 has no official pretrained segmentation weights (`yolo12n-seg.pt` etc. do not exist in Ultralytics Hub). As a workaround, detection backbone weights (`yolo12n.pt`) are used for pretraining, and Ultralytics adapts them during training. This works during training (model is in memory with the full segmentation head), but when the trained weights are reloaded from disk for inference validation, the output format of the segmentation head differs from what the validator expects, causing the crash.

### Practical Impact

| Phase | Status |
|-------|--------|
| Training | ✓ Completed successfully |
| Training metrics (mAP50, mAP50-95, per-class) | ✓ Available in `report.txt` |
| Inference timing (FPS, latency) | ✗ Not available |
| TensorRT export | ✗ Not attempted (no valid inference baseline) |

The 4 affected models (`yolo12n/s/m/l` pretrained seg) are excluded from inference benchmark comparisons. Their training accuracy metrics remain valid and are included in accuracy analysis.

### Workaround

The trained weights (`best.pt`) are valid and fully usable. Inference benchmarking can be retried at any time with a different Ultralytics version without retraining:

```bash
# Create a venv with a different Ultralytics version
pip install ultralytics==<version>

# Re-run inference for affected models
python scripts/infer.py --weights results/rtx5090/core_comparison/yolo12_seg_nano_pretrained/train/weights/best.pt \
    --format pytorch --precision fp32 --imgsz 640 --batch 1 \
    --arch yolo12 --size nano --task segment --approach pretrained \
    --experiment core_comparison --device rtx5090
```

Repeat for `small`, `medium`, `large`. The dashboard will pick up the new `report_*.txt` files automatically.

---

## Weighted Sampler Silent Failure — yolo26 Medium Pretrained Balanced

**Affected model:** `yolo26_seg_medium_pretrained_balanced` (class_imbalance experiment)
**Status:** Data point unreliable — exclude from class imbalance analysis

### Symptom

The `yolo26_seg_medium_pretrained_balanced` training run produced results **pixel-perfect identical** to `yolo26_seg_medium_pretrained` (core_comparison) across all 374 epochs — every loss and metric value matches to 5+ decimal places, with only the wall-clock `time` column differing by ~30 seconds per epoch.

### Root Cause

The weighted sampler works via a monkey-patch of `ultralytics.data.build.build_dataloader` using a module-level global (`_original_build_dataloader`). When multiple balanced runs execute sequentially in the same Python process, the patch/restore cycle can result in the patch being applied to an already-patched function, or not being applied at all, depending on how Ultralytics internally references its dataloader builder. For this specific run the patch had no effect — training proceeded with the default uniform sampler.

### Practical Impact

- The balanced result for yolo26 medium pretrained is a duplicate of the unbalanced result
- It must be **excluded** from class imbalance analysis comparisons
- All other balanced models show measurable differences from their unbalanced counterparts and are considered valid

### Resolution

To obtain a valid result, re-run this specific model in isolation (fresh process, no prior balanced runs):

```bash
python -c "
from scripts.train import train_model
train_model('yolo26', 'medium', 'segment', 'pretrained_balanced', 'class_imbalance')
"
```

### Environment

| Component | Version |
|-----------|---------|
| Ultralytics | 8.4.26 |
| PyTorch | 2.7.0+cu128 |
| Python | 3.12.3 |
| Hardware | NVIDIA RTX 5090 |
