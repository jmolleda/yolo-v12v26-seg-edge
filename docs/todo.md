# TODO

## Address Dataset Class Imbalance

### Problem

The training dataset (2,188 images, 9,423 annotations) has significant class imbalance:

| Class | Count | Ratio vs largest |
|-------|------:|:---:|
| Solda | 2,536 | 1.0x |
| IV-2 | 2,224 | 0.88x |
| IV-1B | 1,920 | 0.76x |
| IV-4 | 1,108 | 0.44x |
| IV-1A | 774 | 0.31x |
| IV-3 | 393 | 0.15x |
| IV-6 | 250 | 0.10x |
| IV-5 | 218 | 0.09x |

The top 3 classes hold 71% of all annotations. Solda has ~12x more instances than IV-5.

### Why current augmentation does not help

The augmentations in `hiperparametros.yaml` (mosaic, mixup, copy_paste, flips, rotation, HSV shifts, etc.) apply **uniformly** to all images regardless of class content. They increase visual variety but preserve the original class distribution — an underrepresented class remains underrepresented.

### Proposed solutions

1. **Focal Loss** (`fl_gamma: 1.5` in `hiperparametros.yaml`)
   - One-line change, no data manipulation needed
   - Down-weights loss for easy/confident predictions (frequent classes), keeps loss high for hard predictions (rare classes)
   - Does not directly count class instances — responds to model confidence, which correlates with class frequency
   - Will not fully compensate for a 12x imbalance on its own

2. **Image duplication (oversampling)**
   - Duplicate images containing rare classes so all classes appear roughly equally during training
   - Target copies: IV-4 ×2, IV-1A ×3, IV-3 ×6, IV-6 ×10, IV-5 ×11
   - Existing augmentations ensure each copy is transformed differently per epoch, reducing overfitting risk
   - Caveat: images often contain multiple classes (e.g. IV-1B + Solda), so duplicating for a rare class also increases counts of co-occurring classes
   - Increases epoch time proportionally to the number of added images

3. **Targeted augmentation**
   - Apply heavier augmentation only to images containing rare classes
   - More controlled than simple duplication but requires custom logic

4. **Undersampling majority classes**
   - Remove some Solda/IV-2/IV-1B images to balance distribution
   - Simplest approach but loses data

### Notes

- For the benchmark (comparing architectures/formats), the imbalance affects all models equally, so comparisons remain fair
- The imbalance should be documented in the thesis as a dataset limitation
- Per-class mAP from the benchmark results will quantify the actual impact
