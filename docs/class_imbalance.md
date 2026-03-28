# Class Imbalance

## Problem

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

## Why current augmentation does not help

The augmentations in `hiperparametros.yaml` (mosaic, mixup, copy_paste, flips, rotation, HSV shifts, etc.) apply **uniformly** to all images regardless of class content. They increase visual variety but preserve the original class distribution — an underrepresented class remains underrepresented.

## Solution: Weighted Sampling (Experiment 5)

**Implemented** via `scripts/weighted_sampler.py`. Monkey-patches the Ultralytics dataloader to sample images with probability inversely proportional to their class frequency using `WeightedRandomSampler`.

- Each image's weight = max inverse-frequency of any class it contains
- Images with rare classes (IV-5, IV-6, IV-3) are sampled ~10x more often
- Total epoch length stays the same (2,188 draws), only the mix changes
- Validation is unaffected — mAP reflects true performance

**Experiment 5** trains all model sizes — nano, small, medium, large — (both architectures, both approaches) with balanced sampling and compares per-class mAP against unbalanced baselines from Experiment 1.

## Other approaches considered but not used (yet)

1. **Focal Loss** (`fl_gamma`) — Would confound the weighted sampling experiment. Discussed as future work.
2. **Image duplication** — Co-occurrence problem (duplicating for IV-5 also inflates Solda), increases disk/epoch time.
3. **Targeted augmentation** — AugmenTory library doesn't support rebalancing; Albumentations doesn't handle YOLO polygons natively.
4. **Undersampling** — Loses data from an already small dataset.

## Notes

- For the benchmark (comparing architectures/formats), the imbalance affects all models equally, so comparisons remain fair
- The imbalance should be documented in the thesis as a dataset limitation
- Per-class mAP from the benchmark results will quantify the actual impact
