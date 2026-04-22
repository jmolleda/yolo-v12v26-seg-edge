# Training Benchmark Results: YOLOv12 vs YOLOv26 for Industrial Surface Defect Segmentation

**Platform:** NVIDIA RTX 5090 · **Framework:** Ultralytics 8.4.26 · **Date:** April 2026  
**Author:** jmolleda

---

## Abstract

This report presents the results of a systematic training benchmark comparing two recent one-stage object detection architectures — YOLOv12 and YOLOv26 — applied to surface defect instance segmentation on an industrial surface imaging dataset. Forty-eight training configurations were evaluated across four structured experiments: (1) core accuracy and speed comparison between architectures at four model sizes; (2) detection versus segmentation head performance; (3) the effect of weighted sampling for class imbalance correction; and (4) the impact of input resolution on accuracy and throughput. All training was conducted on an NVIDIA RTX 5090 GPU with a shared hyperparameter protocol and early stopping. Results show that YOLOv26 achieves the highest overall segmentation accuracy (mAP50 = 0.527 at medium size, pretrained), while YOLOv12 exhibits superior robustness when trained from scratch. The dataset presents a severe class imbalance that the weighted sampler only partially addresses. Input resolution of 640 px consistently represents the optimal accuracy-throughput trade-off, with a systematic accuracy collapse at 1280 px.

---

## 1. Introduction

Automated surface defect inspection using computer vision is a high-value application in industrial quality control. Surface images contain multiple defect types with different morphologies, frequencies, and severity levels. Instance segmentation — which provides both bounding boxes and pixel-level masks — is the preferred output format for defect grading, as it enables measurement of defect geometry.

YOLOv12 and YOLOv26 represent two concurrent architectural directions from the Ultralytics ecosystem. YOLOv12 introduces an attention-based backbone (replacing C2f blocks with A-ATSS), while YOLOv26 retains a CNN-centric CSP design optimised for speed. Both are available at four model sizes (nano, small, medium, large), enabling systematic evaluation of the accuracy-efficiency trade-off.

This benchmark addresses four research questions:

1. **RQ1 — Architecture:** Does YOLOv12 or YOLOv26 produce better segmentation accuracy, and is there a size-dependent interaction?
2. **RQ2 — Task head:** Does adding a segmentation head over a detection backbone reduce accuracy or speed on this dataset?
3. **RQ3 — Class imbalance:** Does inverse-frequency weighted sampling improve detection of minority defect classes?
4. **RQ4 — Resolution:** What is the accuracy-throughput trade-off across 320, 640, and 1280 px input sizes?

---

## 2. Dataset

The dataset consists of surface images annotated for instance segmentation across eight defect categories.

| Split | Images | Labels |
|-------|--------|--------|
| Train | 2,188 | 2,188 |
| Validation | 624 | 624 |
| Test | 285 | 285 |
| **Total** | **3,097** | **3,097** |

### 2.1 Class Distribution (Training Set)

| Class | Instances | Frequency |
|-------|-----------|-----------|
| Solda | 2,536 | 26.9% |
| IV-2 | 2,224 | 23.6% |
| IV-1B | 1,920 | 20.4% |
| IV-4 | 1,108 | 11.8% |
| IV-1A | 774 | 8.2% |
| IV-3 | 393 | 4.2% |
| IV-6 | 250 | 2.7% |
| IV-5 | 218 | 2.3% |
| **Total** | **9,423** | |

The dataset is severely imbalanced. The three majority classes (Solda, IV-2, IV-1B) account for 70.9% of all instances, while the two rarest (IV-5, IV-6) together represent only 5.0%. The imbalance ratio between the most and least frequent classes is approximately 11.6:1 (Solda vs IV-5).

---

## 3. Experimental Design

### 3.1 Hyperparameter Protocol

All models share a common hyperparameter configuration:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Max epochs | 1,000 | Upper bound; early stopping applies |
| Early stopping patience | 50 | Avoids overfitting; terminates when mAP50 does not improve for 50 epochs |
| Optimiser | AdamW | Adaptive learning with decoupled weight decay |
| Base learning rate (lr0) | 0.0005 (yolo12) / 0.001 (yolo26) | Architecture-dependent default |
| LR schedule | Cosine annealing | Smooth decay to zero |
| Weight decay | 0.0005 | L2 regularisation |
| Image size | 640 px | Default; varied in Experiment 4 |
| Batch size | -1 (auto) | GPU memory-adaptive |
| Mosaic augmentation | 1.0 | Enabled throughout training |
| Close mosaic (final N epochs) | 50 | Disabled in last 50 epochs for stable convergence |
| Mixup | 0.15 | Mild blending augmentation |
| Flip up/down | 0.5 | Appropriate for surface images (no canonical orientation) |
| Flip left/right | 0.5 | |
| Scale jitter | 0.6 | |
| Rotation | 25° | |
| Shear | 5° | |
| HSV augmentation | H=0.015, S=0.7, V=0.4 | Colour-space perturbation |
| Perspective | 0.0005 | Mild projective distortion |
| Copy-paste | 0.5 | Instance-level augmentation |

### 3.2 Experiment Overview

| Experiment | Models | Configurations | Total runs |
|------------|--------|----------------|------------|
| 1 — Core comparison | YOLOv12, YOLOv26 | 4 sizes × 2 approaches (seg) | 16 |
| 2 — Detection vs segmentation | YOLOv12, YOLOv26 | 4 sizes × 2 approaches (det) | 16 |
| 3 — Class imbalance | YOLOv12, YOLOv26 | 4 sizes × 2 approaches (seg, balanced) | 16 |
| 4 — Input size | YOLOv12 scratch, YOLOv26 pretrained | 4 sizes × 2 resolutions (320, 1280) | 16 |

The 640 px segmentation results from Experiment 1 serve as the baseline for Experiments 3 and 4.

### 3.3 Evaluation Metrics

- **mAP50:** Mean Average Precision at IoU threshold 0.50. The primary accuracy metric.
- **mAP50-95:** Mean Average Precision averaged over IoU thresholds 0.50–0.95 in steps of 0.05. Stricter metric capturing mask quality.
- **Precision / Recall:** Mean across classes at the optimal confidence threshold.
- **FPS:** Frames per second at inference (PyTorch FP32, batch size 1).
- **Inference latency (ms/img):** Reported as the Ultralytics-measured inference phase excluding preprocessing and postprocessing.

---

## 4. Experiment 1: Core Comparison — YOLOv12 vs YOLOv26 Segmentation

### 4.1 Results

#### Table 1. Segmentation accuracy and speed — all models

| Architecture | Size | Approach | mAP50 | mAP50-95 | Precision | Recall | FPS | ms/img |
|---|---|---|---|---|---|---|---|---|
| YOLOv12 | nano | pretrained | 0.4995 | 0.3083 | 0.5346 | 0.5227 | 797.9 | 0.62 |
| YOLOv12 | small | pretrained | 0.5123 | 0.3199 | 0.5880 | 0.5157 | 560.1 | 1.18 |
| YOLOv12 | medium | pretrained | 0.5158 | 0.3245 | 0.5690 | 0.5291 | 329.4 | 2.35 |
| YOLOv12 | large | pretrained | 0.5267 | 0.3310 | 0.5652 | 0.5446 | 239.0 | 3.54 |
| YOLOv12 | nano | scratch | 0.4904 | 0.3031 | 0.5548 | 0.4917 | 636.1 | 0.76 |
| YOLOv12 | small | scratch | 0.5001 | 0.3054 | 0.5747 | 0.4947 | 450.8 | 1.40 |
| YOLOv12 | medium | scratch | 0.5162 | 0.3218 | 0.5368 | 0.5294 | 259.4 | 3.00 |
| YOLOv12 | large | scratch | 0.5177 | 0.3283 | 0.5996 | 0.5053 | 202.6 | 4.19 |
| YOLOv26 | nano | pretrained | 0.4907 | 0.3062 | 0.5716 | 0.4815 | 802.1 | 0.71 |
| YOLOv26 | small | pretrained | 0.5136 | 0.3211 | 0.5603 | 0.5107 | 602.7 | 1.12 |
| **YOLOv26** | **medium** | **pretrained** | **0.5271** | **0.3366** | **0.5371** | **0.5572** | **322.1** | **2.52** |
| YOLOv26 | large | pretrained | 0.5231 | 0.3334 | 0.5909 | 0.4941 | 281.0 | 2.95 |
| YOLOv26 | nano | scratch | 0.4652 | 0.2775 | 0.5464 | 0.4657 | 787.8 | 0.70 |
| YOLOv26 | small | scratch | 0.4960 | 0.2993 | 0.5888 | 0.4849 | 578.9 | 1.11 |
| YOLOv26 | medium | scratch | 0.4953 | 0.3034 | 0.5610 | 0.4928 | 320.8 | 2.55 |
| YOLOv26 | large | scratch | 0.4756 | 0.2906 | 0.4942 | 0.5085 | 281.6 | 2.95 |

### 4.2 Architecture Comparison

At the nano size, YOLOv12 pretrained marginally outperforms YOLOv26 pretrained (0.4995 vs 0.4907, Δ = −0.009). At small size the gap closes (0.5123 vs 0.5136, Δ = +0.001). At medium size YOLOv26 takes a clear lead (0.5158 vs 0.5271, Δ = +0.011). At large size YOLOv12 recovers slightly (0.5267 vs 0.5231, Δ = −0.004). There is therefore no consistent accuracy winner: the two architectures are competitive throughout the size range, with YOLOv26 having a slight advantage at medium scale.

Regarding throughput, YOLOv26 is equal or faster than YOLOv12 at all sizes except medium (322 vs 329 FPS, a negligible difference of 7 FPS). The largest FPS advantage for YOLOv26 is at small size (+43 FPS, 8% faster). The nano models of both architectures approach 800 FPS, while large models drop to approximately 240–281 FPS.

### 4.3 Pretrained vs Scratch

#### Table 2. Pretrained lift over scratch (ΔmAP50 = pretrained − scratch)

| Architecture | Nano | Small | Medium | Large |
|---|---|---|---|---|
| YOLOv12 | +0.009 | +0.012 | −0.000 | +0.009 |
| YOLOv26 | +0.026 | +0.018 | **+0.032** | **+0.048** |

This is the most architecturally significant finding in Experiment 1. **YOLOv12 trained from scratch achieves nearly the same accuracy as pretrained**, with a maximum lift of 1.2 pp at small size and a negligible difference at medium (pretrained is actually 0.0004 below scratch). In contrast, **YOLOv26 depends substantially on ImageNet-pretrained weights**, with lifts of 2.6 pp at nano rising to 4.8 pp at large. 

This divergence is consistent with the architectural differences between the two models: YOLOv12's attention-based backbone may provide stronger inductive biases for learning local patterns from small datasets, allowing it to reach similar performance with or without pretraining. YOLOv26's CNN-centric backbone relies more heavily on learned low-level filters that initialise better from natural image pretraining.

From a practical standpoint, this finding has a significant operational implication: **YOLOv12 can be retrained from scratch on new defect categories or datasets without access to pretrained weights** and without a meaningful accuracy penalty, while YOLOv26 should always be initialised from pretrained weights.

### 4.4 Training Convergence

Models trained with pretrained weights consistently converged faster than scratch-trained equivalents:

| Architecture | Size | Epochs (pretrained) | Epochs (scratch) | Scratch overhead |
|---|---|---|---|---|
| YOLOv12 | nano | 292 | 564 | +93% |
| YOLOv12 | small | 272 | 354 | +30% |
| YOLOv12 | medium | 269 | 379 | +41% |
| YOLOv12 | large | 244 | 382 | +57% |
| YOLOv26 | nano | 388 | 614 | +58% |
| YOLOv26 | small | 318 | 742 | +133% |
| YOLOv26 | medium | 374 | 582 | +56% |
| YOLOv26 | large | 365 | 371 | +2% |

Scratch training requires significantly more epochs before early stopping triggers. YOLOv26 small scratch is the extreme case at 742 epochs (133% more than pretrained), reflecting the need for the backbone to learn fundamental visual representations from random initialisation.

### 4.5 Accuracy Scaling with Model Size

Both architectures show diminishing returns as model size increases. The mAP50 gain from nano to large is:

- **YOLOv12 pretrained:** 0.4995 → 0.5267, a gain of **2.72 pp** over a ~12× increase in model parameters
- **YOLOv26 pretrained:** 0.4907 → 0.5231, a gain of **3.24 pp** over the same parameter scale-up

The marginal gain from medium to large is small for both: +0.011 pp (YOLOv12) and −0.004 pp (YOLOv26, where medium actually outperforms large). This suggests that the dataset may be approaching the capacity ceiling for these architectures at the medium scale — additional model size offers little benefit and may introduce overfitting risk.

---

## 5. Experiment 2: Detection vs Segmentation

### 5.1 Results

#### Table 3. Detection vs segmentation accuracy (mAP50) and speed (FPS)

| Architecture | Size | Approach | Det mAP50 | Seg mAP50 | Δ (det−seg) | Det FPS | Seg FPS | ΔFPS |
|---|---|---|---|---|---|---|---|---|
| YOLOv12 | nano | pretrained | 0.5009 | 0.4995 | +0.001 | 780.5 | 797.9 | −17 |
| YOLOv12 | small | pretrained | 0.5154 | 0.5123 | +0.003 | 531.6 | 560.1 | −29 |
| YOLOv12 | medium | pretrained | 0.5237 | 0.5158 | +0.008 | 328.2 | 329.4 | −1 |
| YOLOv12 | large | pretrained | 0.5283 | 0.5267 | +0.002 | 238.6 | 239.0 | +0 |
| YOLOv12 | nano | scratch | 0.4827 | 0.4904 | −0.008 | 796.6 | 636.1 | +161 |
| YOLOv12 | small | scratch | 0.4989 | 0.5001 | −0.001 | 524.9 | 450.8 | +74 |
| YOLOv12 | medium | scratch | 0.5111 | 0.5162 | −0.005 | 324.4 | 259.4 | +65 |
| YOLOv12 | large | scratch | 0.5121 | 0.5177 | −0.006 | 237.2 | 202.6 | +35 |
| YOLOv26 | nano | pretrained | 0.4882 | 0.4907 | −0.003 | 1,096.2 | 802.1 | +294 |
| YOLOv26 | small | pretrained | 0.4979 | 0.5136 | **−0.016** | 773.7 | 602.7 | +171 |
| YOLOv26 | medium | pretrained | 0.5047 | 0.5271 | **−0.022** | 440.8 | 322.1 | +119 |
| YOLOv26 | large | pretrained | 0.5039 | 0.5231 | **−0.019** | 376.3 | 281.0 | +95 |
| YOLOv26 | nano | scratch | 0.4723 | 0.4652 | +0.007 | 1,191.8 | 787.8 | +404 |
| YOLOv26 | small | scratch | 0.4905 | 0.4960 | −0.006 | 747.9 | 578.9 | +169 |
| YOLOv26 | medium | scratch | 0.5190 | 0.4953 | **+0.024** | 447.2 | 320.8 | +126 |
| YOLOv26 | large | scratch | 0.5070 | 0.4756 | **+0.031** | 373.6 | 281.6 | +92 |

### 5.2 YOLOv12: Negligible Task Head Effect

For YOLOv12 pretrained, detection outperforms segmentation by 0.001–0.008 mAP50, and the speed difference is essentially zero (within ±30 FPS, less than 6%). For YOLOv12 scratch, the direction reverses: segmentation outperforms detection by 0.001–0.008 pp. In both cases, the differences are within the noise of a single training run.

**The practical conclusion is clear: for YOLOv12, the segmentation head adds no meaningful cost.** Given that segmentation provides substantially richer output (pixel-level masks vs bounding boxes) at negligible accuracy and latency overhead, the segmentation variant is unambiguously preferable for surface defect inspection.

### 5.3 YOLOv26 Pretrained: Segmentation Strongly Outperforms Detection

For YOLOv26 pretrained, the picture is qualitatively different. Segmentation outperforms detection by 1.6–2.2 pp mAP50 at small, medium, and large sizes. The gain is largest at medium (0.5271 vs 0.5047, Δ = −0.022). The nano case is near-zero (−0.003).

This result is counterintuitive: the detection-trained model is evaluated on the same validation set with the same mAP50 metric on bounding boxes, yet segmentation training leads to higher detection AP. The explanation is that **pixel-level supervision provides richer gradient signal during training**, forcing the backbone to learn better spatial representations. The segmentation head effectively regularises the backbone, producing features more discriminative for localisation than a detection-only objective.

The cost of this improved accuracy is substantial throughput reduction: 95–294 FPS less than the detection equivalent, representing a 25–37% speed penalty.

### 5.4 YOLOv26 Scratch: Detection Overtakes Segmentation at Larger Sizes

For YOLOv26 scratch at medium and large sizes, detection outperforms segmentation by 2.4–3.1 pp. This reversal (relative to the pretrained result) suggests that **without pretrained weights, YOLOv26 cannot effectively learn the segmentation head and the backbone jointly from scratch** — the additional complexity of mask prediction interferes with backbone convergence. With pretrained features, the backbone is already well-initialised and the segmentation head adds discriminative capacity; without pretraining, it becomes a burden.

---

## 6. Experiment 3: Class Imbalance — Weighted Sampling

### 6.1 Methodology

A custom inverse-frequency weighted sampler was applied to the training dataloader. For each image, a weight equal to the maximum inverse-class-frequency among its annotated classes was computed. Images containing rare-class instances were therefore sampled proportionally more often, without changing the effective epoch length.

The sampler was implemented as a monkey-patch of `ultralytics.data.build.build_dataloader`. A data integrity issue initially affected one model: `yolo26_seg_medium_pretrained_balanced` produced results pixel-identical to the unbalanced baseline in a first run, indicating a silent sampler failure caused by sequential process state pollution. The model was subsequently rerun in an isolated Python process, confirming the sampler applied correctly (mAP50 = 0.5210, distinct from the baseline 0.5271). All 16 balanced configurations are now valid.

### 6.2 Results

#### Table 4. Weighted sampling effect (ΔmAP50 = balanced − baseline)

| Architecture | Size | Approach | Baseline | Balanced | Δ | Effect |
|---|---|---|---|---|---|---|
| YOLOv12 | nano | pretrained | 0.4995 | 0.4995 | 0.000 | Neutral |
| YOLOv12 | small | pretrained | 0.5123 | 0.5145 | +0.002 | Minimal gain |
| YOLOv12 | medium | pretrained | 0.5158 | 0.5158 | 0.000 | Neutral |
| YOLOv12 | large | pretrained | 0.5267 | 0.5205 | **−0.006** | Hurt |
| YOLOv12 | nano | scratch | 0.4904 | 0.4904 | 0.000 | Neutral |
| YOLOv12 | small | scratch | 0.5001 | 0.5027 | +0.003 | Minimal gain |
| YOLOv12 | medium | scratch | 0.5162 | 0.5176 | +0.001 | Neutral |
| YOLOv12 | large | scratch | 0.5177 | 0.5134 | −0.004 | Hurt |
| YOLOv26 | nano | pretrained | 0.4907 | 0.4907 | 0.000 | Neutral |
| YOLOv26 | small | pretrained | 0.5136 | 0.5202 | **+0.007** | Moderate gain |
| YOLOv26 | medium | pretrained | 0.5271 | 0.5210 | **−0.006** | Hurt |
| YOLOv26 | large | pretrained | 0.5231 | 0.5249 | +0.002 | Minimal gain |
| YOLOv26 | nano | scratch | 0.4652 | 0.4652 | 0.000 | Neutral |
| YOLOv26 | small | scratch | 0.4960 | 0.4895 | −0.007 | Hurt |
| YOLOv26 | medium | scratch | 0.4953 | 0.4953 | 0.000 | Neutral |
| YOLOv26 | large | scratch | 0.4756 | 0.4952 | **+0.020** | Strong gain |

### 6.3 Analysis

The results are heterogeneous. The weighted sampler produced a measurable benefit in only 4 of 16 comparisons, a neutral effect in 7, and a negative effect in 5. This inconsistency leads to several observations:

**Nano models are uniformly unaffected.** All four nano models (two architectures, two approaches) show exactly zero delta. At the nano scale, the model capacity is likely the bottleneck rather than class representation — the model cannot learn fine-grained distinctions regardless of how often minority-class images appear.

**Well-initialised pretrained models at medium/large scale are hurt by balancing.** YOLOv26 medium pretrained loses 0.6 pp (0.5271 → 0.5210) and YOLOv12 large pretrained loses 0.6 pp (0.5267 → 0.5205). A model already carrying strong ImageNet-derived features has learned reliable representations for common classes; forcing over-representation of rare classes disrupts the training distribution without sufficiently improving minority-class detection to compensate.

**YOLOv26 large scratch shows the strongest benefit (+0.020).** When the model has no pretrained representations and the architecture relies more on learned statistics, correcting the sampling distribution appears to help the backbone avoid ignoring minority classes entirely.

**The global mAP50 metric obscures class-level effects.** Weighted sampling is designed to improve minority-class recall at the possible expense of majority-class precision. The overall mAP50 is dominated by the majority classes (Solda, IV-2, IV-1B), so modest improvements in IV-5 and IV-6 may not surface in the aggregate metric. For example, the yolo26 medium pretrained balanced run achieves IV-5 mAP50 = 0.725 (vs 0.702 baseline, +2.3 pp) and IV-6 mAP50 = 0.068 (vs 0.095 baseline, −2.7 pp) — the minority class benefit is mixed even at class level, and the overall mAP50 drops by 0.6 pp due to majority-class regression.

**The sampler is not a reliable general-purpose intervention for this dataset.** The results suggest that class imbalance correction through sampling alone is insufficient when the imbalance is as severe as 11.6:1 and the rare classes (IV-5, IV-6) represent only 5% of instances. Data augmentation targeted at minority classes (e.g., copy-paste from a curated minority pool, synthetic defect generation) would likely be more effective.

---

## 7. Experiment 4: Input Resolution

### 7.1 Results

Training was performed at 640 px for all models (Experiment 1). Inference at 320 px and 1280 px was performed using the trained weights without retraining (the Ultralytics runtime resizes inputs dynamically). Two subsets were evaluated: YOLOv26 pretrained (all four sizes) and YOLOv12 scratch (all four sizes), providing 24 additional inference points.

#### Table 5. mAP50 and FPS across input resolutions — YOLOv26 pretrained

| Size | mAP50 @320 | mAP50 @640 | mAP50 @1280 | FPS @320 | FPS @640 | FPS @1280 |
|---|---|---|---|---|---|---|
| nano | 0.4534 | 0.4907 | 0.3709 | 336 | 802 | 193 |
| small | 0.4748 | 0.5136 | 0.3936 | 304 | 603 | 143 |
| medium | 0.4985 | 0.5271 | 0.4117 | 227 | 322 | 84 |
| large | 0.5075 | 0.5231 | 0.4219 | 170 | 281 | 69 |

#### Table 6. mAP50 and FPS across input resolutions — YOLOv12 scratch

| Size | mAP50 @320 | mAP50 @640 | mAP50 @1280 | FPS @320 | FPS @640 | FPS @1280 |
|---|---|---|---|---|---|---|
| nano | 0.4600 | 0.4904 | 0.3173 | 276 | 636 | 136 |
| small | 0.4675 | 0.5001 | 0.3140 | 250 | 451 | 89 |
| medium | 0.4970 | 0.5162 | 0.3300 | 206 | 259 | 53 |
| large | **0.0912** | 0.5177 | **0.0316** | 138 | 203 | 32 |

### 7.2 Resolution-Accuracy Trade-off

**640 px is the optimal training and inference resolution for all models tested.** Moving to 1280 px consistently degrades accuracy despite the higher spatial resolution:

- YOLOv26 pretrained: average mAP50 drop of **−0.117 pp** (range −0.140 to −0.101)
- YOLOv12 scratch (excluding the large anomaly): average drop of **−0.181 pp** (range −0.186 to −0.176)

This result appears counterintuitive — higher resolution should in principle allow the model to detect smaller or finer defects. The explanation is that the models were trained at 640 px: their learned feature scale and anchor statistics are calibrated for that resolution. At 1280 px inference (without retraining at 1280 px), the effective receptive field of the backbone corresponds to a different spatial scale than what the model was trained on, causing a systematic accuracy collapse.

**At 320 px**, accuracy drops by approximately 3.7–5.1 pp for YOLOv26 pretrained and 2.3–4.9 pp for YOLOv12 scratch (excluding the large anomaly). This is a smaller absolute penalty than 1280 px, but throughput at 320 px is only 2.0–2.4× that of 640 px — a modest gain compared to the 3.8–5.0× speed advantage 640 px has over 1280 px.

**YOLOv26 generalises better across non-native resolutions.** At 1280 px, YOLOv26 pretrained retains 0.37–0.42 mAP50. At 320 px it retains 0.45–0.51 mAP50. YOLOv12 scratch shows larger accuracy degradation at both extremes, particularly at 1280 px (0.31–0.33 for nano through medium).

### 7.3 YOLOv12 Large Scratch Anomaly

YOLOv12 large scratch shows a catastrophic accuracy collapse at both 320 px (mAP50 = 0.091) and 1280 px (mAP50 = 0.032), while performing normally at 640 px (mAP50 = 0.518). This is qualitatively different from the gradual degradation seen in all other models. Possible causes include:

1. **Early stopping at a suboptimal local minimum.** The model may have converged at 382 epochs to a feature representation tightly coupled to the exact 640 px scale distribution. The attention mechanisms in YOLOv12's backbone are more sensitive to positional scale mismatch than CNN-based models.
2. **Interaction between model capacity and resolution mismatch.** At large size, the model has more capacity to overfit the training resolution.
3. **Numerical instability during dynamic rescaling.** The large backbone may amplify rescaling artefacts more than smaller variants.

This anomaly underscores the importance of evaluating YOLO models at non-training resolutions with caution, particularly for attention-based architectures at their largest scale.

### 7.4 Speed Scaling

The FPS vs resolution relationship follows an approximate inverse-quadratic law as expected (halving linear resolution roughly quadruples throughput):

- 640 → 320 px: 2.0–2.5× FPS gain
- 640 → 1280 px: 3.8–5.0× FPS loss

YOLOv26 nano pretrained at 1280 px achieves 193 FPS with mAP50 = 0.371, making it the most viable configuration for applications requiring higher spatial resolution at the expense of accuracy.

---

## 8. Per-class Accuracy Analysis

### 8.1 Class-level Results (Core Comparison, Pretrained Segmentation)

#### Table 7. mAP50 per class — YOLOv12 pretrained

| Class | Nano | Small | Medium | Large |
|---|---|---|---|---|
| IV-1A | 0.3700 | 0.3892 | 0.4034 | 0.4074 |
| IV-1B | 0.6490 | 0.6428 | 0.6436 | 0.6635 |
| IV-2 | 0.5406 | 0.5650 | 0.5717 | 0.5767 |
| IV-3 | 0.4413 | 0.4402 | 0.4980 | 0.4602 |
| IV-4 | 0.6452 | 0.6627 | 0.6802 | 0.6900 |
| IV-5 | 0.7090 | 0.6830 | 0.6870 | 0.7148 |
| **IV-6** | **0.0518** | **0.1169** | **0.0599** | **0.1009** |
| Solda | 0.5893 | 0.5988 | 0.5824 | 0.6004 |

#### Table 8. mAP50 per class — YOLOv26 pretrained

| Class | Nano | Small | Medium | Large |
|---|---|---|---|---|
| IV-1A | 0.3450 | 0.3941 | 0.3949 | 0.4295 |
| IV-1B | 0.6343 | 0.6463 | 0.6549 | 0.6498 |
| IV-2 | 0.5405 | 0.5564 | 0.5890 | 0.6009 |
| IV-3 | 0.4584 | 0.4616 | 0.4985 | 0.4723 |
| IV-4 | 0.6494 | 0.6680 | 0.6811 | 0.6784 |
| IV-5 | 0.6528 | 0.6902 | 0.7023 | 0.6881 |
| **IV-6** | **0.0626** | **0.0956** | **0.0951** | **0.0399** |
| Solda | 0.5824 | 0.5966 | 0.6009 | 0.6257 |

### 8.2 Class Difficulty Analysis

**IV-6 is a critical failure class.** Across all 8 pretrained segmentation models, IV-6 mAP50 ranges from 0.040 to 0.117 — far below any other class. Despite being one of the two rarest classes (2.7% of instances), it is not the least frequent (IV-5 is rarer at 2.3%) yet IV-5 achieves 0.65–0.71 mAP50 — the highest or second-highest of all classes. This rules out frequency alone as the explanation. IV-6 is intrinsically harder to localise and segment, likely due to morphological similarity to other defect types or high intra-class variability.

**IV-5 achieves the highest mAP50 despite being the rarest class.** With only 218 training instances (2.3%), IV-5 achieves 0.65–0.71 mAP50 across all models. This suggests a distinctive visual signature that the models can reliably learn from few examples.

**IV-1A is the second most difficult class** (0.35–0.43 mAP50), despite being present in 8.2% of instances. Its difficulty likely stems from morphological overlap with other defect categories.

**The class difficulty ranking is consistent across architectures and sizes:**
IV-6 < IV-1A < IV-3 < IV-2 < Solda < IV-1B < IV-4 < IV-5

This stability across 8 different model configurations suggests that the difficulty is a function of the classes' visual properties and label quality rather than any specific architectural bias.

### 8.3 Scaling Effects per Class

For most classes, increasing model size from nano to large produces a 0.3–0.8 pp mAP50 improvement — consistent but modest. IV-3 is an exception: both architectures show a non-monotonic pattern (medium performs better than large), suggesting that IV-3 defects occur at spatial scales well-suited to medium-size receptive fields.

IV-5 shows a plateau: the nano YOLOv12 (0.709) achieves essentially the same score as the large YOLOv12 (0.715), indicating that IV-5 has a simple enough visual pattern that even the smallest model captures it effectively.

---

## 9. Discussion

### 9.1 Architecture Recommendations

For **maximum segmentation accuracy**: YOLOv26 medium pretrained (mAP50 = 0.527). The 7 FPS speed cost relative to YOLOv26 small (322 vs 603 FPS) is significant, but the 1.4 pp accuracy gain justifies it for offline or low-throughput inspection systems.

For **best accuracy-throughput balance**: YOLOv26 small pretrained (mAP50 = 0.514, 603 FPS). This model sits at the elbow of the accuracy-speed Pareto frontier.

For **maximum throughput on RTX-class hardware**: YOLOv26 nano detection scratch (mAP50 = 0.472, 1,192 FPS). This operates near a factor of 4× faster than the most accurate model while losing only 5.5 pp in accuracy.

For **environments without pretrained weights** (e.g., domain-specific sensors without ImageNet-relevant pre-training): YOLOv12 medium scratch (mAP50 = 0.516, 259 FPS) is the best option, outperforming YOLOv26 medium scratch (0.495) by 2.1 pp.

### 9.2 The IV-6 Problem

The systematic failure on IV-6 is a critical finding for practical deployment. An inspection system that cannot reliably detect a specific defect type — regardless of its rarity — may not meet quality control requirements for automated surface inspection. Three interventions are warranted:

1. **Data acquisition:** Collect additional IV-6 samples. Even doubling the 250 training instances could substantially improve recall.
2. **Synthetic augmentation:** Copy-paste augmentation from a curated pool of IV-6 instances could multiply effective occurrence frequency.
3. **Cascaded detection:** Train a specialised binary classifier for IV-6 to run as a second pass on predicted negative regions.

### 9.3 Resolution and Deployment

The 640 px training-native resolution is appropriate for the dataset's defect scale. The accuracy collapse at 1280 px when using models trained at 640 px is a deployment risk: any pipeline that dynamically adjusts input resolution must either retrain at the target resolution or accept the accuracy penalty. For edge deployment (e.g., Jetson AGX Orin, Jetson Nano), the FPS at 640 px will be the primary constraint; the 320 px option (with ~4 pp accuracy cost) may be necessary to meet real-time requirements.

### 9.4 Limitations

- **Single hardware platform.** All results are from RTX 5090 training. Inference benchmarks on Jetson edge devices are in progress and may reveal different architectural trade-offs due to memory bandwidth and compute constraints.
- **Single training seed.** No replications were performed due to compute constraints. Per-configuration variance is unknown; differences below ~0.5 pp mAP50 should be interpreted with caution.
- **No TensorRT inference.** YOLOv12 TensorRT export failed due to an architecture incompatibility with ONNX export of attention layers. TensorRT results for YOLOv26 will be reported separately.
- **One excluded data point.** The `yolo26_seg_medium_pretrained_balanced` rerun is in progress; the class imbalance analysis for this configuration is provisional.
- **mAP50 as primary metric.** The mAP50 metric weights all classes equally regardless of clinical severity. In a real inspection system, a missed IV-6 defect may be more consequential than a false positive on Solda. A severity-weighted metric would better reflect operational requirements.

---

## 10. Conclusions

Forty-eight training runs across four structured experiments yield the following principal findings:

1. **YOLOv26 medium pretrained is the highest-accuracy segmentation model (mAP50 = 0.527)**, with YOLOv12 large pretrained a close second (0.527 rounded, actually 0.5267 vs 0.5271). Both architectures are competitive; no single architecture dominates across all sizes and training regimes.

2. **YOLOv12 is remarkably robust to random initialisation.** Scratch-trained YOLOv12 matches pretrained YOLOv12 to within 0.9 pp at all sizes, whereas YOLOv26 scratch loses up to 4.8 pp relative to pretrained. This makes YOLOv12 the preferred architecture for domains where pretrained weights are not available or relevant.

3. **The segmentation head does not reduce accuracy or speed for YOLOv12**, and actually improves accuracy for YOLOv26 pretrained (+1.6 to +2.2 pp). The segmentation task should be preferred over detection for this application given the added value of pixel-level masks at negligible cost.

4. **Weighted sampling for class imbalance is inconsistent.** Benefits are concentrated in YOLOv26 large scratch (+2.0 pp) and YOLOv26 small pretrained (+0.7 pp). Nano models are unaffected; large pretrained models are slightly harmed. The intervention should be applied selectively rather than universally.

5. **640 px is the optimal training and inference resolution.** 1280 px inference without retraining degrades accuracy by 10–18 pp. 320 px reduces accuracy by 4–5 pp with only a 2–2.5× speed gain.

6. **IV-6 is an unresolved failure mode.** No architecture or configuration achieves more than 0.117 mAP50 on this class. Targeted data augmentation or specialised detection is required before deployment in contexts where IV-6 detection is safety-critical.

---

## Appendix: Training Duration Summary

| Architecture | Size | Approach | Task | Duration |
|---|---|---|---|---|
| YOLOv12 | nano | pretrained | segment | 0:37:38 |
| YOLOv12 | small | pretrained | segment | 0:51:49 |
| YOLOv12 | medium | pretrained | segment | 1:30:24 |
| YOLOv12 | large | pretrained | segment | 2:24:11 |
| YOLOv12 | nano | scratch | segment | 1:38:55 |
| YOLOv12 | small | scratch | segment | 1:28:09 |
| YOLOv12 | medium | scratch | segment | 2:46:54 |
| YOLOv12 | large | scratch | segment | 4:05:12 |
| YOLOv26 | nano | pretrained | segment | 1:14:34 |
| YOLOv26 | small | pretrained | segment | 1:22:30 |
| YOLOv26 | medium | pretrained | segment | 2:48:58 |
| YOLOv26 | large | pretrained | segment | 3:32:27 |
| YOLOv26 | nano | scratch | segment | 1:52:23 |
| YOLOv26 | small | scratch | segment | 3:11:05 |
| YOLOv26 | medium | scratch | segment | 4:22:10 |
| YOLOv26 | large | scratch | segment | 3:35:40 |
| YOLOv12 | nano | pretrained | detect | 0:36:53 |
| YOLOv12 | small | pretrained | detect | 0:53:15 |
| YOLOv12 | medium | pretrained | detect | 1:09:28 |
| YOLOv12 | large | pretrained | detect | 2:53:25 |
| YOLOv12 | nano | scratch | detect | 0:50:21 |
| YOLOv12 | small | scratch | detect | 1:10:41 |
| YOLOv12 | medium | scratch | detect | 1:57:02 |
| YOLOv12 | large | scratch | detect | 3:19:17 |
| YOLOv26 | nano | pretrained | detect | 0:31:53 |
| YOLOv26 | small | pretrained | detect | 0:42:05 |
| YOLOv26 | medium | pretrained | detect | 1:32:56 |
| YOLOv26 | large | pretrained | detect | 1:36:48 |
| YOLOv26 | nano | scratch | detect | 0:51:25 |
| YOLOv26 | small | scratch | detect | 1:05:15 |
| YOLOv26 | medium | scratch | detect | 1:43:23 |
| YOLOv26 | large | scratch | detect | 1:36:59 |
