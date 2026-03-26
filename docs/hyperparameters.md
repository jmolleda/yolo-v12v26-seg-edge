# Hyperparameters

Detailed description of all training hyperparameters defined in `hiperparametros.yaml`.

## Training parameters

| Parameter | Value | Ultralytics default | Description |
|-----------|-------|---------|-------------|
| `epochs` | **1000** | 100 | Maximum number of full passes through the training dataset. Each epoch processes all images once. |
| `imgsz` | 640 | 640 | Input image resolution in pixels (images are resized to 640x640). Larger = more detail but slower. |
| `batch` | 16 | 16 | Number of images processed simultaneously per gradient update. Larger = more stable gradients but more VRAM. |
| `patience` | **50** | 100 | Early stopping: if mAP doesn't improve for 50 consecutive epochs, training stops. Prevents wasting time when the model has converged. |

## Optimizer

| Parameter | Value | Ultralytics default | Description |
|-----------|-------|---------|-------------|
| `optimizer` | **AdamW** | auto | Optimizer algorithm. AdamW adds proper weight decay to Adam, preventing overfitting better than standard Adam. |
| `lr0` | **0.001** | 0.01 | Initial learning rate. Controls step size during gradient descent. Too high = unstable (NaN loss), too low = slow convergence. Large models use 0.0005 override. |
| `cos_lr` | **True** | False | Cosine annealing: learning rate follows a cosine curve from `lr0` down to near zero. Starts fast, then fine-tunes gradually at the end. |
| `weight_decay` | 0.0005 | 0.0005 | L2 regularization penalty. Discourages large weights, reducing overfitting. Applied correctly by AdamW (decoupled from gradient). |

## Augmentation — Geometric

| Parameter | Value | Ultralytics default | Description |
|-----------|-------|---------|-------------|
| `mosaic` | 1.0 | 1.0 | 100% probability. Combines 4 random training images into one by placing them in quadrants. Forces the model to learn objects at different positions and scales, especially effective for small object detection. |
| `close_mosaic` | **50** | 10 | Disables mosaic for the last 50 epochs. Lets the model fine-tune on clean, unmodified images before finishing — improves final accuracy. |
| `mixup` | **0.15** | 0.0 | 15% probability. Blends two images together with transparency (alpha compositing). Creates soft transitions between classes, encouraging smoother decision boundaries. |
| `copy_paste` | **0.5** | 0.0 | 50% probability. Cuts segmentation masks of objects from one image and pastes them onto another. Very effective for instance segmentation but computationally expensive — operates on individual masks. |
| `flipud` | **0.5** | 0.0 | 50% probability of vertical flip (upside down). Useful when objects can appear in any vertical orientation. |
| `fliplr` | 0.5 | 0.5 | 50% probability of horizontal flip (mirror). Standard augmentation — most objects look equally valid when mirrored. |
| `scale` | **0.6** | 0.5 | Random scaling factor +/-60% (image zoom between 0.4x and 1.6x). Teaches the model to recognize objects at different sizes. |
| `degrees` | **25.0** | 0.0 | Random rotation up to +/-25 degrees. Helps with objects that aren't always axis-aligned. |
| `shear` | **5** | 0.0 | Random shear up to +/-5 degrees. Simulates perspective distortion by slanting the image. |
| `perspective` | **0.0005** | 0.0 | Very slight random perspective transformation. Simulates camera angle variation. Small value keeps it subtle. |

## Augmentation — Color

| Parameter | Value | Ultralytics default | Description |
|-----------|-------|---------|-------------|
| `hsv_h` | 0.015 | 0.015 | Random hue shift +/-1.5% of the color wheel. Small value — colors shift slightly but objects remain recognizable. |
| `hsv_s` | 0.7 | 0.7 | Random saturation change +/-70%. Large range — images can become very vivid or nearly grayscale. Teaches robustness to lighting/camera variation. |
| `hsv_v` | 0.4 | 0.4 | Random brightness (value) change +/-40%. Simulates different exposure levels — shadows, bright lights. |

## Transfer learning

| Parameter | Value | Ultralytics default | Description |
|-----------|-------|---------|-------------|
| `pretrained` | **False** | True | Default: train from random weights (scratch). The training script overrides this to `True` for transfer learning runs, loading COCO-pretrained weights as the starting point. |

**Bold** values indicate parameters that differ from Ultralytics defaults. 13 out of 22 parameters are customized.
