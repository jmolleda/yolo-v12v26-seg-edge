# Environment

Software versions used across benchmark devices.

## RTX 5090 (antares)

| Component | Version |
|-----------|---------|
| OS | Ubuntu 24.04 (kernel 6.8.0-100-generic) |
| GPU | NVIDIA GeForce RTX 5090 (32 GB VRAM) |
| GPU Driver | 570.211.01 |
| CUDA | 12.8 |
| cuDNN | 90701 |
| Python | 3.12 |
| PyTorch | 2.7.0+cu128 |
| Ultralytics | 8.4.26 |
| TensorRT | N/A (not used on this device) |

## Jetson Orin AGX

| Component | Version |
|-----------|---------|
| OS | Ubuntu (kernel TBD) |
| JetPack | TBD |
| GPU | Integrated Ampere (64 GB shared RAM) |
| CUDA | TBD |
| cuDNN | TBD |
| TensorRT | TBD |
| Python | TBD |
| PyTorch | TBD |
| Ultralytics | TBD |

## Jetson Orin Nano

| Component | Version |
|-----------|---------|
| OS | Ubuntu (kernel TBD) |
| JetPack | TBD |
| GPU | Integrated Ampere (8 GB shared RAM) |
| CUDA | TBD |
| cuDNN | TBD |
| TensorRT | TBD |
| Python | TBD |
| PyTorch | TBD |
| Ultralytics | TBD |

## How to fill in TBD values

Run the following on each device:

```bash
# OS kernel
uname -r

# CUDA
nvcc --version

# cuDNN
python -c "import torch; print(torch.backends.cudnn.version())"

# TensorRT (Jetsons only)
python -c "import tensorrt; print(tensorrt.__version__)"

# JetPack (Jetsons only)
cat /etc/nv_tegra_release
# or
sudo apt-cache show nvidia-jetpack | grep Version

# PyTorch
python -c "import torch; print(torch.__version__)"

# Ultralytics
python -c "import ultralytics; print(ultralytics.__version__)"

# GPU Driver
nvidia-smi | head -3
```

## Development Machine (Windows)

| Component | Version |
|-----------|---------|
| OS | Windows 11 Pro for Workstations 10.0.26200 |
| Python | 3.12 |

This machine is used for development only, not for benchmark execution.
