"""Weighted sampling to address class imbalance in YOLO training.

Monkey-patches the Ultralytics dataloader to sample images with probability
inversely proportional to their class frequency. Images containing rare classes
are sampled more often, while the total epoch length stays the same.

Only affects training (shuffle=True). Validation is left unchanged.
"""

import os
import numpy as np
import yaml
from torch.utils.data import WeightedRandomSampler

# Store original function for restoration
_original_build_dataloader = None


def compute_image_weights(data_yaml_path):
    """Compute per-image sampling weights from YOLO label files.

    For each image, the weight is the max inverse-frequency of any class
    present. This ensures images with rare classes are always upsampled,
    even if they also contain common classes.

    Args:
        data_yaml_path: Path to data.yaml with class names and train path.

    Returns:
        Tuple of (weights list, class_counts dict) for logging.
    """
    with open(data_yaml_path, "r") as f:
        data_config = yaml.safe_load(f)

    nc = data_config["nc"]
    class_names = data_config["names"]

    # Resolve training labels directory.
    # data.yaml train path may be relative (e.g. ../train/images or train/images).
    # Ultralytics strips leading ../ and resolves relative to the yaml directory.
    data_dir = os.path.dirname(os.path.abspath(data_yaml_path))
    train_rel = data_config["train"].lstrip("../").lstrip("./")
    train_images_path = os.path.normpath(os.path.join(data_dir, train_rel))
    train_labels_path = train_images_path.replace("images", "labels")

    # Count class instances across all label files
    label_files = sorted([
        f for f in os.listdir(train_labels_path)
        if f.endswith(".txt")
    ])

    class_counts = np.zeros(nc, dtype=np.float64)
    image_classes = []  # list of sets, one per image

    for label_file in label_files:
        label_path = os.path.join(train_labels_path, label_file)
        classes_in_image = set()
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cls_id = int(parts[0])
                    classes_in_image.add(cls_id)
                    class_counts[cls_id] += 1
        image_classes.append(classes_in_image)

    # Compute inverse frequency weights per class
    # Avoid division by zero for classes with no instances
    class_weights = np.where(
        class_counts > 0,
        class_counts.sum() / class_counts,
        0.0,
    )

    # Per-image weight = max class weight among classes present
    # Images with no annotations get weight 1.0
    image_weights = []
    for classes_in_image in image_classes:
        if classes_in_image:
            weight = max(class_weights[c] for c in classes_in_image)
        else:
            weight = 1.0
        image_weights.append(weight)

    # Log class distribution
    print("\n--- Weighted Sampling: Class Distribution ---")
    for i, name in enumerate(class_names):
        print(f"  {name:8s}: {int(class_counts[i]):5d} instances, "
              f"weight={class_weights[i]:.2f}")
    print(f"  Total images: {len(image_weights)}")
    print("--- Weighted Sampling Active ---\n")

    return image_weights, dict(zip(class_names, class_counts.astype(int).tolist()))


def apply_weighted_sampling(data_yaml_path):
    """Monkey-patch Ultralytics build_dataloader to use weighted sampling.

    Must be called before model.train(). Only affects training dataloaders
    (where shuffle=True). Validation dataloaders are unchanged.

    Args:
        data_yaml_path: Path to data.yaml.
    """
    global _original_build_dataloader

    import ultralytics.data.build as build_module

    if _original_build_dataloader is not None:
        print("Warning: weighted sampling already applied, skipping.")
        return

    _original_build_dataloader = build_module.build_dataloader

    image_weights = compute_image_weights(data_yaml_path)[0]
    weights_tensor = np.array(image_weights, dtype=np.float64)

    original_fn = _original_build_dataloader

    def patched_build_dataloader(dataset, batch, workers, shuffle=True, rank=-1):
        """Wrapper that injects WeightedRandomSampler for training."""
        if shuffle and rank == -1:
            # Training mode: replace shuffle with weighted sampling
            sampler = WeightedRandomSampler(
                weights=weights_tensor,
                num_samples=len(weights_tensor),
                replacement=True,
            )
            loader = original_fn(dataset, batch, workers, shuffle=False, rank=rank)
            # Replace the sampler in the underlying DataLoader
            loader.sampler = sampler
            return loader
        else:
            return original_fn(dataset, batch, workers, shuffle=shuffle, rank=rank)

    build_module.build_dataloader = patched_build_dataloader
    print("Weighted sampling monkey-patch applied.")


def restore_default_sampling():
    """Restore the original Ultralytics build_dataloader function."""
    global _original_build_dataloader

    if _original_build_dataloader is None:
        return

    import ultralytics.data.build as build_module
    build_module.build_dataloader = _original_build_dataloader
    _original_build_dataloader = None
    print("Weighted sampling monkey-patch removed.")
