"""Materialize the JRTIP-revision evaluation data (R4.1 / R4.2 / R4.4).

Creates, from the existing splits and the quarantine manifest:

  data/test_clean/{images,labels}   the 146-image leak-free test subset
  data/calib_train/{images,labels}  624 deterministically sampled TRAIN images
                                    for INT8 calibration (replaces the previous
                                    validation-set calibration)
  data/data_test_clean.yaml         val -> test_clean (evaluation yaml)
  data/data_calib.yaml              val -> calib_train (INT8 export yaml)

Idempotent: re-running rebuilds the directories from scratch.
"""

import os
import random
import shutil

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CALIB_N = 624          # matches the size of the previous (val) calibration set
SEED = 42

NAMES = ["IV-1A", "IV-1B", "IV-2", "IV-3", "IV-4", "IV-5", "IV-6", "Solda"]


def label_for(img):
    return os.path.splitext(img)[0] + ".txt"


def materialize(name, src_split, image_list):
    root = os.path.join(DATA_DIR, name)
    if os.path.isdir(root):
        shutil.rmtree(root)
    os.makedirs(os.path.join(root, "images"))
    os.makedirs(os.path.join(root, "labels"))
    for img in image_list:
        shutil.copy2(os.path.join(DATA_DIR, src_split, "images", img),
                     os.path.join(root, "images", img))
        lbl = label_for(img)
        src_lbl = os.path.join(DATA_DIR, src_split, "labels", lbl)
        if os.path.exists(src_lbl):
            shutil.copy2(src_lbl, os.path.join(root, "labels", lbl))
    return len(image_list)


def write_yaml(fname, val_rel):
    path = os.path.join(DATA_DIR, fname)
    with open(path, "w") as fh:
        fh.write(f"""train: ../train/images
val: {val_rel}
test: ../test/images

nc: 8
names: {NAMES}
""")
    return path


def main():
    # 1. Clean test subset from the manifest (tracked in config/)
    manifest = os.path.join(os.path.dirname(DATA_DIR), "config", "test_clean.txt")
    with open(manifest) as fh:
        clean = [ln.strip() for ln in fh if ln.strip()]
    n = materialize("test_clean", "test", clean)
    print(f"test_clean: {n} images")

    # 2. Deterministic train-drawn calibration set
    train_imgs = sorted(os.listdir(os.path.join(DATA_DIR, "train", "images")))
    rng = random.Random(SEED)
    calib = sorted(rng.sample(train_imgs, CALIB_N))
    n = materialize("calib_train", "train", calib)
    print(f"calib_train: {n} images (seed={SEED})")

    # 3. Data yamls
    print("wrote", write_yaml("data_test_clean.yaml", "../test_clean/images"))
    print("wrote", write_yaml("data_calib.yaml", "../calib_train/images"))


if __name__ == "__main__":
    main()
