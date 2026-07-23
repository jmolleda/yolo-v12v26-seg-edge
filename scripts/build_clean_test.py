"""Build the leak-free (quarantined) test manifest for the JRTIP revision (R4.2).

Recomputes cross-split pHash distances, reports test-set contamination at
several thresholds, then writes:
  - data/test_clean.txt         image filenames of the clean test subset
  - data/test_quarantined.txt   purged test images + reason (nearest split/file/dist)

Threshold for the quarantine is set by QUARANTINE_T below.
"""

import os
import re
from collections import defaultdict

from PIL import Image
import imagehash

DATA_DIR = r"D:\tmp\TFM\ICM\BenchMarks\data"
SPLITS = ["train", "valid", "test"]
QUARANTINE_T = 6   # pHash Hamming distance; adjust after sensitivity readout


def hash_split(split):
    d = os.path.join(DATA_DIR, split, "images")
    out = []
    for f in sorted(os.listdir(d)):
        with Image.open(os.path.join(d, f)) as im:
            out.append((f, imagehash.phash(im)))
    return out


def main():
    hashes = {s: hash_split(s) for s in SPLITS}
    print({s: len(v) for s, v in hashes.items()})

    # For each test image, the minimum distance to any train or valid image.
    ref = hashes["train"] + hashes["valid"]
    nearest = {}
    for f, h in hashes["test"]:
        best = None
        for rf, rh in ref:
            dst = h - rh
            if best is None or dst < best[0]:
                best = (dst, rf)
        nearest[f] = best

    for t in (0, 2, 4, 6, 8, 10, 12):
        n = sum(1 for d, _ in nearest.values() if d <= t)
        print(f"threshold <= {t:2d}: {n:3d} / {len(nearest)} test images contaminated")

    clean = sorted(f for f, (d, _) in nearest.items() if d > QUARANTINE_T)
    quarantined = sorted((f, d, rf) for f, (d, rf) in nearest.items() if d <= QUARANTINE_T)

    with open(os.path.join(DATA_DIR, "test_clean.txt"), "w") as fh:
        fh.write("\n".join(clean) + "\n")
    with open(os.path.join(DATA_DIR, "test_quarantined.txt"), "w") as fh:
        for f, d, rf in quarantined:
            fh.write(f"{f}\tdist={d}\tnearest={rf}\n")

    print(f"\nQuarantine threshold: <= {QUARANTINE_T}")
    print(f"Clean test subset: {len(clean)} images -> data/test_clean.txt")
    print(f"Quarantined:       {len(quarantined)} images -> data/test_quarantined.txt")


if __name__ == "__main__":
    main()
