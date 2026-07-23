"""Leak-free test manifest v2: pHash + capture-adjacency criterion (R4.2).

A test image is quarantined if ANY of:
  A. pHash Hamming distance <= 6 to any train/valid image (near-identical), or
  B. capture adjacency to any train/valid image:
       - foto_<t>: |t_delta| <= 500 (timestamps in centiseconds -> 5 s burst window)
       - IMG_<n>:  |n_delta| <= 2   (consecutive camera shots)
     combined with pHash distance <= 14 as a same-scene sanity check, or
  C. detect_image_<pose>: pHash distance <= 12 to any train/valid detect_image
     (robot re-visits of the same pose; visually verified same scene at 12).

Writes data/test_clean.txt and data/test_quarantined.txt (overwrites v1).
"""

import os
import re
from PIL import Image
import imagehash

DATA_DIR = r"D:\tmp\TFM\ICM\BenchMarks\data"

RF_RE = re.compile(r"^(?P<stem>.+?)_(?:jpg|JPG|jpeg|png)\.rf\.[0-9a-f]{32}\.(jpg|jpeg|png)$")
FOTO_RE = re.compile(r"^foto_(\d+)$")
IMG_RE = re.compile(r"^IMG_(\d+)$")
DETECT_RE = re.compile(r"^detect_image_")

FOTO_WINDOW = 500     # centiseconds (5 s)
IMG_WINDOW = 2
PHASH_STRICT = 6      # criterion A
PHASH_ADJ = 14        # sanity bound for criterion B
PHASH_DETECT = 12     # criterion C


def stem(fname):
    m = RF_RE.match(fname)
    return m.group("stem") if m else os.path.splitext(fname)[0]


def classify(fname):
    st = stem(fname)
    m = FOTO_RE.match(st)
    if m:
        return ("foto", int(m.group(1)))
    m = IMG_RE.match(st)
    if m:
        return ("img", int(m.group(1)))
    if DETECT_RE.match(st):
        return ("detect", None)
    return ("other", None)


def load(split):
    d = os.path.join(DATA_DIR, split, "images")
    out = []
    for f in sorted(os.listdir(d)):
        with Image.open(os.path.join(d, f)) as im:
            out.append((f, classify(f), imagehash.phash(im)))
    return out


def main():
    ref = load("train") + load("valid")
    test = load("test")
    print(f"ref={len(ref)}, test={len(test)}")

    quarantined = []   # (fname, reason)
    clean = []
    for f, (fam, num), h in test:
        reason = None
        for rf, (rfam, rnum), rh in ref:
            dist = h - rh
            if dist <= PHASH_STRICT:
                reason = f"A phash={dist} nearest={rf}"
                break
            if fam == rfam and num is not None and rnum is not None:
                delta = abs(num - rnum)
                window = FOTO_WINDOW if fam == "foto" else IMG_WINDOW
                if delta <= window and dist <= PHASH_ADJ:
                    reason = f"B {fam}-adjacent delta={delta} phash={dist} nearest={rf}"
                    break
            if fam == "detect" and rfam == "detect" and dist <= PHASH_DETECT:
                reason = f"C detect phash={dist} nearest={rf}"
                break
        if reason:
            quarantined.append((f, reason))
        else:
            clean.append(f)

    with open(os.path.join(DATA_DIR, "test_clean.txt"), "w") as fh:
        fh.write("\n".join(clean) + "\n")
    with open(os.path.join(DATA_DIR, "test_quarantined.txt"), "w") as fh:
        for f, r in quarantined:
            fh.write(f"{f}\t{r}\n")

    from collections import Counter
    reasons = Counter(r.split()[0] for _, r in quarantined)
    print(f"quarantined={len(quarantined)} (by criterion: {dict(reasons)})")
    print(f"clean={len(clean)} -> data/test_clean.txt")


if __name__ == "__main__":
    main()
