"""Split-leakage audit for the steel defect dataset (JRTIP major revision, R4.2).

Checks, across the train/valid/test splits:
  1. Source-name collisions: the same original file (name before the Roboflow
     ".rf.<hash>" suffix) appearing in more than one split.
  2. Sequence adjacency: IMG_<n> camera-sequence names whose neighbours
     (n-1 / n+1) live in a different split — a proxy for temporally adjacent
     captures of the same scene being separated by a random split.
  3. Near-duplicates: perceptual-hash (pHash) distance <= threshold between
     images in different splits.
  4. Per-split class and instance counts (from YOLO label files), including
     defect-free images.

Outputs a Markdown report to major-revision/split_audit_report.md in the paper
repo and prints a summary to stdout.
"""

import os
import re
import sys
from collections import defaultdict

from PIL import Image
import imagehash

DATA_DIR = r"D:\tmp\TFM\ICM\BenchMarks\data"
SPLITS = ["train", "valid", "test"]
CLASS_NAMES = ["IV-1A", "IV-1B", "IV-2", "IV-3", "IV-4", "IV-5", "IV-6", "Solda"]
PHASH_THRESHOLD = 6          # Hamming distance on 64-bit pHash
REPORT = r"D:\GitHub\jmolleda\Papers\2026-edge-seg\paper\major-revision\split_audit_report.md"

RF_RE = re.compile(r"^(?P<stem>.+?)_(?:jpg|JPG|jpeg|png)\.rf\.[0-9a-f]{32}\.(jpg|jpeg|png)$")
IMG_SEQ_RE = re.compile(r"^IMG_(?P<num>\d+)$")


def source_stem(fname):
    """Original file stem before Roboflow suffixing, or full name if no match."""
    m = RF_RE.match(fname)
    return m.group("stem") if m else os.path.splitext(fname)[0]


def collect():
    files = {}   # split -> list of filenames
    for s in SPLITS:
        d = os.path.join(DATA_DIR, s, "images")
        files[s] = sorted(os.listdir(d))
    return files


def audit_source_names(files):
    """Same source stem in more than one split."""
    stem_map = defaultdict(set)   # stem -> {splits}
    stem_files = defaultdict(list)
    for s in SPLITS:
        for f in files[s]:
            st = source_stem(f)
            stem_map[st].add(s)
            stem_files[st].append((s, f))
    collisions = {st: sorted(sp) for st, sp in stem_map.items() if len(sp) > 1}
    return collisions, stem_files


def audit_sequences(files):
    """IMG_<n> whose neighbour number is in a different split."""
    seq = {}   # number -> (split, fname)
    for s in SPLITS:
        for f in files[s]:
            m = IMG_SEQ_RE.match(source_stem(f))
            if m:
                seq[int(m.group("num"))] = (s, f)
    adjacent = []
    for n, (s, f) in sorted(seq.items()):
        for nb in (n - 1, n + 1):
            if nb in seq and seq[nb][0] != s and nb > n:
                adjacent.append((n, s, nb, seq[nb][0]))
    return seq, adjacent


def audit_phash(files):
    """Cross-split perceptual-hash near-duplicates."""
    hashes = []   # (split, fname, hash)
    for s in SPLITS:
        d = os.path.join(DATA_DIR, s, "images")
        for f in files[s]:
            with Image.open(os.path.join(d, f)) as im:
                hashes.append((s, f, imagehash.phash(im)))
        print(f"  hashed {s}: {len(files[s])} images")
    pairs = []
    for i in range(len(hashes)):
        s1, f1, h1 = hashes[i]
        for j in range(i + 1, len(hashes)):
            s2, f2, h2 = hashes[j]
            if s1 == s2:
                continue
            dist = h1 - h2
            if dist <= PHASH_THRESHOLD:
                pairs.append((dist, s1, f1, s2, f2))
    pairs.sort()
    return pairs


def audit_labels():
    """Per-split class instance counts + defect-free image counts."""
    stats = {}
    for s in SPLITS:
        d = os.path.join(DATA_DIR, s, "labels")
        cls_counts = defaultdict(int)
        empty = 0
        n_files = 0
        for f in sorted(os.listdir(d)):
            n_files += 1
            path = os.path.join(d, f)
            with open(path) as fh:
                lines = [ln for ln in fh.read().splitlines() if ln.strip()]
            if not lines:
                empty += 1
            for ln in lines:
                cls_counts[int(ln.split()[0])] += 1
        stats[s] = {"files": n_files, "empty": empty, "cls": dict(cls_counts)}
    return stats


def main():
    files = collect()
    print("Collecting source-name collisions...")
    collisions, stem_files = audit_source_names(files)
    print(f"  {len(collisions)} colliding source stems")

    print("Checking IMG_<n> sequence adjacency across splits...")
    seq, adjacent = audit_sequences(files)
    print(f"  {len(seq)} sequence-named images, {len(adjacent)} cross-split adjacent pairs")

    print("Computing per-split label statistics...")
    stats = audit_labels()

    print("Perceptual hashing (this is the slow part)...")
    pairs = audit_phash(files)
    print(f"  {len(pairs)} cross-split pairs with pHash distance <= {PHASH_THRESHOLD}")

    # ── report ──
    lines = []
    lines.append("# Split-Leakage Audit Report (R4.2)\n")
    lines.append(f"Dataset: `{DATA_DIR}`  \nSplits: train/valid/test = "
                 f"{len(files['train'])}/{len(files['valid'])}/{len(files['test'])}  \n"
                 f"pHash threshold: Hamming distance <= {PHASH_THRESHOLD} (64-bit pHash)\n")

    lines.append("## 1. Source-name collisions across splits\n")
    if collisions:
        lines.append(f"**{len(collisions)} source stems appear in more than one split:**\n")
        for st, sp in sorted(collisions.items()):
            lines.append(f"- `{st}` -> {', '.join(sp)}")
            for s, f in stem_files[st]:
                lines.append(f"    - {s}: `{f}`")
    else:
        lines.append("None. Every original source file maps to exactly one split.")
    lines.append("")

    lines.append("## 2. Camera-sequence adjacency across splits\n")
    n_seq_per_split = defaultdict(int)
    for n, (s, f) in seq.items():
        n_seq_per_split[s] += 1
    lines.append(f"IMG_<n> sequence-named images per split: " +
                 ", ".join(f"{s}={n_seq_per_split[s]}" for s in SPLITS) + "\n")
    if adjacent:
        lines.append(f"**{len(adjacent)} consecutive-number pairs straddle splits** "
                     "(consecutive shots may show the same physical area):\n")
        for n, s, nb, s2 in adjacent:
            lines.append(f"- IMG_{n} ({s}) is adjacent to IMG_{nb} ({s2})")
    else:
        lines.append("No consecutive-number pairs straddle splits.")
    lines.append("")

    lines.append("## 3. Cross-split perceptual near-duplicates\n")
    if pairs:
        lines.append(f"**{len(pairs)} cross-split pairs at distance <= {PHASH_THRESHOLD}:**\n")
        lines.append("| dist | split A | file A | split B | file B |")
        lines.append("|---|---|---|---|---|")
        for dist, s1, f1, s2, f2 in pairs:
            lines.append(f"| {dist} | {s1} | `{f1}` | {s2} | `{f2}` |")
    else:
        lines.append("None found.")
    lines.append("")

    lines.append("## 4. Per-split class and instance counts\n")
    header = "| split | images | defect-free | " + " | ".join(CLASS_NAMES) + " | total inst. |"
    lines.append(header)
    lines.append("|" + "---|" * (len(CLASS_NAMES) + 4))
    for s in SPLITS:
        st = stats[s]
        total = sum(st["cls"].values())
        row = (f"| {s} | {st['files']} | {st['empty']} | " +
               " | ".join(str(st["cls"].get(i, 0)) for i in range(len(CLASS_NAMES))) +
               f" | {total} |")
        lines.append(row)
    lines.append("")

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"\nReport written to {REPORT}")


if __name__ == "__main__":
    main()
