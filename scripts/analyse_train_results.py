"""Deep analysis of all benchmark results."""
import os, glob, re

def parse_report(path):
    with open(path) as f:
        content = f.read()
    def get(pattern):
        m = re.search(pattern, content)
        return m.group(1).strip() if m else None

    per_class = {}
    lines = content.split("\n")
    in_pc = False
    for line in lines:
        if "--- Per-class Accuracy ---" in line:
            in_pc = True; continue
        if in_pc:
            if line.startswith("---") or not line.strip():
                in_pc = False; continue
            if "Class" in line and "mAP50" in line:
                continue
            parts = line.split()
            if len(parts) >= 5:
                per_class[parts[0]] = {
                    "P": float(parts[1]), "R": float(parts[2]),
                    "mAP50": float(parts[3]), "mAP50_95": float(parts[4])
                }
    return {
        "experiment": get(r"Experiment: (.+)"),
        "arch": get(r"Architecture: (.+)"),
        "size": get(r"Model size: (.+)"),
        "task": get(r"Task: (.+)"),
        "approach": get(r"Approach: (.+)"),
        "map50": float(get(r"mAP50:\s+([\d.]+)") or 0),
        "map50_95": float(get(r"mAP50-95:\s+([\d.]+)") or 0),
        "precision": float(get(r"P \(mean\):\s+([\d.]+)") or 0),
        "recall": float(get(r"R \(mean\):\s+([\d.]+)") or 0),
        "fps": float(get(r"FPS:\s+([\d.]+)") or 0),
        "inference_ms": float(get(r"Inference:\s+([\d.]+) ms") or 0),
        "imgsz": get(r"Input size: (.+)"),
        "per_class": per_class,
    }

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "rtx5090")
ORDER = ["nano", "small", "medium", "large"]

# Load all reports
reports = [parse_report(f) for f in sorted(glob.glob(os.path.join(ROOT, "*", "*", "report.txt")))]
core   = [r for r in reports if r["experiment"] == "core_comparison"]
det_seg = [r for r in reports if r["experiment"] == "detection_vs_segmentation"]
bal    = [r for r in reports if r["experiment"] == "class_imbalance"]

# Input size reports
is_reports = []
for f in sorted(glob.glob(os.path.join(ROOT, "input_size", "*", "report_*.txt"))):
    r = parse_report(f)
    m = re.search(r"img(\d+)", os.path.basename(f))
    if m:
        r["imgsz"] = m.group(1)
    r["experiment"] = "input_size"
    is_reports.append(r)

def find(lst, **kwargs):
    for r in lst:
        if all(r.get(k) == v for k, v in kwargs.items()):
            return r
    return None

SEP = "=" * 72

# ------------------------------------------------------------------ #
print(SEP)
print("EXPERIMENT 1: CORE COMPARISON — YOLOv12 vs YOLOv26 (Segmentation)")
print(SEP)
header = f"  {'Model':<42} {'mAP50':>7} {'mAP50-95':>9} {'P':>7} {'R':>7} {'FPS':>8} {'ms':>6}"
print(f"\n{header}")
print("-" * 90)
for arch in ["yolo12", "yolo26"]:
    for approach in ["pretrained", "scratch"]:
        for size in ORDER:
            r = find(core, arch=arch, size=size, approach=approach)
            if r:
                lbl = f"{arch} {size} {approach}"
                print(f"  {lbl:<42} {r['map50']:7.4f} {r['map50_95']:9.4f}"
                      f" {r['precision']:7.4f} {r['recall']:7.4f}"
                      f" {r['fps']:8.1f} {r['inference_ms']:6.2f}")
    print()

print("\n--- Pretrained vs Scratch lift (mAP50) ---")
for arch in ["yolo12", "yolo26"]:
    for size in ORDER:
        pre = find(core, arch=arch, size=size, approach="pretrained")
        scr = find(core, arch=arch, size=size, approach="scratch")
        if pre and scr:
            d = pre["map50"] - scr["map50"]
            print(f"  {arch} {size:<7}: pre={pre['map50']:.4f}  scr={scr['map50']:.4f}  Δ={d:+.4f}")
    print()

print("\n--- YOLOv12 vs YOLOv26 (pretrained, mAP50) ---")
for size in ORDER:
    y12 = find(core, arch="yolo12", size=size, approach="pretrained")
    y26 = find(core, arch="yolo26", size=size, approach="pretrained")
    if y12 and y26:
        dm = y26["map50"] - y12["map50"]
        df = y26["fps"] - y12["fps"]
        print(f"  {size:<7}: yolo12={y12['map50']:.4f}/{y12['fps']:.0f}fps  "
              f"yolo26={y26['map50']:.4f}/{y26['fps']:.0f}fps  "
              f"ΔmAP={dm:+.4f} ΔFps={df:+.0f}")

# ------------------------------------------------------------------ #
print(f"\n\n{SEP}")
print("EXPERIMENT 2: DETECTION vs SEGMENTATION")
print(SEP)
print(f"\n  {'Model':<42} {'task':>4}  {'mAP50':>7} {'FPS':>8}  vs seg:  {'mAP50':>7} {'FPS':>8}  {'ΔmAP':>7} {'ΔFps':>6}")
print("-" * 100)
for arch in ["yolo12", "yolo26"]:
    for approach in ["pretrained", "scratch"]:
        for size in ORDER:
            rd = find(det_seg, arch=arch, size=size, approach=approach)
            rs = find(core,    arch=arch, size=size, approach=approach)
            if rd and rs:
                lbl = f"{arch} {size} {approach}"
                dm = rd["map50"] - rs["map50"]
                df = rd["fps"] - rs["fps"]
                print(f"  {lbl:<42} det  {rd['map50']:7.4f} {rd['fps']:8.1f}  "
                      f"seg: {rs['map50']:7.4f} {rs['fps']:8.1f}  {dm:+7.4f} {df:+6.0f}")
    print()

# ------------------------------------------------------------------ #
print(f"\n\n{SEP}")
print("EXPERIMENT 3: CLASS IMBALANCE — Balanced Weighted Sampler")
print(SEP)
print("\n  NOTE: yolo26 medium pretrained_balanced = EXCLUDED (sampler failed)\n")
print(f"  {'Model':<42} {'balanced':>8} {'baseline':>9} {'Δ':>7}  {'fps_bal':>8} {'fps_base':>9} {'ΔFps':>6}")
print("-" * 100)
for arch in ["yolo12", "yolo26"]:
    for approach in ["pretrained", "scratch"]:
        for size in ORDER:
            b = find(bal,  arch=arch, size=size, approach=approach + "_balanced")
            u = find(core, arch=arch, size=size, approach=approach)
            if b and u:
                lbl = f"{arch} {size} {approach}"
                invalid = (arch == "yolo26" and size == "medium" and approach == "pretrained")
                flag = "  ⚠" if invalid else ""
                dm = b["map50"] - u["map50"]
                df = b["fps"] - u["fps"]
                print(f"  {lbl:<42} {b['map50']:8.4f} {u['map50']:9.4f} {dm:+7.4f}"
                      f"  {b['fps']:8.1f} {u['fps']:9.1f} {df:+6.0f}{flag}")
    print()

# ------------------------------------------------------------------ #
print(f"\n\n{SEP}")
print("EXPERIMENT 4: INPUT SIZE — 320 vs 640 (baseline) vs 1280")
print(SEP)
print(f"\n  {'Model':<42}  {'mAP320':>7} {'mAP640':>7} {'mAP1280':>8}  {'fps320':>7} {'fps640':>7} {'fps1280':>8}")
print("-" * 100)
for arch in ["yolo12", "yolo26"]:
    for approach in ["pretrained", "scratch"]:
        for size in ORDER:
            r320  = find(is_reports, arch=arch, size=size, approach=approach, imgsz="320")
            r640  = find(core,       arch=arch, size=size, approach=approach)
            r1280 = find(is_reports, arch=arch, size=size, approach=approach, imgsz="1280")
            if r320 or r640 or r1280:
                lbl = f"{arch} {size} {approach}"
                m320  = f"{r320['map50']:.4f}"  if r320  else "  -   "
                m640  = f"{r640['map50']:.4f}"  if r640  else "  -   "
                m1280 = f"{r1280['map50']:.4f}" if r1280 else "  -    "
                f320  = f"{r320['fps']:7.0f}"   if r320  else "    N/A"
                f640  = f"{r640['fps']:7.0f}"   if r640  else "    N/A"
                f1280 = f"{r1280['fps']:8.0f}"  if r1280 else "     N/A"
                print(f"  {lbl:<42}  {m320:>7} {m640:>7} {m1280:>8}  {f320} {f640} {f1280}")
    print()

# ------------------------------------------------------------------ #
print(f"\n\n{SEP}")
print("PER-CLASS ACCURACY — Core Comparison (Pretrained Segmentation)")
print(SEP)
classes = ["IV-1A","IV-1B","IV-2","IV-3","IV-4","IV-5","IV-6","Solda"]
for arch in ["yolo12","yolo26"]:
    print(f"\n  {arch} pretrained:")
    print(f"  {'Class':<8}", end="")
    for size in ORDER:
        print(f"  {size:^19}", end="")
    print()
    print(f"  {'':8}", end="")
    for _ in ORDER:
        print(f"  {'mAP50':>8} {'mAP50-95':>9}", end="")
    print()
    print("  " + "-"*87)
    for cls in classes:
        print(f"  {cls:<8}", end="")
        for size in ORDER:
            r = find(core, arch=arch, size=size, approach="pretrained")
            pc = r["per_class"].get(cls, {}) if r else {}
            m50    = f"{pc['mAP50']:.4f}"    if pc else "  -  "
            m5095  = f"{pc['mAP50_95']:.4f}" if pc else "  -  "
            print(f"  {m50:>8} {m5095:>9}", end="")
        print()
    print()
