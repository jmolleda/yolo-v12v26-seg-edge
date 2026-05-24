"""Generate analysis figures for the IEEE TII paper.

molle3.pdf  Accuracy-Latency Pareto scatter  (3-panel, double-column)
molle4.pdf  Power-Efficiency scatter          (FPS/W vs mAP50-95, AGX only)
molle5.pdf  Quantisation cost-benefit scatter (speedup vs accuracy loss, AGX)

Run from the BenchMarks root:
    python scripts/gen_analysis_figures.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ── paths ─────────────────────────────────────────────────────────────────────
RTX_CSV  = "results/rtx5090/benchmark_results.csv"
AGX_CSV  = "results/jetson_agx/benchmark_results.csv"
NANO_CSV = "results/jetson_nano/benchmark_results.csv"
OUT_DIR  = r"D:\GitHub\jmolleda\Papers\2026-edge-seg\paper"
FONT     = "DejaVu Sans"

# ── colours consistent with molle2.pdf ────────────────────────────────────────
C_26PRE = "#1a4e8c"   # dark blue   — YOLOv26 pretrained
C_26SCR = "#5b9bd5"   # mid blue    — YOLOv26 scratch
C_12SCR = "#e07b39"   # orange      — YOLOv12 scratch

GROUPS = {
    ("yolo26", "pretrained"): (C_26PRE, "YOLOv26 pretrained"),
    ("yolo26", "scratch"):    (C_26SCR, "YOLOv26 scratch"),
    ("yolo12", "scratch"):    (C_12SCR, "YOLOv12 scratch"),
}

SIZE_PT = {"nano": 30, "small": 55, "medium": 90, "large": 130}

# ── helpers ───────────────────────────────────────────────────────────────────

def load_all():
    dfs = []
    for path, dlabel in [
        (RTX_CSV,  "RTX 5090"),
        (AGX_CSV,  "Jetson AGX"),
        (NANO_CSV, "Jetson Nano"),
    ]:
        df = pd.read_csv(path)
        df["device_label"] = dlabel
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df["size_mb"] = df["model_size"].str.replace(" MB", "", regex=False).astype(float)
    df["size_label"] = df.apply(_size_label, axis=1)
    return df


def _size_label(row):
    mb, arch = row["size_mb"], row["architecture"]
    t = (12, 40, 52) if arch == "yolo12" else (12, 40, 58)
    if mb < t[0]:   return "nano"
    elif mb < t[1]: return "small"
    elif mb < t[2]: return "medium"
    else:           return "large"


def core_seg_b1(df):
    return df[
        (df["experiment"] == "core_comparison") &
        (df["task"] == "segment") &
        (df["batch"] == 1)
    ].copy()


def pareto_front(xs, ys):
    """Indices of Pareto-optimal (maximise both) points, sorted by x."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    n = len(xs)
    keep = np.ones(n, bool)
    for i in range(n):
        for j in range(n):
            if i != j and xs[j] >= xs[i] and ys[j] >= ys[i] and (xs[j] > xs[i] or ys[j] > ys[i]):
                keep[i] = False
                break
    idx = np.where(keep)[0]
    return idx[np.argsort(xs[idx])]


# ── Fig. molle3: Pareto scatter ───────────────────────────────────────────────

def make_molle3(df):
    sub = core_seg_b1(df)
    # Keep FP32 (pytorch) and FP16 (TRT only) — skip INT8 to reduce clutter
    sub = sub[~((sub["format"] == "tensorrt") & (sub["precision"] == "int8"))]

    # Per-device axis configuration
    DEVICE_CFG = {
        "RTX 5090":    {"xscale": "log",    "xticks": [100, 200, 400, 800]},
        "Jetson AGX":  {"xscale": "linear", "xticks": [20, 30, 40, 50, 60]},
        "Jetson Nano": {"xscale": "log",    "xticks": [6, 10, 20, 40]},
    }

    devices = ["RTX 5090", "Jetson AGX", "Jetson Nano"]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.7))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.26,
                        wspace=0.30)

    for ax, dlabel in zip(axes, devices):
        ddata = sub[sub["device_label"] == dlabel]

        for (arch, approach), (color, _) in GROUPS.items():
            grp = ddata[
                (ddata["architecture"] == arch) &
                (ddata["approach"] == approach)
            ]
            if grp.empty:
                continue

            for _, row in grp.iterrows():
                marker = "o" if row["format"] == "pytorch" else "s"
                sz = SIZE_PT.get(row["size_label"], 50)
                fill = color if row["approach"] == "pretrained" else "none"
                ec   = color
                ax.scatter(row["fps"], row["map50_95"],
                           s=sz, marker=marker,
                           facecolors=fill, edgecolors=ec,
                           linewidths=0.8, zorder=3)

            # Pareto frontier
            xs = grp["fps"].values
            ys = grp["map50_95"].values
            pidx = pareto_front(xs, ys)
            if len(pidx) >= 2:
                ax.plot(xs[pidx], ys[pidx],
                        color=color, linewidth=0.9, linestyle="--",
                        zorder=2, alpha=0.8)

        cfg = DEVICE_CFG[dlabel]
        ax.set_xscale(cfg["xscale"])
        ax.xaxis.set_major_locator(mticker.FixedLocator(cfg["xticks"]))
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.xaxis.set_major_formatter(mticker.FixedFormatter(
            [str(t) for t in cfg["xticks"]]
        ))
        ax.set_title(dlabel, fontsize=7, fontfamily=FONT, pad=3)
        ax.set_xlabel("FPS", fontsize=6.5, fontfamily=FONT)
        if ax is axes[0]:
            ax.set_ylabel("mAP50-95", fontsize=6.5, fontfamily=FONT)
        ax.set_ylim(top=0.34)
        ax.tick_params(labelsize=6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Legend
    legend_els = []
    for (arch, approach), (color, label) in GROUPS.items():
        fill = color if approach == "pretrained" else "none"
        legend_els.append(
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=fill, markeredgecolor=color,
                   markeredgewidth=0.8, markersize=5, label=label)
        )
    legend_els += [
        Line2D([0], [0], color="grey", linewidth=0.9, linestyle="--",
               label="Pareto frontier"),
        Line2D([0], [0], marker="s", color="grey", markersize=5, linestyle="none",
               label="TRT FP16"),
        Line2D([0], [0], marker="o", color="grey", markersize=5, linestyle="none",
               label="PyTorch FP32"),
    ]
    fig.legend(handles=legend_els, loc="lower center", ncol=3,
               fontsize=5.5, frameon=False,
               bbox_to_anchor=(0.5, 0.01), handletextpad=0.4, columnspacing=1.0)

    out = os.path.join(OUT_DIR, "molle3.pdf")
    fig.savefig(out, format="pdf", dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")


# ── Fig. molle4: Power-Efficiency scatter (AGX) ───────────────────────────────

def make_molle4(df):
    sub = core_seg_b1(df)
    sub = sub[
        (sub["device_label"] == "Jetson AGX") &
        sub["fps_per_watt"].notna() &
        sub["map50_95"].notna()
    ]
    # Skip INT8 (≈FP16 on AGX)
    sub = sub[~((sub["format"] == "tensorrt") & (sub["precision"] == "int8"))]

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    fig.subplots_adjust(left=0.14, right=0.97, top=0.93, bottom=0.20)

    for (arch, approach), (color, label) in GROUPS.items():
        grp = sub[
            (sub["architecture"] == arch) &
            (sub["approach"] == approach)
        ]
        if grp.empty:
            continue
        for _, row in grp.iterrows():
            marker = "o" if row["format"] == "pytorch" else "s"
            sz = SIZE_PT.get(row["size_label"], 50)
            fill = color if row["approach"] == "pretrained" else "none"
            ax.scatter(row["fps_per_watt"], row["map50_95"],
                       s=sz, marker=marker,
                       facecolors=fill, edgecolors=color,
                       linewidths=0.8, zorder=3)

        # Pareto frontier
        xs = grp["fps_per_watt"].values
        ys = grp["map50_95"].values
        pidx = pareto_front(xs, ys)
        if len(pidx) >= 2:
            ax.plot(xs[pidx], ys[pidx],
                    color=color, linewidth=0.9, linestyle="--",
                    zorder=2, alpha=0.8)

    ax.set_xlabel("FPS / W", fontsize=7, fontfamily=FONT)
    ax.set_ylabel("mAP50-95", fontsize=7, fontfamily=FONT)
    ax.set_title("Power Efficiency — Jetson AGX Orin", fontsize=7.5,
                 fontfamily=FONT, pad=3)
    ax.tick_params(labelsize=6.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_els = []
    for (arch, approach), (color, label) in GROUPS.items():
        fill = color if approach == "pretrained" else "none"
        legend_els.append(
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=fill, markeredgecolor=color,
                   markeredgewidth=0.8, markersize=5, label=label)
        )
    legend_els += [
        Line2D([0], [0], marker="o", color="grey", markersize=5, linestyle="none",
               label="PyTorch FP32"),
        Line2D([0], [0], marker="s", color="grey", markersize=5, linestyle="none",
               label="TRT FP16"),
    ]
    ax.legend(handles=legend_els, fontsize=5.5, frameon=False,
              loc="lower right", handletextpad=0.4)

    out = os.path.join(OUT_DIR, "molle4.pdf")
    fig.savefig(out, format="pdf", dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")


# ── Fig. molle5: Quantisation cost-benefit (AGX) ─────────────────────────────

def make_molle5(df):
    """Scatter: x = mAP50-95 drop vs FP32 baseline (pp), y = FPS speedup ratio.

    Only TRT points (FP16 and INT8) vs their pytorch FP32 baseline,
    on Jetson AGX.  Positive x = accuracy gain, negative = loss.
    """
    sub = core_seg_b1(df)
    agx = sub[sub["device_label"] == "Jetson AGX"].copy()

    # Build (arch, size_label, approach) → fp32 FPS and mAP
    fp32 = agx[agx["format"] == "pytorch"].set_index(
        ["architecture", "size_label", "approach"]
    )[["fps", "map50_95"]].rename(columns={"fps": "fps_fp32", "map50_95": "map_fp32"})

    trt = agx[agx["format"] == "tensorrt"].copy()
    trt = trt.join(fp32, on=["architecture", "size_label", "approach"], how="inner")
    trt["speedup"]   = trt["fps"] / trt["fps_fp32"]
    trt["map_delta"] = (trt["map50_95"] - trt["map_fp32"]) * 100  # pp

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    fig.subplots_adjust(left=0.14, right=0.97, top=0.90, bottom=0.16)

    prec_style = {
        "fp16": {"marker": "o", "color": C_26PRE,  "label": "TRT FP16"},
        "int8": {"marker": "s", "color": C_12SCR,  "label": "TRT INT8"},
    }
    arch_marker = {"yolo26": "o", "yolo12": "^"}

    for (arch, approach), (color, _) in GROUPS.items():
        for prec, pstyle in prec_style.items():
            grp = trt[
                (trt["architecture"] == arch) &
                (trt["approach"] == approach) &
                (trt["precision"] == prec)
            ]
            if grp.empty:
                continue
            for _, row in grp.iterrows():
                sz = SIZE_PT.get(row["size_label"], 50)
                ec = pstyle["color"]
                fill = color
                mkr = arch_marker.get(arch, "o")
                ax.scatter(row["map_delta"], row["speedup"],
                           s=sz, marker=mkr,
                           facecolors=fill, edgecolors=ec,
                           linewidths=1.0, zorder=3)

    ax.axvline(0, color="#aaaaaa", linewidth=0.8, linestyle=":")
    ax.axhline(1, color="#aaaaaa", linewidth=0.8, linestyle=":")

    ax.set_xlabel("mAP50-95 change vs PyTorch FP32 (pp)", fontsize=6.5, fontfamily=FONT)
    ax.set_ylabel("FPS speedup vs FP32", fontsize=6.5, fontfamily=FONT)
    ax.set_title("TRT Quantisation — Jetson AGX Orin", fontsize=7.5,
                 fontfamily=FONT, pad=3)
    ax.tick_params(labelsize=6.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend: colour = arch+approach group, marker edge = precision, shape = arch
    legend_els = []
    for (arch, approach), (color, label) in GROUPS.items():
        mkr = arch_marker.get(arch, "o")
        legend_els.append(
            Line2D([0], [0], marker=mkr, color="w",
                   markerfacecolor=color, markeredgecolor=color,
                   markeredgewidth=0.8, markersize=6, label=label)
        )
    legend_els += [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="grey", markeredgecolor=C_26PRE,
               markeredgewidth=1.0, markersize=5, label="TRT FP16 edge"),
        Line2D([0], [0], marker="s", color="w",
               markerfacecolor="grey", markeredgecolor=C_12SCR,
               markeredgewidth=1.0, markersize=5, label="TRT INT8 edge"),
    ]
    ax.legend(handles=legend_els, fontsize=5.2, frameon=False,
              loc="upper left", handletextpad=0.4, labelspacing=0.3)

    out = os.path.join(OUT_DIR, "molle5.pdf")
    fig.savefig(out, format="pdf", dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_all()

    print("Generating molle3.pdf (Pareto scatter, 3-panel)...")
    make_molle3(df)

    print("Generating molle4.pdf (Power-efficiency scatter, AGX)...")
    make_molle4(df)

    print("Generating molle5.pdf (Quantisation cost-benefit, AGX)...")
    make_molle5(df)

    print("\nDone. Copy PDFs to the paper repository.")


if __name__ == "__main__":
    main()
