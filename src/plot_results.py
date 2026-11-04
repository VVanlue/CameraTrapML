"""
Generates the four figures planned in Section 7 of the proposal:
  1. Bar chart of peak memory by quantization level
  2. Line plots of latency and throughput vs batch size
  3. Accuracy comparison table across quantization levels and resolutions (CSV, printable)
  4. Scatter plot of throughput vs accuracy

Usage:
    python src/plot_results.py --raw_csv results/mobilenet_v2_raw_results.csv --out_dir figures
"""

import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---- styling (validated categorical palette, light surface) ----------------
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
LEVEL_ORDER = ["fp32", "fp16", "int8", "int4", "int3", "int2", "1.58bit", "1bit"]
LEVEL_COLOR = dict(zip(LEVEL_ORDER, SERIES))           # colour follows the level, never its rank
LEVEL_BITS = {"fp32": 32, "fp16": 16, "int8": 8, "int4": 4, "int3": 3, "int2": 2, "1.58bit": 1.58, "1bit": 1}
METHOD_STYLE = {"qat": "-", "ptq": "--", "-": "-"}
SEQ_BLUE = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlesize": 13,
    "axes.labelsize": 11, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "lines.linewidth": 2, "lines.markersize": 7, "font.size": 10.5,
})


def variant_order(v):
    lvl, _, meth = v.partition("-")
    return (LEVEL_ORDER.index(lvl) if lvl in LEVEL_ORDER else 99, meth != "qat")


def save(fig, out, name, note=None):
    if note:
        fig.text(0.01, 0.005, note, fontsize=8, color=INK2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03 if note else 0, 1, 1))
    fig.savefig(out / name, dpi=200)
    plt.close(fig)
    print("  wrote", out / name)


def color_of(variant):
    return LEVEL_COLOR.get(variant.split("-")[0], INK2)


# ---- figures ----------------------------------------------------------------

def fig_accuracy_vs_bits(s, out, res):
    d = s[s.resolution == res].drop_duplicates("variant")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    fp = d[d.level.isin(["fp32", "fp16"])]
    for ax, metric, title in [(axes[0], "accuracy", "Top-1 accuracy"), (axes[1], "f1_macro", "Macro F1")]:
        for method, label, col in [("qat", "QAT (quantization-aware training)", SERIES[0]),
                                   ("ptq", "PTQ (post-training only)", SERIES[1])]:
            m = d[d.method == method]
            m = pd.concat([fp, m]).copy()
            m["bits"] = m.level.map(LEVEL_BITS)
            m = m.sort_values("bits", ascending=False)
            if (m.method == method).sum() == 0:
                continue
            ax.errorbar(m.bits, m[metric], yerr=1.96 * m[f"{metric}_std"].fillna(0), marker="o",
                        ls=METHOD_STYLE[method], color=col, label=label, capsize=3)
        ax.set_xscale("log", base=2)
        ax.set_xticks([32, 16, 8, 4, 3, 2, 1.58, 1])
        ax.set_xticklabels(["fp32", "fp16", "int8", "int4", "int3", "int2", "1.58\nbit", "1bit"], fontsize=9)
        ax.minorticks_off()
        ax.invert_xaxis()
        ax.set_xlabel("Weight precision (log-spaced bits per weight)")
        ax.set_title(f"{title} vs bit-width")
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("Score on held-out test set")
    axes[0].legend(loc="lower left")
    save(fig, out, "01_accuracy_vs_bits.png", f"Error bars: 95% bootstrap CI over test images. Input {res}px.")


def fig_storage(s, out, res):
    d = s[(s.resolution == res)].drop_duplicates("variant").copy()
    d = d[d.method.isin(["qat", "-"])]
    d = d.sort_values("variant", key=lambda c: c.map(variant_order))
    fig, ax = plt.subplots(figsize=(8, 4.4))
    bars = ax.barh(d.level, d.storage_mb, color=[color_of(v) for v in d.variant], height=0.6,
                   edgecolor=SURFACE, linewidth=2)
    for b, v in zip(bars, d.storage_mb):
        ax.text(b.get_width(), b.get_y() + b.get_height() / 2, f"  {v:.2f} MB", va="center", color=INK, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Model file size on disk (MB)")
    ax.set_title("Storage: what quantization really buys")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, d.storage_mb.max() * 1.25)
    save(fig, out, "02_storage_by_level.png",
         "int4…1bit = bit-packed file (measured bytes). First conv + classifier kept at 8 bits; scales/biases fp32.")


def fig_memory_breakdown(s, out, res):
    d = s[(s.resolution == res) & s.method.isin(["qat", "-"])]
    bss = sorted(d.batch_size.unique())
    pick = [bss[0], bss[-1]] if len(bss) > 1 else bss
    fig, axes = plt.subplots(1, len(pick), figsize=(6 * len(pick), 4.4), sharey=True, squeeze=False)
    for ax, bs in zip(axes[0], pick):
        x = d[d.batch_size == bs].sort_values("variant", key=lambda c: c.map(variant_order))
        idx = np.arange(len(x))
        load = x.model_load_mb_mean.clip(lower=0)
        act = x.activation_mb_mean.clip(lower=0)
        ax.bar(idx, load, color=SERIES[0], width=0.65, label="Model load (weights + runtime)",
               edgecolor=SURFACE, linewidth=2)
        ax.bar(idx, act, bottom=load, color=SERIES[1], width=0.65, label="Activations / workspace",
               edgecolor=SURFACE, linewidth=2)
        ax.errorbar(idx, x.peak_inference_mb_mean, yerr=x.peak_inference_mb_std.fillna(0),
                    fmt="none", ecolor=INK, capsize=3, lw=1)
        ax.set_xticks(idx)
        ax.set_xticklabels(x.level, rotation=30)
        ax.set_title(f"Peak inference RAM, batch {bs}")
        ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel("MB above process baseline")
    axes[0][0].legend(loc="upper right")
    save(fig, out, "03_memory_breakdown.png",
         "Private data memory (excludes library code), mean of fresh-process repeats; error bar = std. Sampled every 1 ms.")


def _lines_vs_batch(s, out, res, metric, ylabel, title, fname):
    d = s[(s.resolution == res) & s.method.isin(["qat", "-"])]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for v in sorted(d.variant.unique(), key=variant_order):
        x = d[d.variant == v].sort_values("batch_size")
        ax.errorbar(x.batch_size, x[f"{metric}_mean"], yerr=x[f"{metric}_std"].fillna(0), marker="o",
                    color=color_of(v), label=v.split("-")[0], capsize=3)
    ax.set_xscale("log", base=2)
    bss = sorted(d.batch_size.unique())
    ax.set_xticks(bss)
    ax.set_xticklabels([str(b) for b in bss])
    ax.set_xlabel("Batch size (images per forward pass)")
    ax.set_ylabel(ylabel)
    vals = d[f"{metric}_mean"].dropna()
    if len(vals) and vals.max() / max(vals.min(), 1e-9) > 15:   # e.g. slow fp16 would flatten every other line
        ax.set_yscale("log")
        ax.set_ylabel(ylabel + " (log scale)")
    ax.set_title(title)
    ax.legend(ncol=2, fontsize=9, loc="best")
    save(fig, out, fname, "Mean ± std across fresh-process repeats; each repeat = adaptive trials (<2% SEM).")


def fig_tradeoff(s, out, res, xcol, xlabel, title, fname, logx=False):
    bs0 = s.batch_size.min()
    d = s[(s.resolution == res) & (s.batch_size == bs0)].drop_duplicates("variant")
    fig, ax = plt.subplots(figsize=(8, 5))
    for _, r in d.iterrows():
        xv = r[xcol] if xcol in r else np.nan
        mk = "o" if r.method in ("qat", "-") else "s"
        ax.scatter(xv, r.accuracy, s=80, color=color_of(r.variant), marker=mk,
                   edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate(r.variant, (xv, r.accuracy), textcoords="offset points", xytext=(6, 4), fontsize=8.5,
                    color=INK2)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    save(fig, out, fname, "Up and to the left is better. Circles = QAT/float, squares = PTQ.")


def fig_per_class_heatmap(acc_dir, arch, out, res):
    rows = {}
    for f in sorted(acc_dir.glob(f"{arch}_*_res{res}.json")):
        v = f.stem[len(arch) + 1: -len(f"_res{res}")]
        if v.endswith("-ptq"):
            continue
        rows[v] = json.loads(f.read_text()).get("per_class_f1", {})
    if not rows:
        return
    df = pd.DataFrame(rows).T
    df = df.loc[sorted(df.index, key=variant_order)]
    df.index = [v.split("-")[0] for v in df.index]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seqblue", SEQ_BLUE)
    fig, ax = plt.subplots(figsize=(max(10, 0.5 * df.shape[1]), 0.55 * df.shape[0] + 2.2))
    im = ax.imshow(df.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(df.shape[1]))
    ax.set_xticklabels([c.replace("_", " ") for c in df.columns], rotation=60, ha="right", fontsize=8.5)
    ax.set_yticks(range(df.shape[0]))
    ax.set_yticklabels(df.index)
    ax.grid(False)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            v = df.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                    color="#ffffff" if v > 0.6 else INK)
    fig.colorbar(im, ax=ax, fraction=0.025, label="F1")
    ax.set_title("Per-species F1 by quantization level (QAT)")
    save(fig, out, "08_per_class_f1_heatmap.png")


def fig_confusions(acc_dir, arch, out, res):
    wanted = ["fp32", "int8-qat", "int4-qat", "int2-qat", "1bit-qat"]
    files = [(v, acc_dir / f"{arch}_{v}_res{res}_preds.npz") for v in wanted]
    files = [(v, f) for v, f in files if f.exists()]
    if not files:
        return
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seqblue", SEQ_BLUE)
    fig, axes = plt.subplots(1, len(files), figsize=(4.6 * len(files), 4.8), squeeze=False)
    for ax, (v, f) in zip(axes[0], files):
        z = np.load(f, allow_pickle=True)
        k = len(z["classes"])
        cm = np.zeros((k, k))
        for t, p in zip(z["labels"], z["preds"]):
            cm[t, p] += 1
        cm = cm / cm.sum(1, keepdims=True).clip(min=1)
        ax.imshow(cm, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(v.split("-")[0])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("predicted")
        ax.grid(False)
    axes[0][0].set_ylabel("true species")
    save(fig, out, "09_confusion_matrices.png",
         "Row-normalised. A clean diagonal = correct; mass off-diagonal = species being confused.")


def fig_resolution(s, out):
    if s.resolution.nunique() < 2:
        return
    bs0 = s.batch_size.min()
    d = s[(s.batch_size == bs0) & s.method.isin(["qat", "-"])]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for v in sorted(d.variant.unique(), key=variant_order):
        x = d[d.variant == v].sort_values("resolution")
        if len(x) < 2:
            continue
        axes[0].plot(x.resolution, x.accuracy, marker="o", color=color_of(v), label=v.split("-")[0])
        axes[1].errorbar(x.resolution, x.latency_ms_per_image_mean, yerr=x.latency_ms_per_image_std.fillna(0),
                         marker="o", color=color_of(v), label=v.split("-")[0], capsize=3)
    axes[0].set_title("Accuracy vs input resolution")
    axes[0].set_ylabel("Test accuracy")
    axes[1].set_title(f"Latency vs input resolution (batch {bs0})")
    axes[1].set_ylabel("ms per image")
    for ax in axes:
        ax.set_xlabel("Input size (px, square)")
    axes[1].legend(fontsize=9)
    save(fig, out, "11_resolution_effect.png")


def fig_devices(s_all, out, res):
    if s_all.tag.nunique() < 2:
        return
    bs0 = s_all.batch_size.min()
    d = s_all[(s_all.resolution == res) & (s_all.batch_size == bs0) & s_all.method.isin(["qat", "-"])]
    tags = list(d.tag.unique())
    levels = [lv for lv in LEVEL_ORDER if lv in set(d.level)]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    w = 0.8 / len(tags)
    for i, t in enumerate(tags):
        x = d[d.tag == t].set_index("level").reindex(levels)
        ax.bar(np.arange(len(levels)) + i * w, x.latency_ms_per_image_mean, width=w,
               yerr=x.latency_ms_per_image_std.fillna(0), color=SERIES[i], label=t,
               edgecolor=SURFACE, linewidth=2, capsize=3)
    ax.set_xticks(np.arange(len(levels)) + w * (len(tags) - 1) / 2)
    ax.set_xticklabels(levels)
    ax.set_ylabel("ms per image")
    ax.set_title(f"Latency by device (batch {bs0}, {res}px)")
    ax.legend()
    ax.grid(axis="x", visible=False)
    save(fig, out, "12_device_comparison.png")


def write_tables(s, out, res):
    bs0 = s.batch_size.min()
    bsmax = s.batch_size.max()
    d = s[(s.resolution == res)]
    a = d[d.batch_size == bs0].set_index("variant")
    b = d[d.batch_size == bsmax].set_index("variant")
    t = pd.DataFrame({
        "accuracy": a.accuracy, "acc_std": a.accuracy_std, "macro_F1": a.f1_macro,
        "storage_MB": a.storage_mb,
        f"latency_ms_bs{bs0}": a.latency_ms_per_image_mean, f"lat_std_bs{bs0}": a.latency_ms_per_image_std,
        f"throughput_bs{bsmax}": b.throughput_images_per_sec_mean,
        f"thr_std_bs{bsmax}": b.throughput_images_per_sec_std,
        f"peak_RAM_MB_bs{bs0}": a.peak_inference_mb_mean, f"ram_std_bs{bs0}": a.peak_inference_mb_std,
    })
    t = t.loc[sorted(t.index, key=variant_order)]
    t.to_csv(out / "summary_table.csv", float_format="%.4f")
    fp = t.loc["fp32"] if "fp32" in t.index else None
    lines = ["| model | accuracy | macro F1 | size (MB) | ms/img (bs %d) | img/s (bs %d) | peak RAM MB |" % (bs0, bsmax),
             "|---|---|---|---|---|---|---|"]
    for v, r in t.iterrows():
        comp = f" ({fp.storage_MB / r.storage_MB:.1f}× smaller)" if fp is not None and v != "fp32" else ""
        lines.append(f"| {v} | {r.accuracy:.3f} ± {r.acc_std:.3f} | {r.macro_F1:.3f} | {r.storage_MB:.2f}{comp} | "
                     f"{r[f'latency_ms_bs{bs0}']:.1f} ± {r[f'lat_std_bs{bs0}']:.1f} | "
                     f"{r[f'throughput_bs{bsmax}']:.1f} ± {r[f'thr_std_bs{bsmax}']:.1f} | "
                     f"{r[f'peak_RAM_MB_bs{bs0}']:.1f} ± {r[f'ram_std_bs{bs0}']:.1f} |")
    (out / "summary_table.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="mobilenet_v2")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--out_dir", default="figures")
    p.add_argument("--tag", default=None, help="device tag for single-device figures (default: first found)")
    p.add_argument("--resolution", type=int, default=None, help="main resolution (default: 224 if present)")
    a = p.parse_args()

    rd, out = Path(a.results_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(rd / f"{a.arch}_*_summary.csv")))
    if not files:
        raise SystemExit(f"No {a.arch}_*_summary.csv in {rd}. Run run_sweep.py first.")
    s_all = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    s_all["method"] = s_all["method"].fillna("-")
    tag = a.tag or ("laptop" if "laptop" in set(s_all.tag) else s_all.tag.iloc[0])
    s = s_all[s_all.tag == tag]
    res = a.resolution or (224 if 224 in set(s.resolution) else int(s.resolution.iloc[0]))
    print(f"Plotting {a.arch}, device tag '{tag}', main resolution {res}px")

    fig_accuracy_vs_bits(s, out, res)
    fig_storage(s, out, res)
    fig_memory_breakdown(s, out, res)
    _lines_vs_batch(s, out, res, "latency_ms_per_image", "ms per image", "Latency vs batch size",
                    "04_latency_vs_batch.png")
    _lines_vs_batch(s, out, res, "throughput_images_per_sec", "images / second", "Throughput vs batch size",
                    "05_throughput_vs_batch.png")
    s_lat = s.rename(columns={"latency_ms_per_image_mean": "lat"})
    fig_tradeoff(s_lat, out, res, "lat", f"ms per image (batch {s.batch_size.min()})",
                 "Accuracy vs latency", "06_accuracy_vs_latency.png")
    fig_tradeoff(s, out, res, "storage_mb", "Model size on disk (MB, log scale)",
                 "Accuracy vs model size", "07_accuracy_vs_storage.png", logx=True)
    fig_per_class_heatmap(rd / "accuracy", a.arch, out, res)
    fig_confusions(rd / "accuracy", a.arch, out, res)
    _lines_vs_batch(s, out, res, "peak_inference_mb", "MB above baseline", "Peak inference RAM vs batch size",
                    "10_memory_vs_batch.png")
    fig_resolution(s, out)
    fig_devices(s_all, out, res)
    write_tables(s, out, res)


if __name__ == "__main__":
    main()
