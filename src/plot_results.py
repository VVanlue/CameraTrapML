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
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_memory_bar(df: pd.DataFrame, out_dir: Path):
    summary = df.groupby("quant_level")["peak_memory_mb"].mean().reindex(["fp32", "int8", "int4"]).dropna()
    fig, ax = plt.subplots()
    ax.bar(summary.index, summary.values)
    ax.set_ylabel("Peak memory (MB)")
    ax.set_title("Peak memory by quantization level")
    fig.tight_layout()
    fig.savefig(out_dir / "memory_by_quant_level.png", dpi=150)
    plt.close(fig)


def plot_latency_throughput_vs_batch(df: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for quant_level, group in df.groupby("quant_level"):
        summary = group.groupby("batch_size")[["latency_ms_per_image", "throughput_images_per_sec"]].mean()
        axes[0].plot(summary.index, summary["latency_ms_per_image"], marker="o", label=quant_level)
        axes[1].plot(summary.index, summary["throughput_images_per_sec"], marker="o", label=quant_level)

    axes[0].set_xlabel("Batch size")
    axes[0].set_ylabel("Latency (ms/image)")
    axes[0].set_title("Latency vs batch size")
    axes[0].legend()

    axes[1].set_xlabel("Batch size")
    axes[1].set_ylabel("Throughput (images/sec)")
    axes[1].set_title("Throughput vs batch size")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "latency_throughput_vs_batch.png", dpi=150)
    plt.close(fig)


def make_accuracy_table(df: pd.DataFrame, out_dir: Path):
    table = df.groupby(["quant_level", "resolution"])[["accuracy", "f1_macro"]].mean().reset_index()
    table.to_csv(out_dir / "accuracy_table.csv", index=False)
    return table


def plot_throughput_vs_accuracy(df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots()
    for quant_level, group in df.groupby("quant_level"):
        summary = group.groupby("batch_size").agg(
            throughput=("throughput_images_per_sec", "mean"),
            accuracy=("accuracy", "mean"),
        )
        ax.scatter(summary["throughput"], summary["accuracy"], label=quant_level, s=60)

    ax.set_xlabel("Throughput (images/sec)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Throughput vs accuracy trade-off")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "throughput_vs_accuracy.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_csv", required=True, help="Path to a *_raw_results.csv from run_sweep.py")
    parser.add_argument("--out_dir", default="figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.raw_csv)

    plot_memory_bar(df, out_dir)
    plot_latency_throughput_vs_batch(df, out_dir)
    table = make_accuracy_table(df, out_dir)
    plot_throughput_vs_accuracy(df, out_dir)

    print(f"Figures saved to {out_dir}/")
    print("\nAccuracy table:")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
