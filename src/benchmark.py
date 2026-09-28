"""
Benchmarking harness. For a given model + dataloader, measures:
  - peak memory (MB, process RSS delta during inference)
  - per-image latency (ms)
  - throughput (images/sec)
  - accuracy and macro F1

Run standalone for a single config, or import run_benchmark() from
run_sweep.py to sweep quantization level x batch size x resolution.
"""

import argparse
import gc
import time
from dataclasses import dataclass, asdict

import psutil
import torch
from sklearn.metrics import accuracy_score, f1_score


@dataclass
class BenchmarkResult:
    arch: str
    quant_level: str
    batch_size: int
    resolution: int
    peak_memory_mb: float
    latency_ms_per_image: float
    throughput_images_per_sec: float
    accuracy: float
    f1_macro: float


def _peak_rss_mb() -> float:
    return psutil.Process().memory_info().rss / 1e6


def measure_memory_and_latency(model, timing_loader, batch_size: int, n_warmup: int = 2) -> dict:
    """Runs the model over the small fixed timing subset (see
    dataset.sample_timing_subset) and returns memory + latency stats.
    Uses a DataLoader already built with the target batch_size."""
    model.eval()
    gc.collect()

    # Warm-up (excluded from timing — first-call overhead, e.g. lazy
    # kernel init, would otherwise skew results)
    with torch.no_grad():
        for i, (images, _) in enumerate(timing_loader):
            if i >= n_warmup:
                break
            model(images)

    gc.collect()
    mem_before = _peak_rss_mb()

    n_images = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for images, _ in timing_loader:
            model(images)
            n_images += images.size(0)
    elapsed = time.perf_counter() - t0

    mem_after = _peak_rss_mb()

    return {
        "peak_memory_mb": max(mem_after - mem_before, 0.0),
        "latency_ms_per_image": (elapsed / n_images) * 1000 if n_images else float("nan"),
        "throughput_images_per_sec": n_images / elapsed if elapsed > 0 else float("nan"),
    }


def measure_accuracy(model, val_loader) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1_macro": f1_score(all_labels, all_preds, average="macro"),
    }


def run_benchmark(
    model,
    val_loader,
    timing_loader,
    arch: str,
    quant_level: str,
    batch_size: int,
    resolution: int,
) -> BenchmarkResult:
    """Full benchmark for one (arch, quant_level, batch_size, resolution) config."""
    mem_lat = measure_memory_and_latency(model, timing_loader, batch_size)
    acc = measure_accuracy(model, val_loader)

    return BenchmarkResult(
        arch=arch,
        quant_level=quant_level,
        batch_size=batch_size,
        resolution=resolution,
        peak_memory_mb=mem_lat["peak_memory_mb"],
        latency_ms_per_image=mem_lat["latency_ms_per_image"],
        throughput_images_per_sec=mem_lat["throughput_images_per_sec"],
        accuracy=acc["accuracy"],
        f1_macro=acc["f1_macro"],
    )


if __name__ == "__main__":
    # Standalone single-config run, useful for your Week 2 "harness produces
    # one validated baseline result" milestone.
    import sys
    sys.path.insert(0, ".")
    from dataset import load_ena24, make_loader, sample_timing_subset
    from models import load_finetuned

    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True, choices=["mobilenet_v2", "resnet18"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", default="data/ena24")
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    full_ds, _, val_ds = load_ena24(args.data_dir, args.resolution)
    num_classes = len(full_ds.classes)
    val_loader = make_loader(val_ds, batch_size=32)
    timing_subset = sample_timing_subset(val_ds)
    timing_loader = make_loader(timing_subset, batch_size=args.batch_size)

    model = load_finetuned(args.arch, num_classes, args.checkpoint)
    result = run_benchmark(model, val_loader, timing_loader, args.arch, "fp32", args.batch_size, args.resolution)
    print(asdict(result))
