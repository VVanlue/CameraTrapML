"""
Runs the experimental sweep across quantization level x batch size x
resolution, for all 7 levels: fp32, int8, int4, int3, int2, 1.58bit, 1bit.

Only fp32 and int8 give you real, physically-measured memory/latency
differences. int4/int3/int2/1.58bit/1bit report REAL accuracy/F1 but
SIMULATED memory (theoretical, not measured) and near-FP32 latency,
because no mainstream framework executes CNNs below int8 — see
quantize.py's module docstring. The results CSV includes a
peak_memory_is_theoretical column so you can distinguish real from
theoretical rows when you plot and discuss this.

Start small for a progress demo:

    python src/run_sweep.py --arch mobilenet_v2 \
        --checkpoint checkpoints/mobilenet_v2_fp32_res224.pt \
        --quant_levels fp32 int8 int4 1bit \
        --batch_sizes 1 --resolutions 224 --trials 1

then widen once you know it runs end to end.
"""

import argparse
import copy
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from dataset import load_ena24, make_loader, sample_timing_subset
from models import load_finetuned
from quantize import (
    QUANT_LEVELS,
    load_finetuned_into_quantizable,
    quantize_int8_static,
    apply_quantization,
    theoretical_param_memory_bytes,
)
from benchmark import run_benchmark

REAL_MEMORY_LEVELS = {"fp32", "int8"}  # everything else uses a theoretical estimate


def build_model_for_level(arch, num_classes, checkpoint, quant_level, calibration_loader=None):
    if quant_level == "fp32":
        return load_finetuned(arch, num_classes, checkpoint)

    if quant_level == "int8":
        model = load_finetuned_into_quantizable(arch, num_classes, checkpoint)
        return quantize_int8_static(model, calibration_loader)

    # int4, int3, int2, 1.58bit, 1bit — all simulated
    model = load_finetuned(arch, num_classes, checkpoint)
    return apply_quantization(model, quant_level)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True, choices=["mobilenet_v2", "resnet18"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", default="data/ena24")
    parser.add_argument("--quant_levels", nargs="+", default=QUANT_LEVELS,
                         help=f"Any of: {QUANT_LEVELS}")
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--resolutions", nargs="+", type=int, default=[128, 224, 384])
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--out_dir", default="results")
    args = parser.parse_args()

    for level in args.quant_levels:
        if level not in QUANT_LEVELS:
            raise ValueError(f"Unknown quant level {level!r}. Choose from {QUANT_LEVELS}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for resolution in args.resolutions:
        full_ds, _, val_ds = load_ena24(args.data_dir, resolution)
        num_classes = len(full_ds.classes)
        val_loader = make_loader(val_ds, batch_size=32)
        calibration_loader = make_loader(val_ds, batch_size=8)
        timing_subset = sample_timing_subset(val_ds)

        for quant_level in args.quant_levels:
            base_model = build_model_for_level(
                args.arch, num_classes, args.checkpoint, quant_level, calibration_loader
            )
            theoretical_mem_mb = theoretical_param_memory_bytes(base_model, quant_level) / 1e6 \
                if quant_level not in REAL_MEMORY_LEVELS else None

            for batch_size in args.batch_sizes:
                timing_loader = make_loader(timing_subset, batch_size=batch_size)

                for trial in range(1, args.trials + 1):
                    model = copy.deepcopy(base_model)
                    result = run_benchmark(
                        model, val_loader, timing_loader,
                        args.arch, quant_level, batch_size, resolution,
                    )
                    row = asdict(result)
                    row["trial"] = trial
                    row["peak_memory_is_theoretical"] = quant_level not in REAL_MEMORY_LEVELS
                    if theoretical_mem_mb is not None:
                        row["peak_memory_mb"] = theoretical_mem_mb  # override measured RSS with the theoretical estimate
                    rows.append(row)
                    print(f"[{quant_level} | bs={batch_size} | res={resolution} | trial={trial}] "
                          f"mem={row['peak_memory_mb']:.1f}MB"
                          f"{' (theoretical)' if row['peak_memory_is_theoretical'] else ''} "
                          f"lat={row['latency_ms_per_image']:.2f}ms "
                          f"thr={row['throughput_images_per_sec']:.1f}img/s "
                          f"acc={row['accuracy']:.3f} f1={row['f1_macro']:.3f}")

    raw_df = pd.DataFrame(rows)
    raw_path = out_dir / f"{args.arch}_raw_results.csv"
    raw_df.to_csv(raw_path, index=False)

    group_cols = ["arch", "quant_level", "batch_size", "resolution"]
    metric_cols = ["peak_memory_mb", "latency_ms_per_image", "throughput_images_per_sec", "accuracy", "f1_macro"]
    agg_df = raw_df.groupby(group_cols)[metric_cols].agg(["mean", "std"]).reset_index()
    agg_path = out_dir / f"{args.arch}_aggregated_results.csv"
    agg_df.to_csv(agg_path, index=False)

    print(f"\nRaw results: {raw_path}")
    print(f"Aggregated (mean +/- std): {agg_path}")
    print("\nNote: rows for int4/int3/int2/1.58bit/1bit use a THEORETICAL memory "
          "estimate (peak_memory_is_theoretical=True), not measured RSS. "
          "Latency/throughput for those levels reflect float32 compute, not "
          "real low-bit execution — report this limitation explicitly.")


if __name__ == "__main__":
    main()