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
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

from benchmark import run_in_fresh_process
from dataset import load_ena24, make_loader, preload_tensors
from models import load_meta
from quantize import QUANT_LEVELS, STORAGE_BITS, export_torchscript, load_model_for_level, set_backend


def variant_name(level, method):
    return level if level in ("fp32", "fp16") else f"{level}-{method}"


def model_path_for(models_dir, arch, level, method, res):
    if level in ("fp32", "fp16"):
        return models_dir / f"{arch}_fp32_res{res}.pt"
    return models_dir / f"{arch}_{level}_{method}_res{res}.pt"


def deployable_path(models_dir, arch, level, method, res, nc, backend, eager=False):
    """TorchScript file used for timing/memory (same loader for every level).
    fp32/fp16 TorchScript files are created on first use."""
    if eager:
        return model_path_for(models_dir, arch, level, method, res)
    if level in ("fp32", "fp16"):
        ts = models_dir / f"{arch}_{level}_res{res}.ts"
        if not ts.exists():
            m = load_model_for_level(arch, nc, level, str(models_dir / f"{arch}_fp32_res{res}.pt"), backend)
            export_torchscript(m, ts, res, half=(level == "fp16"))
        return ts
    ts = models_dir / f"{arch}_{level}_{method}_res{res}.ts"
    return ts if ts.exists() else model_path_for(models_dir, arch, level, method, res)


def storage_bytes(models_dir, arch, level, method, res, cache_dir):
    if level == "fp32":
        return (models_dir / f"{arch}_fp32_res{res}.pt").stat().st_size
    if level == "fp16":
        sd = torch.load(models_dir / f"{arch}_fp32_res{res}.pt", map_location="cpu", weights_only=True)
        path = cache_dir / f"{arch}_fp16_res{res}.pt"
        torch.save({k: v.half() if v.is_floating_point() else v for k, v in sd.items()}, path)
        return path.stat().st_size
    qpk = models_dir / f"{arch}_{level}_{method}_res{res}.qpk"
    return qpk.stat().st_size if qpk.exists() else np.nan


@torch.no_grad()
def accuracy_report(model, level, test_ds, n_boot=1000, seed=0):
    loader = make_loader(test_ds, 64, num_workers=2)
    preds, labels = [], []
    for x, y in loader:
        if level == "fp16":
            x = x.half()
        preds += model(x).argmax(1).tolist()
        labels += y.tolist()
    preds, labels = np.array(preds), np.array(labels)
    rng = np.random.default_rng(seed)
    boots_acc, boots_f1 = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(labels), len(labels))
        boots_acc.append((preds[i] == labels[i]).mean())
        boots_f1.append(f1_score(labels[i], preds[i], average="macro", zero_division=0))
    per_class = f1_score(labels, preds, average=None, labels=np.arange(len(test_ds.dataset.classes)),
                         zero_division=0)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "accuracy_std": float(np.std(boots_acc)), "f1_macro_std": float(np.std(boots_f1)),
        "accuracy_ci95": [float(np.percentile(boots_acc, 2.5)), float(np.percentile(boots_acc, 97.5))],
        "f1_macro_ci95": [float(np.percentile(boots_f1, 2.5)), float(np.percentile(boots_f1, 97.5))],
        "per_class_f1": dict(zip(test_ds.dataset.classes, map(float, per_class))),
        "n_test": int(len(labels)),
    }, preds, labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", required=True, choices=["mobilenet_v2", "resnet18"])
    p.add_argument("--models_dir", default="models")
    p.add_argument("--levels", nargs="+", default=QUANT_LEVELS)
    p.add_argument("--methods", nargs="+", default=["qat"], help="qat and/or ptq (for int8..1bit)")
    p.add_argument("--batch_sizes", nargs="+", type=int, default=[1, 4, 8, 16])
    p.add_argument("--resolutions", nargs="+", type=int, default=[224])
    p.add_argument("--repeats", type=int, default=3, help="fresh processes per config")
    p.add_argument("--threads", type=int, default=os.cpu_count())
    p.add_argument("--min_trials", type=int, default=5)
    p.add_argument("--max_trials", type=int, default=30)
    p.add_argument("--target_rel_sem", type=float, default=0.02)
    p.add_argument("--n_timing", type=int, default=96)
    p.add_argument("--backend", default=None)
    p.add_argument("--eager", action="store_true", help="benchmark eager models instead of TorchScript")
    p.add_argument("--skip_accuracy", action="store_true",
                   help="reuse accuracy JSONs already in results/accuracy (e.g. on a slow Pi)")
    p.add_argument("--tag", default="laptop", help="device label, e.g. laptop, pi4, docker_1cpu")
    p.add_argument("--data_dir", default=None)
    p.add_argument("--out_dir", default="results")
    args = p.parse_args()

    backend = set_backend(args.backend)
    models_dir, out_dir = Path(args.models_dir), Path(args.out_dir)
    cache_dir, acc_dir = out_dir / "cache", out_dir / "accuracy"
    for d in (cache_dir, acc_dir):
        d.mkdir(parents=True, exist_ok=True)

    rows = []
    t_start = time.time()
    for res in args.resolutions:
        meta_path = models_dir / f"{args.arch}_res{res}_meta.json"
        if not meta_path.exists():
            print(f"[skip res {res}] no {meta_path} — train the FP32 baseline at this resolution first")
            continue
        meta = load_meta(meta_path)
        nc = len(meta["classes"])
        test_ds = None
        timing_path = cache_dir / f"timing_res{res}_n{args.n_timing}.pt"
        if not args.skip_accuracy or not timing_path.exists():
            _, _, _, test_ds = load_ena24(args.data_dir or meta["data_dir"], res)
        if not timing_path.exists():
            x, _ = preload_tensors(test_ds, args.n_timing)
            torch.save(x, timing_path)

        for level in args.levels:
            methods = ["-"] if level in ("fp32", "fp16") else args.methods
            for method in methods:
                variant = variant_name(level, method)
                mpath = model_path_for(models_dir, args.arch, level, method, res)
                if not mpath.exists():
                    print(f"[skip] {variant} res{res}: {mpath} not found")
                    continue

                acc_file = acc_dir / f"{args.arch}_{variant}_res{res}.json"
                if args.skip_accuracy and acc_file.exists():
                    acc = json.loads(acc_file.read_text())
                elif args.skip_accuracy:
                    acc = {"accuracy": np.nan, "f1_macro": np.nan, "accuracy_std": np.nan, "f1_macro_std": np.nan}
                else:
                    model = load_model_for_level(args.arch, nc, level, str(mpath), backend)
                    acc, preds, labels = accuracy_report(model, level, test_ds)
                    del model
                    acc_file.write_text(json.dumps(acc, indent=2))
                    np.savez(acc_dir / f"{args.arch}_{variant}_res{res}_preds.npz",
                             preds=preds, labels=labels, classes=np.array(meta["classes"]))
                print(f"\n{variant} @ {res}px: test acc {acc['accuracy']:.4f} (±{acc['accuracy_std']:.4f}) "
                      f"F1 {acc['f1_macro']:.4f}")
                size_b = storage_bytes(models_dir, args.arch, level, method, res, cache_dir)
                run_path = deployable_path(models_dir, args.arch, level, method, res, nc, backend, args.eager)

                for bs in args.batch_sizes:
                    for rep in range(1, args.repeats + 1):
                        r = run_in_fresh_process(dict(
                            arch=args.arch, num_classes=nc, level=level, model_path=str(run_path),
                            timing_tensor=str(timing_path), batch_size=bs, threads=args.threads,
                            min_trials=args.min_trials, max_trials=args.max_trials,
                            target_rel_sem=args.target_rel_sem, backend=backend))
                        if not r.get("ok"):
                            print(f"  [error] bs={bs} rep={rep}: {r.get('error')}\n{r.get('traceback', '')}")
                            continue
                        r.pop("trial_throughputs", None)
                        row = dict(arch=args.arch, tag=args.tag, variant=variant, level=level,
                                   method=method, batch_size=bs, resolution=res, repeat=rep,
                                   storage_mb=size_b / 1e6, storage_bits=STORAGE_BITS[level],
                                   accuracy=acc["accuracy"], f1_macro=acc["f1_macro"],
                                   accuracy_std=acc["accuracy_std"], f1_macro_std=acc["f1_macro_std"],
                                   **{k: v for k, v in r.items() if k != "ok"})
                        rows.append(row)
                        print(f"  bs={bs:2d} rep={rep} | {r['latency_ms_per_image']:7.2f} ms/img | "
                              f"{r['throughput_images_per_sec']:7.1f} img/s | batch p95 "
                              f"{r['batch_latency_ms_p95']:7.1f} ms | load {r['model_load_mb']:6.1f} MB | "
                              f"peak {r['peak_inference_mb']:6.1f} MB | trials {r['n_trials']}")

                # write as we go so a crash doesn't lose everything
                pd.DataFrame(rows).to_csv(out_dir / f"{args.arch}_{args.tag}_raw.csv", index=False)

    if not rows:
        print("No results — check that models exist.")
        return
    raw = pd.DataFrame(rows)
    raw_path = out_dir / f"{args.arch}_{args.tag}_raw.csv"
    raw.to_csv(raw_path, index=False)

    keys = ["arch", "tag", "variant", "level", "method", "batch_size", "resolution"]
    metrics = ["latency_ms_per_image", "throughput_images_per_sec", "batch_latency_ms_p50",
               "batch_latency_ms_p95", "model_load_mb", "peak_inference_mb", "activation_mb",
               "peak_rss_total_mb", "peak_rss_incl_code_mb", "n_trials"]
    g = raw.groupby(keys)
    summary = g[metrics].agg(["mean", "std", "count"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    for m in metrics:  # 95% CI half-width (t approx with n repeats)
        n = summary[f"{m}_count"].clip(lower=1)
        summary[f"{m}_ci95"] = 1.96 * summary[f"{m}_std"].fillna(0) / np.sqrt(n)
    first = g[["storage_mb", "storage_bits", "accuracy", "f1_macro", "accuracy_std", "f1_macro_std"]].first()
    summary = summary.join(first).reset_index()
    sum_path = out_dir / f"{args.arch}_{args.tag}_summary.csv"
    summary.to_csv(sum_path, index=False)

    print(f"\nDone in {(time.time() - t_start) / 60:.1f} min")
    print(f"Raw:     {raw_path}\nSummary: {sum_path}")
    show = summary[summary.batch_size == summary.batch_size.min()][
        ["variant", "resolution", "accuracy", "f1_macro", "storage_mb", "latency_ms_per_image_mean",
         "latency_ms_per_image_std", "peak_inference_mb_mean", "peak_inference_mb_std"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
