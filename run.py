#!/usr/bin/env python
"""
Single entry point for the project. Every stage is a thin wrapper around a
script in src/, so `python run.py <stage> --help` shows that script's options.

    python run.py demo                      # classify examples/ with the shipped FP32 model (no dataset needed)
    python run.py smoke                     # 5-minute install check on a tiny synthetic dataset
    python run.py all --images_dir data/raw/images --annotations data/raw/ena24.json
                                            # full pipeline: prepare -> train -> quantize -> sweep -> plot

Individual stages (extra arguments are passed straight through):

    prepare    src/prepare_ena24.py      raw ENA24 -> data/ena24_crops, data/ena24, data/ena24_boxes.json
    train      src/train_baseline.py     fine-tune the FP32 baseline -> models/
    quantize   src/train_quantized.py    PTQ / QAT for int8 ... 1bit -> models/
    sweep      src/run_sweep.py          accuracy + latency/throughput/memory -> data/results/
    plot       src/plot_results.py       figures + tables -> data/figures/   (same as `python result.py`)
    predict    src/predict.py            headless detector + classifier on a folder of photos
    app        src/app.py                local web demo (needs gradio)
    detector   src/train_detector.py     optional: train our own ENA24 animal detector
    export     src/export_for_backend.py re-convert quantized models for another CPU backend (e.g. ARM)
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

STAGES = {
    "prepare": SRC / "prepare_ena24.py",
    "train": SRC / "train_baseline.py",
    "quantize": SRC / "train_quantized.py",
    "sweep": SRC / "run_sweep.py",
    "plot": SRC / "plot_results.py",
    "predict": SRC / "predict.py",
    "app": SRC / "app.py",
    "detector": SRC / "train_detector.py",
    "export": SRC / "export_for_backend.py",
    "smoke": ROOT / "scripts" / "smoke_test.py",
}


def call(script: Path, *args) -> None:
    cmd = [sys.executable, str(script), *map(str, args)]
    print("\n$ " + " ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def run_all(argv) -> None:
    p = argparse.ArgumentParser(prog="run.py all", description="Run the whole pipeline end to end.")
    p.add_argument("--images_dir", help="raw ENA24 images (omit together with --annotations to skip prepare)")
    p.add_argument("--annotations", help="raw ena24.json")
    p.add_argument("--arch", default="mobilenet_v2", choices=["mobilenet_v2", "resnet18"])
    p.add_argument("--resolution", type=int, default=224)
    p.add_argument("--epochs", type=int, default=8, help="FP32 fine-tuning epochs")
    p.add_argument("--methods", nargs="+", default=["qat", "ptq"], choices=["qat", "ptq"])
    p.add_argument("--batch_sizes", nargs="+", type=int, default=[1, 4, 8, 16])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    a = p.parse_args(argv)
    dev = ["--device", a.device] if a.device else []
    ckpt = f"models/{a.arch}_fp32_res{a.resolution}.pt"

    if a.images_dir and a.annotations:
        call(STAGES["prepare"], "--images_dir", a.images_dir, "--annotations", a.annotations)
    elif a.images_dir or a.annotations:
        p.error("--images_dir and --annotations must be given together")
    call(STAGES["train"], "--arch", a.arch, "--resolution", a.resolution, "--epochs", a.epochs, *dev)
    for method in a.methods:
        call(STAGES["quantize"], "--arch", a.arch, "--resolution", a.resolution, "--method", method,
             "--fp32_checkpoint", ckpt, *dev)
    call(STAGES["sweep"], "--arch", a.arch, "--resolutions", a.resolution, "--methods", *a.methods,
         "--batch_sizes", *a.batch_sizes, "--repeats", a.repeats)
    call(STAGES["plot"], "--arch", a.arch, "--resolution", a.resolution)


def run_demo(argv) -> None:
    p = argparse.ArgumentParser(prog="run.py demo",
                                description="Classify the photos in examples/ with every model in models/.")
    p.add_argument("--images", default="examples")
    p.add_argument("--model", default=None, help="one model name from `python run.py predict --images . --list`")
    p.add_argument("--detector", default="auto", choices=["auto", "megadetector", "ena24", "none"])
    a = p.parse_args(argv)
    model = ["--model", a.model] if a.model else []
    call(STAGES["predict"], "--images", a.images, "--detector", a.detector, *model)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return
    stage, rest = sys.argv[1], sys.argv[2:]
    try:
        if stage == "all":
            run_all(rest)
        elif stage == "demo":
            run_demo(rest)
        elif stage in STAGES:
            call(STAGES[stage], *rest)
        else:
            sys.exit(f"unknown stage {stage!r}. Choose from: all, demo, {', '.join(STAGES)}")
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
