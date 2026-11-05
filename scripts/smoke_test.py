"""
5-minute end-to-end check that your install works, BEFORE spending hours on
real training. Builds a tiny fake dataset (coloured shapes), then runs every
stage of the project on it: baseline training, PTQ + QAT for every level,
the sweep (fresh-process memory/latency), plots, and the detection pipeline.

    python scripts/smoke_test.py

Everything goes into smoke_test_tmp/ and can be deleted afterwards. Accuracy
numbers here are meaningless; you are only checking nothing crashes.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
TMP = ROOT / "smoke_test_tmp"
PY = sys.executable


def make_fake_dataset(root: Path, classes=("Bobcat", "Red_Fox", "Wild_Turkey"), per_class=60):
    rng = np.random.default_rng(0)
    for ci, c in enumerate(classes):
        d = root / c
        d.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            img = Image.fromarray(rng.integers(60, 120, (96, 96, 3), dtype=np.uint8))
            dr = ImageDraw.Draw(img)
            col = [(220, 60, 60), (60, 200, 60), (60, 60, 220)][ci]
            x, y = rng.integers(10, 50, 2)
            if ci == 0:
                dr.ellipse([x, y, x + 35, y + 35], fill=col)
            elif ci == 1:
                dr.rectangle([x, y, x + 35, y + 35], fill=col)
            else:
                dr.polygon([(x, y + 35), (x + 18, y), (x + 35, y + 35)], fill=col)
            img.save(d / f"{ci * 1000 + i}_{i}.jpg")


def run(*args):
    cmd = [PY, *map(str, args)]
    print("\n$ " + " ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, cwd=TMP, check=True)


def main():
    if TMP.exists():
        shutil.rmtree(TMP)
    data = TMP / "data" / "fake"
    make_fake_dataset(data)
    src = ROOT / "src"
    res = 96
    run(src / "train_baseline.py", "--arch", "mobilenet_v2", "--data_dir", data, "--resolution", res,
        "--epochs", 2, "--batch_size", 16, "--workers", 0, "--no_pretrained")
    levels = ["int8", "int4", "int2", "1.58bit", "1bit"]
    run(src / "qat_train.py", "--arch", "mobilenet_v2", "--resolution", res, "--method", "ptq",
        "--fp32_checkpoint", f"models/mobilenet_v2_fp32_res{res}.pt", "--levels", *levels, "--workers", 0,
        "--calib_batches", 4)
    run(src / "qat_train.py", "--arch", "mobilenet_v2", "--resolution", res, "--method", "qat", "--epochs", 1,
        "--fp32_checkpoint", f"models/mobilenet_v2_fp32_res{res}.pt", "--levels", *levels, "--workers", 0,
        "--calib_batches", 4, "--batch_size", 16)
    run(src / "run_sweep.py", "--arch", "mobilenet_v2", "--resolutions", res, "--batch_sizes", 1, 8,
        "--repeats", 2, "--methods", "qat", "ptq", "--min_trials", 3, "--max_trials", 5, "--n_timing", 32,
        "--tag", "laptop")
    run(src / "plot_results.py", "--arch", "mobilenet_v2", "--resolution", res)
    sample = next((data / "Bobcat").glob("*.jpg"))
    run(src / "predict.py", "--images", sample, "--detector", "none", "--model", f"mobilenet_v2 int4 QAT @{res}")
    print("\nSMOKE TEST PASSED — see smoke_test_tmp/figures and smoke_test_tmp/predictions")


if __name__ == "__main__":
    main()
