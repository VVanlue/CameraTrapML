#!/usr/bin/env python
"""
Plot the results: reads the sweep output in data/results/ and writes every
figure and summary table to data/figures/.

    python result.py                         # mobilenet_v2, defaults
    python result.py --arch resnet18
    python result.py --results_dir data/results --out_dir data/figures --resolution 224

All options are those of src/plot_results.py (`python result.py --help`).
The preliminary (P1) results in data/results/preliminary/ were produced by an
earlier version of the pipeline and are kept as-is for reference.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    cmd = [sys.executable, str(ROOT / "src" / "plot_results.py"), *sys.argv[1:]]
    sys.exit(subprocess.run(cmd, cwd=ROOT).returncode)
