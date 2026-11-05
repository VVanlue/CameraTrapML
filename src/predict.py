"""
Headless version of the app, for batches of photos and for constrained
hardware (Raspberry Pi, a 1-CPU Docker container, ...). No browser needed.

    python src/predict.py --images path/to/folder --model "mobilenet_v2 int8 QAT @224" \
        --out_dir predictions --threads 1

Writes predictions/<name>_boxed.jpg (green boxes) and predictions/predictions.csv,
and prints end-to-end timing (detector + classifier) per image and a summary,
which is the number a field deployment actually cares about.
"""

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import psutil
import torch
from PIL import Image

from pipeline import CameraTrapPipeline, Classifier, find_classifier_models, make_detector


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True, help="an image file or a folder of images")
    p.add_argument("--model", default=None, help="name as listed by --list (default: first)")
    p.add_argument("--list", action="store_true", help="list available classifier models and exit")
    p.add_argument("--models_dir", default="models")
    p.add_argument("--detector", default="auto", choices=["auto", "megadetector", "ena24", "none"])
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--threads", type=int, default=0, help="0 = all cores; 1 emulates a single-core board")
    p.add_argument("--backend", default=None)
    p.add_argument("--out_dir", default="predictions")
    a = p.parse_args()

    if a.threads:
        torch.set_num_threads(a.threads)
    models = find_classifier_models(a.models_dir)
    if a.list or not models:
        print("\n".join(models) or "no models found")
        return
    name = a.model or next(iter(models))
    arch, level, path, res = models[name]
    meta = json.loads((Path(a.models_dir) / f"{arch}_res{res}_meta.json").read_text())

    rss0 = psutil.Process().memory_info().rss / 2**20
    pipe = CameraTrapPipeline(make_detector(a.detector, a.models_dir),
                              Classifier(arch, level, path, res, meta["classes"], a.backend))
    rss_loaded = psutil.Process().memory_info().rss / 2**20

    src = Path(a.images)
    files = [src] if src.is_file() else sorted(f for f in src.rglob("*") if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    totals, peak = [], rss_loaded
    with open(out / "predictions.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "box_x1", "box_y1", "box_x2", "box_y2", "det_conf", "species", "species_conf",
                    "detector_ms", "classifier_ms", "total_ms"])
        for i, f in enumerate(files):
            img = Image.open(f)
            t0 = time.perf_counter()
            annotated, dets, t = pipe.run(img, a.conf)
            total = (time.perf_counter() - t0) * 1000
            if i > 0:                      # first image includes one-time warm-up
                totals.append(total)
            peak = max(peak, psutil.Process().memory_info().rss / 2**20)
            annotated.save(out / f"{f.stem}_boxed.jpg", quality=90)
            for d in dets or [None]:
                w.writerow([f.name] + ([*map(lambda v: round(v, 1), d.box), round(d.det_conf, 3), d.species,
                                        round(d.species_conf, 3)] if d else [""] * 7) +
                           [round(t["detector_ms"], 1), round(t["classifier_ms"], 1), round(total, 1)])
            print(f"{f.name}: " + (", ".join(f"{d.species} {d.species_conf:.0%}" for d in dets) or "nothing")
                  + f"  [{total:.0f} ms]")

    print(f"\nModel: {name} | detector: {pipe.detector.name} | threads: {torch.get_num_threads()}")
    if totals:
        print(f"End-to-end per image: mean {statistics.mean(totals):.0f} ms, "
              f"median {statistics.median(totals):.0f} ms over {len(totals)} images (first excluded)")
    print(f"RAM: pipeline load {rss_loaded - rss0:.0f} MB, peak process {peak:.0f} MB")
    print(f"Outputs in {out}/")


if __name__ == "__main__":
    main()
