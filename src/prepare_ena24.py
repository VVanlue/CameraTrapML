"""
Convert the raw LILA BC ENA24-detection release into the folder layouts the
rest of the project uses.

Raw release (https://lila.science/datasets/ena24detection):
    ena24/images/<id>.jpg          (unzipped image archive)
    ena24.json                     (COCO Camera Traps annotations, with boxes)

Outputs (pick with --mode):
    crops   data/ena24_crops/<Class_Name>/<imageid>_<annid>.jpg
            One image per bounding box, cropped around the animal with a small
            margin. THIS is what the classifier should be trained on, because
            the app runs "detector finds the animal -> classifier names the crop".
    full    data/ena24/<Class_Name>/<imageid>.jpg
            Whole frames, labelled with the (first) box's class. This is the
            layout your original runs used.
    both    both of the above.

It also writes data/ena24_boxes.json: a cleaned list of {file, width, height,
boxes:[[x1,y1,x2,y2,class_name], ...]} used by train_detector.py and by
evaluate_pipeline.py.

Usage:
    python src/prepare_ena24.py --images_dir data/raw/images \
        --annotations data/raw/ena24.json --mode both
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def clean_name(name: str) -> str:
    """'Eastern Gray Squirrel' -> 'Eastern_Gray_Squirrel' (matches your class list)."""
    name = re.sub(r"[^A-Za-z0-9]+", "_", name.strip())
    return name.strip("_")


def to_xyxy(bbox, w, h):
    """COCO bbox [x, y, w, h]; ENA24 uses absolute pixels, but handle normalized too."""
    x, y, bw, bh = bbox
    if max(x, y, bw, bh) <= 1.0:  # normalized coordinates
        x, y, bw, bh = x * w, y * h, bw * w, bh * h
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(float(w), x + bw), min(float(h), y + bh)
    return x1, y1, x2, y2


def expand_box(x1, y1, x2, y2, w, h, margin):
    bw, bh = x2 - x1, y2 - y1
    x1 -= bw * margin
    x2 += bw * margin
    y1 -= bh * margin
    y2 += bh * margin
    return max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images_dir", required=True, help="folder with the raw ENA24 .jpg files")
    p.add_argument("--annotations", required=True, help="ena24.json (COCO Camera Traps)")
    p.add_argument("--mode", choices=["crops", "full", "both"], default="both")
    p.add_argument("--crops_out", default="data/ena24_crops")
    p.add_argument("--full_out", default="data/ena24")
    p.add_argument("--boxes_out", default="data/ena24_boxes.json")
    p.add_argument("--margin", type=float, default=0.10, help="extra context around each crop (fraction of box size)")
    p.add_argument("--min_box_px", type=int, default=24, help="skip boxes smaller than this (too tiny to classify)")
    p.add_argument("--drop_classes", nargs="*", default=["Human"], help="classes to exclude")
    args = p.parse_args()

    images_dir = Path(args.images_dir)
    coco = json.loads(Path(args.annotations).read_text())
    cats = {c["id"]: clean_name(c["name"]) for c in coco["categories"]}
    images = {im["id"]: im for im in coco["images"]}
    anns_by_image = defaultdict(list)
    for a in coco["annotations"]:
        if "bbox" in a and a.get("category_id") in cats:
            anns_by_image[a["image_id"]].append(a)

    drop = {clean_name(c) for c in args.drop_classes}
    crop_counts, full_counts = Counter(), Counter()
    boxes_index = []
    missing = 0

    for image_id, anns in tqdm(anns_by_image.items(), desc="images"):
        meta = images.get(image_id)
        if meta is None:
            continue
        src = images_dir / meta["file_name"]
        if not src.exists():
            src = images_dir / Path(meta["file_name"]).name
        if not src.exists():
            missing += 1
            continue

        img = Image.open(src).convert("RGB")
        w, h = img.size
        record = {"file": str(src), "width": w, "height": h, "boxes": []}

        for a in anns:
            cls = cats[a["category_id"]]
            if cls in drop:
                continue
            x1, y1, x2, y2 = to_xyxy(a["bbox"], w, h)
            if (x2 - x1) < args.min_box_px or (y2 - y1) < args.min_box_px:
                continue
            record["boxes"].append([x1, y1, x2, y2, cls])

            if args.mode in ("crops", "both"):
                cx1, cy1, cx2, cy2 = expand_box(x1, y1, x2, y2, w, h, args.margin)
                out = Path(args.crops_out) / cls
                out.mkdir(parents=True, exist_ok=True)
                img.crop((cx1, cy1, cx2, cy2)).save(out / f"{Path(src).stem}_{a['id']}.jpg", quality=95)
                crop_counts[cls] += 1

        if not record["boxes"]:
            continue
        boxes_index.append(record)

        if args.mode in ("full", "both"):
            cls = record["boxes"][0][4]
            out = Path(args.full_out) / cls
            out.mkdir(parents=True, exist_ok=True)
            img.save(out / f"{Path(src).stem}.jpg", quality=95)
            full_counts[cls] += 1

    Path(args.boxes_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.boxes_out).write_text(json.dumps(boxes_index))

    print(f"Images with usable boxes: {len(boxes_index)} (missing files: {missing})")
    if crop_counts:
        print(f"Crops written to {args.crops_out}: {sum(crop_counts.values())}")
        for k, v in sorted(crop_counts.items()):
            print(f"  {k:25s} {v}")
    if full_counts:
        print(f"Full frames written to {args.full_out}: {sum(full_counts.values())}")
    print(f"Box index: {args.boxes_out}")


if __name__ == "__main__":
    main()
