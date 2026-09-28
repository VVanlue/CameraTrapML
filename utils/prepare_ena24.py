"""
Convert the raw LILA BC ENA24 release (images/ + a COCO-style annotations
JSON) into an ImageFolder-compatible layout for torchvision:

    data/ena24/<class_name>/<image>.jpg

LILA's camera-trap releases are typically distributed as a folder of
images plus a COCO-format JSON (images, annotations, categories). This
script reads that JSON and copies (or symlinks) each image into a
folder named after its category.

If your particular ENA24 download already comes as class-per-folder,
you don't need this script at all — point dataset.py straight at it.

Usage:
    python utils/prepare_ena24.py \
        --images_dir raw/ena24/images \
        --annotations raw/ena24/ena24.json \
        --out_dir data/ena24 \
        --mode symlink
"""

import argparse
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True, help="Folder of raw downloaded images")
    parser.add_argument("--annotations", required=True, help="Path to COCO-style annotations JSON")
    parser.add_argument("--out_dir", required=True, help="Output ImageFolder root, e.g. data/ena24")
    parser.add_argument("--mode", choices=["copy", "symlink"], default="symlink",
                         help="symlink is faster and saves disk space; copy is safer for sharing")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.annotations) as f:
        coco = json.load(f)

    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    image_id_to_filename = {img["id"]: img["file_name"] for img in coco["images"]}

    # ENA24 images may have multiple annotations (multiple animals); we
    # assign each image to the category of its first annotation for a
    # simple single-label classification setup.
    image_id_to_category = {}
    for ann in coco["annotations"]:
        image_id_to_category.setdefault(ann["image_id"], ann["category_id"])

    n_ok, n_missing = 0, 0
    for image_id, category_id in image_id_to_category.items():
        filename = image_id_to_filename.get(image_id)
        if filename is None:
            continue
        src = images_dir / filename
        if not src.exists():
            n_missing += 1
            continue

        class_name = cat_id_to_name[category_id].replace(" ", "_")
        class_dir = out_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        dst = class_dir / src.name

        if args.mode == "symlink":
            if not dst.exists():
                dst.symlink_to(src.resolve())
        else:
            if not dst.exists():
                shutil.copy2(src, dst)
        n_ok += 1

    print(f"Prepared {n_ok} images into {out_dir} ({n_missing} referenced images were missing on disk).")
    print(f"Classes found: {sorted(set(cat_id_to_name.values()))}")


if __name__ == "__main__":
    main()
