"""
OPTIONAL: train our own "find the animal" detector on ENA24's bounding boxes,
for a fully offline pipeline that doesn't depend on MegaDetector.

    python src/train_detector.py --boxes data/ena24_boxes.json --epochs 6 --device cuda

Model: torchvision Faster R-CNN with a MobileNetV3-Large FPN backbone
(COCO-pretrained), retrained as a single-class "animal" detector — the
species is the classifier's job. Saves models/detector_ena24.pt.
Needs a GPU to be practical (~10-15 min/epoch on a Colab T4); on CPU use
--max_images 1500 --epochs 3 for something usable in an hour or two.
"""

import argparse
import json
import random
import re
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import (FasterRCNN_MobileNet_V3_Large_FPN_Weights,
                                          fasterrcnn_mobilenet_v3_large_fpn)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.functional import hflip, to_tensor
from tqdm import tqdm


class BoxDataset(Dataset):
    def __init__(self, records, train):
        self.records, self.train = records, train

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        img = Image.open(r["file"]).convert("RGB")
        boxes = torch.tensor([b[:4] for b in r["boxes"]], dtype=torch.float32)
        x = to_tensor(img)
        if self.train and random.random() < 0.5:
            x = hflip(x)
            w = x.shape[-1]
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        return x, {"boxes": boxes, "labels": torch.ones(len(boxes), dtype=torch.int64)}


def group_split(records, val_frac=0.15, seed=0, group=10):
    def key(r):
        m = re.match(r"(\d+)", Path(r["file"]).stem)
        return int(m.group(1)) // group if m else Path(r["file"]).stem
    keys = sorted({key(r) for r in records}, key=str)
    random.Random(seed).shuffle(keys)
    val_keys = set(keys[: int(len(keys) * val_frac)])
    return [r for r in records if key(r) not in val_keys], [r for r in records if key(r) in val_keys]


@torch.no_grad()
def recall_at_iou(model, loader, device, iou=0.5, conf=0.25):
    from torchvision.ops import box_iou
    model.eval()
    hit = tot = fp = 0
    for xs, ts in loader:
        outs = model([x.to(device) for x in xs])
        for o, t in zip(outs, ts):
            pb = o["boxes"][o["scores"] >= conf].cpu()
            tot += len(t["boxes"])
            if len(pb) and len(t["boxes"]):
                m = box_iou(t["boxes"], pb)
                hit += int((m.max(1).values >= iou).sum())
                fp += int((m.max(0).values < iou).sum())
            else:
                fp += len(pb)
    return hit / max(tot, 1), fp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--boxes", default="data/ena24_boxes.json")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--max_images", type=int, default=0)
    p.add_argument("--out", default="models/detector_ena24.pt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    records = json.loads(Path(a.boxes).read_text())
    if a.max_images:
        records = records[: a.max_images]
    tr, va = group_split(records)
    collate = lambda b: tuple(zip(*b))
    tl = DataLoader(BoxDataset(tr, True), a.batch_size, shuffle=True, num_workers=2, collate_fn=collate)
    vl = DataLoader(BoxDataset(va, False), a.batch_size, num_workers=2, collate_fn=collate)

    model = fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT)
    in_f = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, 2)   # background + animal
    model.to(a.device)
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.SGD(params, lr=a.lr, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * len(tl))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    best = -1
    for ep in range(1, a.epochs + 1):
        model.train()
        for xs, ts in tqdm(tl, desc=f"epoch {ep}", leave=False):
            xs = [x.to(a.device) for x in xs]
            ts = [{k: v.to(a.device) for k, v in t.items()} for t in ts]
            loss = sum(model(xs, ts).values())
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        rec, fps = recall_at_iou(model, vl, a.device)
        print(f"epoch {ep}: val recall@IoU0.5 {rec:.3f} | false positives {fps}")
        if rec > best:
            best = rec
            torch.save(model.state_dict(), a.out)
            print(f"  saved {a.out}")


if __name__ == "__main__":
    main()
