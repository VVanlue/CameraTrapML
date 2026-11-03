"""
Fine-tune an ImageNet-pretrained MobileNetV2 or ResNet18 on ENA24 and
save the resulting FP32 checkpoint. This is the baseline every
quantized version gets compared against.

Usage:
    python src/train_baseline.py --arch mobilenet_v2 --data_dir data/ena24 --epochs 5
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from dataset import class_counts, load_ena24, make_loader
from models import build_model, save_meta


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        preds += model(x.to(device)).argmax(1).cpu().tolist()
        labels += y.tolist()
    return {"accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro", zero_division=0)}


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, desc="train"):
    model.train()
    total, n = 0.0, 0
    for x, y in tqdm(loader, desc=desc, leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


def class_weights(train_ds, num_classes, device):
    """ENA24 is imbalanced (lots of deer/birds, few bears/horses). Inverse-sqrt
    frequency weights help macro-F1 without wrecking accuracy."""
    counts = torch.tensor(class_counts(train_ds), dtype=torch.float).clamp(min=1)
    w = counts.sum() / counts
    w = w.sqrt()
    return (w / w.mean()).to(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=["mobilenet_v2", "resnet18"], required=True)
    p.add_argument("--data_dir", default="data/ena24_crops")
    p.add_argument("--resolution", type=int, default=224)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_pretrained", action="store_true", help="start from random weights (offline testing only)")
    p.add_argument("--max_train_images", type=int, default=0, help="debug: cap the training set")
    p.add_argument("--out_dir", default="models")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    classes, train_ds, val_ds, test_ds = load_ena24(args.data_dir, args.resolution)
    if args.max_train_images:
        train_ds.indices = train_ds.indices[: args.max_train_images]
    nc = len(classes)
    train_loader = make_loader(train_ds, args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = make_loader(val_ds, 64, num_workers=args.workers)
    test_loader = make_loader(test_ds, 64, num_workers=args.workers)

    model = build_model(args.arch, nc, pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=steps, pct_start=0.15)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_ds, nc, device), label_smoothing=0.05)

    print(f"Fine-tuning {args.arch} @ {args.resolution}px on {nc} classes | "
          f"train {len(train_ds)} / val {len(val_ds)} / test {len(test_ds)} | device {device}")

    ckpt = out_dir / f"{args.arch}_fp32_res{args.resolution}.pt"
    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device, f"epoch {epoch}")
        m = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}/{args.epochs} | loss {loss:.4f} | val_acc {m['accuracy']:.4f} | "
              f"val_f1 {m['f1_macro']:.4f} | {time.time() - t0:.0f}s")
        if m["f1_macro"] > best_f1:
            best_f1 = m["f1_macro"]
            torch.save(model.state_dict(), ckpt)
            print(f"  -> saved best checkpoint {ckpt}")

    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    test = evaluate(model, test_loader, device)
    print(f"TEST accuracy {test['accuracy']:.4f} | TEST macro-F1 {test['f1_macro']:.4f}")

    save_meta(out_dir / f"{args.arch}_res{args.resolution}_meta.json",
              arch=args.arch, resolution=args.resolution, classes=classes,
              data_dir=args.data_dir, best_val_f1=best_f1, test=test, seed=args.seed)
    (out_dir / "classes.txt").write_text("\n".join(classes))


if __name__ == "__main__":
    main()
