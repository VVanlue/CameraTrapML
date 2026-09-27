"""
Fine-tune an ImageNet-pretrained MobileNetV2 or ResNet18 on ENA24 and
save the resulting FP32 checkpoint. This is the baseline every
quantized version gets compared against.

Usage:
    python src/train_baseline.py --arch mobilenet_v2 --data_dir data/ena24 --epochs 5
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from dataset import load_ena24, make_loader
from models import build_model


def evaluate(model, loader, device) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1_macro": f1_score(all_labels, all_preds, average="macro"),
    }


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=["mobilenet_v2", "resnet18"], required=True)
    parser.add_argument("--data_dir", default="data/ena24")
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out_dir", default="checkpoints")
    parser.add_argument("--device", default="cpu", help="cpu (matches the field-hardware constraint) or cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    full_ds, train_ds, val_ds = load_ena24(args.data_dir, args.resolution)
    num_classes = len(full_ds.classes)
    train_loader = make_loader(train_ds, args.batch_size, shuffle=True)
    val_loader = make_loader(val_ds, args.batch_size, shuffle=False)

    model = build_model(args.arch, num_classes, pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    print(f"Fine-tuning {args.arch} on {num_classes} classes, {len(train_ds)} train / {len(val_ds)} val images")

    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        metrics = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        print(f"Epoch {epoch}/{args.epochs} | loss {train_loss:.4f} | "
              f"val_acc {metrics['accuracy']:.4f} | val_f1 {metrics['f1_macro']:.4f} | {elapsed:.1f}s")

        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            ckpt_path = out_dir / f"{args.arch}_fp32_res{args.resolution}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> saved new best checkpoint to {ckpt_path}")

    # Also save class list alongside the checkpoint so downstream scripts
    # know num_classes / label order without re-scanning the dataset.
    classes_path = out_dir / f"{args.arch}_classes.txt"
    classes_path.write_text("\n".join(full_ds.classes))
    print(f"Done. Best val F1: {best_f1:.4f}. Classes saved to {classes_path}")


if __name__ == "__main__":
    main()
