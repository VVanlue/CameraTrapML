"""
Produce a REAL quantized model for each level, by PTQ or QAT.

    python src/qat_train.py --arch mobilenet_v2 --resolution 224 \
        --fp32_checkpoint models/mobilenet_v2_fp32_res224.pt \
        --levels int8 int4 int3 int2 1.58bit 1bit --method qat --device cuda

For each level this writes to models/:
    <arch>_<level>_<method>_res<R>.pt     quantized state_dict (int weights, loads on CPU)
    <arch>_<level>_<method>_res<R>.qpk    bit-packed n-bit deployment file (size = real storage)
    <arch>_<level>_<method>_res<R>.json   test accuracy/F1, file sizes, unique codes per layer

How QAT works here (talk-track version)
---------------------------------------
Start from the trained FP32 network. BatchNorm is folded into the conv
weights. Every conv/linear weight goes through a quantizer in the forward
pass, so the loss "sees" the rounding error. The backward pass uses the
straight-through estimator (treat rounding as identity) so gradients still
flow to the latent full-precision weights. Activations get 8-bit fake quant.
A frozen FP32 copy of the model acts as a teacher (knowledge distillation),
which is the standard trick that makes 2-bit / ternary / binary trainable.
After training we snap the weights to the grid and convert to integer
kernels (see quantize.py).

Training runs fine on a GPU (Colab is enough); conversion always happens on CPU.
"""

import argparse
import copy
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from dataset import fixed_subset, load_ena24, make_loader
from models import build_model, load_meta, load_state
from quantize import (INTEGER_LEVELS, bake_and_convert, calibrate, count_unique_weight_codes,
                      export_packed, export_torchscript, freeze_activation_observers, load_packed,
                      prepare_quant_model, set_backend)

# Lower bit-widths need more fine-tuning to recover.
DEFAULT_EPOCHS = {"int8": 2, "int4": 4, "int3": 4, "int2": 6, "1.58bit": 6, "1bit": 10}
DEFAULT_LR = {"int8": 2e-5, "int4": 1e-4, "int3": 1e-4, "int2": 2e-4, "1.58bit": 2e-4, "1bit": 3e-4}


@torch.no_grad()
def evaluate(model, loader, device="cpu", half=False):
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        x = x.to(device)
        if half:
            x = x.half()
        preds += model(x).argmax(1).cpu().tolist()
        labels += y.tolist()
    return {"accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro", zero_division=0)}


def kd_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.7):
    soft = F.kl_div(F.log_softmax(student_logits / T, 1), F.softmax(teacher_logits / T, 1),
                    reduction="batchmean") * T * T
    hard = F.cross_entropy(student_logits, labels, label_smoothing=0.05)
    return alpha * soft + (1 - alpha) * hard


def qat_finetune(model, teacher, train_loader, val_loader, epochs, lr, device, use_kd=True):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs * len(train_loader)))
    best_state, best_f1 = None, -1
    for ep in range(1, epochs + 1):
        model.train()
        if ep > max(1, (3 * epochs) // 4):
            freeze_activation_observers(model)   # stop moving activation ranges for the last quarter
        t0, tot, n = time.time(), 0.0, 0
        for x, y in tqdm(train_loader, desc=f"  QAT epoch {ep}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            out = model(x)
            if use_kd:
                with torch.no_grad():
                    t_out = teacher(x)
                loss = kd_loss(out, t_out, y)
            else:
                loss = F.cross_entropy(out, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * x.size(0)
            n += x.size(0)
        m = evaluate(model, val_loader, device)   # fake-quant eval (same math as the integer model)
        print(f"  epoch {ep}: loss {tot / max(n, 1):.4f} | val_acc {m['accuracy']:.4f} | "
              f"val_f1 {m['f1_macro']:.4f} | {time.time() - t0:.0f}s")
        if m["f1_macro"] > best_f1:
            best_f1, best_state = m["f1_macro"], copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", required=True, choices=["mobilenet_v2", "resnet18"])
    p.add_argument("--resolution", type=int, default=224)
    p.add_argument("--fp32_checkpoint", required=True)
    p.add_argument("--meta", default=None, help="defaults to <arch>_res<R>_meta.json next to the fp32 checkpoint")
    p.add_argument("--data_dir", default=None, help="defaults to the one recorded in the meta file")
    p.add_argument("--levels", nargs="+", default=INTEGER_LEVELS)
    p.add_argument("--method", choices=["ptq", "qat"], default="qat")
    p.add_argument("--epochs", type=int, default=0, help="0 = per-level defaults")
    p.add_argument("--lr", type=float, default=0, help="0 = per-level defaults")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--no_kd", action="store_true")
    p.add_argument("--no_progressive", action="store_true",
                   help="QAT each level from FP32 instead of from the previous (higher-bit) QAT model")
    p.add_argument("--all_layers_lowbit", action="store_true",
                   help="also quantize the first conv and the classifier to n bits (default keeps them 8-bit)")
    p.add_argument("--quantize_depthwise", action="store_true",
                   help="also quantize MobileNetV2 depthwise convs to n bits (default keeps them 8-bit)")
    p.add_argument("--backend", default=None, help="x86 | fbgemm | qnnpack (default: auto for this CPU)")
    p.add_argument("--calib_batches", type=int, default=32)
    p.add_argument("--max_train_images", type=int, default=0)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default="models")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    backend = set_backend(args.backend)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = args.meta or Path(args.fp32_checkpoint).parent / f"{args.arch}_res{args.resolution}_meta.json"
    meta = load_meta(meta_path)
    classes = meta["classes"]
    data_dir = args.data_dir or meta["data_dir"]
    nc = len(classes)

    _, train_ds, val_ds, test_ds = load_ena24(data_dir, args.resolution)
    if args.max_train_images:
        train_ds.indices = train_ds.indices[: args.max_train_images]
    train_loader = make_loader(train_ds, args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = make_loader(val_ds, 64, num_workers=args.workers)
    test_loader = make_loader(test_ds, 64, num_workers=args.workers)
    # PTQ calibrates on held-out val images; QAT also uses them for its initial activation ranges.
    calib_loader = make_loader(fixed_subset(val_ds, 512), 16, num_workers=args.workers)

    teacher = build_model(args.arch, nc, pretrained=False)
    load_state(teacher, args.fp32_checkpoint)
    teacher = teacher.to(args.device).eval()

    # Progressive QAT: walk from high to low precision and start each level from
    # the previous level's trained weights (int8 -> int4 -> ... -> 1bit). Each
    # step is a small jump, which is what makes ternary/binary trainable.
    levels = sorted(args.levels, key=lambda l: INTEGER_LEVELS.index(l) if l in INTEGER_LEVELS else -1)
    prev_state = None
    for level in levels:
        if level not in INTEGER_LEVELS:
            print(f"skip {level}: fp32/fp16 need no quantization step")
            continue
        tag = f"{args.arch}_{level}_{args.method}_res{args.resolution}"
        print(f"\n=== {tag} (backend {backend}) ===")
        model = prepare_quant_model(args.arch, nc, args.fp32_checkpoint, level, backend,
                                    first_last_8bit=not args.all_layers_lowbit,
                                    depthwise_8bit=not args.quantize_depthwise).to(args.device)
        if args.method == "qat" and prev_state is not None and not args.no_progressive:
            model.load_state_dict(prev_state, strict=False)
            model.apply(torch.ao.quantization.enable_observer)   # previous run froze them; let them adapt again
            print("  initialised from the previous level's QAT weights (progressive)")
        calibrate(model, calib_loader, args.calib_batches, args.device)

        if args.method == "qat":
            epochs = args.epochs or DEFAULT_EPOCHS[level]
            lr = args.lr or DEFAULT_LR[level]
            model = qat_finetune(model, teacher, train_loader, val_loader, epochs, lr,
                                 args.device, use_kd=not args.no_kd)

        latent = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        torch.save(latent, out_dir / f"{tag}.latent.pt")   # lets export_for_backend.py re-convert for ARM
        if args.method == "qat":   # latent float weights + activation ranges, minus level-specific quantizer state
            prev_state = {k: v for k, v in latent.items() if "weight_fake_quant" not in k}
        qmodel, layer_info = bake_and_convert(model)        # -> CPU, real integer kernels
        test = evaluate(qmodel, test_loader, "cpu")
        print(f"  INTEGER MODEL test acc {test['accuracy']:.4f} | test F1 {test['f1_macro']:.4f}")

        torch.save(qmodel.state_dict(), out_dir / f"{tag}.pt")
        flags = {"first_last_8bit": not args.all_layers_lowbit, "depthwise_8bit": not args.quantize_depthwise}
        (out_dir / f"{tag}.json").write_text(json.dumps(flags))   # replaced by the full report below
        pack_stats = export_packed(qmodel, layer_info, out_dir / f"{tag}.qpk")
        export_torchscript(qmodel, out_dir / f"{tag}.ts", args.resolution)
        # Proof the n-bit file is complete: rebuild from ONLY the packed bits, compare logits.
        xb = next(iter(test_loader))[0][:8]
        with torch.no_grad():
            reloaded = load_packed(args.arch, nc, level, out_dir / f"{tag}.qpk", backend)
            identical = bool(torch.equal(reloaded(xb), qmodel(xb)))
        print(f"  packed file reloads to an identical model: {identical}")
        codes = count_unique_weight_codes(qmodel)
        report = {
            "arch": args.arch, "level": level, "method": args.method, "resolution": args.resolution,
            "backend": backend, **flags,
            "test": test, **pack_stats, "packed_reload_identical": identical,
            "state_dict_bytes": (out_dir / f"{tag}.pt").stat().st_size,
            "max_unique_codes_at_target_level": max([c for n, c in codes.items()
                                                     if layer_info.get(n, {}).get("level") == level] or [0]),
            "layers_at_target_level": sum(1 for v in layer_info.values() if v["level"] == level),
            "layers_kept_8bit": sum(1 for v in layer_info.values() if v["level"] == "int8"),
            "classes": classes,
        }
        (out_dir / f"{tag}.json").write_text(json.dumps(report, indent=2))
        print(f"  packed n-bit file {pack_stats['packed_file_bytes'] / 1e6:.2f} MB "
              f"(avg {pack_stats['avg_weight_bits']:.2f} bits/weight) | "
              f"{report['layers_at_target_level']} layers at {level} (max {report['max_unique_codes_at_target_level']} "
              f"distinct integer values), {report['layers_kept_8bit']} kept 8-bit")


if __name__ == "__main__":
    main()
