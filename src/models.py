"""
Model construction utilities.

Builds ImageNet-pretrained MobileNetV2 or ResNet18 from torchvision and
swaps the final classification head to match the number of wildlife
classes in the ENA24 dataset.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import quantization as qmodels

SUPPORTED_MODELS = ("mobilenet_v2", "resnet18")


def _swap_head(model: nn.Module, arch: str, num_classes: int) -> nn.Module:
    if arch == "mobilenet_v2":
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    else:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_model(arch: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if arch not in SUPPORTED_MODELS:
        raise ValueError(f"arch must be one of {SUPPORTED_MODELS}, got {arch!r}")
    if arch == "mobilenet_v2":
        w = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v2(weights=w)
    else:
        w = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=w)
    return _swap_head(model, arch, num_classes)


def build_quantizable(arch: str, num_classes: int) -> nn.Module:
    if arch == "mobilenet_v2":
        model = qmodels.mobilenet_v2(weights=None, quantize=False)
    elif arch == "resnet18":
        model = qmodels.resnet18(weights=None, quantize=False)
    else:
        raise ValueError(arch)
    return _swap_head(model, arch, num_classes)


def load_state(model: nn.Module, checkpoint_path: str, strict: bool = True) -> nn.Module:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=strict)
    return model


def load_finetuned(arch: str, num_classes: int, checkpoint_path: str, device: str = "cpu") -> nn.Module:
    model = load_state(build_model(arch, num_classes, pretrained=False), checkpoint_path)
    return model.to(device).eval()


def save_meta(path: Path, **meta):
    Path(path).write_text(json.dumps(meta, indent=2))


def load_meta(path) -> dict:
    return json.loads(Path(path).read_text())


if __name__ == "__main__":
    for arch in SUPPORTED_MODELS:
        m = build_model(arch, num_classes=22, pretrained=False)
        q = build_quantizable(arch, num_classes=22)
        q.load_state_dict(m.state_dict())  # proves FP32 checkpoints load into the quantizable model
        out = m(torch.randn(1, 3, 224, 224))
        print(f"{arch}: output {tuple(out.shape)}, params {sum(p.numel() for p in m.parameters())/1e6:.2f}M — OK")
