"""
Model construction utilities.

Builds ImageNet-pretrained MobileNetV2 or ResNet18 from torchvision and
swaps the final classification head to match the number of wildlife
classes in the ENA24 dataset.
"""

import torch
import torch.nn as nn
from torchvision import models


SUPPORTED_MODELS = ("mobilenet_v2", "resnet18")


def build_model(arch: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Build a MobileNetV2 or ResNet18 with a new classification head.

    Args:
        arch: "mobilenet_v2" or "resnet18"
        num_classes: number of wildlife classes in your dataset
        pretrained: load ImageNet weights (recommended, then fine-tune)

    Returns:
        torch.nn.Module ready for fine-tuning
    """
    if arch not in SUPPORTED_MODELS:
        raise ValueError(f"arch must be one of {SUPPORTED_MODELS}, got {arch!r}")

    if arch == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)

    elif arch == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    return model


def load_finetuned(arch: str, num_classes: int, checkpoint_path: str, device: str = "cpu") -> nn.Module:
    """Rebuild a model architecture and load fine-tuned weights from disk."""
    model = build_model(arch, num_classes, pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    # Quick sanity check: build each model and run one dummy forward pass.
    # This is your Week 1 "pipeline is alive" checkpoint.
    for arch in SUPPORTED_MODELS:
        m = build_model(arch, num_classes=23)  # ENA24 has ~23 classes; adjust to your label set
        dummy = torch.randn(1, 3, 224, 224)
        out = m(dummy)
        print(f"{arch}: output shape {tuple(out.shape)} — OK")
