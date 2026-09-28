"""
Quantization utilities.

INT8: uses torchvision's *quantizable* MobileNetV2/ResNet18 architectures
(torchvision.models.quantization.*), which already implement the
fuse_model() step needed for PyTorch's post-training static quantization
(Jacob et al. 2018 — the same scheme our proposal cites). This gives a
real int8 model with genuine memory and latency changes.

INT4: mainstream frameworks (torch, ONNX Runtime) do not support int4
execution for standard convolutional architectures the way they do for
transformer/linear layers. Rather than skip INT4 entirely, this module
implements a *simulated* INT4 quantization: weights are quantized to 4
bits per channel and immediately dequantized back to float32, so you can
still measure the ACCURACY impact of 4-bit weights. Because computation
still happens in float32, this does NOT give you a real latency/memory
number for INT4 — only a theoretical memory estimate (see
theoretical_param_memory_bytes). Report this distinction explicitly in
your write-up; it's exactly the "INT4 may not fully work" risk your
proposal already planned for.
"""

from typing import Callable

import torch
import torch.nn as nn
from torchvision.models.quantization import mobilenet_v2 as q_mobilenet_v2
from torchvision.models.quantization import resnet18 as q_resnet18


# ---------------------------------------------------------------------------
# INT8 (real, static, via torchvision quantizable architectures)
# ---------------------------------------------------------------------------

def build_quantizable_model(arch: str, num_classes: int) -> nn.Module:
    """Build the quantizable (fuse_model-capable) version of the architecture,
    with an unpretrained head sized for our number of classes."""
    if arch == "mobilenet_v2":
        model = q_mobilenet_v2(weights=None, quantize=False)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif arch == "resnet18":
        model = q_resnet18(weights=None, quantize=False)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported arch: {arch}")
    return model


def load_finetuned_into_quantizable(arch: str, num_classes: int, fp32_checkpoint: str) -> nn.Module:
    """Load your fine-tuned FP32 weights into the quantizable architecture.
    Layer names match the regular torchvision model, so this should load
    cleanly; strict=False guards against minor key mismatches from the
    added quant/dequant stubs (which have no parameters of their own)."""
    model = build_quantizable_model(arch, num_classes)
    state_dict = torch.load(fp32_checkpoint, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[load_finetuned_into_quantizable] missing={missing} unexpected={unexpected}")
    return model


def quantize_int8_static(
    model: nn.Module,
    calibration_loader,
    num_calibration_batches: int = 10,
) -> nn.Module:
    """
    Post-training static INT8 quantization.

    Args:
        model: a quantizable model from build_quantizable_model / load_finetuned_into_quantizable
        calibration_loader: DataLoader providing representative (image, label) batches
        num_calibration_batches: how many batches to run for calibration
    """
    model.eval()
    model.fuse_model()  # fuses conv+bn(+relu) — required before quantizing

    model.qconfig = torch.quantization.get_default_qconfig("fbgemm")  # x86 CPU backend
    torch.quantization.prepare(model, inplace=True)

    with torch.no_grad():
        for i, (images, _) in enumerate(calibration_loader):
            if i >= num_calibration_batches:
                break
            model(images)

    torch.quantization.convert(model, inplace=True)
    return model


# ---------------------------------------------------------------------------
# INT4 (simulated — see module docstring for the important caveat)
# ---------------------------------------------------------------------------

def _quantize_dequantize_tensor_int4(weight: torch.Tensor) -> torch.Tensor:
    """Symmetric per-output-channel fake quantization to 4 bits (range [-8, 7])."""
    qmin, qmax = -8, 7
    out_channels = weight.shape[0]
    flat = weight.view(out_channels, -1)
    max_abs = flat.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
    scale = max_abs / qmax

    q = torch.clamp(torch.round(flat / scale), qmin, qmax)
    dq = (q * scale).view_as(weight)
    return dq


def fake_quantize_int4(model: nn.Module) -> nn.Module:
    """Apply simulated 4-bit weight quantization to every Conv2d/Linear layer,
    in place, and return the same model with dequantized (float32) weights."""
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                module.weight.copy_(_quantize_dequantize_tensor_int4(module.weight))
    return model


def theoretical_param_memory_bytes(model: nn.Module, bits_per_param: int) -> float:
    """Estimate parameter storage if every parameter were packed at
    bits_per_param bits (e.g. 32 for fp32, 8 for int8, 4 for int4).
    Use this for the INT4 memory bar since real packed int4 storage isn't
    produced by fake_quantize_int4 (weights stay float32 in memory)."""
    n_params = sum(p.numel() for p in model.parameters())
    return n_params * bits_per_param / 8


if __name__ == "__main__":
    # Smoke test with random weights (no dataset needed).
    m = build_quantizable_model("mobilenet_v2", num_classes=23)
    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out_before = m(dummy)
    fake_quantize_int4(m)
    with torch.no_grad():
        out_after = m(dummy)
    print("INT4 simulation ran OK. Output shapes:", out_before.shape, out_after.shape)
    print("Theoretical INT4 memory (MB):", theoretical_param_memory_bytes(m, 4) / 1e6)
