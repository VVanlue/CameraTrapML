"""
Quantization utilities.

Supported levels: fp32, int8, int4, int3, int2, 1.58bit, 1bit

INT8 is REAL: uses torchvision's quantizable MobileNetV2/ResNet18
architectures (torchvision.models.quantization.*) with PyTorch's
post-training static quantization (Jacob et al. 2018). Genuine int8
tensors, genuine memory/latency differences.

INT4, INT3, INT2, 1.58bit (ternary), and 1bit (binary) are SIMULATED.
No mainstream framework (torch, ONNX Runtime) executes CNN convolutions
at these precisions the way they do for transformers/LLMs. Weights are
quantized to the target precision and immediately dequantized back to
float32, so:
  - Accuracy / F1 impact IS real and meaningful to report.
  - Latency / throughput at these levels will look close to FP32,
    because compute still happens in float32. This is expected, not a
    bug — report it as a known limitation, not a finding that "INT2 is
    as fast as FP32."
  - Memory should be reported using theoretical_param_memory_bytes()
    (an estimate of packed storage), not the measured process RSS,
    since the in-memory tensors are still float32.

1.58bit (ternary, weights in {-1, 0, +1}) and 1bit (binary, weights in
{-1, +1}) follow the simplified schemes from Li et al. 2016 (Ternary
Weight Networks) and Rastegari et al. 2016 (XNOR-Net / Binary Weight
Networks) respectively — using per-channel mean absolute value as the
scale rather than the papers' full optimization, which is a reasonable
simplification for this project's scope.
"""

import torch
import torch.nn as nn
from torchvision.models.quantization import mobilenet_v2 as q_mobilenet_v2
from torchvision.models.quantization import resnet18 as q_resnet18


QUANT_LEVELS = ["fp32", "int8", "int4", "int3", "int2", "1.58bit", "1bit"]

# Bit width used for the THEORETICAL memory estimate at each level.
THEORETICAL_BITS = {
    "fp32": 32,
    "int8": 8,
    "int4": 4,
    "int3": 3,
    "int2": 2,
    "1.58bit": 1.58,  # log2(3), the information-theoretic width of a ternary value
    "1bit": 1,
}

# Integer levels handled by the generic symmetric n-bit fake quantizer.
_SIMULATED_INT_BITS = {"int4": 4, "int3": 3, "int2": 2}


# ---------------------------------------------------------------------------
# INT8 (real, static, via torchvision quantizable architectures)
# ---------------------------------------------------------------------------

def build_quantizable_model(arch: str, num_classes: int) -> nn.Module:
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
    model = build_quantizable_model(arch, num_classes)
    state_dict = torch.load(fp32_checkpoint, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[load_finetuned_into_quantizable] missing={missing} unexpected={unexpected}")
    return model


def quantize_int8_static(model: nn.Module, calibration_loader, num_calibration_batches: int = 10) -> nn.Module:
    model.eval()
    model.fuse_model()
    model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
    torch.quantization.prepare(model, inplace=True)
    with torch.no_grad():
        for i, (images, _) in enumerate(calibration_loader):
            if i >= num_calibration_batches:
                break
            model(images)
    torch.quantization.convert(model, inplace=True)
    return model


# ---------------------------------------------------------------------------
# Simulated low-bit quantization: INT4 / INT3 / INT2
# ---------------------------------------------------------------------------

def _quantize_dequantize_symmetric(weight: torch.Tensor, bits: int) -> torch.Tensor:
    """Symmetric per-output-channel fake quantization to `bits` bits."""
    qmax = 2 ** (bits - 1) - 1
    qmin = -(2 ** (bits - 1))
    out_channels = weight.shape[0]
    flat = weight.view(out_channels, -1)
    max_abs = flat.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
    scale = max_abs / qmax
    q = torch.clamp(torch.round(flat / scale), qmin, qmax)
    return (q * scale).view_as(weight)


def fake_quantize_n_bit(model: nn.Module, bits: int) -> nn.Module:
    """Apply simulated symmetric n-bit weight quantization to every
    Conv2d/Linear layer, in place."""
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                module.weight.copy_(_quantize_dequantize_symmetric(module.weight, bits))
    return model


# ---------------------------------------------------------------------------
# Simulated 1.58-bit (ternary) and 1-bit (binary)
# ---------------------------------------------------------------------------

def _ternary_quantize_tensor(weight: torch.Tensor, threshold_ratio: float = 0.7) -> torch.Tensor:
    """Per-output-channel ternary quantization: weights become {-scale, 0, +scale}.
    Simplified version of Li et al. 2016 (Ternary Weight Networks): scale is
    the mean absolute weight per channel, and values within threshold_ratio *
    scale of zero are pruned to exactly zero."""
    out_channels = weight.shape[0]
    flat = weight.view(out_channels, -1)
    scale = flat.abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
    threshold = threshold_ratio * scale
    q = torch.zeros_like(flat)
    q[flat > threshold] = 1.0
    q[flat < -threshold] = -1.0
    return (q * scale).view_as(weight)


def _binary_quantize_tensor(weight: torch.Tensor) -> torch.Tensor:
    """Per-output-channel binary quantization: weights become {-scale, +scale}.
    Simplified version of Rastegari et al. 2016 (Binary Weight Networks):
    scale is the mean absolute weight per channel, sign gives the +/-."""
    out_channels = weight.shape[0]
    flat = weight.view(out_channels, -1)
    scale = flat.abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
    sign = torch.sign(flat)
    sign[sign == 0] = 1.0  # torch.sign(0) == 0; map to +1 so no weight is exactly zero
    return (sign * scale).view_as(weight)


def fake_quantize_ternary(model: nn.Module) -> nn.Module:
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                module.weight.copy_(_ternary_quantize_tensor(module.weight))
    return model


def fake_quantize_binary(model: nn.Module) -> nn.Module:
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                module.weight.copy_(_binary_quantize_tensor(module.weight))
    return model


# ---------------------------------------------------------------------------
# Memory estimate (used for every simulated level, and as a sanity check for int8)
# ---------------------------------------------------------------------------

def theoretical_param_memory_bytes(model: nn.Module, level: str) -> float:
    """Estimate parameter storage if every parameter were packed at the
    bit width associated with `level`. Use this for INT4/INT3/INT2/
    1.58bit/1bit memory reporting, since fake quantization leaves
    weights as float32 in actual memory."""
    bits = THEORETICAL_BITS[level]
    n_params = sum(p.numel() for p in model.parameters())
    return n_params * bits / 8


# ---------------------------------------------------------------------------
# Single dispatch point used by run_sweep.py
# ---------------------------------------------------------------------------

def apply_quantization(model: nn.Module, level: str) -> nn.Module:
    """Apply simulated quantization for any non-fp32, non-int8 level.
    (fp32 needs no change; int8 goes through quantize_int8_static, which
    needs a calibration_loader and is called separately.)"""
    if level in _SIMULATED_INT_BITS:
        return fake_quantize_n_bit(model, _SIMULATED_INT_BITS[level])
    if level == "1.58bit":
        return fake_quantize_ternary(model)
    if level == "1bit":
        return fake_quantize_binary(model)
    raise ValueError(f"apply_quantization does not handle level={level!r} "
                      f"(fp32 needs no call; int8 uses quantize_int8_static)")


if __name__ == "__main__":
    # Smoke test with random weights (no dataset needed).
    dummy = torch.randn(1, 3, 224, 224)
    for level in ["int4", "int3", "int2", "1.58bit", "1bit"]:
        m = build_quantizable_model("mobilenet_v2", num_classes=22)
        apply_quantization(m, level)
        with torch.no_grad():
            out = m(dummy)
        mem_mb = theoretical_param_memory_bytes(m, level) / 1e6
        print(f"{level}: output shape {tuple(out.shape)}, theoretical memory {mem_mb:.2f} MB")