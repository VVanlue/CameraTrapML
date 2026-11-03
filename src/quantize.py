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

import pickle
import platform
import warnings
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.ao.quantization as tq
from torch.ao.quantization import FakeQuantizeBase, QConfig

from models import build_quantizable, build_model, load_state

# torch >= 2.10 warns that eager-mode torch.ao.quantization will move to torchao.
# It still works (tested on 2.15); silence the wall of warnings.
warnings.filterwarnings("ignore", module=r"torch\.ao\..*")
warnings.filterwarnings("ignore", message=r".*torch\.ao\.quantization.*")

QUANT_LEVELS = ["fp32", "fp16", "int8", "int4", "int3", "int2", "1.58bit", "1bit"]
INTEGER_LEVELS = ["int8", "int4", "int3", "int2", "1.58bit", "1bit"]

# (mode, bits) for the weight quantizer
LEVEL_SPEC = {
    "int8": ("int", 8),
    "int4": ("int", 4),
    "int3": ("int", 3),
    "int2": ("int", 2),
    "1.58bit": ("ternary", 2),
    "1bit": ("binary", 1),
}
# Bits per weight in the packed file (ternary uses base-3 packing: 5 weights/byte)
STORAGE_BITS = {"fp32": 32, "fp16": 16, "int8": 8, "int4": 4, "int3": 3, "int2": 2, "1.58bit": 1.6, "1bit": 1}


def default_backend() -> str:
    m = platform.machine().lower()
    if m.startswith(("arm", "aarch64")):
        return "qnnpack"
    return "x86" if "x86" in torch.backends.quantized.supported_engines else "fbgemm"


def set_backend(backend: Optional[str] = None) -> str:
    backend = backend or default_backend()
    torch.backends.quantized.engine = backend
    return backend


# ---------------------------------------------------------------------------
# Low-bit weight quantizer (per output channel, symmetric, STE)
# ---------------------------------------------------------------------------

def _int_range(bits):
    return -(2 ** (bits - 1)), 2 ** (bits - 1) - 1


def _best_int_scale(w2d: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-channel scale choosing the clipping ratio that minimises MSE.
    Max-abs scaling (what the old code did) wastes most of a 2-3 bit grid on
    rare outliers; MSE clipping is the standard fix."""
    qmin, qmax = _int_range(bits)
    max_abs = w2d.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    if bits >= 8:
        return max_abs / qmax
    best_err, best_scale = None, None
    for ratio in torch.linspace(1.0, 0.25, 16):
        s = max_abs * ratio / qmax
        q = torch.clamp(torch.round(w2d / s), qmin, qmax) * s
        err = (q - w2d).pow(2).mean(dim=1, keepdim=True)
        if best_err is None:
            best_err, best_scale = err, s
        else:
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_scale = torch.where(better, s, best_scale)
    return best_scale


def quantize_weight(w: torch.Tensor, mode: str, bits: int, scale: Optional[torch.Tensor] = None):
    """Returns (dequantized_weight, int_codes, per_channel_scale)."""
    shape = w.shape
    w2d = w.reshape(shape[0], -1)
    if mode == "int":
        qmin, qmax = _int_range(bits)
        s = _best_int_scale(w2d, bits) if scale is None else scale.view(-1, 1)
        q = torch.clamp(torch.round(w2d / s), qmin, qmax)
    elif mode == "ternary":
        a = w2d.abs()
        delta = 0.7 * a.mean(dim=1, keepdim=True)          # TWN threshold
        mask = (a > delta).float()
        if scale is None:
            s = (a * mask).sum(1, keepdim=True) / mask.sum(1, keepdim=True).clamp(min=1)
            s = s.clamp(min=1e-8)
        else:
            s = scale.view(-1, 1)
        q = torch.sign(w2d) * mask
    elif mode == "binary":
        s = w2d.abs().mean(dim=1, keepdim=True).clamp(min=1e-8) if scale is None else scale.view(-1, 1)
        q = torch.where(w2d >= 0, torch.ones_like(w2d), -torch.ones_like(w2d))
    else:
        raise ValueError(mode)
    return (q * s).view(shape), q.view(shape), s.view(-1)


class LowBitWeightFakeQuant(FakeQuantizeBase):
    """Drop-in weight quantizer for torch.ao QAT/PTQ. Exposes the attributes
    torch.ao.quantization.convert() needs (dtype, qscheme, ch_axis,
    calculate_qparams) so the converted model holds real integer weights."""

    def __init__(self, mode: str = "int", bits: int = 4, ch_axis: int = 0, **kwargs):
        super().__init__()
        self.mode, self.bits, self.ch_axis = mode, bits, ch_axis
        self.dtype = torch.qint8
        self.qscheme = torch.per_channel_symmetric
        if mode == "int":
            self.quant_min, self.quant_max = _int_range(bits)
        else:
            self.quant_min, self.quant_max = -1, 1
        self.register_buffer("scale", torch.ones(1))
        self.register_buffer("zero_point", torch.zeros(1, dtype=torch.int64))
        self.register_buffer("frozen", torch.tensor(False))

    def forward(self, w):
        if self.observer_enabled[0] == 1 and not bool(self.frozen):
            with torch.no_grad():
                _, _, s = quantize_weight(w.detach(), self.mode, self.bits)
            self.scale = s.detach().clone()
            self.zero_point = torch.zeros_like(s, dtype=torch.int64)
        if self.fake_quant_enabled[0] == 1:
            wq, _, _ = quantize_weight(w, self.mode, self.bits, self.scale.to(w.device))
            return w + (wq - w).detach()  # straight-through estimator
        return w

    @torch.jit.export
    def calculate_qparams(self):
        return self.scale.detach().cpu(), self.zero_point.detach().cpu()

    def extra_repr(self):
        return f"mode={self.mode}, bits={self.bits}, frozen={bool(self.frozen)}"


# ---------------------------------------------------------------------------
# Building the quantization-ready model
# ---------------------------------------------------------------------------

def _qconfig(level: str, backend: str) -> QConfig:
    mode, bits = LEVEL_SPEC[level]
    act = tq.get_default_qat_qconfig(backend).activation
    return QConfig(activation=act, weight=LowBitWeightFakeQuant.with_args(mode=mode, bits=bits))


def _first_last_names(model: nn.Module):
    convs_linears = [n for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    return convs_linears[0], convs_linears[-1]


def _is_depthwise(m: nn.Module) -> bool:
    conv = m[0] if isinstance(m, nn.Sequential) and len(m) and isinstance(m[0], nn.Conv2d) else m
    return isinstance(conv, nn.Conv2d) and conv.groups > 1 and conv.groups == conv.in_channels


def prepare_quant_model(arch: str, num_classes: int, fp32_checkpoint: Optional[str], level: str,
                        backend: str, first_last_8bit: bool = True, depthwise_8bit: bool = True) -> nn.Module:
    """FP32 weights -> quantizable arch -> fold BatchNorm into conv -> insert
    quantizers. Returned model is in train() mode and ready for QAT (or for
    calibration, for PTQ)."""
    model = build_quantizable(arch, num_classes)
    if fp32_checkpoint:
        load_state(model, fp32_checkpoint)
    model.eval()
    first, last = _first_last_names(model)
    model.fuse_model(is_qat=False)          # folds BN into conv weights; conv+relu fused
    model.qconfig = _qconfig(level, backend)
    if first_last_8bit and level != "int8":
        # Fusion keeps the module name (e.g. "features.0.0" becomes a fused
        # ConvReLU2d at the same name), so we can target first/last directly.
        q8 = _qconfig("int8", backend)
        mods = dict(model.named_modules())
        for name in (first, last):
            mods[name].qconfig = q8
    if depthwise_8bit and level != "int8":
        # MobileNetV2's depthwise 3x3 filters have only 9 weights each; at 1-2 bits
        # they lose almost all information and training cannot recover (BNN
        # literature keeps them higher-precision for the same reason). They are
        # ~3% of the weights, so the size cost is small; avg bits/weight is reported.
        q8 = _qconfig("int8", backend)
        for name, m in model.named_modules():
            if _is_depthwise(m):
                m.qconfig = q8
    model.train()
    tq.prepare_qat(model, inplace=True)
    return model


def freeze_activation_observers(model: nn.Module):
    for m in model.modules():
        if isinstance(m, FakeQuantizeBase) and not isinstance(m, LowBitWeightFakeQuant):
            m.disable_observer()


@torch.no_grad()
def calibrate(model: nn.Module, loader, num_batches: int = 20, device="cpu"):
    """PTQ: run data through the model so activation observers learn ranges."""
    was_training = model.training
    model.eval()
    for i, (x, _) in enumerate(loader):
        if i >= num_batches:
            break
        model(x.to(device))
    model.train(was_training)


@torch.no_grad()
def bake_and_convert(model: nn.Module) -> (nn.Module, Dict[str, dict]):
    """Snap every weight exactly onto its integer grid, freeze scales, and
    convert to a true quantized model. Returns (quantized_model, layer_info)."""
    model = model.cpu().eval()
    info = {}
    for name, m in model.named_modules():
        fq = getattr(m, "weight_fake_quant", None)
        if isinstance(fq, LowBitWeightFakeQuant):
            wq, q, s = quantize_weight(m.weight.data, fq.mode, fq.bits)
            m.weight.data.copy_(wq)
            fq.scale = s.clone()
            fq.zero_point = torch.zeros_like(s, dtype=torch.int64)
            fq.frozen.fill_(True)
            level = {"ternary": "1.58bit", "binary": "1bit"}.get(fq.mode, f"int{fq.bits}")
            info[name] = {"level": level, "mode": fq.mode, "bits": fq.bits}
    qmodel = tq.convert(model, inplace=False)
    return qmodel.eval(), info


# ---------------------------------------------------------------------------
# Saving / loading
# ---------------------------------------------------------------------------

def build_empty_quantized(arch: str, num_classes: int, level: str, backend: str,
                          first_last_8bit: bool = True, depthwise_8bit: bool = True) -> nn.Module:
    """Rebuild the converted-model structure (no trained weights) so a saved
    quantized state_dict can be loaded into it."""
    model = prepare_quant_model(arch, num_classes, None, level, backend, first_last_8bit, depthwise_8bit)
    model.eval()
    with torch.no_grad():
        model(torch.randn(1, 3, 64, 64))   # initialise scales so convert() works
    qmodel, _ = bake_and_convert(model)
    return qmodel


def _layer_flags(path) -> dict:
    """Read first_last_8bit / depthwise_8bit from the report JSON saved next to a model."""
    from pathlib import Path
    import json
    j = Path(path).with_suffix(".json")
    if j.exists():
        r = json.loads(j.read_text())
        return {"first_last_8bit": r.get("first_last_8bit", True), "depthwise_8bit": r.get("depthwise_8bit", True)}
    return {"first_last_8bit": True, "depthwise_8bit": True}


def load_model_for_level(arch: str, num_classes: int, level: str, path: str,
                         backend: Optional[str] = None) -> nn.Module:
    """Load any level for inference (CPU). fp32/fp16 load the FP32 checkpoint."""
    if level in ("fp32", "fp16"):
        m = build_model(arch, num_classes, pretrained=False)
        load_state(m, path)
        m.eval()
        return m.half() if level == "fp16" else m
    backend = set_backend(backend)
    qmodel = build_empty_quantized(arch, num_classes, level, backend, **_layer_flags(path))
    state = torch.load(path, map_location="cpu", weights_only=False)
    qmodel.load_state_dict(state)
    return qmodel.eval()


# ---------------------------------------------------------------------------
# Bit packing: the real n-bit storage format
# ---------------------------------------------------------------------------

def pack_codes(codes: np.ndarray, level: str) -> bytes:
    codes = codes.astype(np.int64).ravel()
    if level == "1.58bit":                      # base-3, 5 trits per byte (3^5 = 243 <= 256)
        t = (codes + 1).astype(np.uint8)
        pad = (-len(t)) % 5
        t = np.concatenate([t, np.zeros(pad, np.uint8)]).reshape(-1, 5).astype(np.uint16)
        packed = t[:, 0] + 3 * t[:, 1] + 9 * t[:, 2] + 27 * t[:, 3] + 81 * t[:, 4]
        return packed.astype(np.uint8).tobytes()
    if level == "1bit":
        return np.packbits((codes > 0).astype(np.uint8)).tobytes()
    bits = int(level.replace("int", ""))
    u = (codes + 2 ** (bits - 1)).astype(np.uint64)      # offset to unsigned
    bitarr = ((u[:, None] >> np.arange(bits, dtype=np.uint64)) & 1).astype(np.uint8)
    return np.packbits(bitarr.ravel()).tobytes()


def unpack_codes(buf: bytes, level: str, n: int) -> np.ndarray:
    b = np.frombuffer(buf, dtype=np.uint8)
    if level == "1.58bit":
        b = b.astype(np.int64)
        t = np.stack([(b // 3 ** k) % 3 for k in range(5)], axis=1).ravel()[:n]
        return t - 1
    if level == "1bit":
        return np.unpackbits(b)[:n].astype(np.int64) * 2 - 1
    bits = int(level.replace("int", ""))
    bitarr = np.unpackbits(b)[: n * bits].reshape(n, bits).astype(np.uint64)
    u = (bitarr << np.arange(bits, dtype=np.uint64)).sum(1)
    return u.astype(np.int64) - 2 ** (bits - 1)


def _quantized_weight_modules(qmodel):
    for name, m in qmodel.named_modules():
        if "quantized" not in type(m).__module__ or not callable(getattr(m, "weight", None)):
            continue
        try:
            w = m.weight()
        except Exception:
            continue
        if isinstance(w, torch.Tensor) and w.is_quantized:
            yield name, m, w


def export_packed(qmodel: nn.Module, layer_info: Dict[str, dict], path: str) -> dict:
    """Write the deployable n-bit file: packed integer weights + fp32 per-channel
    scales + fp32 biases (+ fp32 for anything not quantized). Verifies the
    round trip and returns size statistics."""
    blob, n_weights, weight_bits = {}, 0, 0.0
    for name, m, w in _quantized_weight_modules(qmodel):
        level = layer_info.get(name, {}).get("level", "int8")
        codes = w.int_repr().numpy()
        packed = pack_codes(codes, level)
        assert np.array_equal(unpack_codes(packed, level, codes.size), codes.ravel().astype(np.int64)), \
            f"packing round-trip failed for {name}"
        bias = m.bias()
        blob[name] = {
            "level": level, "shape": tuple(codes.shape), "codes": packed,
            "scale": w.q_per_channel_scales().float().numpy() if w.qscheme() in
            (torch.per_channel_affine, torch.per_channel_symmetric) else np.array([w.q_scale()], np.float32),
            "bias": None if bias is None else bias.detach().float().numpy(),
        }
        n_weights += codes.size
        weight_bits += codes.size * STORAGE_BITS[level]
    # activation scale/zero-points and any non-quantized params (tiny)
    extras = {k: v for k, v in qmodel.state_dict().items()
              if isinstance(v, torch.Tensor) and not v.is_quantized and v.numel() < 4096
              and ("scale" in k or "zero_point" in k)}
    blob["_activation_qparams"] = {k: v.numpy() for k, v in extras.items()}
    data = pickle.dumps(blob, protocol=5)   # raw bytes, no base64-style bloat
    with open(path, "wb") as f:
        f.write(data)
    return {"packed_file_bytes": len(data), "n_quantized_weights": n_weights,
            "avg_weight_bits": weight_bits / max(n_weights, 1)}


def load_packed(arch: str, num_classes: int, level: str, path: str, backend: Optional[str] = None) -> nn.Module:
    """Rebuild a runnable integer model from ONLY the bit-packed .qpk file.
    Used to prove the n-bit file is a complete, deployable format."""
    backend = set_backend(backend)
    with open(path, "rb") as f:
        blob = pickle.load(f)
    qmodel = build_empty_quantized(arch, num_classes, level, backend, **_layer_flags(path))
    mods = dict(qmodel.named_modules())
    for name, rec in blob.items():
        if name == "_activation_qparams":
            continue
        n = int(np.prod(rec["shape"]))
        codes = torch.from_numpy(unpack_codes(rec["codes"], rec["level"], n).reshape(rec["shape"])).to(torch.int8)
        scales = torch.from_numpy(rec["scale"]).double()
        qw = torch._make_per_channel_quantized_tensor(codes, scales, torch.zeros(len(scales), dtype=torch.int64), 0)
        bias = None if rec["bias"] is None else torch.from_numpy(rec["bias"])
        mods[name].set_weight_bias(qw, bias)
    for key, val in blob["_activation_qparams"].items():
        mod_name, attr = key.rsplit(".", 1)
        m = mods[mod_name]
        cur = getattr(m, attr)
        if isinstance(cur, torch.Tensor):
            cur.copy_(torch.as_tensor(val))
        else:
            setattr(m, attr, type(cur)(np.asarray(val).item()))
    return qmodel.eval()


def export_torchscript(model: nn.Module, path: str, resolution: int, half: bool = False) -> str:
    """Freeze any level into a self-contained TorchScript file. This is the
    deployment format: loading it does NOT rebuild a float model first, so
    load-time RAM is comparable across fp32 / fp16 / int8 / ... / 1bit."""
    model = model.eval()
    x = torch.randn(1, 3, resolution, resolution)
    if half:
        x = x.half()
    with torch.no_grad():
        ts = torch.jit.freeze(torch.jit.trace(model, x))
    torch.jit.save(ts, str(path))
    return str(path)


def fp_file_bytes(model: nn.Module, level: str, path: str) -> int:
    """Save an fp32/fp16 state_dict and return its size on disk."""
    sd = {k: (v.half() if level == "fp16" and v.is_floating_point() else v) for k, v in model.state_dict().items()}
    torch.save(sd, path)
    import os
    return os.path.getsize(path)


def count_unique_weight_codes(qmodel) -> Dict[str, int]:
    """Sanity check for the talk: how many distinct integer values each layer
    actually uses (int4 <= 16, ternary <= 3, binary == 2)."""
    return {name: int(torch.unique(w.int_repr()).numel()) for name, _, w in _quantized_weight_modules(qmodel)}


if __name__ == "__main__":
    # Smoke test on random weights: every level converts, runs on integer
    # kernels, packs losslessly, and uses only the allowed integer codes.
    backend = set_backend()
    print(f"quantized engine: {backend}")
    x = torch.randn(2, 3, 224, 224)
    for level in INTEGER_LEVELS:
        m = prepare_quant_model("mobilenet_v2", 22, None, level, backend)
        m.eval()
        with torch.no_grad():
            m(x)
        q, info = bake_and_convert(m)
        with torch.no_grad():
            out = q(x)
        stats = export_packed(q, info, f"/tmp/_smoke_{level}.qpk")
        codes = count_unique_weight_codes(q)
        q2 = load_packed("mobilenet_v2", 22, level, f"/tmp/_smoke_{level}.qpk", backend)
        with torch.no_grad():
            same = torch.equal(q2(x), out)
        print(f"{level:8s} reload-from-packed identical={same} | out {tuple(out.shape)} | packed {stats['packed_file_bytes']/1e6:.2f} MB "
              f"| avg bits {stats['avg_weight_bits']:.2f} | max unique codes/layer (excl. first/last) "
              f"{max(list(codes.values())[1:-1])}")
