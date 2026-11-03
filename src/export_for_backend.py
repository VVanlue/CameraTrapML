"""
Re-convert already-trained quantized models for a different CPU backend,
WITHOUT retraining. Use it before copying models to a Raspberry Pi (ARM ->
QNNPACK engine); models converted on an Intel/AMD laptop (x86/FBGEMM engine)
do run on ARM, but their activation rounding differs slightly.

    python src/export_for_backend.py --arch mobilenet_v2 --resolution 224 --backend qnnpack \
        --out_dir models_pi

Reads   models/<arch>_<level>_<method>_res<R>.latent.pt   (saved by qat_train.py)
Writes  <out_dir>/ the same file names (.pt, .ts, .qpk, .json) + fp32 files + meta,
        so you can copy <out_dir> to the Pi and point --models_dir at it.
"""

import argparse
import json
import shutil
from pathlib import Path

import torch

from models import load_meta
from quantize import (bake_and_convert, export_packed, export_torchscript, prepare_quant_model,
                      set_backend, _layer_flags)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", required=True)
    p.add_argument("--resolution", type=int, default=224)
    p.add_argument("--backend", default="qnnpack")
    p.add_argument("--models_dir", default="models")
    p.add_argument("--out_dir", default="models_pi")
    a = p.parse_args()

    src, dst = Path(a.models_dir), Path(a.out_dir)
    dst.mkdir(parents=True, exist_ok=True)
    backend = set_backend(a.backend)
    meta = load_meta(src / f"{a.arch}_res{a.resolution}_meta.json")
    nc = len(meta["classes"])
    for f in [f"{a.arch}_res{a.resolution}_meta.json", f"{a.arch}_fp32_res{a.resolution}.pt"]:
        shutil.copy(src / f, dst / f)

    for latent in sorted(src.glob(f"{a.arch}_*_res{a.resolution}.latent.pt")):
        tag = latent.name[: -len(".latent.pt")]
        level = tag.split("_")[-3]
        flags = _layer_flags(src / f"{tag}.pt")
        model = prepare_quant_model(a.arch, nc, None, level, backend, **flags)
        state = torch.load(latent, map_location="cpu", weights_only=True)
        # weight scales are recomputed from the weights at conversion, so skip them
        state = {k: v for k, v in state.items() if "weight_fake_quant" not in k}
        model.load_state_dict(state, strict=False)
        model.apply(torch.ao.quantization.disable_observer)   # keep the trained activation ranges
        qmodel, info = bake_and_convert(model)
        torch.save(qmodel.state_dict(), dst / f"{tag}.pt")
        export_packed(qmodel, info, dst / f"{tag}.qpk")
        export_torchscript(qmodel, dst / f"{tag}.ts", a.resolution)
        report = json.loads((src / f"{tag}.json").read_text())
        report["backend"] = backend
        (dst / f"{tag}.json").write_text(json.dumps(report, indent=2))
        print(f"re-converted {tag} for {backend}")
    print(f"Copy {dst}/ to the device and use --models_dir {dst}")


if __name__ == "__main__":
    main()
