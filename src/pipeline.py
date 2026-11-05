"""
The end-user pipeline: photo in -> green boxes + species names out.

    Stage 1  DETECTOR   finds every animal (and person/vehicle) in the frame.
    Stage 2  CLASSIFIER our ENA24 MobileNetV2/ResNet18, at whatever quantization
                        level you pick, names the species in each box.

This two-stage design is the standard in camera-trap ML (it is how MegaDetector
+ species classifiers are used in practice), and it keeps the quantization
study clean: the thing we quantize is exactly the classifier.

Detector backends (pick with detector=...):
    "megadetector"  Microsoft/AI for Good MegaDetector via `pip install PytorchWildlife`.
                    Pretrained on millions of camera-trap images; no training needed.
    "ena24"         Our own Faster R-CNN (MobileNetV3) trained on ENA24 boxes with
                    src/train_detector.py. Fully offline, small.
    "auto"          megadetector if installed, else ena24 if trained, else whole-image.
    "none"          classify the whole frame and draw the box around it (fallback).
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from dataset import build_transforms
from quantize import load_model_for_level, set_backend

GREEN = (0, 230, 64)


@dataclass
class Detection:
    box: List[float]               # x1, y1, x2, y2 in pixels
    det_conf: float
    det_class: str                 # animal | person | vehicle
    species: str = ""
    species_conf: float = 0.0
    top3: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------

class MegaDetector:
    NAMES = {0: "animal", 1: "person", 2: "vehicle"}

    def __init__(self, version: Optional[str] = None, device="cpu"):
        from PytorchWildlife.models import detection as pw_detection
        errors = []
        self.model = None
        candidates = [("MegaDetectorV6", version or "MDV6-yolov9-c"), ("MegaDetectorV6", "MDV6-yolov10-c"),
                      ("MegaDetectorV5", None)]
        for cls_name, ver in candidates:
            cls = getattr(pw_detection, cls_name, None)
            if cls is None:
                continue
            try:
                self.model = cls(device=device, pretrained=True, version=ver) if ver else cls(device=device, pretrained=True)
                self.name = f"{cls_name} {ver or ''}".strip()
                break
            except Exception as e:  # weights name differs between PytorchWildlife releases
                errors.append(f"{cls_name}/{ver}: {e}")
        if self.model is None:
            raise RuntimeError("Could not load MegaDetector: " + " | ".join(errors))

    def __call__(self, img: Image.Image, conf: float) -> List[Detection]:
        arr = np.asarray(img.convert("RGB"))
        try:
            res = self.model.single_image_detection(arr, det_conf_thres=conf)
        except TypeError:
            res = self.model.single_image_detection(arr)
        det = res["detections"]
        out = []
        for box, c, k in zip(det.xyxy, det.confidence, det.class_id):
            if float(c) >= conf:
                out.append(Detection([float(v) for v in box], float(c), self.NAMES.get(int(k), "animal")))
        return out


class ENA24Detector:
    def __init__(self, weights="models/detector_ena24.pt", device="cpu"):
        from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
        self.model = fasterrcnn_mobilenet_v3_large_fpn(weights=None, weights_backbone=None, num_classes=2)
        self.model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
        self.model.eval().to(device)
        self.device, self.name = device, "ENA24 Faster R-CNN (ours)"

    @torch.inference_mode()
    def __call__(self, img: Image.Image, conf: float) -> List[Detection]:
        from torchvision.transforms.functional import to_tensor
        out = self.model([to_tensor(img.convert("RGB")).to(self.device)])[0]
        return [Detection(b.tolist(), float(s), "animal")
                for b, s in zip(out["boxes"].cpu(), out["scores"].cpu()) if float(s) >= conf]


class WholeImage:
    name = "none (whole frame)"

    def __call__(self, img, conf):
        w, h = img.size
        return [Detection([0, 0, w, h], 1.0, "animal")]


def make_detector(kind="auto", models_dir="models", device="cpu"):
    errors = []
    if kind in ("auto", "megadetector"):
        try:
            return MegaDetector(device=device)
        except Exception as e:
            errors.append(f"megadetector: {e}")
            if kind == "megadetector":
                raise
    if kind in ("auto", "ena24"):
        w = Path(models_dir) / "detector_ena24.pt"
        if w.exists():
            return ENA24Detector(str(w), device)
        errors.append(f"ena24: {w} not found (run src/train_detector.py)")
        if kind == "ena24":
            raise FileNotFoundError(errors[-1])
    if errors:
        print("[detector] falling back to whole-frame mode:\n  " + "\n  ".join(errors))
    return WholeImage()


# ---------------------------------------------------------------------------
# Classifier (any quantization level)
# ---------------------------------------------------------------------------

def find_classifier_models(models_dir="models"):
    """{display name: (arch, level, path, resolution)} for every model on disk."""
    found = {}
    for meta in sorted(Path(models_dir).glob("*_res*_meta.json")):
        m = json.loads(meta.read_text())
        arch, res = m["arch"], m["resolution"]
        fp = Path(models_dir) / f"{arch}_fp32_res{res}.pt"
        if fp.exists():
            found[f"{arch} fp32 @{res}"] = (arch, "fp32", str(fp), res)
            found[f"{arch} fp16 @{res}"] = (arch, "fp16", str(fp), res)
        for f in sorted(Path(models_dir).glob(f"{arch}_*_*_res{res}.pt")):
            parts = f.stem.split("_")
            # <arch parts>_<level>_<method>_res<R>
            level, method = parts[-3], parts[-2]
            if method in ("qat", "ptq"):
                found[f"{arch} {level} {method.upper()} @{res}"] = (arch, level, str(f), res)
    order = ["fp32", "fp16", "int8", "int4", "int3", "int2", "1.58bit", "1bit"]
    return dict(sorted(found.items(), key=lambda kv: (kv[1][0], kv[1][3], order.index(kv[1][1])
                                                      if kv[1][1] in order else 99, kv[0])))


class Classifier:
    def __init__(self, arch, level, path, resolution, classes, backend=None):
        set_backend(backend)
        self.model = load_model_for_level(arch, len(classes), level, path, backend)
        self.level, self.classes, self.tf = level, classes, build_transforms(resolution)
        self.size_mb = self._size(path, level)

    @staticmethod
    def _size(path, level):
        qpk = Path(path).with_suffix(".qpk")
        if qpk.exists():
            return qpk.stat().st_size / 1e6
        b = Path(path).stat().st_size / 1e6
        return b / 2 if level == "fp16" else b

    @torch.inference_mode()
    def __call__(self, crops: List[Image.Image]):
        if not crops:
            return []
        x = torch.stack([self.tf(c.convert("RGB")) for c in crops])
        if self.level == "fp16":
            x = x.half()
        probs = torch.softmax(self.model(x).float(), 1)
        out = []
        for p in probs:
            top = torch.topk(p, min(3, len(self.classes)))
            out.append([(self.classes[i], float(v)) for v, i in zip(top.values, top.indices)])
        return out


def crop_with_margin(img, box, margin=0.10):
    """Same 10% context margin the classifier saw in training (prepare_ena24.py)."""
    x1, y1, x2, y2 = box
    w, h = img.size
    bw, bh = x2 - x1, y2 - y1
    return img.crop((max(0, int(x1 - bw * margin)), max(0, int(y1 - bh * margin)),
                     min(w, int(x2 + bw * margin)), min(h, int(y2 + bh * margin))))


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _font(size):
    for name in ("DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def draw_detections(img: Image.Image, dets: List[Detection]) -> Image.Image:
    img = img.convert("RGB").copy()
    d = ImageDraw.Draw(img, "RGBA")
    W, H = img.size
    lw = max(3, int(min(W, H) / 200))
    font = _font(max(16, int(min(W, H) / 28)))
    for det in dets:
        x1, y1, x2, y2 = det.box
        d.rectangle([x1, y1, x2, y2], outline=GREEN, width=lw)
        label = det.species.replace("_", " ") if det.species else det.det_class
        text = f"{label}  {det.species_conf:.0%}" if det.species else label
        tb = d.textbbox((0, 0), text, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        tx = min(max(0, x1), W - tw - 8)
        ty = y1 - th - 10 if y1 - th - 10 > 0 else y1 + lw + 2   # above the box, or inside if no room
        d.rectangle([tx, ty, tx + tw + 8, ty + th + 8], fill=(0, 0, 0, 170))
        d.text((tx + 4, ty + 2 - tb[1]), text, fill=GREEN, font=font)
    return img


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

class CameraTrapPipeline:
    def __init__(self, detector, classifier: Classifier):
        self.detector, self.classifier = detector, classifier

    def run(self, img: Image.Image, det_conf=0.25, classify_classes=("animal",), max_dets=15):
        img = img.convert("RGB")
        t0 = time.perf_counter()
        dets = sorted(self.detector(img, det_conf), key=lambda d: -d.det_conf)[:max_dets]
        t1 = time.perf_counter()
        to_classify = [d for d in dets if d.det_class in classify_classes]
        preds = self.classifier([crop_with_margin(img, d.box) for d in to_classify])
        t2 = time.perf_counter()
        for d, top in zip(to_classify, preds):
            d.species, d.species_conf, d.top3 = top[0][0], top[0][1], top
        for d in dets:
            if d.det_class != "animal":
                d.species, d.species_conf = d.det_class.capitalize(), d.det_conf
        timing = {"detector_ms": (t1 - t0) * 1000, "classifier_ms": (t2 - t1) * 1000,
                  "n_boxes": len(dets)}
        return draw_detections(img, dets), dets, timing
