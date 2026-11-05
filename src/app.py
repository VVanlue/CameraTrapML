"""
Local web app: drop in a camera-trap photo, get green boxes + species names.

    python src/app.py                     # opens http://127.0.0.1:7860 in your browser
    python src/app.py --detector ena24    # force our own detector
    python src/app.py --share             # temporary public link (e.g. to demo from a Pi)

Everything runs on your machine; nothing is uploaded anywhere (unless --share).
The model dropdown lists every classifier you have trained (fp32, fp16, int8,
int4 ... 1bit), so during the talk you can run the SAME photo through fp32 and
1-bit and show the audience the difference live.
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")   # keep everything local

sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr

from pipeline import CameraTrapPipeline, Classifier, find_classifier_models, make_detector

def build_app(models_dir, detector_kind, backend):
    models = find_classifier_models(models_dir)
    if not models:
        raise SystemExit(f"No trained classifiers found in {models_dir}/. Train one first (README step 3).")
    detector = make_detector(detector_kind, models_dir)
    cache = {}

    def get_classifier(name):
        if name not in cache:
            arch, level, path, res = models[name]
            meta = json.loads((Path(models_dir) / f"{arch}_res{res}_meta.json").read_text())
            cache[name] = Classifier(arch, level, path, res, meta["classes"], backend)
        return cache[name]

    def predict(image, model_name, conf):
        if image is None:
            return None, [], "Upload a photo to start."
        clf = get_classifier(model_name)
        pipe = CameraTrapPipeline(detector, clf)
        annotated, dets, t = pipe.run(image, det_conf=conf)
        rows = []
        for i, d in enumerate(dets, 1):
            alts = ", ".join(f"{s.replace('_', ' ')} {p:.0%}" for s, p in d.top3[1:]) if d.top3 else ""
            rows.append([i, d.species.replace("_", " "), f"{d.species_conf:.1%}", f"{d.det_conf:.2f}", alts])
        if not dets:
            msg = f"No animal found above confidence {conf:.2f}. Try lowering the slider."
        else:
            msg = (f"**{len(dets)} detection(s)** · detector ({detector.name}) {t['detector_ms']:.0f} ms · "
                   f"classifier ({model_name}, {clf.size_mb:.2f} MB) {t['classifier_ms']:.0f} ms")
        return annotated, rows, msg

    with gr.Blocks(title="Camera Trap Classifier") as demo:
        gr.Markdown("# Wildlife camera trap: find & name the animal\n"
                    "Upload a trail-camera photo. The detector boxes each animal and the "
                    "ENA24 classifier names it. Switch the quantization level to compare.")
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(type="pil", label="Camera-trap photo", height=420)
                model_dd = gr.Dropdown(list(models), value=list(models)[0], label="Classifier (quantization level)")
                conf = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Detection confidence threshold")
                btn = gr.Button("Detect animals", variant="primary")
            with gr.Column(scale=1):
                out_img = gr.Image(type="pil", label="Result", height=420)
                status = gr.Markdown()
                table = gr.Dataframe(headers=["#", "species", "confidence", "detector score", "runner-ups"],
                                     label="Detections", wrap=True)
        ex_dir = Path("examples")
        if ex_dir.exists():
            exs = [[str(p)] for p in sorted(ex_dir.glob("*.jp*g"))[:8]]
            if exs:
                gr.Examples(exs, inputs=[inp])
        btn.click(predict, [inp, model_dd, conf], [out_img, table, status])
        inp.upload(predict, [inp, model_dd, conf], [out_img, table, status])
        model_dd.change(predict, [inp, model_dd, conf], [out_img, table, status])
    return demo


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models_dir", default="models")
    p.add_argument("--detector", default="auto", choices=["auto", "megadetector", "ena24", "none"])
    p.add_argument("--backend", default=None, help="quantized engine: x86 | fbgemm | qnnpack")
    p.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to reach it from another device on your network")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true")
    p.add_argument("--no_browser", action="store_true")
    a = p.parse_args()
    demo = build_app(a.models_dir, a.detector, a.backend)
    demo.launch(server_name=a.host, server_port=a.port, share=a.share, inbrowser=not a.no_browser)


if __name__ == "__main__":
    main()
