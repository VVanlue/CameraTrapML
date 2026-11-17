"""Build docs/presentation/CameraTrapML_final.pptx from the results in data/.

    uv sync --extra docs
    python scripts/make_slides.py .

Titles and numbers are filled in from data/results/mobilenet_v2_colab_summary.csv, so re-running this after a
new sweep updates the deck. Speaker notes say who presents each slide. Edit the generated deck freely
(e.g. in Google Slides) and export it to PDF for submission.
"""
import io
import re
import sys
from pathlib import Path

import pandas as pd
import qrcode
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

REPO = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent)
OUT = REPO / "docs/presentation/CameraTrapML_final.pptx"
FIG = REPO / "data/figures/mobilenet_v2"
REPO_URL = "https://github.com/VVanlue/CameraTrapML"

INK, INK2, ACCENT, GREY, BG = (RGBColor(0x0B, 0x0B, 0x0B), RGBColor(0x52, 0x51, 0x4E),
                               RGBColor(0x2A, 0x78, 0xD6), RGBColor(0xE4, 0xE3, 0xDF), RGBColor(0xFC, 0xFC, 0xFB))
ORANGE = RGBColor(0xEB, 0x68, 0x34)

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
W, H = prs.slide_width, prs.slide_height
BLANK = prs.slide_layouts[6]
n_slide = [0]

# ---- numbers straight from the results ---------------------------------------
s = pd.read_csv(REPO / "data/results/mobilenet_v2_colab_summary.csv")
s1 = s[s.batch_size == s.batch_size.min()].set_index("variant")
acc = s1.accuracy
lat = s1.latency_ms_per_image_mean
size = s1.storage_mb
ram = s1.peak_inference_mb_mean


def text(slide, x, y, w, h, body, size=20, color=INK, bold=False, align=PP_ALIGN.LEFT, heads=False):
    """heads=True: lines that are not bullets / numbered items are bold section headings."""
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    lines = body if isinstance(body, list) else [body]
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = line
        head = heads and line and not re.match(r"[•\d]", line)
        r.font.size, r.font.color.rgb, r.font.bold = Pt(size), (ACCENT if head else color), bold or bool(head)
        p.space_after = Pt(size * 0.45)
    return tb


def slide(title, speaker, notes, subtitle=None):
    sl = prs.slides.add_slide(BLANK)
    sl.background.fill.solid()
    sl.background.fill.fore_color.rgb = BG
    n_slide[0] += 1
    text(sl, Inches(0.6), Inches(0.35), Inches(12.1), Inches(1.0), title, 26, INK, True)
    if subtitle:
        text(sl, Inches(0.6), Inches(1.15), Inches(12.1), Inches(0.5), subtitle, 16, INK2)
    bar = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.6), Inches(1.05 if not subtitle else 1.65),
                              Inches(1.2), Emu(38000))
    bar.fill.solid(); bar.fill.fore_color.rgb = ACCENT; bar.line.fill.background()
    text(sl, Inches(12.2), Inches(7.1), Inches(0.9), Inches(0.35), str(n_slide[0]), 11, INK2, align=PP_ALIGN.RIGHT)
    sl.notes_slide.notes_text_frame.text = f"[{speaker}]\n{notes}"
    return sl


def figure(sl, path, x, y, w=None, h=None):
    return sl.shapes.add_picture(str(path), x, y, width=w, height=h)


def takeaway(sl, body):
    box = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.6), Inches(6.45), Inches(12.1), Inches(0.6))
    box.fill.solid(); box.fill.fore_color.rgb = RGBColor(0xEA, 0xF2, 0xFC); box.line.fill.background()
    tf = box.text_frame
    tf.margin_left = Inches(0.2)
    r = tf.paragraphs[0].add_run()
    r.text = "Takeaway: " + body
    r.font.size, r.font.color.rgb, r.font.bold = Pt(17), INK, True


def box(sl, x, y, w, h, label, fill=RGBColor(0xFF, 0xFF, 0xFF), color=INK, size=15, bold=True):
    b = sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    b.fill.solid(); b.fill.fore_color.rgb = fill
    b.line.color.rgb = ACCENT; b.line.width = Pt(1.5)
    tf = b.text_frame
    tf.word_wrap = True
    for i, line in enumerate(label.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); r.text = line
        r.font.size, r.font.color.rgb, r.font.bold = Pt(size if i == 0 else size - 3), color, bold and i == 0
    return b


def arrow(sl, a, b):
    c = sl.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, a.left + a.width, a.top + a.height // 2,
                                b.left, b.top + b.height // 2)
    c.line.color.rgb = INK2; c.line.width = Pt(2)
    c.line._get_or_add_ln().append(
        __import__("lxml.etree", fromlist=["SubElement"]).fromstring(
            '<a:tailEnd xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" type="triangle"/>'))


# ---- 1 title -----------------------------------------------------------------
sl = prs.slides.add_slide(BLANK)
sl.background.fill.solid(); sl.background.fill.fore_color.rgb = BG
n_slide[0] += 1
text(sl, Inches(0.8), Inches(2.0), Inches(11.5), Inches(1.2), "Quantization in Wildlife Camera Traps", 44, INK, True)
text(sl, Inches(0.8), Inches(3.2), Inches(11.5), Inches(1.1),
     "How small can a wildlife-spotting AI model get before it stops being useful?", 24, INK2)
text(sl, Inches(0.8), Inches(4.6), Inches(11.5), Inches(1.2),
     ["Ben Bush  ·  Victoria Vanlue", "CSCE 585 — Machine Learning Systems, Fall 2025"], 20, INK)
sl.notes_slide.notes_text_frame.text = "[Ben] Introduce both team members and the one-line question."

dlat = None

# ---- 2 problem -----------------------------------------------------------------
sl = slide("Camera traps take more photos than people can label", "Ben",
           "Camera traps are motion-triggered cameras used to study wildlife. They take thousands of photos, and "
           "someone has to say which animal is in each one. A trained model can do that automatically, but in the "
           "field it often has to run on a small, cheap computer with no internet. A full-size model can be too big "
           "or too slow for that. Quantization is the usual way to shrink a model - our question is what it costs.")
text(sl, Inches(0.6), Inches(1.5), Inches(7.0), Inches(4.8), [
    "• Labeling wildlife photos by hand is slow and needs experts",
    "• AI models can do it, but in the field they run on small, cheap computers",
    "• Quantization makes models smaller — but does it hurt accuracy, and does it make them faster?",
], 22)
figure(sl, REPO / "data/predictions_slide.jpg" if (REPO / "data/predictions_slide.jpg").exists()
       else REPO / "examples/7458.jpg", Inches(7.9), Inches(1.6), w=Inches(4.8))

# ---- 3 questions + contribution ----------------------------------------------
sl = slide("Our questions, and what we built", "Ben",
           "Three questions from our proposal: size, speed, accuracy. What we built: we tested every level from "
           "full size down to 1 bit, compared two ways of shrinking (with and without retraining), saved real "
           "compressed files so the sizes are measured, and built a working demo.")
text(sl, Inches(0.6), Inches(1.5), Inches(6.0), Inches(4.8), [
    "Questions",
    "1.  How much smaller does the model get?",
    "2.  Does it run faster and use less memory?",
    "3.  How much accuracy do we lose — and can retraining win it back?",
], 20, heads=True)
text(sl, Inches(6.9), Inches(1.5), Inches(5.9), Inches(4.8), [
    "What we built",
    "• Tested 8 sizes, from 32-bit (full) down to 1-bit",
    "• Compared shrinking with and without retraining",
    "• Real compressed model files, so sizes are measured",
    "• A demo app, plus a test on a simulated small device",
], 20, heads=True)

# ---- 4 system design -------------------------------------------------------------
sl = slide("How it works: our experiment vs. the demo app", "Victoria",
           "Two different things on this slide. Top row is the DEMO app only. Bottom row is OUR EXPERIMENT, which is where all of our results come from. Top row: the app. MegaDetector, a free and widely used camera-trap model, draws a box around each animal. "
           "Our model then names the species in each box. We only shrink our model, so the comparison stays fair. "
           "Bottom row: how we shrink it. Quantization means storing each number in the model with fewer bits. "
           "PTQ just rounds the numbers after training. QAT rounds them and then retrains briefly so the model "
           "adapts. Finally we pack the small numbers tightly into a file.")
tag = lambda sl, y, t, c: text(sl, Inches(0.6), y, Inches(12), Inches(0.4), t, 14, c, True)
tag(sl, Inches(1.25), "DEMO APP ONLY — MegaDetector is an off-the-shelf animal finder; it is not part of our measurements", INK2)
y = Inches(1.7)
b1 = box(sl, Inches(0.6), y, Inches(2.2), Inches(1.2), "Camera-trap photo")
b2 = box(sl, Inches(3.4), y, Inches(2.6), Inches(1.2), "MegaDetector\ndraws a box around\neach animal")
b3 = box(sl, Inches(6.6), y, Inches(2.6), Inches(1.2), "Our small model\nnames the species", RGBColor(0xEA, 0xF2, 0xFC))
b4 = box(sl, Inches(9.8), y, Inches(2.9), Inches(1.2), "Labeled photo\n(22 species)")
for a, b in [(b1, b2), (b2, b3), (b3, b4)]:
    arrow(sl, a, b)
tag(sl, Inches(3.45), "OUR EXPERIMENT — all results come from here (animals cut out using the dataset's own labeled boxes)", ACCENT)
y2 = Inches(3.9)
q = [box(sl, Inches(0.6 + i * 3.1), y2, Inches(2.7), Inches(1.5), t, size=15) for i, t in enumerate([
    "1. Train the full model\n(32-bit numbers)",
    "2. Round its numbers\nto fewer bits\n(8, 4, 3, 2, 1)",
    "3. PTQ: stop here\nQAT: retrain briefly\nso it adapts",
    "4. Save a small file\n(bits packed tightly)"])]
for a, b in zip(q, q[1:]):
    arrow(sl, a, b)
text(sl, Inches(0.6), Inches(5.6), Inches(12), Inches(0.7), [
    "Quantization = storing each number in the model with fewer bits.   "
    "PTQ = shrink after training.   QAT = shrink, then retrain."], 15, INK2)

# ---- 5 methodology -------------------------------------------------------------------
sl = slide("How we tested it", "Victoria",
           "Data: the public ENA24 camera-trap dataset, about 9,700 photos, 22 species. We cut out each animal. "
           "Photos taken in the same burst stay together, so the test set never contains near-copies of training "
           "photos. We changed the number of bits, PTQ vs QAT, and how many photos are processed at once. Every "
           "timing test runs in a fresh program and is repeated 3 times. We trained on a Colab GPU but timed "
           "everything on regular CPUs, because that is what field devices have.")
text(sl, Inches(0.6), Inches(1.5), Inches(6.0), Inches(4.8), [
    "Setup",
    "• ENA24 dataset: 9,676 camera-trap photos, 22 species",
    "• Model: MobileNetV2, a small, popular image model",
    "• Fair split: photos from the same burst stay together",
    "What we changed",
    "• Bits per number: 32, 16, 8, 4, 3, 2, 1.58, 1",
    "• PTQ vs QAT,  and photos per batch (1–16)",
], 18, heads=True)
text(sl, Inches(6.9), Inches(1.5), Inches(5.9), Inches(4.8), [
    "What we measured",
    "• Accuracy (% of test animals named correctly)",
    "• File size, memory use, time per photo",
    "How we kept it fair",
    "• Each timing run in a fresh program, repeated 3×",
    "• Timed on CPUs: a Colab server and a simulated 1-core, 1 GB device",
    "• Chart labels: fp32 = 32-bit (full) · int8 = 8-bit · … · 1bit = 1-bit",
], 18, heads=True)

# ---- 5b reproducibility: notebook + results ------------------------------------------------
sl = slide("All results are saved as CSV files and explained in a notebook", "Ben",
           "Everything is public and reproducible. The Colab notebook runs the whole experiment from scratch in about "
           "3 hours on a free GPU. Every measurement is saved as CSV files in data/results. The results walkthrough "
           "notebook loads those CSVs and explains each finding with tables and charts; it needs no GPU and no dataset "
           "and runs in seconds, so we can open it live. Show it now if there is time: "
           "notebooks/results_walkthrough.ipynb")
nb_shot = REPO / "docs/presentation/assets/notebook_screenshot.png"
if nb_shot.exists():
    figure(sl, nb_shot, Inches(0.6), Inches(1.6), w=Inches(5.5))
text(sl, Inches(6.3), Inches(1.4), Inches(6.5), Inches(5.0), [
    "Run the experiment (≈3 h, free Colab GPU)",
    "• notebooks/colab_run.ipynb  →  trains, shrinks and times all 14 models",
    "Results, saved as CSV",
    "• data/results/mobilenet_v2_colab_summary.csv — 56 rows",
    "• data/results/mobilenet_v2_docker_1cpu_1gb_summary.csv",
    "• data/results/accuracy/ — scores for every model and species",
    "Explain the results (runs in seconds, shown live)",
    "• notebooks/results_walkthrough.ipynb  →  tables, charts, findings",
    "• Left: part of it — the file-size table, loaded from the CSV",
], 16, heads=True)

# ---- 6 result: accuracy ----------------------------------------------------------------
sl = slide("With retraining, accuracy holds down to 3 bits; without it, it collapses", "Victoria",
           f"Blue line = QAT (with retraining), orange = PTQ (no retraining). The full model gets {acc['fp32']:.1%}. "
           f"With retraining, 8-, 4- and 3-bit stay within about one point of that (3-bit: {acc['int3-qat']:.1%}). "
           f"Even 1-bit keeps {acc['1bit-qat']:.1%}. Without retraining, 8-bit already drops to "
           f"{acc['int8-ptq']:.1%}, and at 3 bits and below the model is basically guessing. "
           f"One oddity: 2-bit ({acc['int2-qat']:.1%}) did worse than 1.58-bit - we don't know why yet.")
figure(sl, FIG / "01_accuracy_vs_bits.png", Inches(0.6), Inches(1.3), w=Inches(12.1))
takeaway(sl, "below 8 bits, retraining is the difference between ~89% and ~10% accuracy.")

# ---- 7 result: storage ------------------------------------------------------------------
sl = slide(f"The model file can be {size['fp32'] / size['1bit-qat']:.0f}× smaller; 3-bit is the sweet spot", "Ben",
           f"Each dot is one model: further left = smaller file, higher = more accurate. The full model is "
           f"{size['fp32']:.1f} MB. The 3-bit model is {size['int3-qat']:.2f} MB, about "
           f"{size['fp32'] / size['int3-qat']:.0f} times smaller, with the same accuracy. The 1-bit model is "
           f"{size['1bit-qat']:.2f} MB. These are real file sizes, not estimates.")
figure(sl, FIG / "07_accuracy_vs_storage.png", Inches(2.4), Inches(1.3), h=Inches(5.05))
takeaway(sl, f"3-bit with retraining: same accuracy as the full model at {size['int3-qat']:.1f} MB instead of "
             f"{size['fp32']:.1f} MB.")

# ---- 8 result: latency ------------------------------------------------------------------
sl = slide(f"8-bit runs {lat['fp32'] / lat['int8-qat']:.1f}× faster, but going below 8 bits adds no speed", "Ben",
           f"Time per photo on a CPU (lower is better). 8-bit takes {lat['int8-qat']:.0f} ms vs {lat['fp32']:.0f} ms "
           f"for the full model, and uses {ram['int8-qat']:.0f} MB of memory instead of {ram['fp32']:.0f}. "
           "Below 8 bits the lines sit on top of 8-bit: today's software only has fast math for 8-bit numbers, so "
           f"smaller models still run at 8-bit speed. Surprise: 16-bit is {lat['fp16'] / lat['fp32']:.0f} times "
           "slower than the full model, because CPUs have no fast 16-bit math.")
figure(sl, FIG / "04_latency_vs_batch.png", Inches(0.6), Inches(1.3), h=Inches(5.05))
text(sl, Inches(9.8), Inches(1.6), Inches(3.3), Inches(4.5), [
    "Time per photo",
    f"32-bit   {lat['fp32']:.0f} ms",
    f"8-bit     {lat['int8-qat']:.0f} ms",
    f"1-bit     {lat['1bit-qat']:.0f} ms",
    f"16-bit   {lat['fp16']:.0f} ms (!)",
], 17, INK2)
takeaway(sl, "speed and memory gains stop at 8 bits; going lower only makes the file smaller.")

# ---- 9 device comparison (if available) -------------------------------------------------
DEV = REPO / "data/results/mobilenet_v2_docker_1cpu_1gb_summary.csv"
if (FIG / "11_device_comparison.png").exists() and DEV.exists():
    dv = pd.read_csv(DEV)
    d1 = dv[dv.batch_size == dv.batch_size.min()].set_index("variant")
    dlat, dram = d1.latency_ms_per_image_mean, d1.peak_inference_mb_mean
    sl = slide(f"On a small 1-core device, 8-bit is {dlat['fp32'] / dlat['int8-qat']:.1f}× faster", "Victoria",
               "To mimic a small field device, we limited the computer to 1 processor core and 1 GB of memory "
               f"(a Docker container on a MacBook). Full model: {dlat['fp32']:.0f} ms per photo; 8-bit: "
               f"{dlat['int8-qat']:.0f} ms. Memory: {dram['fp32']:.0f} MB vs {dram['int8-qat']:.0f} MB. Same pattern "
               "as before: below 8 bits there is no extra speed. Note: one MacBook core is faster than a real field "
               "board, so real devices would be slower overall.")
    figure(sl, FIG / "11_device_comparison.png", Inches(0.6), Inches(1.3), h=Inches(5.05))
    takeaway(sl, f"on weak hardware 8-bit pays off most ({dlat['fp32']:.0f} → {dlat['int8-qat']:.0f} ms per photo).")

# ---- 10 end to end ---------------------------------------------------------------------------
sl = slide("Demo: in the full app, finding the animal takes most of the time", "Victoria",
           "This slide is the DEMO, not the experiment. In the full app, MegaDetector takes about a second per photo, while our model takes 20-40 milliseconds "
           "at any size. So shrinking only our model barely changes the total time. Accuracy check on our example "
           "photos: the full and 8-bit models got all four right; the 1-bit model called the turkey a red fox. "
           "Live demo if time allows: python run.py app")
imgs = sorted((REPO / "docs/presentation/assets/demo_fp32").glob("*_boxed.jpg"))[:2]
for i, p in enumerate(imgs):
    figure(sl, p, Inches(0.6 + i * 4.2), Inches(1.5), w=Inches(4.0))
text(sl, Inches(9.1), Inches(1.5), Inches(3.7), Inches(4.6), [
    "Time per photo",
    "Finding the animal  ≈ 950 ms",
    "Naming the species  ≈ 20–40 ms",
    "",
    "32-bit and 8-bit: 4 of 4 correct",
    "1-bit: turkey → “red fox”",
], 17)
text(sl, Inches(0.6), Inches(5.75), Inches(8.3), Inches(0.5), "Demo only — the detector (MegaDetector) is off-the-shelf and not part of our measurements.", 14, INK2)
takeaway(sl, "to speed up the whole app, the detector is the next thing to shrink.")

# ---- 11 limitations ---------------------------------------------------------------------------
sl = slide("What we couldn't test, and what's next", "Ben",
           "Be upfront. We used one model and one dataset. Going below 8 bits only saves space because current "
           "software has no faster low-bit math - that's a software limit, not a model limit. We didn't test on a "
           "real field device. The 2-bit result is still unexplained. And we didn't shrink the detector.")
text(sl, Inches(0.6), Inches(1.5), Inches(6.0), Inches(4.8), [
    "Limitations",
    "• One model (MobileNetV2) and one dataset (ENA24)",
    "• Below 8 bits saves space, not time — current software lacks faster math",
    "• Simulated a small device instead of testing a real one",
    "• The 2-bit dip is unexplained",
], 19, heads=True)
text(sl, Inches(6.9), Inches(1.5), Inches(5.9), Inches(4.8), [
    "Next steps",
    "• Try a second model (ResNet18) and dataset",
    "• Use software built for very low-bit math",
    "• Shrink the detector too",
    "• Test on a real Raspberry Pi",
], 19, heads=True)

# ---- 12 conclusion --------------------------------------------------------------------------
sl = slide("Is it worth it? Yes — if you retrain, and 8-bit is enough for speed", "Ben",
           "Answer the three questions, point to the QR code for the code and data, and invite questions.")
text(sl, Inches(0.6), Inches(1.5), Inches(8.3), Inches(4.8), [
    f"Size:  up to {size['fp32'] / size['1bit-qat']:.0f}× smaller; 3-bit is {size['fp32'] / size['int3-qat']:.0f}× smaller with no accuracy loss",
    f"Speed:  8-bit is {lat['fp32'] / lat['int8-qat']:.1f}× faster" + (f" ({dlat['fp32'] / dlat['int8-qat']:.1f}× on a small device)" if dlat is not None else "") + "; fewer bits add no speed",
    "Accuracy:  with retraining, full accuracy down to 3 bits; without it, accuracy collapses",
    "All code, data and results are public — scan the QR code",
], 20)
qr = qrcode.make(REPO_URL)
buf = io.BytesIO(); qr.save(buf, format="PNG"); buf.seek(0)
sl.shapes.add_picture(buf, Inches(9.6), Inches(1.6), width=Inches(3.0))
text(sl, Inches(9.0), Inches(4.7), Inches(4.0), Inches(0.5), REPO_URL.replace("https://", ""), 14, INK2,
     align=PP_ALIGN.CENTER)

# ---- backup -------------------------------------------------------------------------------------
for fname, title in [("08_per_class_f1_heatmap.png", "Backup: accuracy for each species (with retraining)"),
                     ("03_memory_breakdown.png", "Backup: where the memory goes"),
                     ("09_confusion_matrices.png", "Backup: which species get mixed up")]:
    if (FIG / fname).exists():
        sl = slide(title, "either", "Backup slide for questions.")
        figure(sl, FIG / fname, Inches(0.6), Inches(1.3), w=Inches(12.1))
        pic = sl.shapes[-1]
        if pic.height > Inches(5.9):
            ratio = Inches(5.9) / pic.height
            pic.height, pic.width = Inches(5.9), int(pic.width * ratio)

OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(OUT)
print("wrote", OUT, "-", len(prs.slides), "slides")
