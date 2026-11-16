# CameraTrapML
## Quantization in Wildlife Camera Traps
*Measuring memory, latency, and accuracy trade-offs in compressed image classifiers — is quantization worth it?*

CSCE 585 — Machine Learning Systems (Fall 2025)

## Group Info
- Ben Bush — Technical Lead
  - Email: `<add @email.sc.edu>`
- Victoria Vanlue — Data Lead
  - Email: `<add @email.sc.edu>`

## Project Summary/Abstract
Camera traps produce far more images than people can label, and the models that
label them often have to run offline on old laptops or small edge devices. We
fine-tune ImageNet-pretrained MobileNetV2 / ResNet18 classifiers on the ENA24
camera-trap dataset (22 classes) and quantize them from FP32 down to FP16, INT8,
INT4, INT3, INT2, ternary (1.58-bit) and binary (1-bit) weights, using both
post-training quantization (PTQ) and quantization-aware training (QAT). For every
level we measure accuracy / macro-F1, on-disk size, peak inference RAM, latency
and throughput across batch sizes, and we ship a detector → classifier demo that
runs any of the quantized models on real camera-trap photos.

## Problem Description
- **Problem:** find the precision level at which a wildlife species classifier
  becomes small and fast enough for field hardware without losing the accuracy
  that makes it useful.
- **Motivation**
  - Manual identification of camera-trap images is slow, costly and error-prone.
  - Field devices are often offline and memory/CPU-constrained; an FP32 model may not fit or be fast enough.
  - Quantization is the standard compression tool, but its accuracy cost on small, imbalanced wildlife datasets is not well measured.
- **Challenges**
  - Accuracy collapses below 8 bits without QAT (MobileNetV2's depthwise filters have only 9 weights each).
  - Mainstream runtimes only have INT8 CPU kernels, so sub-8-bit models need a custom bit-packed storage format and run on INT8 kernels.
  - Memory and latency measurements are noisy; each config runs in a fresh process with repeats and adaptive trial counts.

## Contribution
### [`Extension of existing work`]
We use the integer-arithmetic quantization scheme of Jacob et al. (2018) and the
camera-trap model families of Hussain et al. (2022), and extend them as follows:
- A systematic sweep from FP32 to 1-bit, PTQ vs. QAT (progressive, with knowledge distillation), across batch size and resolution on ENA24.
- A real bit-packed n-bit model format (`.qpk`) that reloads into an identical integer model, so storage savings are measured rather than estimated.
- An end-to-end detector → quantized classifier demo (web app + headless CLI + 1-CPU/1 GB Docker "field device").

## References
BibTeX in [`docs/references.bib`](docs/references.bib). The proposal (P0) is in [`docs/proposal.md`](docs/proposal.md).

---

## Dependencies
- Python 3.10+ (developed and tested with 3.12)
- PyTorch ≥ 2.1, torchvision ≥ 0.16 (CPU is enough for inference and the sweep; a GPU is recommended for training/QAT)
- macOS, Linux or Windows. Quantized engine is chosen automatically (`qnnpack` on ARM/Apple Silicon, `x86`/`fbgemm` on Intel/AMD).
- Full list: [`pyproject.toml`](pyproject.toml) / [`uv.lock`](uv.lock) (or [`requirements.txt`](requirements.txt) for pip).

```bash
uv sync                                  # creates .venv from uv.lock
uv sync --extra megadetector             # optional: MegaDetector for the demo
# pip alternative: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

## Directory Structure
```
|- run.py                     # single entry point: demo / smoke / all / individual stages
|- result.py                  # plots data/results/ -> data/figures/
|- src/
|   |- models.py              # MobileNetV2 / ResNet18 construction (float + quantizable)
|   |- dataset.py             # ENA24 ImageFolder loading, burst-aware grouped splits
|   |- prepare_ena24.py       # raw LILA ENA24 -> class folders (crops / full frames) + boxes
|   |- train_baseline.py      # FP32 fine-tuning -> models/
|   |- quantize.py            # n-bit weight quantizer, int conversion, bit packing
|   |- train_quantized.py     # PTQ / QAT for int8 ... 1bit -> models/
|   |- benchmark.py           # fresh-process latency / throughput / memory measurement
|   |- run_sweep.py           # full experiment -> data/results/
|   |- plot_results.py        # figures + tables -> data/figures/
|   |- pipeline.py            # detector -> classifier pipeline (green boxes + species)
|   |- predict.py             # headless pipeline for folders of photos
|   |- app.py                 # local Gradio web demo
|   |- train_detector.py      # optional: our own ENA24 animal detector
|   |- export_for_backend.py  # re-convert quantized models for ARM (Raspberry Pi)
|- data/
|   |- results/               # sweep CSVs + per-model accuracy JSON (tracked)
|   |   |- preliminary/       # P1 preliminary sweep (earlier pipeline version)
|   |- figures/               # report figures + summary tables (tracked)
|   |   |- preliminary/
|   |- raw/, ena24*/          # dataset (not tracked; see "How to Run")
|- models/                    # shipped FP32 baseline + metadata; quantized models are generated here
|- examples/                  # sample ENA24 photos for the demo
|- notebooks/colab_run.ipynb  # run the full experiment on a Colab GPU
|- scripts/smoke_test.py      # 5-minute end-to-end check on synthetic data
|- docs/                      # proposal, references
|- Dockerfile.constrained     # 1-CPU / 1 GB "field device" emulation
|- pyproject.toml, uv.lock, requirements.txt
```

## How to Run
Every stage is `python run.py <stage> [options]`; `python run.py <stage> --help`
shows that stage's options.

**0. Check the install (≈5 min, no dataset needed)**
```bash
python run.py smoke     # trains / quantizes / benchmarks / plots on a tiny synthetic dataset
python run.py demo      # classifies examples/ with the shipped FP32 baseline -> data/predictions/
```

**1. Download the dataset** — [ENA24-detection on LILA BC](https://lila.science/datasets/ena24detection):
put the images in `data/raw/images/` and the annotations at `data/raw/ena24.json`.

**2. Run the whole pipeline**
```bash
python run.py all --images_dir data/raw/images --annotations data/raw/ena24.json --device cuda
```
This runs, in order (each can also be run alone):
```bash
python run.py prepare  --images_dir data/raw/images --annotations data/raw/ena24.json   # -> data/ena24_crops, data/ena24
python run.py train    --arch mobilenet_v2 --epochs 8                                   # -> models/mobilenet_v2_fp32_res224.pt
python run.py quantize --arch mobilenet_v2 --method qat --fp32_checkpoint models/mobilenet_v2_fp32_res224.pt
python run.py quantize --arch mobilenet_v2 --method ptq --fp32_checkpoint models/mobilenet_v2_fp32_res224.pt
python run.py sweep    --arch mobilenet_v2 --methods qat ptq --batch_sizes 1 4 8 16 --repeats 3
python result.py                                                                        # -> data/figures/
```
Note: `train` overwrites the shipped baseline in `models/` with a newly trained,
crop-based one (the shipped model was trained on whole frames — see
`models/mobilenet_v2_res224_meta.json`). Use `--arch resnet18` for the second architecture
and `--resolutions 128 224 384` (after training at those resolutions) for the resolution study.

**On Google Colab instead:** open [`notebooks/colab_run.ipynb`](https://colab.research.google.com/github/VVanlue/CameraTrapML/blob/colab/notebooks/colab_run.ipynb),
select a GPU runtime and `Run all`. It downloads ENA24, runs every stage, saves progress to Google Drive
and gives you a zip of `data/results/`, `data/figures/` and `models/` to unzip into the repo.

**3. Constrained "field device" run (optional)**
```bash
docker build -f Dockerfile.constrained -t cameratrap-edge .
docker run --rm --cpus=1 --memory=1g -v "$(pwd)/models:/app/models" \
    -v "$(pwd)/data/results:/app/data/results" cameratrap-edge
python result.py        # adds the device-comparison figure
```

**Reproducibility:** data splits are deterministic (stratified, burst-grouped, seed 42);
training and QAT use `--seed` (default 0); every benchmark config runs in a fresh process.

## Results
Preliminary (P1) sweep with an earlier version of the pipeline — MobileNetV2, 224 px,
PTQ only, sub-8-bit levels simulated (`data/results/preliminary/`):

| level | accuracy | macro-F1 |
|---|---|---|
| fp32 | 0.987 | 0.983 |
| int8 | 0.980 | 0.975 |
| int4 | 0.167 | 0.090 |
| int2 | 0.081 | 0.017 |
| 1bit | 0.034 | 0.003 |

INT8 PTQ is nearly lossless, but PTQ collapses below 8 bits — the motivation for the
QAT pipeline. The final sweep (`python run.py all ...`) writes the full results and
figures to `data/results/` and `data/figures/`.

## Demo
- Video presentation: `<add YouTube link>`
- Live demo: `python run.py app` opens a local web app; choose any trained model
  (fp32 … 1bit) from the dropdown and drop in a photo from `examples/`.
