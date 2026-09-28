# CameraTrapML
Machine Learning Systems (CSCE 585)
# Quantization in Wildlife Camera Traps

Pipeline for measuring memory, latency, and accuracy trade-offs when
quantizing MobileNetV2 / ResNet18 wildlife classifiers to INT8 and INT4.

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Get the data

Download ENA24 from LILA BC / Hugging Face. If it arrives as raw images +
a COCO-style annotations JSON, convert it into ImageFolder layout:

```bash
python utils/prepare_ena24.py \
    --images_dir raw/ena24/images \
    --annotations raw/ena24/ena24.json \
    --out_dir data/ena24
```

If your download already comes pre-sorted into class folders, skip this
and point everything at that directory directly.

## 2. Sanity check the pipeline

```bash
python src/models.py      # builds both architectures, runs one dummy forward pass
python src/dataset.py --data_dir data/ena24   # confirms the dataset loads and reports class counts
```

## 3. Fine-tune the FP32 baseline

```bash
python src/train_baseline.py --arch mobilenet_v2 --data_dir data/ena24 --epochs 5
python src/train_baseline.py --arch resnet18 --data_dir data/ena24 --epochs 5
```

Saves checkpoints to `checkpoints/<arch>_fp32_res224.pt`.

## 4. Run a single benchmark (fast sanity check)

```bash
python src/benchmark.py --arch mobilenet_v2 \
    --checkpoint checkpoints/mobilenet_v2_fp32_res224.pt \
    --data_dir data/ena24 --resolution 224 --batch_size 1
```

## 5. Run the full sweep

Start small, then widen:

```bash
python src/run_sweep.py --arch mobilenet_v2 \
    --checkpoint checkpoints/mobilenet_v2_fp32_res224.pt \
    --quant_levels fp32 int8 --batch_sizes 1 --resolutions 224 --trials 1
```

Full grid (matches Section 7 of the proposal):

```bash
python src/run_sweep.py --arch mobilenet_v2 \
    --checkpoint checkpoints/mobilenet_v2_fp32_res224.pt \
    --quant_levels fp32 int8 int4 \
    --batch_sizes 1 4 8 \
    --resolutions 128 224 384 \
    --trials 3
```

Writes `results/<arch>_raw_results.csv` and `results/<arch>_aggregated_results.csv`.

## 6. Generate figures

```bash
python src/plot_results.py --raw_csv results/mobilenet_v2_raw_results.csv --out_dir figures
```

Produces the memory bar chart, latency/throughput vs batch size line
plots, the accuracy table, and the throughput-vs-accuracy scatter plot.

## Important caveat on INT4

Mainstream tools (`torch.quantization`, ONNX Runtime) don't support real
INT4 execution for standard CNNs the way they do for INT8. `src/quantize.py`
implements a **simulated** INT4: weights are quantized to 4 bits and
immediately dequantized back to float32, so accuracy impact is measurable,
but latency/memory numbers under `int4` reflect FP32 compute, not real
4-bit execution. Report the theoretical memory estimate
(`quantize.theoretical_param_memory_bytes`) separately from measured
latency/throughput for this level, and say so explicitly in the report —
this is the INT4 risk our proposal already flagged as a possible
fallback-to-INT8 scenario.

## Repo layout

```
src/
  models.py         # model construction (MobileNetV2 / ResNet18)
  dataset.py         # ENA24 loading, transforms, timing subset
  train_baseline.py  # fine-tuning -> FP32 checkpoint
  quantize.py         # INT8 static quantization + simulated INT4
  benchmark.py        # memory / latency / throughput / accuracy measurement
  run_sweep.py         # full experimental sweep -> CSV
  plot_results.py      # figures from CSV results
utils/
  prepare_ena24.py    # COCO -> ImageFolder conversion
checkpoints/           # fine-tuned FP32 weights (created by train_baseline.py)
results/                # raw + aggregated CSVs (created by run_sweep.py)
figures/                 # PNG figures (created by plot_results.py)
```
