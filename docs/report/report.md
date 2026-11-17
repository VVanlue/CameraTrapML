---
title: "Quantization in Wildlife Camera Traps: Memory, Latency, and Accuracy Trade-offs from FP32 to 1-bit"
author:
  - Ben Bush
  - Victoria Vanlue
date: "CSCE 585 — Machine Learning Systems, Fall 2025"
bibliography: ../references.bib
link-citations: true
---

**Links.**
Code and data: <https://github.com/VVanlue/CameraTrapML> ·
Slides: `docs/presentation/` in the repository ·
Video: `<add YouTube link>` ·
Raw results: `data/results/` · Figures: `data/figures/mobilenet_v2/` ·
Results walkthrough notebook: `notebooks/results_walkthrough.ipynb`

# Abstract

Camera traps produce far more images than people can label, and the classifiers that label them often have to
run offline on small, CPU-only devices. We study how far a wildlife species classifier can be compressed before
it stops being useful. We fine-tune an ImageNet-pretrained MobileNetV2 on animal crops from the ENA24 camera-trap
dataset (22 classes) and quantize it to FP16, INT8, INT4, INT3, INT2, ternary (1.58-bit) and binary (1-bit)
weights, with both post-training quantization (PTQ) and quantization-aware training (QAT). We measure accuracy and
macro-F1 with bootstrap confidence intervals, bit-packed model size, peak inference memory, latency and throughput
in fresh processes. With QAT, accuracy stays at the FP32 level (89.6%) down to 3 bits while the model file shrinks
8.9×, and a ternary model keeps 84.5% at 13.8× smaller; PTQ alone collapses below 4 bits. INT8 runs 2.3× faster
(7.6× on a single ARM core) and uses 2–4× less RAM than FP32, but lower bit-widths give no further speed or memory gain on CPU because PyTorch
only provides 8-bit integer kernels, and FP16 is 19× *slower* than FP32 on CPU. In an end-to-end
detector → classifier pipeline, the detector rather than the classifier dominates latency.

# 1. Introduction

A single camera-trap deployment can produce hundreds of thousands of images, most of which must be sorted by
species before ecologists can use them. Manual labeling is slow, costly and requires expertise, and deep-learning
classifiers can do most of this work [@miao2019insights]. In practice, however, these models frequently need to run
in the field — on old laptops or small embedded boards, often without network access [@hussain2022iot]. A
full-precision (FP32) model may be too large or too slow for such hardware.

Quantization — storing and computing with fewer bits per weight — is the standard tool for shrinking models
[@jacob2018quantization], but its benefits and costs depend on the model, the data and the runtime. We ask:

- **RQ1** How much storage and memory does quantization save compared to FP32?
- **RQ2** How does quantization level affect latency and throughput, and does this depend on batch size?
- **RQ3** How much accuracy and macro-F1 is lost, and how much can quantization-aware training recover?

Our contributions are (i) a systematic sweep from FP32 to 1-bit, comparing PTQ and progressive QAT with knowledge
distillation, on a real camera-trap dataset; (ii) a bit-packed model format that reloads into an identical
integer model, so storage savings are measured rather than estimated; (iii) a fresh-process benchmarking harness
with repeats and confidence intervals; and (iv) an end-to-end detector → quantized-classifier demo and a
constrained (1 CPU, 1 GB) device emulation. All code, data and figures are released.

# 2. Related Work

**Integer quantization.** Jacob et al. [-@jacob2018quantization] introduced the affine INT8 scheme with
quantization-aware training that PyTorch's `torch.ao.quantization` implements, evaluated on MobileNet-style
networks. We use their scheme as our INT8 backbone and extend it to sub-8-bit weights.

**Extremely low-bit weights.** Ternary Weight Networks [@li2016ternary] and XNOR-Net / Binary Weight Networks
[@rastegari2016xnor] constrain weights to {−1, 0, +1} and {−1, +1} with a per-channel scale. We adopt their
quantizers (with per-channel mean-|w| scaling) for our 1.58-bit and 1-bit levels, and use knowledge distillation
[@hinton2015distilling] from the FP32 model during QAT.

**Camera-trap classification.** Miao et al. [-@miao2019insights] show that CNNs reach high accuracy on camera-trap
species classification but do not consider compression. Hussain et al. [-@hussain2022iot] deploy MobileNetV2 and
ResNet18 on edge devices for camera traps, focusing on the device and networking architecture rather than on
quantization. MegaDetector [@beery2019megadetector] is the standard first-stage animal detector; we use it to crop
animals before species classification, mirroring practical camera-trap pipelines.

# 3. System Design

**Pipeline.** All measurements in Section 5 use animal crops cut from the dataset's labeled boxes; the detector
below is used only for the end-to-end demo (Section 5.5). Our system is a two-stage pipeline: MegaDetector V6 finds animals in a photo, and our classifier
names the species of each cropped detection (10% context margin). Only the classifier is quantized, which isolates
the effect of quantization.

**Classifier.** An ImageNet-pretrained MobileNetV2 [@torchvision] with a 22-way head, fine-tuned on ENA24 crops for
8 epochs (AdamW, one-cycle LR 3·10⁻⁴, label smoothing 0.05, inverse-square-root class weights for the imbalanced
classes).

**Quantization.** BatchNorm is folded into the convolutions, and every convolution / linear weight passes through a
per-output-channel symmetric quantizer, with 8-bit fake-quantized activations. For INT2–INT4 the per-channel scale
is chosen by an MSE-optimal clipping search; ternary and binary use the TWN / BWN quantizers. Following common
practice, the first convolution, the classifier and MobileNetV2's depthwise 3×3 convolutions (≈3% of the weights)
stay at 8 bits; the resulting average bit-width is reported (e.g. 1.29 bits/weight for the binary model).

- **PTQ** only calibrates activation ranges on 512 held-out validation images.
- **QAT** fine-tunes with the straight-through estimator and distillation from the frozen FP32 teacher (T = 4,
  α = 0.7), progressively: INT8 → INT4 → INT3 → INT2 → 1.58-bit → 1-bit, each level initialised from the previous
  one (2–10 epochs per level).

The trained model is converted to a true integer model with `torch.ao.quantization.convert`. Its integer weight
codes are then bit-packed into a `.qpk` file (base-3 packing for ternary); we verify that every packed file
reloads into a model with bit-identical outputs.

# 4. Experimental Setup

**Data.** ENA24-detection [@lila_ena24]: 9,676 images with bounding boxes. We crop each annotated animal and
drop the *Human* class, leaving 22 classes (21 animal species plus *Vehicle*). The train / validation / test split
(70 / 15 / 15%) is stratified by class and keeps photo bursts together, so near-duplicate frames never appear on
both sides; the test set has 1,475 crops.

**Variants.** FP32, FP16 and INT8 / INT4 / INT3 / INT2 / 1.58-bit / 1-bit, each with PTQ and QAT — 14 models,
all at 224 × 224 input.

**Metrics.** Top-1 accuracy and macro-F1 on the test set with 95% bootstrap confidence intervals (1,000
resamples) and per-class F1; model file size; peak inference memory (private data memory, sampled every 1 ms);
latency per image and throughput at batch sizes 1, 4, 8 and 16.

**Controls.** Every benchmark configuration runs in a freshly spawned process, with warm-up batches and adaptive
trial counts (5–30 trials, stopping when the standard error falls below 2%), repeated 3 times. Splits use a fixed
seed, and training and QAT are seeded.

**Hardware.** Training and QAT ran on a Google Colab NVIDIA T4 GPU. All timing and memory measurements ran on CPU,
the deployment target: the Colab virtual machine's CPU, and (Section 5.4) a Docker container limited to one CPU
core and 1 GB RAM on an Apple M4 laptop.

# 5. Results

| model | accuracy (95% CI) | macro-F1 | size (MB) | ms / image (batch 1) | peak RAM (MB) |
|---|---|---|---|---|---|
| FP32 | 0.896 ± 0.008 | 0.874 | 9.26 | 43.7 | 39.8 |
| FP16 | 0.896 ± 0.008 | 0.875 | 4.67 | 841.7 | 13.9 |
| INT8 QAT | 0.882 ± 0.008 | 0.859 | 2.37 | 19.3 | 10.0 |
| INT4 QAT | 0.887 ± 0.008 | 0.863 | 1.31 | 16.1 | 10.2 |
| INT3 QAT | 0.894 ± 0.008 | 0.869 | 1.04 | 19.4 | 10.2 |
| INT2 QAT | 0.736 ± 0.011 | 0.654 | 0.78 | 19.2 | 10.3 |
| 1.58-bit QAT | 0.845 ± 0.009 | 0.819 | 0.67 | 14.4 | 10.0 |
| 1-bit QAT | 0.798 ± 0.011 | 0.757 | 0.51 | 19.3 | 10.1 |
| INT8 PTQ | 0.821 ± 0.010 | 0.787 | 2.37 | 19.2 | 10.1 |
| INT4 PTQ | 0.711 ± 0.011 | 0.666 | 1.31 | 14.4 | 10.2 |
| INT3 / INT2 / 1.58 / 1-bit PTQ | 0.04 – 0.12 | < 0.05 | 1.04 – 0.51 | ≈ 19 | ≈ 10 |

: **Table 1.** MobileNetV2 on ENA24 crops, 224 px. Latency and RAM are means over 3 fresh-process repeats on the
Colab CPU (standard deviations in `data/figures/mobilenet_v2/summary_table.md`).

## 5.1 Accuracy (RQ3)

![**Figure 1.** Top-1 accuracy (left) and macro-F1 (right) against weight precision, for QAT (blue) and PTQ
(orange). Error bars are 95% bootstrap confidence intervals over test images. With QAT, accuracy is flat from FP32
down to INT3; PTQ degrades from INT8 and collapses at INT3.](../../data/figures/mobilenet_v2/01_accuracy_vs_bits.png)

With QAT, INT8, INT4 and INT3 stay within about one point of FP32 (89.6%), and their confidence intervals overlap
FP32's. Ternary keeps 84.5% and binary 79.8%. PTQ is markedly worse: it already loses 7.5 points at INT8 and 18.5
at INT4, and falls to near chance (4–12%) at three bits and below. Retraining is therefore not optional below
8 bits: at INT3 it is the difference between 89.4% and 12.4% accuracy.

The PTQ loss at INT8 is larger than often reported. MobileNetV2 has been reported to be sensitive to post-training
quantization, largely because of its depthwise convolutions [@nagel2019datafree]; our calibration set is also small
(512 images). We did not isolate the cause.

INT2-QAT (73.6%) is an unexpected dip below ternary (84.5%). Per-class F1 (`data/figures/mobilenet_v2/08_per_class_f1_heatmap.png`) shows
that much of the loss comes from a few classes: *Horse* drops from 1.00 to 0.13 F1. One hypothesis is the asymmetric 2-bit grid
{−2, −1, 0, 1} compared with ternary's symmetric {−1, 0, +1}, but we did not test it. Across low-bit levels, the
species that lose the most are mostly those that are already hardest for FP32 (e.g. *Woodchuck*, FP32 F1 0.72 →
0.48 at 1-bit), with *Horse* as the notable exception.

## 5.2 Storage (RQ1)

![**Figure 2.** Test accuracy against model file size (log scale). Circles are QAT or float models, squares PTQ.
Up and to the left is better. QAT models dominate PTQ models at every size, and INT3-QAT matches FP32 accuracy at
one ninth of the size.](../../data/figures/mobilenet_v2/07_accuracy_vs_storage.png)

Bit-packed files shrink from 9.26 MB (FP32) to 2.37 MB (INT8, 3.9×), 1.04 MB (INT3, 8.9×), 0.67 MB (ternary,
13.8×) and 0.51 MB (binary, 18.2×). The ratios are below the nominal 32 / *b* because the 8-bit layers, per-channel
scales and biases are kept. INT3-QAT is the best accuracy-per-byte point.

## 5.3 Latency, throughput and memory (RQ1, RQ2)

![**Figure 3.** Latency per image against batch size on the Colab CPU (log scale; mean ± std over 3 fresh-process
repeats). All integer models cluster together, about 2.3× below FP32 at batch 1; FP16 is an order of magnitude
slower than FP32.](../../data/figures/mobilenet_v2/04_latency_vs_batch.png)

INT8 cuts latency at batch 1 from 43.7 to 19.3 ms (2.3×), raises throughput at batch 16 from 26 to 84 images/s,
and reduces peak inference memory from 40 to 10 MB. Going below 8 bits gives **no further** speed or memory
benefit: all integer models run at 14–19 ms and about 10 MB. This is a property of the software stack, not of the
models. PyTorch only has 8-bit integer CPU kernels, so INT4 … 1-bit models run on the INT8 kernels with their
weights restricted to the smaller grid; real gains would require dedicated low-bit kernels. FP16 is 19× slower
than FP32 (842 ms), because CPUs lack fast half-precision convolution kernels: halving the model size is not free
on field hardware. Batching helps little: per-image latency improves by at most about 1.6× from batch 1 to 16, and
not monotonically on the shared 2-vCPU cloud machine (FP32, for example, is fastest at batch 4). The integer models
range from 14 to 19 ms with no systematic trend with bit-width.

## 5.4 Constrained device

To approximate a small field device, we re-timed the same model files in a Docker container limited to one CPU
core (`--cpus=1`, one PyTorch thread) and 1 GB of RAM on an Apple M4 laptop (ARM64 Linux, QNNPACK integer engine).
Accuracy is unchanged, since the models are identical.

![**Figure 4.** Latency per image at batch 1 on the Colab CPU (blue) and in the 1-core / 1 GB container (orange),
log scale, mean ± std over 3 fresh-process repeats. On the constrained ARM device every integer model runs at about
7 ms, 7.6× faster than FP32.](../../data/figures/mobilenet_v2/11_device_comparison.png)

On one core, INT8 runs at 7.1 ms per image versus 53.6 ms for FP32 (7.6×), and peak inference memory falls from
41 to 18 MB. As on Colab, all sub-8-bit models run at the same speed as INT8 (7.1–7.3 ms), and FP16 is again slower
than FP32 (164 ms, 3.1×). At batch 8 the INT8 advantage shrinks to 2.4× (7.3 vs 17.7 ms), because batching helps
FP32 much more than INT8 on a single thread. The INT8 speedup is therefore much larger here than on the x86 Colab
CPU (2.3×), but the comparison changes the instruction set, integer engine and core count at once, so we cannot
attribute the difference to any single factor.

Caveats: one Apple M4 core is considerably faster than a typical field board, so absolute times are optimistic
(the ratios are more transferable). Timing inputs were a fixed synthetic batch of the same shape as the test images,
because the dataset was not on the laptop; CNN latency does not depend on pixel values.

## 5.5 End-to-end pipeline

On an Apple M4 CPU, MegaDetector V6 takes about 950 ms per photo, while the classifier takes 20–40 ms at any
precision. On four example photos, FP32 and INT8-QAT label every animal correctly (88–99% confidence); 1-bit QAT
mislabels a wild turkey as a red fox. End to end, quantizing only the classifier therefore changes total latency
by only a few percent: the detector is the bottleneck.

# 6. Discussion and Limitations

**What the results support.** For storage-constrained deployments, low-bit QAT is clearly worth it: INT3 keeps FP32
accuracy at one ninth of the size, and ternary trades 5 points for 14× compression. For latency and RAM on
today's CPUs, INT8 is the useful level; lower precision only saves storage.

**Limitations.**
(i) One architecture (MobileNetV2) and one dataset (ENA24); we did not run ResNet18 or a second dataset.
(ii) Sub-8-bit models execute on 8-bit kernels, so we cannot measure what native low-bit kernels would achieve.
(iii) Timing was done on a cloud server CPU and a 1-core container on a fast laptop CPU, not on real field boards.
(iv) The INT2 anomaly and the size of the INT8 PTQ loss are not explained.
(v) QAT used a single seed, so run-to-run variance in training is not captured (confidence intervals cover test
sampling only).
(vi) The detector is not quantized, although it dominates end-to-end latency.

**Future work.** Evaluate ResNet18 and other camera-trap datasets; use real low-bit inference kernels (bit-serial or
lookup-table); quantize the detector; and measure on a Raspberry Pi (the repository already includes an ARM export
path).

# 7. Conclusion

Quantization is worth it for camera-trap classifiers, provided the model is retrained. QAT preserves FP32 accuracy
down to 3 bits and enables up to 18× smaller models, while post-training quantization alone fails below 4 bits.
Speed and memory improve only up to INT8 on current CPU software, and in a full pipeline the detector, not the
classifier, limits latency. Everything needed to reproduce these results is available with
`python run.py all` or the accompanying Colab notebook.

# References
