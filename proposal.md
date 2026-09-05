# Quantization in Wildlife Camera Traps

**Measuring memory, latency, and quality trade-offs in compressed image classifiers, is quantization worth it?**

Ben Bush, CS, Technical Lead
Victoria Vanlue, CS, Data Lead

---

## 1. Team and Responsibilities

Both members of the team will build the quantization pipeline, put together the pretrained models, the benchmarking harness, handle the dataset, set up the experiment design, and run the analysis once results start coming in.
Our Technical Lead will be responsible for monitoring the meeting of deadlines for what we build, while our data lead will be responsible for keeping the schedule for completing our experiments and report.

We will meet once a week between classes, plus check in asynchronously whenever something comes up. For keeping track of decisions and who did what, we're using GitHub Issues for discussions and choices along the way, and commit history for the actual work. We will also keep record of milestones through the README.

## 2. Feedback Received and Responses

| Feedback received | Source/date | Team response | Change made or reason not adopted |
|---|---|---|---|
| Review relevant published literature | Class discussion, Sept. 1 | Accepted | Added and discussed relevant papers in Related Work. |

## 3. Problem and Motivation

With all the pictures that camera traps take, it would be a lot of effort for one person to sort through each picture. It would also take a lot of experience for someone to be able to identify each animal. Identification is a slow process, costly, and easy to make mistakes on.

A trained image classifier can do most of that work automatically, but these models often need to run on old laptops or small embedded devices out in the field, sometimes with no internet connection at all. A full precision FP32 model might be too big or too slow to actually be useful in that setting.

In this project we hope to find what combination of memory, speed, and accuracy, will allow a classifier to be deployed. Quantization is one obvious way to shrink a model down and speed it up, but it can also hurt accuracy, and we want to extensively measure that trade off.

We're also in a good position to be testers. We both have access to multiple devices with various amounts of RAM, as well as virtual machines where we can allocate GPUs. So we can test on different machines, as well as work under about the same constraints as people who would actually use this technology.

## 4. Research Questions and Hypotheses

**RQ1:** How much memory do INT8 and INT4 quantization actually save compared to an FP32 baseline for our wildlife classifier?

**RQ2:** How does quantization level affect latency and throughput per image, and does that change depending on batch size?

**RQ3:** How much accuracy and F1 score do we lose going to INT8 or INT4, and does that loss get worse at higher resolutions?

**H1:** INT8 should give solid memory and speed improvements without hurting accuracy much.

## 5. Related Work

**Jacob et al. (2018), "Quantization and Training of Neural Networks for Efficient Integer Arithmetic Only Inference," CVPR 2018.**
This is the paper behind the INT8 scheme that PyTorch's torch.quantization is built on, tested on MobileNet style models for ImageNet and COCO. We're basically borrowing their method as the backbone of our INT8 pipeline. Where we differ is that we're applying it to a wildlife classifier running on regular laptop hardware instead of standard benchmarks, and we're going further by also testing INT4 and sweeping batch size and resolution, which they don't do.

**Hussain et al. (2022), "An IoT System Using Deep Learning to Classify Camera Trap Images on the Edge," Computers (MDPI).**
This paper builds an edge deployed camera trap classifier using MobileNetV2, ResNet18, and a few other architectures, aimed at cutting the cost of wildlife monitoring. It's useful to us because it shows these same model families actually work for low power wildlife classification, which is why we picked them too. The difference is that they're focused on the deployment architecture itself, the device, the networking, while we're specifically isolating what quantization does to the same models, which isn't something they looked at.

**Miao et al. (2019), "Insights and Approaches Using Deep Learning to Classify Wildlife," Scientific Reports.**
This one trains a CNN, VGG16 and ResNet50, on a large camera trap dataset and reports strong accuracy numbers along with an interpretation of what features the network picks up on. We're using this as a rough benchmark for what full precision accuracy on wildlife images should look like, so we have something to compare our accuracy losses against. They don't touch compression or speed at all, which is the whole point of our project.

## 6. Proposed System or Approach

Our proposed system starts with ImageNet-pretrained MobileNetV2 and ResNet18 models. We will fine-tune each model on the ENA24 camera-trap dataset so its output classes match the wildlife labels. We will then create FP32, INT8, and INT4 versions of the fine-tuned models using PyTorch quantization or ONNX Runtime. Each version will run through our benchmarking harness across quantization levels, batch sizes, and image resolutions. We will compare memory use, latency, throughput, accuracy, and F1 score.

| Reused | Our contribution | Out of scope |
|---|---|---|
| ImageNet-pretrained MobileNetV2 and ResNet18 from PyTorch and torchvision | The benchmarking harness itself | Quantization-aware training |
| Standard quantization tools, torch.quantization and ONNX Runtime | A sweep across quantization level, batch size, and resolution to find the best combination | Custom inference kernels |
| ENA24 camera-trap dataset (LILA BC-Hugging Face) | The final analysis comparing efficiency against accuracy | Multi-GPU serving |

Ben is handling model integration and the quantization pipeline. Victoria is handling dataset preparation and leading the analysis. We will both review and interpret the results together.

## 7. Evaluation Plan

Baseline is the unquantized FP32 model. We're using the ENA24 hold-out split for accuracy. We're also using a separate set of 50 to 100 images just for timing latency.

**How the questions map to experiments**

- RQ1 (memory): peak memory measured at each quantization level, FP32, INT8, INT4.
- RQ2 (latency and throughput): measured across quantization level and batch size, 1, 4, and 8.
- RQ3 (accuracy cost): accuracy and F1 measured across quantization level and image resolution, 128x128, 224x224, and 384x384.

**Baselines, ablations, and factors**

Baseline is FP32. We're comparing it against INT8 and INT4 versions of the same architecture. Hardware, software versions, and the dataset all stay fixed across every run. What we're varying is quantization level, batch size, and resolution.

**Metrics that we're tracking**

- Peak memory (GB)
- Per image latency (ms)
- Throughput (images per second)
- Classification accuracy (%) and F1 score

**Hardware and software**

An 8GB RAM laptop and a 16GB RAM PC, both CPU only, no dedicated GPU. This actually matches the kind of low spec field hardware we're trying to test for in the first place. Exact CPU model, OS, and library versions for each machine get logged as part of the reproducibility plan below.

**Trials**

Every configuration runs 3 times on the same hardware and software, and we report latency and throughput as mean plus or minus standard deviation.

**Plots and tables that we're planning to make**

- Bar chart of peak memory by quantization level
- Line plots of latency and throughput against batch size
- Accuracy comparison table across quantization levels and resolutions
- Scatter plot of throughput against accuracy

**What counts as success**

We will count a quantization level as a success if it noticeably cuts memory and/or improves throughput while keeping accuracy within a small margin of the FP32 baseline.

## 8. Expected Deliverables

- Source code for the quantization pipeline and the benchmarking harness, including data loading, timing, and memory profiling
- FP32, INT8, and INT4 versions of MobileNetV2 and ResNet18
- Config files for every experimental condition
- A script that runs the full sweep with one command
- Raw results, per run CSV or JSON logs covering memory, latency, throughput, accuracy, and F1 for all 3 trials of every configuration
- Aggregated results, mean plus or minus standard deviation summary tables
- The final figures, memory bar chart, latency and throughput line plots, accuracy table, and the throughput vs accuracy scatter plot
- A README with setup instructions, how to reproduce everything, and a summary of what we found
- A small demo showing a quantized classifier actually running on our constrained hardware
- The final written report

## 9. Timeline and Milestones

This project runs to the end of semester deadline in early December.

| Week / Dates | Task | Evidence of completion | Owner(s) |
|---|---|---|---|
| Week 1, Sep 3 to 9 | Set up the environment, install PyTorch, torchvision, ONNX Runtime, and quantization tools. Download and check the ENA24 dataset. Load the pretrained models and run one sample FP32 inference. | Environment reproduces from a committed requirements file, sample inference logged as a closed GitHub issue | Ben (tooling), Victoria (dataset) |
| Week 2, Sep 10 to 16 | Run the full ENA24 held out split through the FP32 model to get our baseline. Start the benchmark harness with timing and memory hooks. | Script runs end to end, produces one validated baseline accuracy and F1 result | Ben (harness), Victoria (eval) |
| Week 3, Sep 17 to 23 | Implement INT8 quantization, check that the converted model's outputs look sane compared to FP32 on a small sample, and wire it into the harness. | INT8 model passes a correctness check within tolerance | Ben |
| Week 4, Sep 24 to 30 | Implement INT4 quantization, fix whatever conversion issues come up, extend the harness to sweep batch size and resolution. | INT4 model runs through the harness on at least one configuration | Ben |
| Week 5, Oct 1 to 7 | Run pilot experiments on a small subset of configurations on both machines to check feasibility and timing. Lock in the success margins from Section 7. | One plot shows whether the design actually works or has problems | Team |
| Week 6, Oct 8 to 14 | Run the full sweep for FP32 and INT8, 3 trials each, on fixed hardware and software. | Versioned raw results for FP32 and INT8 committed to the repo | Victoria (execution), Ben (harness support) |
| Week 7, Oct 15 to 21 | Run the full sweep for INT4, 3 trials each, finishing the raw results. | Versioned raw results for every configuration, with an experiment log | Victoria, Ben |
| Week 8, Oct 22 to 28 | Aggregate the data, compute mean plus or minus standard deviation, build the memory bar chart and the latency and throughput line plots. | Aggregated results table and two draft figures | Victoria |
| Week 9, Oct 29 to Nov 4 | Build the accuracy table and the throughput vs accuracy scatter plot, figure out the best configuration overall. | All four planned plots and tables done in draft form | Victoria, Ben |
| Week 10, Nov 5 to 11 | Build the demo and finalize the README. | Demo runs end to end on both of our machines | Ben |
| Week 11, Nov 12 to 18 | Draft the final report using the finished figures and results, review it together. | First full draft shared, both of us have commented on it | Team |
| Week 12, Nov 19 to 25 | Revise based on feedback, clean up the repo, confirm one of us can reproduce a result using only the README. | Reproducibility check passes, final code and report are ready | Team |
| Week 13, Nov 26 to Dec 2 | Handle anything left over, proofread the report, package everything up. | Every deliverable from Section 7 is actually in the final submission | Team |

## 10. Risks and Mitigations

| Risk | Warning sign | Mitigation | Fallback |
|---|---|---|---|
| Experiments aren't efficient with 8GB of RAM with no GPU | Out of memory errors during setup in weeks 1 and 2 | We test memory usage, choose the best neural network, and keep batch sizes small on the less efficient machine | Run the heavier configs on the 16GB PC only, and note the hardware split as a limitation |
| Trouble downloading or accessing the ENA24 dataset | The download fails, is incomplete, or is too large to store | Start the download in week 1 and keep a working subset ready | Use a smaller documented subset and note the limits on external validity |
| INT4 or INT8 conversion fails or produces garbage output for some layers | Conversion errors or NaN outputs in weeks 3 and 4 | Test the conversion pipeline early on a small sample, use per channel quantization where it's supported | Fall back to an INT8 only comparison and report INT4 as something we attempted but couldn't finish |
| The full sweep is just too slow on CPU only consumer hardware | Pilot runs in week 5 take way longer than expected | Cut down the factorial design, fewer batch sizes or resolutions, run long sweeps overnight | Reduce trials per config or narrow down to the most informative subset, with narrower claims in the report |

## 11. Reproducibility Plan

Everything on an 4GB-16GB RAM laptops/PCs. We will be using windows, mac, and linux OS.

All dependency versions, Python, PyTorch, torchvision, ONNX Runtime, will be in a committed requirements file.

Every run will write a small metadata file next to its results with the random seed, library versions, git commit hash, and a timestamp.

We will keep one config file for each experimental condition and commit it to the repo. The commands needed to reproduce will also be included in the README.

We will put results and aggregated summary tables into version control, so that every figure in the final report can be regenerated.

## 12. References

1. Jacob, B., et al. (2018). *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference.* CVPR. https://openaccess.thecvf.com/content_cvpr_2018/html/Jacob_Quantization_and_Training_CVPR_2018_paper.html

2. Hussain, et al. (2022). *An IoT System Using Deep Learning to Classify Camera Trap Images on the Edge.* Computers, 11(1), 13. https://www.mdpi.com/2073-431X/11/1/13

3. Miao, Z., et al. (2019). *Insights and Approaches Using Deep Learning to Classify Wildlife.* Scientific Reports, 9, 8137. https://doi.org/10.1038/s41598-019-44565-w

4. LILA BC. *ENA24-detection camera-trap dataset.* https://lila.science/datasets/ena24detection

5. PyTorch. *Torchvision models documentation.* https://docs.pytorch.org/vision/stable/models.html

6. ONNX Runtime. *Quantize ONNX models.* https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html

_An AI assistant was used to create the weekly plan_
