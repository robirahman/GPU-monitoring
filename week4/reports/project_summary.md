# GPU Telemetry Fingerprinting — Project Summary
**Platform:** 2× NVIDIA H100 80GB HBM3, NVLink 4.0 (NV6) | CUDA 12.8, PyTorch 2.10
**Date:** 2026-03-16

---

## What Was Done

Six experiments were run to characterise, classify, and adversarially probe GPU workloads using only hardware telemetry (pynvml, 100–200 ms polling). All training used 2-GPU DDP via torchrun with AMP (FP16).

---

## Tests / Experiments

| # | Script | What it sweeps | Key variable |
|---|--------|---------------|-------------|
| 1 | `scale_to_bottleneck.py` — batch sweep | batch size 1 → 1024 | GPU occupancy |
| 2 | `scale_to_bottleneck.py` — width sweep | model width n_ch 8 → 512 | arithmetic intensity |
| 3 | `scale_dataset.py` | dataset size 256 → 1 M samples | total NVLink traffic |
| 4 | `scale_model_accuracy.py` | n_ch 8 → 512 + test accuracy | accuracy vs hardware cost |
| 5 | `edge_cases.py` | 6 adversarial workloads | classifier confusion |
| 6 | `run_classifiers.py` | RF / SVM / LR on 80 runs | telemetry classification |

---

## Main Findings

**1. Batch size changes occupancy, not arithmetic intensity.**
AI is fixed by model channel width. Increasing batch 1→1024 raises TFLOPS from 0.44 → 128.5 (GPU fills up) but AI stays at ~330 FLOP/B for n_ch=64. NVLink allreduce cost stays constant at 0.15 ms (18 MB / 124 GB/s) regardless of batch.

**2. Model width drives all three performance regimes.**

| n_ch | Regime | AI | TFLOPS | NVLink comm% |
|------|--------|----|--------|-------------|
| 8 | memory-bound | 45 | 0.4 | 0% |
| 64 | compute-bound (FP32) | 330 | 32 | 5% |
| 128 | compute-bound (FP16) | 595 | 110 | 16% |
| 256 | NVLink-bound begins | 990 | 242 | 32% |
| 512 | NVLink-bound | 1483 | 347 | 46% |

**3. Dataset size has zero effect on per-step hardware profile.**
With n_ch=128 fixed, every step looks identical (AI=595, TFLOPS=110, comm=16%) regardless of dataset size. Only cumulative NVLink traffic scales: 0.3 GB (256 samples) → 1.15 TB (1 M samples). Useful for billing/auditing total compute.

**4. Accuracy peaks exactly at the NVLink bottleneck onset.**
n_ch=128 is Pareto-optimal: 24.8% accuracy, 6.3 ms/step, 3.92 %acc/ms. n_ch=256 drops to 23.2% (32% NVLink drag), n_ch=512 collapses to 10.4% (severe overfitting: 287 M params, 20 K training examples per GPU).

**5. Telemetry classifiers are highly accurate.**

| Task | Classes | Random Forest | Logistic Reg |
|------|---------|--------------|-------------|
| Binary (training vs rest) | 2 | **100.0%** | **100.0%** |
| Three-way | 3 | **100.0%** | **100.0%** |
| Full workload label | 15 | **95.6%** | 92.6% |
| Category | 7 | 98.75% | 98.75% |
| Sliding window 30 s | — | **100.0%** | **100.0%** |

80 runs, 21 617 telemetry rows, 73 features per run.

**6. Six adversarial edge cases reveal the classifier's weak points.**

| Case | Attack | Closest confusion | L2 distance |
|------|--------|------------------|-------------|
| EC-1 Phantom Train | inference + fake allreduce (NVLink present, no backward) | Baseline Train | 2.22 |
| EC-2 Silent Train | training with `no_sync` (no NVLink, backward present) | EC-3 | 0.79 |
| EC-3 Sparse Sync | grad accum ×16 (allreduce hidden in noise) | EC-2 | 0.79 |
| EC-4 Mining-Like | large-batch inference mimicking GPU-bound mining | all others | >7.0 |
| EC-5 Frozen Backbone | head-only backward (tiny NVLink, short bwd) | Baseline Inference | **0.86** |
| EC-6 Low-Intensity | tiny model + small batch (all features in inference range) | EC-2 | 1.06 |

EC-5 is the hardest case: L2=0.86 from true inference, indistinguishable by util, power, or NVLink alone.
EC-4 is the easiest to detect: power=220 W vs 100 W baseline, util=20% vs 2%, completely isolated.
No single feature distinguishes all cases — the joint vector (fwd_ms, bwd_ms, allreduce_ms, NVLink_MB) is the minimum required.

---

## Directory Map

```
week4/
├── Scripts
│   ├── scale_to_bottleneck.py     Exp 1+2: batch and width sweeps
│   ├── scale_dataset.py           Exp 3: dataset size sweep
│   ├── scale_model_accuracy.py    Exp 4: accuracy vs hardware cost
│   ├── edge_cases.py              Exp 5: 6 adversarial edge cases
│   ├── make_scaling_plots.py      Improved roofline plots (fwd/bwd split)
│   ├── run_classifiers.py         Exp 6: sklearn classifier suite
│   ├── run_nvlink_tests.py        NVLink bandwidth/latency benchmarks
│   ├── ddp_training_characterize.py  DDP step-timing characterisation
│   ├── run_eda.py                 Exploratory data analysis
│   └── workloads_w4.py            Workload collection harness
│
├── results/
│   ├── scaling/                   Exp 1+2 CSVs and JSON
│   ├── dataset_scale/             Exp 3 per-config result + telemetry parquet
│   │   └── n{256..1048576}/
│   ├── model_accuracy/            Exp 4 per-config result + epoch curves
│   │   └── nch{8..512}/
│   ├── edge_cases/                Exp 5 per-case telemetry JSON + worker results
│   │   ├── {CASE}_telemetry.json
│   │   └── {CASE}_worker_result.json
│   ├── ddp/                       DDP characterisation parquets + CSV
│   ├── nvlink/                    NVLink benchmark CSVs
│   ├── classifier_results.csv     All classifier scores
│   └── run_features.csv           73-feature vectors for all 80 runs
│
├── plots/  (54 PNG files)
│   ├── scaling_roofline_combined.png   All configs, fwd▲ bwd■, NVLink arrows
│   ├── scaling_roofline_regimes.png    Width-sweep zoom + step-breakdown bars
│   ├── scaling_step_breakdown.png      Batch + width sweeps timing
│   ├── dataset_scale_roofline.png      Dataset sweep (constant per-step AI)
│   ├── dataset_scale_nvlink.png        Total NVLink traffic vs dataset size
│   ├── dataset_scale_timing.png        Step timing across dataset sizes
│   ├── dataset_scale_telemetry.png     pynvml traces across dataset sizes
│   ├── model_accuracy_roofline.png     Roofline coloured by test accuracy
│   ├── model_accuracy_vs_size.png      Accuracy + TFLOPS + NVLink% vs params
│   ├── model_accuracy_curves.png       Loss/accuracy per epoch for all widths
│   ├── model_accuracy_tradeoff.png     Step breakdown + Pareto scatter
│   ├── edge_cases_telemetry.png        pynvml time-series for all 8 cases
│   ├── edge_cases_feature_scatter.png  Feature-space scatter (util/power/NVLink)
│   ├── edge_cases_confusion.png        L2 distance matrix between cases
│   ├── edge_cases_step_breakdown.png   fwd/bwd/allreduce bars per case
│   ├── cm_*.png                        Confusion matrices (RF/SVM/LR × 4 tasks)
│   ├── feat_importance_*.png           Feature importance per task
│   ├── classifier_summary.png          Accuracy bar chart all tasks/models
│   ├── sliding_window_accuracy.png     Accuracy vs window length
│   ├── pca_projection.png / tsne_projection.png  Workload clusters
│   └── (ddp, nvlink, eda plots …)
│
├── reports/
│   ├── comprehensive_report.md    Full technical reference (all 6 experiments)
│   ├── full_analysis_report.md    Plain-language guide (Parts 0-12)
│   ├── scaling_bottleneck.md      Exp 1+2 report
│   ├── dataset_scale_report.md    Exp 3 report
│   ├── model_accuracy_report.md   Exp 4 report
│   ├── edge_cases_report.md       Exp 5 adversarial analysis
│   └── project_summary.md         ← this file
│
└── paper/
    └── paper.md                   Academic paper (conference submission draft)
```

---

## Key Numbers at a Glance

| Metric | Value |
|--------|-------|
| H100 FP16 peak | 1979 TFLOPS |
| H100 FP32 peak | 67 TFLOPS |
| HBM3 bandwidth | 3350 GB/s |
| NVLink 4 measured | 124 GB/s |
| Best achieved TFLOPS | 347 (n_ch=512, 17.5% of FP16) |
| Binary classifier accuracy | 100% (RF + LR) |
| 15-class classifier accuracy | 95.6% (RF) |
| Best accuracy / Pareto point | n_ch=128, 24.8%, 6.3 ms/step |
| Closest adversarial pair | EC-5 ↔ Inference, L2=0.86 |
| Total telemetry rows | 21 617 across 80 runs |
| Total plots | 54 PNG files |
