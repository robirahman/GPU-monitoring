# Week 4: Data Collection Round 2 and Exploratory Analysis

## Overview

Week 3 produced a solid dataset: **85 runs across 14 workloads** on A100 SXM4, with Tier 1 (pynvml) + basic DCGM fields at 1 Hz. The collaborator's Week 3 report already demonstrated a binary classifier achieving 100% per-run accuracy and 96.8% per-sample F1 using just GPU utilization CV and SM clock std.

Week 4 focuses on **filling data gaps, deeper exploratory analysis, and building the baseline classifier** as outlined in the project plan.

---

## What Was Completed in Week 3

### Data collected (85 parquet files on A100 SXM4 40GB)
| Category | Workloads | Runs |
|----------|-----------|------|
| Baseline | idle | 10 |
| ML Training (FP32) | pytorch_resnet_cifar10, pytorch_mlp_cifar10, gpt2_wikitext2, bert_sst2 | 7 each |
| ML Training (AMP) | pytorch_resnet_cifar10_amp, gpt2_wikitext2_amp, bert_sst2_amp | 9, 3, 3 |
| ML Inference | resnet50_inference | 7 |
| Scientific HPC | cufft_benchmark, nbody_sim | 6 each |
| Crypto Mining | mining_ethash_proxy | 6 |
| Rendering | rendering_proxy, blender_bmw | 6, 3 |

### Infrastructure built
- Telemetry pipeline: `collect_telemetry.py` (Tier 1 + basic DCGM), `run_workload.py` (orchestrator)
- 16 workloads in registry (14 collected, 2 require special hardware)
- Data exploration script: `scripts/explore_data.py`
- Visualization tools: `analysis/visualize_telemetry.py`
- Classifier scaffold: `classifier/temporal_classifier.py`
- Collaborator's Week 3 report with preliminary findings: `analysis/Week3_Report.md`

### Key findings from Week 3 analysis
- **GPU utilization CV (coefficient of variation)** is the single most discriminating feature
- Training workloads show high variability (CV > 30%) due to data loading pauses, epoch boundaries
- Inference/HPC/mining show sustained, stable utilization (low CV)
- SM clock std is the second most useful feature
- Binary rule: `GPU util CV > 30% AND SM clock std > 150 MHz` → Training (works across all 11 initial workloads)

### What we couldn't collect
- **DCGM profiling fields** (tensor_active, fp16/32/64 pipes, SM occupancy, DRAM bandwidth): blocked by Vast.ai virtualization
- **Tier 3 NCU**: blocked (RmProfilingAdminOnly=1)
- **FFmpeg NVENC**: A100 lacks hardware video encoder — only 1 invalid run
- **GROMACS**: installed build has no GPU support (see `cloud_provider_notes.md`)

---

## Step 1: Fill Remaining Data Gaps

### 1a. Edge-case workloads (high priority)

The project plan calls for edge cases that stress the classifier. These are not yet collected:

- **Mixed-precision BERT and GPT-2 with different batch sizes**: We have AMP variants but only at default batch sizes. Try `--batch-size 16` and `--batch-size 64` for GPT-2, `--batch-size 8` and `--batch-size 64` for BERT, to test whether batch size affects the telemetry signature enough to confuse the classifier.

- **DataLoader-bottlenecked training**: Run a training workload with `num_workers=0` to create a CPU-bottlenecked pattern (low GPU util with periodic spikes). This is a realistic scenario that the classifier must handle.

- **Short training runs**: Run GPT-2/BERT for only 50 steps instead of 300-400. Tests whether the classifier works with limited observation windows.

- **Mixed workloads**: Run training + inference simultaneously on the same GPU (e.g., background inference while training). This tests the classifier on overlapping signatures.

### 1b. Tier 2 DCGM profiling fields (medium priority)

Test whether DCGM profiling fields work on a different provider:

1. **Rent a 1-hour A100 on RunPod** (~$1.14) and test:
   ```bash
   dcgmi dmon -e 1004 -c 5  # tensor core utilization
   ```
2. If it works, run 1 collection pass per workload category (training, inference, HPC, mining, rendering) to get tensor core / pipe utilization data.
3. If RunPod also blocks profiling fields, try Massed Compute.

### 1c. Tier 3 NCU (low priority for Week 4)

Defer full NCU collection to a later week. If time permits, do a 1-hour test on Massed Compute to confirm NCU access.

---

## Step 2: Exploratory Data Analysis

The project plan specifies these EDA tasks for Week 4. Use `scripts/explore_data.py` and `analysis/visualize_telemetry.py` as starting points.

### 2a. Time-series plots (per workload category)

Generate time-series plots for each metric across workload types. Focus on:
- GPU utilization over time (shows epoch periodicity for training)
- Memory usage trajectory (training grows then stabilizes; inference is flat)
- Power draw patterns (training has more variance)
- SM clock behavior (training shows more transitions)

Save plots to `analysis/plots/`.

### 2b. Summary statistics table

For each workload, compute:
- Mean, std, CV, min, max for all numeric metrics
- Autocorrelation at lags 1, 5, 10, 30 seconds
- Dominant FFT frequency (captures epoch periodicity)

This becomes the basis for the **signal comparison table** (Deliverable 4 in the project plan).

### 2c. PCA / t-SNE visualization

Compute per-run aggregate features (mean, std, CV for each metric → ~45 features per run) and visualize:
- PCA with 2 components, colored by workload category
- t-SNE with perplexity=10-30, colored by workload category
- These plots show whether workload classes are naturally separable in feature space

### 2d. Correlation matrix

Plot a heatmap of pairwise correlations between all metrics. Identify:
- Highly correlated features (candidates for removal)
- Features with low correlation to workload type (less useful for classification)

---

## Step 3: Build the Baseline Classifier

The scaffold exists in `classifier/temporal_classifier.py`. This week, make it functional and evaluate it.

### 3a. Feature extraction pipeline

The `TemporalFeatureExtractor` class already computes:
- Aggregate stats (mean, std, min, max, CV) for each metric
- Autocorrelation at lags [1, 2, 5, 10, 20, 50]
- Rolling window variance
- Memory trajectory slope
- Power CV, utilization duty cycle

Verify these features work on the full dataset and add:
- **Dominant FFT frequency** for GPU utilization (captures epoch periodicity)
- **Memory growth rate** (MB/min over the run — key training-specific feature)
- **Power draw during high-util vs. low-util windows** (proxy for forward/backward pass asymmetry)

### 3b. Train and evaluate classifiers

Using the 85-run dataset:

1. **Split by run** (not by time window) to prevent data leakage: 70/15/15 train/val/test
2. Train: Random Forest (already scaffolded), XGBoost, SVM, Logistic Regression
3. Evaluate three classification tasks:
   - **Binary**: ML training vs. everything else
   - **Three-way**: ML training vs. ML inference vs. non-ML
   - **Full multi-class**: all 14 workload labels (or 7 categories)
4. Report: accuracy, per-class precision/recall/F1, confusion matrices
5. Extract and plot feature importance from tree models

### 3c. Window-based classification

In addition to per-run classification, test sliding-window classification:
- Extract features from 30s, 60s, 120s windows within each run
- Train classifiers on windowed features
- Report accuracy vs. window size (this informs minimum detection time)

---

## Step 4: Write the Signal Comparison Table (Draft)

Start Deliverable 4: a table mapping workload types to their telemetry signatures.

| Metric | Training (FP32) | Training (AMP) | Inference | HPC (cuFFT) | HPC (N-body) | Mining | Rendering | Idle |
|--------|----------------|----------------|-----------|-------------|--------------|--------|-----------|------|
| GPU util mean | | | | | | | | |
| GPU util CV | | | | | | | | |
| Mem used mean | | | | | | | | |
| Power mean | | | | | | | | |
| SM clock std | | | | | | | | |
| ... | | | | | | | | |

Fill in observed values from the EDA. Highlight the metrics that best distinguish each category.

---

## Step 5: One-Page Summary

Write a 1-page summary answering: **"Which metrics distinguish ML training from inference, and from non-ML workloads, at Tier 1 resolution?"**

This is called for in the project plan and should reference:
- The Week 3 finding that GPU util CV + SM clock std are sufficient for binary classification
- Whether temporal features (autocorrelation, FFT periodicity) add discriminative power
- Which workload pairs are hardest to distinguish (likely training vs. inference, or training vs. rendering)
- What Tier 2 metrics (if obtainable) would add

---

## Deliverables Checklist

- [ ] Edge-case workloads collected (different batch sizes, DataLoader-bottlenecked, short runs)
- [ ] DCGM profiling fields tested on RunPod (1-hour test)
- [ ] Time-series plots for all workload categories saved to `analysis/plots/`
- [ ] Summary statistics table and correlation matrix
- [ ] PCA/t-SNE visualization of workload feature space
- [ ] Baseline classifier trained and evaluated (binary, 3-way, multi-class)
- [ ] Feature importance ranking produced
- [ ] Window-size sensitivity analysis (30s, 60s, 120s)
- [ ] Draft signal comparison table started
- [ ] 1-page summary: "Which metrics distinguish ML training?"

## Budget Estimate

| Item | Cost |
|------|------|
| Week 3 Vast.ai (2x A100, ~8 hrs) | ~$12 |
| RunPod DCGM test (1 hr A100) | ~$1.50 |
| Edge-case collection (if on new instance, ~2 hrs) | ~$2-3 |
| **Week 4 running total** | **~$15-17 of $1,000 budget** |

Most budget remains reserved for Weeks 6-8 adversarial experiments.

---

## Notes on What Comes After Week 4

Per the project plan:
- **Week 5**: Feature engineering (sliding windows, ~120 features), train RF/XGBoost/SVM/LR classifiers, target >85% binary accuracy
- **Week 6**: Add cross-metric features, test window sizes, train 1D CNN/LSTM, Tier 1 vs. Tier 3 accuracy comparison, cross-GPU generalization
- **Week 7**: Design and implement 5 adversarial disguise strategies
- **Week 8**: Adversarial robustness testing, 2 rounds of adversary-defender iteration

The classifier work started in Week 4 feeds directly into Week 5's formal feature engineering and classifier evaluation.
