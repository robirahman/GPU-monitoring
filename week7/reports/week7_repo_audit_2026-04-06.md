# Week 7 Repository Audit and Findings

**Date:** 2026-04-06
**Scope:** `week7/` in `Tajdari-S/Telemetry`

## Executive summary

This audit reviewed the Week 7 pipeline, result tables, plots, and generated reports. The repository clearly contains a substantial B200-focused experiment suite, but it also contains several internal inconsistencies between script headers, constants, output paths, and conclusions.

The strongest technical findings are:

1. **The hardware/transport characterization is strong and internally consistent.** B200 NVLink measurements reach **726.78 GB/s** unidirectional and **1429.61 GB/s** bidirectional, with minimum RTT around **23.42 µs**. Single-GPU to 2-GPU throughput improves from **29,008 img/s** to **44,372 img/s** in the published Week 7 table. These values are coherent across the CSV outputs and the generated comparison report.
2. **The single-GPU classifier result is weak, not strong.** The published binary classifier accuracy is only **0.667**, with **0 recall for the training class** on the tiny evaluation split.
3. **The edge-case classifier is also weak.** Leave-one-out accuracy is **0.50** across **10 cases**, and the prediction vector is heavily biased toward predicting “training”.
4. **Some telemetry rows are suspicious or incomplete.** In `single_gpu_summary.csv`, `mlp_training` has **0 mean GPU utilization** while still showing nontrivial power draw and memory allocation. That makes it look more like a failed or undersampled capture than a reliable training signature.
5. **Several scripts were repurposed from Week 4/H100 code and were not fully normalized for Week 7/B200.** Multiple headers still say “Week 4”, some constants still reference H100 in places that should be B200-specific, and output-path organization in code does not cleanly match the result tree committed in the repository.
6. **Some narrative conclusions overstate generalization.** The comparison report claims architecture-independent transfer and near-hardware-independent binary classification, but the actual published Week 7 classification numbers do not support that claim without major qualification.

## 1. Pipeline structure and test configuration

`week7/run_week7.sh` defines a 10-step execution plan:

1. NVLink characterization
2. DDP training characterization
3. Edge cases
4. Model width scaling
5. Dataset size scaling
6. EDA
7. Classifier suite (“week4 style”)
8. Feature engineering (“week5 style”)
9. Classifier training on windows (“week5 style”)
10. Report generation

This is a well-scoped orchestration script and makes the experiment intent easy to follow.

### Strengths
- The pipeline covers hardware, workload telemetry, adversarial edge cases, scaling, and reporting.
- It explicitly separates “week4-style” classification from “week5-style” feature engineering, which is useful for longitudinal comparison.

### Weaknesses
- Some step scripts still carry Week 4 naming and comments, which makes provenance harder to trust.
- The codebase mixes “single_gpu_tests / two_gpu_tests / edge_cases” organization with older `week7/results` and `week7/plots` conventions.

## 2. Data and result inventory

### 2.1 Single-GPU summary

The committed `single_gpu_summary.csv` contains 9 workloads:
- `cufft_hpc`
- `idle`
- `inference`
- `mining_proxy`
- `mlp_training`
- `nbody_sim`
- `rendering`
- `resnet_amp`
- `resnet_fp32`

#### What looks good
- The workload set spans baseline, inference, HPC, proxy mining/rendering, and training.
- Several rows show meaningful separation in memory usage and power.
- `cufft_hpc` and `nbody_sim` look like plausible high-intensity non-training signatures.
- `resnet_amp` and `resnet_fp32` show distinct memory, power, and clock profiles.

#### What looks questionable
- `mlp_training` reports:
  - `gpu_utilization_pct_mean = 0.0`
  - `gpu_utilization_pct_std = 0.0`
  - `mem_used_mb_mean = 3578`
  - `power_draw_w_mean ≈ 193.8`
  - `sm_clock_mhz_mean = 1965`

  A true training workload with zero utilization and zero utilization variance is not credible as a production-quality telemetry signature. It suggests either failed collection, overly sparse sampling, or a label/capture mismatch.

- `mining_proxy` and `rendering` also show zero GPU utilization but large memory allocations and ~195–199 W power draw. These may be intentional synthetic or memory-resident proxy states, but they weaken any classifier that expects actual dynamic compute utilization.

### 2.2 Single-GPU classifier

The single-GPU classifier result is:
- **Accuracy:** `0.6667`
- `non_training` recall: `1.0`
- `training` recall: `0.0`

This means the model missed the positive class completely on the published evaluation slice. That is the opposite of a robust training detector.

The top feature importances are led by:
- `gpu_utilization_pct_cv`
- `mem_utilization_pct_cv`
- `mem_used_mb_mean`
- `mem_used_mb_cv`

That feature ranking is directionally sensible, but the evaluation size is too small and the result is too weak to support confident generalization claims.

### 2.3 Model scaling (single GPU)

The `model_scaling.csv` table looks internally consistent:
- parameters scale from **71,570** to **287,916,554**
- mean step time increases from **1.46 ms** to **16.87 ms**
- achieved TFLOPS increase from **0.0063** to **2.18**

This is a clean monotonic scaling curve and is one of the stronger parts of the Week 7 result set.

## 3. Two-GPU / NVLink findings

### 3.1 NVLink bandwidth

The NVLink sweep is one of the most credible datasets in the repo. It shows:
- **1 MB:** 59.15 GB/s
- **64 MB:** 571.53 GB/s
- **4096 MB:** **726.78 GB/s**
- **Bidirectional 1024 MB:** **1429.61 GB/s** total

This is a healthy saturating bandwidth curve and strongly supports the core claim that Week 7 is measuring a significantly faster interconnect than the Week 4 H100 NV6 setup.

### 3.2 NVLink latency

The latency table is tight and stable:
- best RTT ≈ **23.42 µs**
- the rest of the sizes are clustered around **23.6–25.3 µs**

That consistency makes the latency measurement look credible.

### 3.3 Throughput comparison

The committed throughput table shows:
- **1 GPU:** 29,008 img/s
- **2 GPU:** 44,372 img/s
- implied speedup: **1.53×**

This is important because it contradicts the older “DataParallel underperforms” narrative inherited from prior weeks. In Week 7’s own committed results, 2-GPU DataParallel is **better than** single GPU, even if it is still far from ideal 2× scaling.

### 3.4 Two-GPU model width and batch sweeps

The `model_width_sweep.csv` and `batch_size_sweep.csv` files show orderly scaling:
- width sweep TFLOPS rise from **0.00115** to **2.20**
- batch sweep TFLOPS rise from **0.0388** at batch 8 to **2.115** at batch 1024

The overall pattern is coherent and supports the plot-generation scripts.

## 4. Edge-case findings

The edge-case table is valuable because it tries to stress the detector under adversarial or ambiguous states.

### Positive aspects
- The case inventory is broad: baseline training/inference plus EC1–EC8.
- It includes memory-heavy idle and AMP masking style cases, which are relevant for B200-specific failure analysis.

### Negative aspects
- The classifier performance is poor:
  - **LOO accuracy: 0.50**
  - the prediction vector is heavily skewed toward predicting “training”

This means the edge-case detector is currently not reliable enough to support strong claims about adversarial robustness.

### Labeling concern
The markdown summary describes some cases as “no” for training, but the classifier JSON’s `true` vector marks several of those edge cases as positive. That does not necessarily prove a bug, but it does mean the repo needs a clearer statement of the operational definition of “training” versus “non-training-like telemetry”.

## 5. Findings on plots and generated reports

### 5.1 Comparison plots and summary report

The generated `comparison.md` is readable and useful, but several of its conclusions are too strong relative to the data.

#### Supported conclusions
- Week 7 B200 NVLink is much faster than Week 4 H100 NV6.
- Week 7 single- and two-GPU throughput numbers are materially higher than earlier results.
- The repo is trying to connect telemetry, transport, scaling, and classifier behavior across weeks.

#### Overstated or weakly supported conclusions
- The report says binary classification is “hardware-independent” and suggests cross-generation transfer with minimal retraining.
- But the published Week 7 single-session classifier accuracy is only **66.7%**, and edge-case LOO accuracy is only **50%**.

Those two facts mean the current Week 7 data support the opposite conclusion: **transfer/generalization remains unresolved and probably needs more data, larger evaluation sets, or better labeling.**

### 5.2 Plot-generation logic is rich, but interpretation must be cautious

`generate_roofline_tier1.py` is ambitious and well-structured. It attempts to generate:
- single- and two-GPU rooflines
- B200 vs H100 comparison roofline
- Tier 1 heatmaps
- CV analysis
- PCA
- power-vs-util scatter
- clock analysis
- NVLink roofline plots

This is analytically strong in design.

However, several interpretations are only as good as the underlying telemetry quality. If key workloads like `mlp_training` have zero GPU utilization, then downstream plots that treat all rows as equally valid may be misleading.

## 6. Code and consistency audit

This repo section shows clear evidence of iterative reuse rather than clean consolidation.

### 6.1 Week 4/H100 leftovers inside Week 7 code

Examples:
- `run_nvlink_tests.py` begins with a docstring that says **“Week 4 Step 4: NVLink Characterization (2x H100 80GB)”**, even though it writes into Week 7 B200 paths and the Week 7 results clearly represent B200 measurements.
- `run_classifiers.py` begins with **“Week 4 Step 3: Baseline Classifier Suite”** while being located in `week7/`.
- `scale_model_accuracy.py` still hardcodes several H100 labels/constants in a way that does not match the rest of Week 7.

### 6.2 Suspicious constants in `scale_model_accuracy.py`

The most obvious consistency problem is this block:
- `B200_FP16_TFLOPS = 4500.0`
- `H100_FP16_TFLOPS = 4500.0`
- `B200_FP32_TFLOPS = 140.0`
- `H100_FP32_TFLOPS = 140.0`
- `B200_MEM_BW_GBPS = 8000.0`
- `H100_MEM_BW_GBPS = 8000.0`

Those H100 constants are set equal to B200 values, which is not physically correct and means any report or plot using them as comparative hardware ceilings is misconfigured.

### 6.3 Pathing mismatch

`run_nvlink_tests.py` writes to:
- `week7/results/nvlink/`
- `week7/plots/`
- `week7/data/`

But the committed Week 7 repo structure also uses:
- `week7/two_gpu_tests/results/`
- `week7/single_gpu_tests/results/`
- `week7/edge_cases/results/`

That suggests either:
- multiple generations of organization are coexisting, or
- the current scripts do not exactly reproduce the published tree.

Either way, reproducibility would improve if the code and tree were normalized to one layout.

## 7. Overall assessment

### What is strongest
- NVLink bandwidth and latency characterization
- Throughput comparison tables
- Single- and two-GPU scaling tables
- Breadth of experiment coverage and report generation

### What is weakest
- Single-GPU classifier credibility
- Edge-case detector performance
- Internal consistency of narrative claims versus actual metrics
- Clean separation between old Week 4/H100 logic and new Week 7/B200 logic

## 8. Recommended actions

1. **Fix naming and constants first.** Remove Week 4/H100 leftovers from Week 7 scripts and correct all B200/H100 ceiling constants.
2. **Revalidate suspect telemetry rows.** In particular, rerun or inspect `mlp_training`, `mining_proxy`, and `rendering` captures with higher sampling confidence and trace-level logging.
3. **Harden evaluation splits.** The current single-GPU classifier result is too small and weak for strong claims.
4. **Clarify labeling policy for edge cases.** Explicitly define which cases are positive, why, and what “training” means operationally.
5. **Tone down the current summary conclusions.** The repo should state that transport/scaling results are strong, while classification transfer and adversarial robustness remain open problems.
6. **Normalize output layout.** Use one consistent path scheme for all Week 7 scripts and generated artifacts.

## Final verdict

**Week 7 is a promising hardware-characterization and telemetry-analysis package, but not yet a convincing end-to-end proof of robust B200 training detection.**

The transport and scaling results are strong. The classifier results are not. The repo will benefit substantially from one cleanup pass focused on consistency, telemetry validity, and claim calibration.
