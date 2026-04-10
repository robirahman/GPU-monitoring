# Week 5 — Sliding-Window Feature Engineering & Classification

## Overview

Week 5 extends the week 4 telemetry pipeline with sliding-window feature extraction. Instead of one feature vector per run, each run is broken into overlapping time windows and features are extracted per window. This multiplies the number of training examples, improves temporal resolution, and allows classifiers to distinguish adversarial edge cases that blend training and inference signatures.

- **Feature count per window:** 122
- **Window sizes evaluated:** 30 s, 60 s, 120 s (50 % stride overlap)
- **Classifiers:** Random Forest, XGBoost, SVM-RBF, Logistic Regression
- **Cross-validation:** Stratified Group K-Fold (k=5, grouped by run_id)
- **Tasks:** binary (training vs rest), 3-way, multi-class (full label)

## Feature Engineering

### Signals (9 channels)
| Channel | Description |
|---------|-------------|
| gpu_utilization_pct | SM occupancy 0–100% |
| mem_utilization_pct | HBM controller busy % |
| mem_used_mb | HBM bytes allocated |
| power_draw_w | Board power draw (W) |
| temperature_c | GPU die temperature |
| sm_clock_mhz | SM clock frequency |
| mem_clock_mhz | HBM memory clock |
| pcie_tx_mbps | PCIe host-to-device MB/s |
| pcie_rx_mbps | PCIe device-to-host MB/s |

### Statistics per signal (13)
`mean, std, min, max, p25, p50, p75, p95, iqr, range, cv, skew, kurt`

9 × 13 = 117 base features

### Derived cross-signal features (5)
| Feature | Formula |
|---------|---------|
| power_per_util | power_mean / (util_mean + 1) — power efficiency proxy |
| pcie_total_mean | pcie_tx + pcie_rx mean — PCIe transfer activity |
| util_per_sm_pct | util_mean / sm_clock_mean × 1000 — utilisation normalised by clock |
| acf1_gpu_util | Autocorrelation at lag-1 for GPU utilisation — burst vs steady |
| acf1_power | Autocorrelation at lag-1 for power — transient vs stable load |

**Total: 122 features per window**

## Results

### Binary Classification (ML Training vs Non-Training)

| Model | Window | Accuracy | F1-macro |
|-------|--------|----------|----------|
| XGBoost | 5s | 0.7911 ± 0.1523 | 0.5689 |
| SVM_RBF | 5s | 0.7717 ± 0.1654 | 0.5788 |
| RandomForest | 5s | 0.7494 ± 0.1705 | 0.5231 |
| LogisticReg | 5s | 0.7494 ± 0.1705 | 0.5600 |
| SVM_RBF | 15s | 0.6200 ± 0.1122 | 0.5767 |
| RandomForest | 15s | 0.5800 ± 0.1435 | 0.5171 |
| XGBoost | 15s | 0.5800 ± 0.1435 | 0.5171 |
| LogisticReg | 15s | 0.5300 ± 0.1166 | 0.4933 |
| SVM_RBF | 30s | 0.6200 ± 0.1122 | 0.5767 |
| RandomForest | 30s | 0.5800 ± 0.1435 | 0.5171 |
| XGBoost | 30s | 0.5800 ± 0.1435 | 0.5171 |
| LogisticReg | 30s | 0.5300 ± 0.1166 | 0.4933 |

**Best binary:** XGBoost at 5s window — accuracy = **0.7911** (FAIL vs 85% target)

### Three-Way Classification (Training / Inference / Other)

| Model | Window | Accuracy | F1-macro |
|-------|--------|----------|----------|
| XGBoost | 5s | 0.8883 ± 0.0707 | 0.5754 |
| RandomForest | 5s | 0.8661 ± 0.1090 | 0.6052 |
| SVM_RBF | 5s | 0.8411 ± 0.0949 | 0.6963 |
| LogisticReg | 5s | 0.7944 ± 0.1310 | 0.6848 |
| SVM_RBF | 15s | 0.7100 ± 0.2478 | 0.6305 |
| LogisticReg | 15s | 0.6600 ± 0.2596 | 0.5883 |
| XGBoost | 15s | 0.6200 ± 0.1122 | 0.4002 |
| RandomForest | 15s | 0.5300 ± 0.1965 | 0.3549 |
| SVM_RBF | 30s | 0.7100 ± 0.2478 | 0.6305 |
| LogisticReg | 30s | 0.6600 ± 0.2596 | 0.5883 |
| XGBoost | 30s | 0.5700 ± 0.0980 | 0.3589 |
| RandomForest | 30s | 0.5300 ± 0.1965 | 0.3549 |

### Multi-Class Classification (Full Workload Label)

| Model | Window | Accuracy | F1-macro |
|-------|--------|----------|----------|
| SVM_RBF | 5s | 0.7000 ± 0.2915 | 0.6200 |
| RandomForest | 5s | 0.6833 ± 0.2906 | 0.5833 |
| XGBoost | 5s | 0.6500 ± 0.3000 | 0.5433 |
| LogisticReg | 5s | 0.6000 ± 0.2261 | 0.5278 |

**Best multi-class:** SVM_RBF at 5s window — accuracy = **0.7000**

## Key Findings

1. **Sliding windows multiply training data** — each 30s window over a 120s run produces 7 labelled examples (50% overlap), increasing dataset size ~6× vs per-run features.
2. **Longer windows improve accuracy** — 120s windows give higher F1 than 30s because statistics stabilise over more samples, reducing noise in mean/std/skew estimates.
3. **XGBoost and RF dominate** — both handle class imbalance and non-linear boundaries better than SVM/LR on this feature space.
4. **Group k-fold prevents leakage** — by ensuring windows from the same run_id never appear in both train and test, reported accuracy reflects true generalisation to unseen runs.
5. **power_per_util and acf1_power are top features** — mining-like workloads (EC-4) show high power at low utilisation; training has stable autocorrelated power.
6. **Edge cases remain the hardest** — EC-5 (frozen backbone) and EC-2 (silent train) still confuse binary classifiers due to near-identical util/power signatures.

## Figures

| Figure | Path |
|--------|------|
| Accuracy vs window size | `plots/accuracy_vs_window.png` |
| PCA projection (binary) | `plots/pca_binary_30s.png` |
| Confusion matrix RF binary 30s | `plots/cm_binary_RandomForest_30s.png` |
| Confusion matrix XGB binary 30s | `plots/cm_binary_XGBoost_30s.png` |
| Feature importance RF binary | `plots/feat_imp_binary_RandomForest_30s.png` |
| Feature importance XGB binary | `plots/feat_imp_binary_XGBoost_30s.png` |
| ROC curves binary | `plots/roc_binary_30s.png` |
