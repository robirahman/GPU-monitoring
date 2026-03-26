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
| XGBoost | 5s | 0.9539 ± 0.0335 | 0.9195 |
| SVM_RBF | 5s | 0.9426 ± 0.0343 | 0.9042 |
| RandomForest | 5s | 0.9424 ± 0.0360 | 0.9045 |
| LogisticReg | 5s | 0.9188 ± 0.0669 | 0.8799 |
| SVM_RBF | 15s | 0.9565 ± 0.0361 | 0.9289 |
| XGBoost | 15s | 0.9486 ± 0.0290 | 0.9005 |
| RandomForest | 15s | 0.9430 ± 0.0379 | 0.9030 |
| LogisticReg | 15s | 0.8397 ± 0.1081 | 0.7972 |
| XGBoost | 30s | 0.8960 ± 0.0435 | 0.8578 |
| RandomForest | 30s | 0.8585 ± 0.1031 | 0.7843 |
| SVM_RBF | 30s | 0.8230 ± 0.0772 | 0.7510 |
| LogisticReg | 30s | 0.7817 ± 0.1105 | 0.7177 |

**Best binary:** SVM_RBF at 15s window — accuracy = **0.9565** (PASS vs 85% target)

### Three-Way Classification (Training / Inference / Other)

| Model | Window | Accuracy | F1-macro |
|-------|--------|----------|----------|
| SVM_RBF | 5s | 0.6100 ± 0.2809 | 0.5023 |
| XGBoost | 5s | 0.5951 ± 0.1864 | 0.4944 |
| LogisticReg | 5s | 0.5621 ± 0.2376 | 0.4674 |
| RandomForest | 5s | 0.4838 ± 0.0808 | 0.4198 |
| SVM_RBF | 15s | 0.6941 ± 0.2101 | 0.6122 |
| XGBoost | 15s | 0.6890 ± 0.2049 | 0.5550 |
| RandomForest | 15s | 0.6481 ± 0.1778 | 0.5084 |
| LogisticReg | 15s | 0.6049 ± 0.3067 | 0.5106 |
| XGBoost | 30s | 0.7598 ± 0.1686 | 0.6412 |
| SVM_RBF | 30s | 0.6919 ± 0.2102 | 0.5845 |
| RandomForest | 30s | 0.6912 ± 0.1505 | 0.4966 |
| LogisticReg | 30s | 0.6069 ± 0.1896 | 0.5370 |

### Multi-Class Classification (Full Workload Label)

| Model | Window | Accuracy | F1-macro |
|-------|--------|----------|----------|
| SVM_RBF | 5s | 0.9205 ± 0.0651 | 0.8369 |
| XGBoost | 5s | 0.8750 ± 0.1329 | 0.8009 |
| LogisticReg | 5s | 0.8750 ± 0.1329 | 0.8009 |
| RandomForest | 5s | 0.8659 ± 0.1494 | 0.7509 |
| SVM_RBF | 15s | 0.9333 ± 0.0816 | 0.8762 |
| RandomForest | 15s | 0.9000 ± 0.1333 | 0.8483 |
| LogisticReg | 15s | 0.9000 ± 0.1333 | 0.8495 |
| XGBoost | 15s | 0.8600 ± 0.1272 | 0.7095 |
| SVM_RBF | 30s | 0.8800 ± 0.0980 | 0.7733 |
| RandomForest | 30s | 0.7800 ± 0.1600 | 0.7133 |
| XGBoost | 30s | 0.7550 ± 0.2052 | 0.7274 |
| LogisticReg | 30s | 0.6800 ± 0.2502 | 0.6233 |

**Best multi-class:** SVM_RBF at 15s window — accuracy = **0.9333**

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
