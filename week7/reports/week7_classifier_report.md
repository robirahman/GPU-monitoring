# Week 7 -- Sliding-Window Classification on B200

## Overview

Week 7 replicates the Week 5 sliding-window methodology on 2x NVIDIA B200 (Blackwell). Each run is broken into overlapping time windows and 125 features are extracted per window.

- **Hardware:** 2x NVIDIA B200 (183 GB HBM3e, 8 TB/s, 1000W TDP)
- **Feature count per window:** 125
- **Window sizes evaluated:** 5s, 15s, 30s
- **Classifiers:** Random Forest, XGBoost, SVM-RBF, Logistic Regression
- **Cross-validation:** Stratified Group K-Fold (k=5, grouped by run_id)
- **Short-run handling:** runs shorter than window size are tagged and excluded from headline metrics

## Detection Window / Delay

The minimum detection window is **5s** (the smallest evaluated window size). At 1 Hz NVML sampling, this requires 5 telemetry samples. True detection delay = window_size + classification_time (<1ms), so:

- **5s window:** detection delay ~ 5s
- **15s window:** detection delay ~ 15s
- **30s window:** detection delay ~ 30s

## Results

### Binary Classification (ML Training vs Non-Training)

| Model | Window | Accuracy | F1-macro | F1-weighted |
|-------|--------|----------|----------|-------------|
| LogisticReg | 5s | 0.9390 +/- 0.0649 | 0.9368 | 0.9389 |
| SVM_RBF | 5s | 0.9290 +/- 0.0616 | 0.9274 | 0.9297 |
| RandomForest | 5s | 0.8876 +/- 0.0642 | 0.8783 | 0.8850 |
| XGBoost | 5s | 0.8742 +/- 0.0734 | 0.8599 | 0.8689 |
| LogisticReg | 15s | 0.8800 +/- 0.2400 | 0.8750 | 0.8650 |
| RandomForest | 15s | 0.8300 +/- 0.2358 | 0.7607 | 0.7936 |
| SVM_RBF | 15s | 0.7800 +/- 0.2205 | 0.6464 | 0.7221 |
| XGBoost | 15s | 0.7100 +/- 0.2764 | 0.5985 | 0.6896 |

**Best binary:** LogisticReg at 5s window -- accuracy = **0.9390** (PASS vs 85% target)

### Three-Way Classification

| Model | Window | Accuracy | F1-macro | F1-weighted |
|-------|--------|----------|----------|-------------|
| SVM_RBF | 5s | 0.9341 +/- 0.0815 | 0.8080 | 0.9316 |
| LogisticReg | 5s | 0.9209 +/- 0.0789 | 0.8629 | 0.9207 |
| RandomForest | 5s | 0.8900 +/- 0.0785 | 0.7891 | 0.8845 |
| XGBoost | 5s | 0.8853 +/- 0.0793 | 0.8473 | 0.8791 |
| LogisticReg | 15s | 0.8800 +/- 0.2400 | 0.8750 | 0.8650 |
| RandomForest | 15s | 0.8300 +/- 0.2358 | 0.7607 | 0.7936 |
| SVM_RBF | 15s | 0.7800 +/- 0.2205 | 0.6464 | 0.7221 |
| XGBoost | 15s | 0.7100 +/- 0.2764 | 0.5985 | 0.6896 |

### Multi-Class Classification

| Model | Window | Accuracy | F1-macro | F1-weighted |
|-------|--------|----------|----------|-------------|
| LogisticReg | 5s | 0.9654 +/- 0.0283 | 0.8000 | 0.9612 |
| SVM_RBF | 5s | 0.9645 +/- 0.0343 | 0.7988 | 0.9545 |
| RandomForest | 5s | 0.9595 +/- 0.0350 | 0.7721 | 0.9519 |
| XGBoost | 5s | 0.9534 +/- 0.0239 | 0.7321 | 0.9477 |
| RandomForest | 15s | 1.0000 +/- 0.0000 | 1.0000 | 1.0000 |
| LogisticReg | 15s | 0.9429 +/- 0.1143 | 0.8833 | 0.9667 |
| SVM_RBF | 15s | 0.8286 +/- 0.3429 | 0.8250 | 0.8500 |
| XGBoost | 15s | 0.8000 +/- 0.4000 | 0.8000 | 0.8000 |

## Normalized Power Feature

Week 7 adds `power_pct_tdp` = power_draw_w / TDP * 100. This enables cross-GPU comparison:
- Raw watts are GPU-specific (B200 TDP=1000W, H100=700W) and cannot be compared directly
- Normalized power (% of TDP) is transferable: 50% TDP means similar thermal budget usage
- For cross-GPU transfer learning, use `power_pct_tdp` instead of `power_draw_w`

## Figures

| Figure | Path |
|--------|------|
| Accuracy vs window size | `plots/accuracy_vs_window.png` |
| PCA projection (binary, 5s) | `plots/pca_binary_5s.png` |
| PCA projection (binary, 15s) | `plots/pca_binary_15s.png` |
| PCA projection (binary, 30s) | `plots/pca_binary_30s.png` |
| CM (illustrative fold) | `plots/cm_binary_RandomForest_5s.png` |
| ROC (illustrative fold) | `plots/roc_binary_5s.png` |
| Feature importance | `plots/feat_imp_binary_RandomForest_5s.png` |
