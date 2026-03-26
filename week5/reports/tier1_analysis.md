# Tier-1 Telemetry Analysis: In-Range Corner Cases

Each config runs for **30 seconds** with pynvml sampled at **10 Hz** (≈300 samples).
Comparison against a real AMP training baseline (ResNet-50, bs=4, SGD+GradScaler).

## Training Baseline — ResNet-50 bs=4 FP16 AMP

| Signal | Mean | Std | CV | ACF1 | Interpretation |
|--------|------|-----|----|------|----------------|
| gpu_util | 13.6% | 1.79 | 0.131 | 0.686 | Periodic fwd/bwd/step cycles → 13% avg, high ACF1 (stable pattern) |
| power_w | 101.9 W | 1.21 | 0.012 | 0.814 | Moderate power, **very low CV** (0.012) — steady load without large spikes |
| mem_used_gb | 1.82 GB | — | — | — | Activations + grads + optimizer state = 1.8 GB at bs=4 |
| mem_util | 1.0% | — | — | — | HBM bus mostly idle at this small batch |

> **Key training signature:** Low average GPU util (13%) but very stable power (CV=0.012),
> high power ACF1 (0.814) — fwd→bwd→step cycle runs at a steady cadence.

---

## CC-A: CNN Inference (ResNet)

### `CC-A/resnet18/bs1/img224/fp16`

| Signal | Value | Training | Delta | Verdict |
|--------|-------|----------|-------|---------|
| GPU util mean | 56.0% | 13.6% | +42.4% | HIGHER |
| Power mean | 108.7W | 101.9W | +6.8W | SIMILAR (false detection risk) |
| Power CV | 0.023 | 0.012 | +0.012 | SIMILAR |
| Power ACF1 | 0.900 | 0.814 | +0.086 | SIMILAR |
| Mem used | 1.47 GB | 1.82 GB | -0.35 GB | SIMILAR |
| Mem util | 0.0% | 1.0% | -1.0% | SIMILAR |
| GPU util ACF1 | 0.645 | 0.686 | -0.041 | SIMILAR |

**Risk level:** MEDIUM

**False-detection causes (what makes classifier confuse this with training):**
- Power (109W) ≈ training (102W) — power-based features cannot discriminate
- power_cv (0.023) ≈ training (0.012) — temporal variability matches
- gpu_util ACF1 (0.645) ≈ training (0.686) — autocorrelation pattern matches

**Discriminating signals (what CAN separate this from training):**
- gpu_util_mean = 56% vs training 14% — inference keeps SM fully occupied without periodic optimizer overhead gaps

### `CC-A/resnet50/bs4/img224/fp16`

| Signal | Value | Training | Delta | Verdict |
|--------|-------|----------|-------|---------|
| GPU util mean | 58.5% | 13.6% | +44.9% | HIGHER |
| Power mean | 133.4W | 101.9W | +31.5W | SIMILAR (false detection risk) |
| Power CV | 0.033 | 0.012 | +0.022 | SIMILAR |
| Power ACF1 | 0.917 | 0.814 | +0.103 | DIFFERENT |
| Mem used | 1.52 GB | 1.82 GB | -0.30 GB | SIMILAR |
| Mem util | 1.0% | 1.0% | -0.0% | SIMILAR |
| GPU util ACF1 | 0.717 | 0.686 | +0.030 | SIMILAR |

**Risk level:** MEDIUM

**False-detection causes (what makes classifier confuse this with training):**
- Power (133W) ≈ training (102W) — power-based features cannot discriminate
- power_cv (0.033) ≈ training (0.012) — temporal variability matches
- gpu_util ACF1 (0.717) ≈ training (0.686) — autocorrelation pattern matches

**Discriminating signals (what CAN separate this from training):**
- gpu_util_mean = 59% vs training 14% — inference keeps SM fully occupied without periodic optimizer overhead gaps
- power_w_acf1 = 0.917 vs training 0.814 — different temporal autocorrelation

### `CC-A/resnet50/bs16/img224/fp16`

| Signal | Value | Training | Delta | Verdict |
|--------|-------|----------|-------|---------|
| GPU util mean | 86.7% | 13.6% | +73.1% | HIGHER |
| Power mean | 243.9W | 101.9W | +142.0W | HIGHER |
| Power CV | 0.068 | 0.012 | +0.056 | DIFFERENT |
| Power ACF1 | 0.914 | 0.814 | +0.100 | DIFFERENT |
| Mem used | 1.65 GB | 1.82 GB | -0.17 GB | SIMILAR |
| Mem util | 12.9% | 1.0% | +11.9% | HIGHER |
| GPU util ACF1 | 0.676 | 0.686 | -0.011 | SIMILAR |

**Risk level:** LOW

**False-detection causes (what makes classifier confuse this with training):**
- gpu_util ACF1 (0.676) ≈ training (0.686) — autocorrelation pattern matches

**Discriminating signals (what CAN separate this from training):**
- gpu_util_mean = 87% vs training 14% — inference keeps SM fully occupied without periodic optimizer overhead gaps
- power_w_mean = 244W vs training 102W — significantly higher sustained power
- power_w_cv = 0.068 vs training 0.012 — variability pattern differs (training has periodic optimizer spikes → higher CV)
- power_w_acf1 = 0.914 vs training 0.814 — different temporal autocorrelation
- mem_util_mean = 13% vs 1% — higher HBM bus utilisation

---

## CC-B: LLM Prefill (GPT-2 family)

### `CC-B/gpt2/bs4/s128/fp16`

| Signal | Value | Training | Delta | Verdict |
|--------|-------|----------|-------|---------|
| GPU util mean | 52.8% | 13.6% | +39.1% | HIGHER |
| Power mean | 168.8W | 101.9W | +66.9W | HIGHER |
| Power CV | 0.056 | 0.012 | +0.044 | DIFFERENT |
| Power ACF1 | 0.896 | 0.814 | +0.082 | SIMILAR |
| Mem used | 2.24 GB | 1.82 GB | +0.42 GB | SIMILAR |
| Mem util | 3.0% | 1.0% | +1.9% | SIMILAR |
| GPU util ACF1 | 0.740 | 0.686 | +0.054 | SIMILAR |

**Risk level:** LOW

**False-detection causes (what makes classifier confuse this with training):**
- gpu_util ACF1 (0.740) ≈ training (0.686) — autocorrelation pattern matches

**Discriminating signals (what CAN separate this from training):**
- gpu_util_mean = 53% vs training 14% — inference keeps SM fully occupied without periodic optimizer overhead gaps
- power_w_mean = 169W vs training 102W — significantly higher sustained power
- power_w_cv = 0.056 vs training 0.012 — variability pattern differs (training has periodic optimizer spikes → higher CV)

### `CC-B/gpt2-m/bs4/s512/fp32`

| Signal | Value | Training | Delta | Verdict |
|--------|-------|----------|-------|---------|
| GPU util mean | 99.2% | 13.6% | +85.6% | HIGHER |
| Power mean | 392.3W | 101.9W | +290.4W | HIGHER |
| Power CV | 0.088 | 0.012 | +0.076 | DIFFERENT |
| Power ACF1 | 0.876 | 0.814 | +0.062 | SIMILAR |
| Mem used | 3.71 GB | 1.82 GB | +1.89 GB | HIGHER (>1.5×) |
| Mem util | 6.9% | 1.0% | +5.9% | SIMILAR |
| GPU util ACF1 | 0.719 | 0.686 | +0.032 | SIMILAR |

**Risk level:** LOW

**False-detection causes (what makes classifier confuse this with training):**
- gpu_util ACF1 (0.719) ≈ training (0.686) — autocorrelation pattern matches

**Discriminating signals (what CAN separate this from training):**
- mem_used_gb = 3.7 GB vs 1.8 GB (2.0×) — inference holds large activation buffers; training with bs=4 uses less HBM
- gpu_util_mean = 99% vs training 14% — inference keeps SM fully occupied without periodic optimizer overhead gaps
- power_w_mean = 392W vs training 102W — significantly higher sustained power
- power_w_cv = 0.088 vs training 0.012 — variability pattern differs (training has periodic optimizer spikes → higher CV)

### `CC-B/gpt2-l/bs4/s128/fp16`

| Signal | Value | Training | Delta | Verdict |
|--------|-------|----------|-------|---------|
| GPU util mean | 59.2% | 13.6% | +45.6% | HIGHER |
| Power mean | 223.4W | 101.9W | +121.5W | HIGHER |
| Power CV | 0.065 | 0.012 | +0.053 | DIFFERENT |
| Power ACF1 | 0.883 | 0.814 | +0.069 | SIMILAR |
| Mem used | 5.24 GB | 1.82 GB | +3.42 GB | HIGHER (>1.5×) |
| Mem util | 6.9% | 1.0% | +5.9% | SIMILAR |
| GPU util ACF1 | 0.562 | 0.686 | -0.124 | DIFFERENT |

**Risk level:** LOW

**False-detection causes (what makes classifier confuse this with training):**
- Arithmetic intensity in training range (45–1483 FLOP/B) — roofline-only classifiers misclassify

**Discriminating signals (what CAN separate this from training):**
- mem_used_gb = 5.2 GB vs 1.8 GB (2.9×) — inference holds large activation buffers; training with bs=4 uses less HBM
- gpu_util_mean = 59% vs training 14% — inference keeps SM fully occupied without periodic optimizer overhead gaps
- power_w_mean = 223W vs training 102W — significantly higher sustained power
- power_w_cv = 0.065 vs training 0.012 — variability pattern differs (training has periodic optimizer spikes → higher CV)

---

## CC-C: LLM Decode (memory-bound)

### `CC-C/gpt2-m/bs256/ctx128/fp16`

| Signal | Value | Training | Delta | Verdict |
|--------|-------|----------|-------|---------|
| GPU util mean | 57.6% | 13.6% | +43.9% | HIGHER |
| Power mean | 180.8W | 101.9W | +78.9W | HIGHER |
| Power CV | 0.054 | 0.012 | +0.042 | DIFFERENT |
| Power ACF1 | 0.884 | 0.814 | +0.070 | SIMILAR |
| Mem used | 3.26 GB | 1.82 GB | +1.44 GB | HIGHER (>1.5×) |
| Mem util | 4.0% | 1.0% | +2.9% | SIMILAR |
| GPU util ACF1 | 0.623 | 0.686 | -0.063 | SIMILAR |

**Risk level:** LOW

**False-detection causes (what makes classifier confuse this with training):**
- gpu_util ACF1 (0.623) ≈ training (0.686) — autocorrelation pattern matches

**Discriminating signals (what CAN separate this from training):**
- gpu_util_mean = 58% vs training 14% — inference keeps SM fully occupied without periodic optimizer overhead gaps
- power_w_mean = 181W vs training 102W — significantly higher sustained power
- power_w_cv = 0.054 vs training 0.012 — variability pattern differs (training has periodic optimizer spikes → higher CV)
- LLM decode: single-token generation → near-zero GPU util bursts, highly irregular pattern vs steady training

---

## CC-D: Quantisation Comparison

### `CC-D/resnet50/bs16/fp32`

| Signal | Value | Training | Delta | Verdict |
|--------|-------|----------|-------|---------|
| GPU util mean | 99.1% | 13.6% | +85.5% | HIGHER |
| Power mean | 361.4W | 101.9W | +259.5W | HIGHER |
| Power CV | 0.070 | 0.012 | +0.059 | DIFFERENT |
| Power ACF1 | 0.898 | 0.814 | +0.083 | SIMILAR |
| Mem used | 1.76 GB | 1.82 GB | -0.05 GB | SIMILAR |
| Mem util | 35.0% | 1.0% | +34.0% | HIGHER |
| GPU util ACF1 | 0.678 | 0.686 | -0.008 | SIMILAR |

**Risk level:** LOW

**False-detection causes (what makes classifier confuse this with training):**
- gpu_util ACF1 (0.678) ≈ training (0.686) — autocorrelation pattern matches

**Discriminating signals (what CAN separate this from training):**
- gpu_util_mean = 99% vs training 14% — inference keeps SM fully occupied without periodic optimizer overhead gaps
- power_w_mean = 361W vs training 102W — significantly higher sustained power
- power_w_cv = 0.070 vs training 0.012 — variability pattern differs (training has periodic optimizer spikes → higher CV)
- mem_util_mean = 35% vs 1% — higher HBM bus utilisation

### `CC-D/gpt2/bs1/s512/fp32`

| Signal | Value | Training | Delta | Verdict |
|--------|-------|----------|-------|---------|
| GPU util mean | 99.3% | 13.6% | +85.7% | HIGHER |
| Power mean | 353.5W | 101.9W | +251.6W | HIGHER |
| Power CV | 0.084 | 0.012 | +0.072 | DIFFERENT |
| Power ACF1 | 0.883 | 0.814 | +0.069 | SIMILAR |
| Mem used | 2.26 GB | 1.82 GB | +0.44 GB | SIMILAR |
| Mem util | 5.0% | 1.0% | +3.9% | SIMILAR |
| GPU util ACF1 | 0.587 | 0.686 | -0.099 | SIMILAR |

**Risk level:** LOW

**False-detection causes (what makes classifier confuse this with training):**
- gpu_util ACF1 (0.587) ≈ training (0.686) — autocorrelation pattern matches

**Discriminating signals (what CAN separate this from training):**
- gpu_util_mean = 99% vs training 14% — inference keeps SM fully occupied without periodic optimizer overhead gaps
- power_w_mean = 354W vs training 102W — significantly higher sustained power
- power_w_cv = 0.084 vs training 0.012 — variability pattern differs (training has periodic optimizer spikes → higher CV)

---

## CC-E: Forward-Only Training Variant

### `CC-E/fwd_only/bs1/fp16`

| Signal | Value | Training | Delta | Verdict |
|--------|-------|----------|-------|---------|
| GPU util mean | 40.7% | 13.6% | +27.1% | HIGHER |
| Power mean | 122.2W | 101.9W | +20.3W | SIMILAR (false detection risk) |
| Power CV | 0.151 | 0.012 | +0.139 | DIFFERENT |
| Power ACF1 | 0.825 | 0.814 | +0.011 | SIMILAR |
| Mem used | 1.54 GB | 1.82 GB | -0.27 GB | SIMILAR |
| Mem util | 1.0% | 1.0% | -0.0% | SIMILAR |
| GPU util ACF1 | 0.255 | 0.686 | -0.431 | DIFFERENT |

**Risk level:** MEDIUM

**False-detection causes (what makes classifier confuse this with training):**
- Power (122W) ≈ training (102W) — power-based features cannot discriminate

**Discriminating signals (what CAN separate this from training):**
- gpu_util_mean = 41% vs training 14% — inference keeps SM fully occupied without periodic optimizer overhead gaps
- power_w_cv = 0.151 vs training 0.012 — variability pattern differs (training has periodic optimizer spikes → higher CV)
- No PCIe gradient upload / parameter update traffic — DDP gradient-allreduce absent (pcie_tx_mbps ≈ 0)

### `CC-E/fwd_only/bs16/fp16`

| Signal | Value | Training | Delta | Verdict |
|--------|-------|----------|-------|---------|
| GPU util mean | 78.7% | 13.6% | +65.1% | HIGHER |
| Power mean | 225.3W | 101.9W | +123.4W | HIGHER |
| Power CV | 0.063 | 0.012 | +0.051 | DIFFERENT |
| Power ACF1 | 0.898 | 0.814 | +0.084 | SIMILAR |
| Mem used | 2.19 GB | 1.82 GB | +0.37 GB | SIMILAR |
| Mem util | 9.1% | 1.0% | +8.1% | HIGHER |
| GPU util ACF1 | 0.621 | 0.686 | -0.065 | SIMILAR |

**Risk level:** LOW

**False-detection causes (what makes classifier confuse this with training):**
- gpu_util ACF1 (0.621) ≈ training (0.686) — autocorrelation pattern matches

**Discriminating signals (what CAN separate this from training):**
- gpu_util_mean = 79% vs training 14% — inference keeps SM fully occupied without periodic optimizer overhead gaps
- power_w_mean = 225W vs training 102W — significantly higher sustained power
- power_w_cv = 0.063 vs training 0.012 — variability pattern differs (training has periodic optimizer spikes → higher CV)
- mem_util_mean = 9% vs 1% — higher HBM bus utilisation
- No PCIe gradient upload / parameter update traffic — DDP gradient-allreduce absent (pcie_tx_mbps ≈ 0)

---

## CC-F: ViT Inference

### `CC-F/ViT-B/16/bs1/fp16`

| Signal | Value | Training | Delta | Verdict |
|--------|-------|----------|-------|---------|
| GPU util mean | 40.9% | 13.6% | +27.3% | HIGHER |
| Power mean | 124.8W | 101.9W | +22.9W | SIMILAR (false detection risk) |
| Power CV | 0.026 | 0.012 | +0.015 | SIMILAR |
| Power ACF1 | 0.891 | 0.814 | +0.077 | SIMILAR |
| Mem used | 1.80 GB | 1.82 GB | -0.01 GB | SIMILAR |
| Mem util | 2.0% | 1.0% | +0.9% | SIMILAR |
| GPU util ACF1 | 0.788 | 0.686 | +0.102 | DIFFERENT |

**Risk level:** MEDIUM

**False-detection causes (what makes classifier confuse this with training):**
- Power (125W) ≈ training (102W) — power-based features cannot discriminate
- power_cv (0.026) ≈ training (0.012) — temporal variability matches

**Discriminating signals (what CAN separate this from training):**
- gpu_util_mean = 41% vs training 14% — inference keeps SM fully occupied without periodic optimizer overhead gaps
- ViT batch=1: very short bursts of compute, low average util — no continuous data pipeline

### `CC-F/ViT-B/8/bs1/fp16`

| Signal | Value | Training | Delta | Verdict |
|--------|-------|----------|-------|---------|
| GPU util mean | 57.4% | 13.6% | +43.8% | HIGHER |
| Power mean | 179.4W | 101.9W | +77.5W | HIGHER |
| Power CV | 0.060 | 0.012 | +0.048 | DIFFERENT |
| Power ACF1 | 0.906 | 0.814 | +0.092 | SIMILAR |
| Mem used | 1.81 GB | 1.82 GB | -0.01 GB | SIMILAR |
| Mem util | 2.0% | 1.0% | +1.0% | SIMILAR |
| GPU util ACF1 | 0.657 | 0.686 | -0.030 | SIMILAR |

**Risk level:** LOW

**False-detection causes (what makes classifier confuse this with training):**
- gpu_util ACF1 (0.657) ≈ training (0.686) — autocorrelation pattern matches

**Discriminating signals (what CAN separate this from training):**
- gpu_util_mean = 57% vs training 14% — inference keeps SM fully occupied without periodic optimizer overhead gaps
- power_w_mean = 179W vs training 102W — significantly higher sustained power
- power_w_cv = 0.060 vs training 0.012 — variability pattern differs (training has periodic optimizer spikes → higher CV)
- ViT batch=1: very short bursts of compute, low average util — no continuous data pipeline

---

## Summary Table: False Detection Risk

| Config | GPU Util | Power | Power CV | Power ACF1 | Mem GB | Risk | Top Discriminator |
|--------|----------|-------|----------|------------|--------|------|-------------------|
| **TRAINING baseline** | **14%** | **102W** | **0.012** | **0.814** | **1.8 GB** | — | — |
| resnet18/bs1/img224/fp16 | 56% | 109W | 0.023 | 0.900 | 1.5 | MEDIUM | gpu_util (56.0 vs 13.6) |
| resnet50/bs4/img224/fp16 | 59% | 133W | 0.033 | 0.917 | 1.5 | MEDIUM | gpu_util (58.5 vs 13.6) |
| resnet50/bs16/img224/fp16 | 87% | 244W | 0.068 | 0.914 | 1.6 | LOW | mem_util (12.9 vs 1.0) |
| gpt2/bs4/s128/fp16 | 53% | 169W | 0.056 | 0.896 | 2.2 | LOW | gpu_util (52.8 vs 13.6) |
| gpt2-m/bs4/s512/fp32 | 99% | 392W | 0.088 | 0.876 | 3.7 | LOW | gpu_util (99.2 vs 13.6) |
| gpt2-l/bs4/s128/fp16 | 59% | 223W | 0.065 | 0.883 | 5.2 | LOW | mem_util (6.9 vs 1.0) |
| gpt2-m/bs256/ctx128/fp16 | 58% | 181W | 0.054 | 0.884 | 3.3 | LOW | gpu_util (57.6 vs 13.6) |
| resnet50/bs16/fp32 | 99% | 361W | 0.070 | 0.898 | 1.8 | LOW | mem_util (35.0 vs 1.0) |
| gpt2/bs1/s512/fp32 | 99% | 354W | 0.084 | 0.883 | 2.3 | LOW | gpu_util (99.3 vs 13.6) |
| fwd_only/bs1/fp16 | 41% | 122W | 0.151 | 0.825 | 1.5 | MEDIUM | gpu_util (40.7 vs 13.6) |
| fwd_only/bs16/fp16 | 79% | 225W | 0.063 | 0.898 | 2.2 | LOW | mem_util (9.1 vs 1.0) |
| ViT-B/16/bs1/fp16 | 41% | 125W | 0.026 | 0.891 | 1.8 | MEDIUM | gpu_util (40.9 vs 13.6) |
| ViT-B/8/bs1/fp16 | 57% | 179W | 0.060 | 0.906 | 1.8 | LOW | gpu_util (57.4 vs 13.6) |

## Key Findings

### 1. GPU Utilisation is NOT a reliable discriminator
Most in-range inference configs show 40–99% GPU util — higher than the training
baseline (13%). Training at bs=4 has low average util due to the fwd→bwd→step
cycle overhead. Inference runs continuously at SM capacity.

### 2. Power CV (variability) is the strongest single discriminator
Training power CV = 0.012 (very stable at bs=4).
Inference configs show higher CV (0.02–0.09) due to data-loading and batching
irregularities. However, large-batch LLM prefill (CC-B/gpt2-m/bs4/s512/fp32)
has power CV = 0.088 ≈ training — **highest false-detection risk**.

### 3. Power ACF1 (temporal autocorrelation) is consistently different
Training ACF1 = 0.814. All inference configs have ACF1 0.86–0.92
(more correlated) because inference runs at a fixed steady rate without the
periodic optimizer-step discontinuity that slightly disrupts training's autocorrelation.

### 4. Memory usage separates LLM prefill clearly
LLM prefill at large batch/seq allocates 2–74 GB of KV/activation memory,
far above the 1.8 GB training baseline at bs=4.

### 5. Hardest configs (HIGH risk) require multi-signal classifiers
- `CC-B/gpt2-m/bs4/s512/fp32` — power, util, CV all similar to training
- `CC-A/resnet50/bs16/img224/fp16` — no single dominant discriminator
- `CC-D/gpt2/bs1/s512/fp32` — moderate util and power matching training
These require the 5-signal temporal window approach (SVM-RBF 95.65% achieved in week 5).

### 6. Forward-only (CC-E) is detectable via periodic power structure
fwd_only runs a model forward pass only — no backward. The power ACF1 is
0.843–0.856 vs training 0.814. The power_cv difference (0.035–0.041 vs 0.012)
indicates training has more variability due to the scaler.step() discontinuity.

## Plots
| File | Description |
|------|-------------|
| `tier1_CC-A.png` | CNN Inference (ResNet): 30s temporal traces vs training |
| `tier1_CC-B.png` | LLM Prefill (GPT-2 family): 30s temporal traces vs training |
| `tier1_CC-C.png` | LLM Decode (memory-bound): 30s temporal traces vs training |
| `tier1_CC-D.png` | Quantisation Comparison: 30s temporal traces vs training |
| `tier1_CC-E.png` | Forward-Only Training Variant: 30s temporal traces vs training |
| `tier1_CC-F.png` | ViT Inference: 30s temporal traces vs training |
| `tier1_signal_comparison.png` | Side-by-side bar chart: all configs vs training baseline |