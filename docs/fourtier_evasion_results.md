# Four-tier evasion analysis: does BMC node power catch the memory-minimization evaders?

**Forward-looking follow-up** (not part of the submitted paper). Reproduce with
`python -m classifier.eval_fourtier_evasion` on data in
`data/h200_nvl_scope/{full_sensor,evaders_4tier}`.

## Question

The NVML-only training-vs-rest classifier is fooled by two evasion families that
run training with a **minimized resident footprint** (`train_gpt2cu -b 1` +
gradient accumulation, so the effective batch/training is unchanged but the
activation memory is ~1/16):

- `adversarial_mm_min_noshape` — tiny footprint, high util, *no* cadence shaping
- `adversarial_mm_freq` — grad-accum 1, so weight-write DRAM traffic is near-continuous

Does adding a fourth, harder-to-spoof sensor — **BMC/DCMI whole-node power** —
catch what NVML misses? And where do the DCGM and oscilloscope tiers land?

## Design

Clean node-local ablation: **every** run — the honest reference corpus AND the
evaders — was captured on the *same* H200 with all four sensors (NVML + DCGM +
scope + BMC). Reference = the honest 4-sensor corpus (`full_sensor/`, evader
families held out); test = `evaders_4tier/`. Only the feature columns change
across tiers, so any difference is attributable to the sensor, not the data.
1911 reference windows (1045 training / 866 non-training), 30s/15s windows.

## Result

Hard evasion rate (fraction of an evader's windows called "rest") is saturated —
the informative view is **mean P(training)** across tiers (evaders are all truly
training, so higher = caught; 0.5 = decision threshold):

| evader family | NVML | +DCGM | +scope | +BMC | all four |
|---|---|---|---|---|---|
| mm_min_noshape | 0.212 | 0.302 | 0.360 | **0.191** | 0.430 |
| mm_freq | 0.096 | 0.221 | 0.272 | **0.116** | 0.369 |
| mm_min_shaped | 0.835 | 0.807 | 0.830 | 0.778 | 0.860 |
| mm_spoof_infer_shaped | 0.835 | 0.807 | 0.829 | 0.778 | 0.860 |

Hard evasion: mm_min_noshape and mm_freq evade **100% on every tier including all
four sensors**; the two shaped/padded families are **detected 100% on NVML alone**.

### 1. BMC node power does *not* catch the evaders
Adding BMC moves P(training) by ≈0 — in fact **−0.02** for mm_min_noshape
(0.212→0.191) and +0.02 for mm_freq. Reason: whole-node power is a workload-
*intensity* proxy that inference drives just as high as training.

| class | BMC node power (window mean) |
|---|---|
| mm_min_noshape (evader) | 790 W |
| mm_freq (evader) | 841 W |
| resnet50_inference (honest **inference**) | 718 W |
| qwen25_7b_fp16 (honest **inference**) | 829 W |
| qwen25_32b_fp16 (honest **inference**) | 938 W |
| phi3_mini_lora (honest **training**) | 553 W |

The evaders draw *more* node power than several honest inference classes, so a
node-power feature cannot separate "training" from "heavy inference" — exactly the
distinction the classifier needs.

### 2. DCGM and scope *do* help, but not enough
Each adds ~+0.1–0.18 to P(training); DCGM's `dram_active` catches mm_freq's
continuous weight-write traffic (dram 0.217 vs 0.138 for mm_min_noshape). All four
tiers together nearly double P(train) — but leave it at **0.37–0.43, still below
0.5**. The memory-minimization evasion survives the full four-sensor stack at the
default threshold. (At a lowered threshold ~0.35 the all-tier classifier would
catch both, at a false-positive cost — the stack pushes the evaders *to* the
boundary, not past it.)

### 3. Why footprint, not power, is the leak
The surviving evaders run at **95–96% util but only ~3.6 GB** resident — a
footprint like resnet50 *inference* (4.8 GB), far below real training
(phi3-LoRA 8.9 GB, online-learning 14 GB, qwen-7B inference 121 GB). The
classifier associates training with a larger footprint, so high-util + tiny-
footprint reads as compute-bound inference. No power sensor addresses footprint.

Conversely the *shaped* families backfire: dropping util to 9% via idle-gap
shaping (and padding the footprint to 5.3 GB in `spoof_infer`) reintroduces a
bursty step cadence the classifier reads as training — detected 100% on NVML.

## Takeaway

The extra physical sensor (BMC node power) does **not** close the memory-
minimization evasion gap; it is redundant with NVML power and, like all power
channels, cannot distinguish training from heavy inference. The evasion attacks
the *memory-footprint* feature, which is orthogonal to every power/thermal
channel. Sensors that see *what kind* of work is happening (DCGM pipe/DRAM
activity, scope ripple structure) help more than sensors that see *how much*
power is drawn — but even all four together are insufficient here at the deployed
threshold. Closing this gap needs a footprint- or cadence-aware feature (or a
lower threshold with its FP cost), not another power probe.
