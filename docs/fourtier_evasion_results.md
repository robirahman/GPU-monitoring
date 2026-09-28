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

## The co-execution evader (MPS) — the decisive case

The above evaders leave *some* residual signal (DCGM/scope raise P(train) toward
the threshold). The natural strongest attack flattens **every** channel at once:
run tiny-batch training concurrently with a steady DRAM-bandwidth cover kernel
under **CUDA MPS**, so they share the GPU (`scripts/covered_train_mps.sh`,
`scripts/cover_kernel.py`; data in `coexec_evaders/`, cover 3 GB and 8 GB at duty
0.3). Tuning note: cover duty 1.0 *starves* training (0 steps/900 s); 0.3 pins
NVML flat-high while training still progresses (~60 s/step).

| evader | NVML | +DCGM | +scope | +BMC | all four |
|---|---|---|---|---|---|
| mm_mps_cover (6.9 GB) | 0.966 | 1.000 | 0.915 | 0.966 | 0.966 |
| mm_mps_cover (12 GB) | 0.975 | 0.975 | 0.938 | 0.975 | 0.963 |

**It evades ~96–97% on every tier, including all four sensors** (P(train)≈0.38–0.44).
And it *neutralizes the DCGM tier* that was the best added discriminator: the cover
drives `dram_active` to **0.83** (HPC-like; mm_freq was 0.22), so DCGM gives no
signal — adding it even *raises* evasion (0.966→1.000). Scope helps only marginally
(ripple is smoothed to ~1.5–1.9 A, steady, vs 9.3 A for the bursty bare evaders).
BMC draws the **highest node power of any class (947–952 W)** — cover + training
combined — yet is useless, because that is the same band as qwen-32B *inference*
(938 W).

## Footprint-min evasion generalizes across model scale

Re-running the tiny-batch (`-b 1 -r 2`) footprint evasion on GPT-2 **355M and 774M**
(`scale_evaders/`) evades **100% on all four tiers** (P(train) 0.25 / 0.23, creeping
up only slightly from 124M's 0.11 as weights+optimizer grow the footprint). Even at
774M the minimized-activation footprint stays inference-like. So this is not a
124M-only artifact.

## Takeaway

Across three evasion families and four sensor tiers, the extra physical/telemetry
sensors do **not** provide robust protection:

- **BMC node power** is redundant with NVML power and, like all power channels,
  cannot separate training from heavy inference (evaders sit at 790–952 W, right
  among honest inference at 718–938 W). It never helps, in any family.
- **DCGM (DRAM activity)** and **scope (ripple)** help against the *simple*
  footprint evaders — but a co-execution adversary flattens exactly those channels
  (dram→0.83, ripple→steady), neutralizing them. Against the co-exec evader all
  four tiers together still leave ~96 % evasion.
- The **memory-footprint** leak (tiny-batch training reading as inference) is
  orthogonal to every power/thermal channel and persists across model scale.

Bottom line: a determined co-execution adversary can flatten every observable
channel while genuinely training, and no combination of these four sensors detects
it at the deployed threshold. Robust detection would need either a fundamentally
different signal (e.g. footprint/cadence-aware features, or isolating the tenant's
own process rather than aggregate node telemetry) or accepting a much lower
threshold and its false-positive cost — not another physical power probe.

*(Reproduce: `python -m classifier.eval_fourtier_evasion`; evader data in
`data/h200_nvl_scope/{evaders_4tier,coexec_evaders,scale_evaders}`.)*
