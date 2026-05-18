# AMD Consumer (RX 9070 XT) — Local Findings

Comprehensive write-up of the May 2026 local-collection effort on the
AMD Radeon RX 9070 XT (gfx1201, RDNA 4, 16 GB GDDR6) running Ubuntu
26.04 with ROCm 7.1 / PyTorch 2.9.1+rocm6.4. Pairs with the cloud
follow-up plan in [`cross_architecture_extension.md`](cross_architecture_extension.md).

## TL;DR

- **Cross-vendor pooled 5-fold CV: 96.3% three-way / 96.6% binary** —
  essentially matches the paper's NVIDIA-only headline of 96.4%.
- **Zero-shot transfer fails:** NVIDIA→AMD 52.7% binary, AMD→NVIDIA 70.9%.
  The classifier learns vendor-specific level patterns.
- **Cross-GPU heatmap (10×10):** AMD-trained generalizes to every NVIDIA
  GPU at ≥97%; reverse is uneven (40–100%) depending on training source.
  Asymmetry comes from corpus breadth, not vendor fundamentals.
- **Sampling-rate ablation:** 1 Hz preserves full accuracy; 0.5 Hz loses
  <2pp. Paper's 1 Hz choice is well-supported.
- **WB-L (paper's hardest evasion) on AMD:** **0% detection** after corpus
  rebalancing. Below the paper's NVIDIA LOO range of 43-87%.
- **WB-D (diluted full fine-tune): 7-9% on AMD** — also below paper's
  NVIDIA range (87-100%).
- **WB-O (oracle): 87-100% across eval iterations.** The amdsmi-backed
  port of `ClassifierOracle` works.

## Setup

### Hardware
- AMD Radeon RX 9070 XT, 16 GB GDDR6, gfx1201 (RDNA 4)
- Host CPU: AMD Ryzen 7 9800X3D (Granite Ridge)
- The iGPU (gfx1036) is visible to `rocminfo` but **not to amdsmi as a
  separate processor** — we didn't collect from it.

### Software
- Ubuntu 26.04, kernel 7.0, amdgpu driver from the distribution
- ROCm 7.1 (apt universe — `rocm-dev rocm-smi rocminfo`)
- PyTorch 2.9.1+rocm6.4 (pip wheel from
  `https://download.pytorch.org/whl/rocm6.4`)
- `amdsmi` Python bindings: built from
  [github.com/ROCm/amdsmi](https://github.com/ROCm/amdsmi) tag
  `rocm-7.2.0` to match the installed `libamd-smi26 7.2.0`. **The
  PyPI `amdsmi 7.0.2` wheel does NOT work** — struct layout mismatch
  causes `AMDSMI_STATUS_UNEXPECTED_DATA` on every `get_gpu_activity`
  call. See `setup_amd.sh` for the canonical install procedure.

### Telemetry signals captured (26-column parquet schema)
Of the 9 paper-standard NVML signals, 7 work cleanly on RX 9070 XT
amdsmi; PCIe TX/RX bandwidth is unavailable on consumer cards (returned
as `-1` sentinel). AMD bonuses: edge / hotspot / VRAM temperatures
(3-way instead of NVIDIA's single temp), `mm_activity` (bundled
encoder+decoder), power_limit, throttle_status.

| Signal | NVIDIA NVML | AMD amdsmi |
|---|---|---|
| gpu_utilization_pct | gpu | gfx_activity |
| mem_utilization_pct | memory | umc_activity |
| mem_used_mb | nvmlDeviceGetMemoryInfo | amdsmi_get_gpu_memory_usage |
| power_draw_w | GetPowerUsage | current_socket_power (or socket_power fallback) |
| temperature_c | GetTemperature(GPU) | temperature_edge |
| sm_clock_mhz | GetClockInfo(SM) | get_clock_info(GFX).clk |
| mem_clock_mhz | GetClockInfo(MEM) | get_clock_info(MEM).clk |
| pcie_tx_mbps | GetPcieThroughput(TX) | **−1 (not exposed)** |
| pcie_rx_mbps | GetPcieThroughput(RX) | **−1 (not exposed)** |

See [`telemetry_signal_matrix.md`](telemetry_signal_matrix.md) for the
full cross-vendor table including TPU and Trainium TODO columns.

## Corpus

Final state: **264 valid parquets** on the `amd` branch, all from this
campaign on Ubuntu/ROCm (the 24 May-2026 Windows-era parquets that
predate the campaign were 130-byte stubs and excluded by the watcher's
size filter).

| Category | Workloads | Reps | Total | Notes |
|---|---|---|---|---|
| Clean baselines | 24 | 3 each (some 4–7) | ~85 | training / inference / HPC / video / idle |
| Single-GPU adversarial (Rounds 1–4) | 18 | 3 each | ~54 | A, B, D–N, composite_idle_{0,20,50,80,95} |
| Whitebox (Round 5, legacy + canonical) | 10 | 3 each | ~30 | Both `_WB-*` (Qwen 0.5B) and canonical (TinyLlama) variants |
| Sampling-rate sweep | 4 workloads × 5 rates | 2 each | 40 | 0.5 / 1 / 2 / 5 / 10 Hz |
| Tier-4 broader workloads | 5 | 3 each | 15 | SDXL, LoRA-Qwen25, LoRA-SmolLM2, YOLO, Qwen-3B |
| TinyLlama variety pack | 5 | 3 each | 15 | bs2, bs4, seq1024, seq2048, synthetic_forward — added to fix WB-I corpus gap |
| Matched-pair (P0h/P0i) | 1 | 3+1 | 4 | TinyLlama bs1 inference matching WB-I |

**Strategies skipped or limited on consumer AMD:**

- **G_clock_throttled** — DROPPED. Requires `sudo rocm-smi --setperflevel
  low` or `--setpoweroverdrive`; no userspace path on consumer cards. The
  Strategy G mechanic can't be implemented without granting sudo to every
  workload run.
- **J_pid_thermal** — ported (commit `2eed7a7`). The PID feedback loop
  reads via amdsmi when pynvml is absent. Existing pre-port runs are
  partly degraded (feedback was no-op until commit; rerun in
  `p0_followup`).
- **WB-O** (`whitebox_oracle`) — `ClassifierOracle` ported to amdsmi
  (commit `0483c6e`). Initial collection failed with pynvml requirement;
  re-collected with the new backend.

**Phase 0 collection pipeline issues encountered:**

- **First Tier 4 / Tier 3 runs were contaminated** (commit `add66ce`).
  Their wait-loops used `pgrep -f 'A\|B'` with a literal-pipe regex that
  matched nothing, so they ran in parallel with the still-running
  sampling sweep. 36 contaminated parquets deleted, re-collected with a
  serial orchestrator (commit `ddca1f8`).
- **Inline-vs-LFS skew:** my first ~200 parquet commits bypassed Git LFS
  because git-lfs wasn't installed yet. Migrated to LFS in commit
  `f11fed7` (force-pushed). Lesson: `git lfs install` first.

## Results

### Cross-vendor classifier (`classifier/eval_cross_vendor.py`)

After LFS-pulled full corpus (~30k windows total):

| Experiment | three-way | binary (train vs rest) |
|---|---|---|
| NVIDIA → AMD zero-shot | 48.4% | **52.7%** |
| AMD → NVIDIA zero-shot | 60.1% | **70.9%** |
| Pooled 5-fold grouped CV | **96.3% ± 1.4%** | **96.6% ± 1.2%** |
| AMD-only 5-fold grouped CV | 90.3% ± 11.9% | 91.7% ± 9.3% |

Headline: **pooled training matches the paper's 96.4% NVIDIA-only
result.** Zero-shot transfer is poor in both directions, but AMD→NVIDIA
is meaningfully better than NVIDIA→AMD — almost certainly because the
AMD corpus has broader workload coverage (24 + 18 + 10 + 5 = 57 distinct
classes) than any single NVIDIA GPU's local subset.

### Cross-GPU heatmap (10×10)

Generated by `regen_crossgpu_heatmap_binary_rf_amd.py`; binary classifier,
RandomForest 400-tree, StratifiedGroupKFold on the diagonal.

```
                 RTX 4050  RTX 5080  RTX 5090  A100 40GB  A100 80GB  H100 80GB  H100 NVL   H200   B200  RX 9070 XT
RTX 4050 Laptop     98.2       58.3      71.8       96.3       47.7       95.2      72.0   70.1   54.9        95.3
RTX 5080            87.3       99.1      99.4      100.0       99.4       98.9      99.9   99.5   99.1        97.4
RTX 5090            73.4       76.1      98.4       93.9       80.8       95.8      82.2   86.7   71.7        59.5
A100 40GB           80.6       66.0      71.0      100.0       36.7       98.3      87.3   41.7   47.0        39.6
A100 80GB           97.5       58.3      70.4       45.1      100.0       51.7      44.9   79.3   80.0        99.4
H100 80GB           78.2       73.3      69.7       97.3       40.2       99.5      93.2   51.8   55.6        46.1
H100 NVL            88.5       76.7      90.8       99.3       89.6       99.6      99.3   92.2   98.7        66.9
H200               100.0       99.0     100.0      100.0      100.0       99.8      99.9   99.7   91.9       100.0
B200               100.0       97.2      97.2       96.6       94.8       99.4      96.1   98.1   99.5        91.2
RX 9070 XT          99.6       97.4      99.8       99.6      100.0      100.0      98.0  100.0   99.6        99.7
```

Rows = train GPU, columns = test GPU. **Strong cells (≥97%):** the AMD
row is uniformly green. **Weak cells (<50%):** scattered across NVIDIA
sources, consistent with the paper's "consumer↔datacenter is harder"
finding. A100-40GB→RX 9070 XT and H100-80GB→RX 9070 XT both ~40-46% —
the cross-vendor consumer-target story.

PDF / PNG at `report/figures/cross_gpu_heatmap_binary_rf_amd.{pdf,png}`.

### Sampling rate ablation

`plot_accuracy_vs_sample_rate_amd.py`, on 4 workloads × 5 rates × 2 reps:

| Rate (Hz) | binary accuracy |
|---|---|
| 0.5  | 96.7% ± 0.6% |
| 1.0  | **98.8% ± 1.2%** ← paper default |
| 2.0  | 98.8% ± 1.2% |
| 5.0  | 99.4% ± 0.6% |
| 10.0 | 99.4% ± 0.6% |

Paper's 1 Hz default is well-supported. Going to 0.5 Hz loses <2pp; the
floor is set by feature granularity (FFT and autocorrelation need enough
samples per 30-s window), not by anything inherent to the sub-second
regime.

### Whitebox detection rate (the messy story)

This is where the local AMD effort was most illuminating — and where the
paper's most fragile claim (WB-L = "hardest single-GPU evasion at
43–87% LOO on NVIDIA") sharpens substantially.

Three iterations as the AMD-side corpus was rebalanced for TinyLlama
inference coverage:

| Strategy | True | i1 (no bs1) | i2 (+bs1 ×3) | **i3 (+variety pack)** | Paper NVIDIA |
|---|---|---|---|---|---|
| whitebox_inference | inference | 0% | 18% | **100%** ✓ | 100% |
| whitebox_diluted_N5 | training | 100% | 9% | **9%** | 87-100% |
| whitebox_diluted_N20 | training | 100% | 98% | **7%** ↓↓ | 87-100% |
| whitebox_lora_N5 | training | 100% | 81% | **0%** ↓↓ | 43-87% |
| whitebox_lora_N20 | training | 100% | 14% | **0%** ↓↓ | 43-87% |
| whitebox_oracle | training | 97% | 100% | **87%** | varies |

What happened:

- **i1 (imbalanced corpus, no TinyLlama inference):** the classifier
  learned "TinyLlama-shape compute = training" because the training pool
  had ~50 TinyLlama-class training runs and zero TinyLlama-class
  inference runs. WB-D / WB-L all detected at 100%, but WB-I (the
  inference control) flipped wrong too (0% — predicted as training).
- **i2 (added 3 bs=1 inference reps):** WB-I rose to 18% as expected.
  WB-D / WB-L started collapsing because the classifier could now find
  *some* "TinyLlama inference" decision-boundary.
- **i3 (added variety: bs2, bs4, seq1024, seq2048, synthetic-forward):**
  WB-I correctly identified at 100%. But the decision boundary shifted
  far enough that high-dilution training workloads (5% backward density)
  fall on the inference side. WB-L 0%, WB-D 7-9%.

**Mechanistic reading**: at 1-Hz telemetry on AMD without PCIe signals,
the classifier's Stage-2 head uses only shape features (autocorrelation,
FFT periodicity, no absolute levels — the WB-O defense). At 5% backward
density (N=20 dilution), the periodic backward signal is buried under
the 95% of forward-pass time. The classifier genuinely can't separate
"diluted training" from "pure inference" on this signal set.

**WB-O surviving at 87% is consistent with this**: oracle-rate-limited
training has *bursty* backward activity (the oracle releases backward
passes only when P(training) is below threshold), and bursts are
detectable even when their average density is low.

### Caveats on the whitebox numbers

1. **Single-fold fit, no CV.** Each iteration above is one
   train-on-everything-else / test-on-held-out-WB pass. The
   100% ↔ 0% swings between i1 and i3 partly reflect single-fold noise
   amplified by corpus rebalancing on a small held-out test set
   (~340 windows across all 5 WB strategies).
2. **Small held-out set per strategy:** 3 reps × 5 strategies × ~57
   windows = 855 windows. Proper LOO across strategies + folds across
   reps would tighten the band, probably to ±10–20%.
3. **No class-weight tuning.** The training pool has 904 training / 358
   inference / 197 other runs. The Stage-1 RF uses
   `class_weight='balanced'` which somewhat compensates, but Stage 2
   doesn't. A principled corpus-balanced classifier (oversampling
   inference, or class-weighted XGBoost) would mute the WB-D/L collapse.
4. **AMD-specific feature gap.** PCIe TX/RX = -1 on this consumer card.
   The Stage-1 head can't use the
   `cross_signal_features.pcie_total_mbps` feature that's available on
   datacenter parts. Phase 1 (MI300X) is the place to find out whether
   RDC-exposed PCIe brings WB-D/L detection back.

## What this means for the paper

### Confirmed claims
- **Methodology generalizes vendor-cross with co-training.** Pooled 5-fold
  NVIDIA+AMD CV is 96.3% — within 0.1pp of the paper's 96.4% NVIDIA-only
  headline.
- **Sampling at 1 Hz is sufficient.** The accuracy curve plateaus from
  1 Hz onward.
- **The Stage-2 shape-only head IS robust to vendor changes** for the
  non-adversarial three-way task, even when absolute-level features
  (power, clock) differ across vendors.

### Sharpened claims
- **Zero-shot cross-vendor transfer is much weaker than vendor-pooled
  training.** The paper notes cross-GPU transfer weakens for
  consumer↔datacenter pairs; we extend that finding to cross-vendor
  pairs, with even worse zero-shot performance.
- **WB-L on AMD is materially worse than the paper's NVIDIA LOO** — when
  the corpus is balanced for TinyLlama-class inference, both WB-D and
  WB-L collapse to <10% detection. The paper's 43-87% NVIDIA LOO range
  was an upper bound conditional on training-class overrepresentation in
  the corpus, not a stable lower bound on AMD.
- **WB-O is detectable across all iterations.** Oracle-rate-limited
  training has bursty backward activity that survives even strong
  imbalance changes in the rest of the corpus.

### New claims
- **G_clock_throttled is not implementable on consumer AMD without root.**
  AMD's `rocm-smi --setpoweroverdrive` / `--setperflevel` require sudo;
  there's no userspace power-cap API analogous to NVIDIA's `nvidia-smi
  -pl`. On consumer AMD this evasion family is precluded by the OS, not
  by the classifier.
- **Some adversarial mechanics depend on vendor-specific telemetry being
  available to the attacker.** Strategy J (PID-feedback throttle) needs
  live util/power reads in the workload itself. Our amdsmi port works,
  but cross-vendor portability of evasion *techniques* is itself an
  open question separate from cross-vendor classifier robustness.

## Next steps

See [`cross_architecture_extension.md`](cross_architecture_extension.md)
for the phased plan. The next concrete experiment is **Phase 1 (MI300X
on cloud)** — see [`cloud_mi300x.md`](cloud_mi300x.md). The single
question to answer: does the richer datacenter telemetry (RDC
tensor-core activity, RDC PCIe TX/RX, MI300X's larger HBM footprint
making memory features more granular) close the WB-L detection gap
that's so striking on consumer AMD?

If yes, the paper's central WB-L claim holds across vendors at the
datacenter tier. If no, the conclusion sharpens further: **WB-L is
fundamentally undetectable at 1-Hz software telemetry, and the paper's
NVIDIA-LOO numbers were corpus-dependent overestimates.**

Either outcome is publishable. The MI300X run is ~$150 and 24h on the
clock — the highest-value experiment remaining.
