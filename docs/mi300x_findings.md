# MI300X (datacenter AMD) — telemetry, collector fixes, and cross-vendor findings

Status as of 2026-06-12. Work lives on branch **`amd-datacenter`**; telemetry
parquets are stored via git-LFS. This note summarizes the datacenter-AMD
collection and the cross-vendor classifier experiments run against it.

## 1. Hardware / environment

- 1× **AMD Instinct MI300X VF** (SR-IOV virtual function), ROCm 7.0, ~192 GB
  HBM, 750 W cap. Single-GPU.
- **VF masking:** these amdsmi fields are unavailable on the VF and read
  `N/A`/`-1`: `mm_activity` (→ encoder/decoder util), `throttle_status`, all
  `voltage_*`, per-HBM temps, fan, **edge `temperature_c`**, and
  `pcie_tx/rx`. Available: gfx/umc activity, power, clocks, hotspot+mem temp,
  ECC, XGMI, energy/PCIe accumulators, per-XCD busy, and the full RDC Tier-2
  rocprofiler counter set. Host-privileged fields await bare-metal
  permissions (open cloud-provider ticket).
- Runtime env: `source scripts/rocm_env.sh` (sets `LD_LIBRARY_PATH` for
  `librdc_bootstrap.so` + PyTorch `libgomp`, and `RDC_HOME`); `rdcd` must run
  (`sudo rdcd -u`, port 50051). The `run_amd_*.sh` scripts source it.

## 2. Collector / pipeline fixes (branch `amd-datacenter`)

The RDC Tier-2 integration was non-functional on ROCm 7 / MI300X. Four bugs
fixed in `scripts/collect_amdsmi_telemetry.py` (`RdcCollector`):

1. **Binding path** hardcoded to the ROCm ≤6 location; ROCm 7 ships bindings
   at `<rocm>/libexec/rdc/python_binding`. Now version-aware.
2. **`process()` misuse** — code iterated its return value, but `RdcReader`
   returns `None` and dispatches via `handle_field()` (raised `'NoneType' object
   is not iterable` every sample). Subclassed to capture values.
3. **Lossy union clobber** — base `process()` truncates DOUBLE fields to int
   in-place; override reads `.dbl` first, preserving fractional precision.
4. **Field-group leak** — `cleanup()` never freed the rdcd field/GPU groups
   (called a non-existent `stop()`), so after ~57 runs rdcd returned
   `RDC_ST_MAX_LIMIT` and silently disabled Tier-2. Now destroys groups +
   disconnects. Validated to 70 sequential cycles.

Also: `workloads/modern_inference.py` whisper used a hardcoded
`/home/claude/.cache` path → now `~/.cache/whisper` (`WHISPER_CACHE`).
Requires `pip install openai-whisper`.

## 3. Phase-1 corpus

`scripts/run_amd_datacenter_corpus.sh --skip-multi-gpu` → 29 workloads × 3
reps = **87 runs**. After re-collecting the 33 runs affected by the RDC leak +
whisper failure: **87/87 carry full Tier-2 profiling, 0 failures.**

- Only non-result: 3× `ffmpeg_vaapi` (~5 idle rows — datacenter cards have no
  VCN video engine). Recommend dropping it from the AMD datacenter registry or
  swapping for a compute proxy.
- **Phase 2** (multi-GPU DDP adversarial) needs an ≥8-GPU node — collected
  separately, not on this 1-GPU VF.

Data-quality sanity (peak Tier-2): gpt2 train tensor ~50% / occ ~23%; resnet50
infer tensor ~46% / occ ~51%; cufft occ ~66% / tensor 0 (FFT has no tensor
cores); idle tensor 0.

## 4. Cross-vendor experiments

All use 30 s / 15 s windows, two-stage RandomForest, grouped by run. The
NVIDIA model is trained on 1,389 NVIDIA runs (NVML). MI300X is harmonized to
the shared schema; the VF-masked edge temp is replaced by hotspot temp and
pcie (−1) → 0. Scripts: `classifier/eval_mi300x_transfer.py`,
`classifier/eval_mi300x_experiments.py`.

### 4.1 Zero-shot transfer NVIDIA → MI300X — does **not** generalize

| Setting | Three-way | Binary is-training |
|---|---|---|
| Paper within-NVIDIA (reference) | ~98.6% | ~98.6% |
| **Zero-shot NVIDIA → MI300X** | **50.2%** | **63.9%** (P 0.37 / R 0.62) |
| **Within-MI300X (5-fold grouped CV)** | **98.9%** | **98.9%** |

The signal is fully present on MI300X (within-vendor 98.9%), but the
NVIDIA-trained model collapses to near-chance zero-shot. **The failure is
cross-vendor domain shift in absolute telemetry** (power up to 750 W, 192 GB
VRAM, different clock/idle ranges) — not missing signal. The classifier
over-predicts "training" on AMD (precision 0.37). Stage-1 (ML-vs-rest) keys on
absolute level features, which differ most across vendors.

### 4.2 Can calibration fix it? Partly.

| NVIDIA → MI300X | Three-way | Binary |
|---|---|---|
| Raw zero-shot | 50.2% | 63.9% |
| Per-device z-score (unconditional) | 54.2% | **54.7%** (hurts) |
| **Idle-baseline calibration** | 60.0% | **75.0%** |

Unconditional per-device z-scoring **hurts** — it removes the absolute
util/power level that separates idle/other from training. **Idle-baseline
calibration** (center each device at its own idle floor, scale by device std)
recovers ~⅓ of the gap (64 → 75% binary) because it preserves "how far above
idle," which is comparable across devices. It does **not** fully close to the
99% ceiling — remaining gap needs label-aware domain adaptation or a few AMD
training examples (few-shot / leave-one-GPU-out).

### 4.3 Tier-2 (rocprofiler) ablation — within-MI300X, binary training detection

| Features | Acc | F1(train) |
|---|---|---|
| Tier-1 only (NVML-equivalent) | 99.65% | 0.993 |
| Tier-2 only (tensor/occ/FLOP/cycles) | 99.05% | 0.983 |
| Tier-1 + Tier-2 | 99.59% | 0.992 |

On this corpus **Tier-1 already saturates**, so Tier-2 adds no marginal
accuracy. But **Tier-2 alone nearly matches full Tier-1** — a strong
independent signal; `rdc_elapsed_cycles`, `rdc_sm_occupancy`,
`rdc_fp32_pipe_active` rank just below `gpu_utilization_pct` in importance.
Tier-2's *marginal* value should appear where Tier-1 fails — the white-box
LoRA / dilution evasions (the paper's <50% case), which this tier set does not
include. A **hard-evasion collection (whitebox LoRA + dilution × 3 reps)** is
in progress to test exactly this; the Tier-2 ablation will be re-run on it.

### 4.4 Tier-2 on the hard evasions (white-box LoRA + dilution) — does NOT rescue detection

Trained a binary is-training detector on the Phase-1 MI300X corpus (which
already includes dilution-style `adversarial_L_diluted_5/10`) and tested
detection on the **held-out** white-box hard-evasion set (24 runs).
Script: `classifier/eval_mi300x_hardevasion.py`.

| Feature set | Recall on disguised training | FPR on inference control |
|---|---|---|
| Tier-1 | **2.8%** | 0.0% |
| Tier-2 | 4.5% | 6.1% |
| Tier-1 + Tier-2 | 2.4% | 0.0% |

The white-box LoRA/dilution evasions **defeat the detector almost entirely**
(recall <5%) — worse than the paper's NVIDIA white-box case (<50%), because
this is a leave-the-family-out test. **Tier-2 does not close the gap**: the
dilution attack hides training in the time-average (≤10% training fraction,
0.22% trainable LoRA params), so 30 s window-aggregate stats — even of tensor
activity — look like inference, and heavy inference (e.g. resnet50) also drives
high tensor %, so tensor activity alone doesn't separate train from infer.
Tier-2 nudges recall 2.8→4.5% but trivially and at higher FPR.

**Implication:** Tier-2's marginal value is NOT free on disguised-training
detection. What's likely needed: (a) train *with* white-box examples
(adversary-defender loop on AMD, as the paper does for NVIDIA), and/or
(b) finer-temporal / burst features rather than 30 s aggregates, since
dilution specifically attacks the time-average. Re-running this with whitebox
included in training (leave-one-whitebox-out) is the fair next test.

## 5. Recommendations / open work

- Re-run the Tier-2 ablation on the hard-evasion set (in progress) — the
  expected place Tier-2 beats Tier-1.
- Few-shot / leave-one-GPU-out: how many MI300X runs in training close the
  64→99% transfer gap?
- Bare-metal re-collection once permissions land (recovers pcie/throttle/
  voltage/edge-temp signals masked on the VF).
- Drop or replace `ffmpeg_vaapi` in the AMD datacenter registry.
- Collect Phase-2 multi-GPU DDP on an ≥8-GPU node.
