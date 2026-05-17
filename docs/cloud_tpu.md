# Next-Steps Doc: Google TPU Corpus (v6e / Trillium)

## Why TPU at all?

The paper claims the methodology generalizes via "physical effects of
computation" — backward passes are necessarily 2× the FLOPs of forward
and consume more energy. If this is fundamental, the same detection
signal should appear on **any** training accelerator, not just GPUs.

TPU has:
- A completely different compute model (systolic-array matmul units,
  scheduled per-step, no per-warp util concept).
- Telemetry exposed through `tpu_info` (Google-published Python library)
  rather than NVML/amdsmi.
- A different memory hierarchy (HBM but addressed differently).

If a NVIDIA+AMD-trained classifier still detects TPU training **zero-shot**,
that's a strong vendor-invariant story. If it doesn't and only the Stage-2
shape-only head (autocorrelation/FFT/periodicity, no levels) transfers,
that's still a clean finding for the paper.

## Existing TPU work in the repo

The `feat/tpu-adaptation` branch has scaffolding code from May 2026:

- `scripts/collect_tpu_telemetry.py` — TPU collector mirroring the NVIDIA
  one, uses `tpu_info` library
- `workloads/tpu_pytorch_training.py` — PyTorch XLA ResNet/CIFAR10 training
- `run_tpu_test.sh` — minimal test script
- `tpu_deployment_notes.md` — GCP zone/quota findings (see below)
- `data/tpu_idle.parquet` — 1 sample idle parquet

**Important:** the branch also *deletes* the AMD work, so don't merge it
into `amd` whole. Cherry-pick the TPU additions on top of `amd`.

## Provisioning

From `feat/tpu-adaptation:tpu_deployment_notes.md` (already in the repo
on that branch), the actually-working combination as of May 2026 is:

- **Zone:** `europe-west4-a`
- **Accelerator:** `v6e-1` (single Trillium chip — cheapest option)
- **Runtime:** `v2-alpha-tpuv6e`

Other zones / generations failed with `LOCATION_POLICY_VIOLATED`,
`QUOTA_EXCEEDED`, or `Insufficient capacity` at that time. Quota for
v5p and v6e-8 in particular needed a manual request.

### Service account (one-time setup)

```bash
# Already done in feat/tpu-adaptation's history if the project is the same:
gcloud iam service-accounts create tpu-service-account \
    --display-name="SPAR TPU monitoring"

# Roles:
for role in tpu.admin storage.admin logging.logWriter monitoring.metricWriter; do
    gcloud projects add-iam-policy-binding focus-surfer-494820-g0 \
        --member="serviceAccount:tpu-service-account@focus-surfer-494820-g0.iam.gserviceaccount.com" \
        --role="roles/${role}"
done
```

(Project ID from the May 2026 notes — replace if using a different
project.)

### Provisioning a v6e-1 (cheapest, ~$2.70/hour)

```bash
ZONE=europe-west4-a
TPU_NAME=spar-tpu-1
ACCEL_TYPE=v6e-1
RUNTIME_VERSION=v2-alpha-tpuv6e

gcloud compute tpus tpu-vm create ${TPU_NAME} \
    --zone=${ZONE} \
    --accelerator-type=${ACCEL_TYPE} \
    --version=${RUNTIME_VERSION} \
    --service-account=tpu-service-account@focus-surfer-494820-g0.iam.gserviceaccount.com

# Once provisioned:
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE}
```

For the Phase 3 full corpus, scale up to **v6e-8** (~$22/hour). Same
incantation with `--accelerator-type=v6e-8`. Quota may need to be raised
first via the GCP console.

## Phase 2 — telemetry probe (1-2 hours, $5)

On the v6e-1 SSH session:

```bash
git clone https://github.com/robirahman/SPAR-GPU-monitoring.git
cd SPAR-GPU-monitoring
git checkout feat/tpu-adaptation   # has the existing TPU scaffold
git lfs install && git lfs pull    # only need NVIDIA/AMD parquets if doing eval here
                                   # for just the probe, skip lfs pull

pip install tpu-info               # Google's library
# tpu_info may need a recent version that supports v6e — check
# https://pypi.org/project/tpu-info/

python - <<'PY'
from tpu_info import metrics, device
print(f"Chip count: {device.get_chip_count()}")
for chip in range(device.get_chip_count()):
    m = metrics.get_chip_usage(chip)
    print(f"chip {chip}: {m}")
    # Inspect the dict — what keys are there? util? memory? power?
PY
```

Expected (from public docs, not yet verified on v6e):

- `tpu_info` is the official Google library, post-`gcloud monitoring` deprecation
- Returns chip-level: TC utilization (%), HBM usage (GB), device temp (°C),
  duty-cycle percentage
- **Does NOT expose: power draw in watts, separate memory clock, PCIe TX/RX**
  (these are not exposed by Google at the host level; you can get
  approximate power via tensorboard plugins but not at 1 Hz polling)

Fill the corresponding column in `docs/telemetry_signal_matrix.md` with
the actual observed signals.

## Phase 3 — full TPU corpus (6–8 hours on v6e-8, ~$160)

After Phase 2 confirms the signal set and we've decided on collector
architecture (likely "standalone collector + downstream normalizer"):

### Adapt the collector

```bash
# In an /amd/ working tree:
git checkout amd
git checkout feat/tpu-adaptation -- scripts/collect_tpu_telemetry.py \
                                    workloads/tpu_pytorch_training.py

# Extend collect_tpu_telemetry.py:
# - Write 26-column parquet matching scripts/collect_amdsmi_telemetry.py schema
# - Signals that TPU doesn't expose: set to -1 sentinel (PCIe-on-AMD precedent)
# - Add `gpu_name` field set to "Google TPU v6e" so vendor detection works
```

### Workloads to collect

Mostly clean workloads — adversarial families are GPU-specific (most
require nvidia-smi / amdsmi controls TPU doesn't have).

```python
# workloads/registry_tpu.py (new file)
WORKLOAD_REGISTRY_TPU = {
    "idle": ...,
    "tpu_resnet_cifar10": ...,                    # XLA ResNet, 10 epochs
    "tpu_resnet_cifar10_bf16": ...,               # bf16 native on TPU
    "tpu_mlp_cifar10": ...,
    "tpu_bert_sst2": ...,
    "tpu_gpt2_wikitext2": ...,
    "tpu_llm_infer_phi3_mini": ...,
    "tpu_llm_infer_qwen25_1p5b": ...,
    "tpu_llm_infer_qwen25_3b": ...,
    "tpu_jax_fft": ...,                           # HPC
    "tpu_lora_finetune_qwen25_0p5b": ...,
}
```

Workloads share most code with the NVIDIA/AMD versions. The launchers
just need:
- `import torch_xla.core.xla_model as xm` and `device = xm.xla_device()`
- `xm.optimizer_step(optimizer)` instead of `optimizer.step()`
- `xm.mark_step()` after each forward/backward to actually execute

For LLM inference, PyTorch XLA's KV-cache support is shakier than CUDA.
Consider using JAX/Flax-based inference (e.g., MaxText) as an alternative
if PyTorch XLA inference is buggy.

### Launch

```bash
nohup ./scripts/run_tpu_corpus.sh --reps 3 \
    > data/collection_logs/tpu_corpus.out 2>&1 &
```

(write this orchestrator analogously to `run_amd_corpus.sh` — same shape,
calls `python scripts/run_workload_tpu.py --workload <X>` per run)

## Risks / unknowns

- **`tpu_info` polling granularity.** Docs suggest 1 Hz is supported but
  not guaranteed. If it caps at 5 s, the 30-s window classifier still has
  6 samples per window, which is OK, but the autocorrelation features
  degrade. Verify in Phase 2.
- **Power not exposed.** TPU pods have wall-power telemetry at the
  chassis level (via Google's cloud monitoring API) but not per-chip at
  1 Hz. The `power_draw_w` column will be `-1` for all TPU rows. The
  classifier's Stage-1 head leans heavily on power; expect lower
  accuracy. Stage-2 (shape only) should be more robust.
- **Cost-overrun risk.** TPU VMs charge from the moment they're CREATED,
  not when you SSH in. Don't leave a v6e-8 idle. Use
  `gcloud compute tpus tpu-vm delete ${TPU_NAME} --zone=${ZONE}` aggressively.
- **XLA compilation time.** First step of each XLA run can take 5–10 min
  to compile. Bake in a `--duration` of 600 s minimum or you'll measure
  compile time, not training.
- **HBM unified addressing.** TPU HBM appears as one pool per chip;
  there's no "VRAM used / total" the way NVML reports for NVIDIA. The
  `mem_used_mb` field will be approximate (whatever `tpu_info` returns
  for HBM usage).

## Cost summary

| Phase | Configuration | $/hr | Duration | Total |
|---|---|---|---|---|
| Phase 2 (telemetry probe) | v6e-1 | $2.70 | 2h | **$5** |
| Phase 3 (full corpus) | v6e-8 | ~$22 | 6–8h | **$130–175** |

**Total recommended TPU spend: ~$180.**
