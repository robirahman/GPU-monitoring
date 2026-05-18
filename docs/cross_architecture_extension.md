# Cross-Architecture Extension: NVIDIA + AMD → MI300X / TPU / Trainium

This is the top-level reading order for extending the paper's
single-vendor (NVIDIA) methodology across vendors and architectures.
Written after the local consumer-AMD corpus was complete (Pass-1 + Pass-2
+ Tier 2 + Tier 3 + Tier 4 + sampling sweep, ~200 LFS parquets on the
`amd` branch).

## The cross-architecture story we want to tell

The paper's headline is **98.6% accuracy distinguishing ML training from
other workloads, on NVIDIA hardware**. The Appendix E future-work bullets
flag three obvious extensions:

1. **Multi-node / multi-server training** — out of scope here (no
   multi-node cloud rentals planned).
2. **Larger model sizes** — partially addressed via the AMD Tier 4
   workloads (Qwen 3B, SDXL).
3. **Other accelerators (AMD, Huawei, TPU, Trainium)** — this is what we
   tackle.

This session's local work answered the **AMD-consumer** question
positively: pooled NVIDIA + RX 9070 XT 5-fold CV is **96.3% three-way /
96.6% binary**, matching the paper's NVIDIA-only headline of 96.4%.
Zero-shot transfer (NVIDIA → AMD with no AMD in train) is poor at 52.7%
binary — the classifier learns vendor-specific signal patterns.

The remaining open questions:
- Does the pattern extend to **AMD datacenter** (MI300X / MI325X)?
- Does it extend to **non-GPU** accelerators (TPU, Trainium)?
- Specifically, does the **Stage-2 shape-only head** (which uses only
  autocorrelation / FFT / periodicity, no absolute levels) transfer
  across architectures where absolute levels (power, clock) differ
  enormously?

## Reading order

| Document | What it is |
|---|---|
| [`docs/telemetry_signal_matrix.md`](telemetry_signal_matrix.md) | Phase 2 deliverable. Which of the 9 paper signals each platform exposes. Pre-filled for NVIDIA/AMD; TODO for TPU/Trainium. |
| [`docs/cloud_mi300x.md`](cloud_mi300x.md) | Datacenter AMD. Provisioning on RunPod / TensorWave, expected costs, full corpus + multi-GPU DDP. **Highest paper-relevance per dollar.** |
| [`docs/cloud_tpu.md`](cloud_tpu.md) | Google TPU v6e (Trillium). Provisioning on GCP, the existing `feat/tpu-adaptation` scaffolding to reuse, expected gaps (no per-chip power at 1 Hz). |
| [`docs/cloud_trainium.md`](cloud_trainium.md) | AWS Trainium2. Provisioning on EC2, Neuron SDK workload porting cost, biggest unknown. |

## Phase ordering (recommended)

Each phase gates the next — don't sink cloud budget into Phase 3 if
Phase 2 reveals that TPU/Trainium can't be made to produce a usable
signal set.

```
PHASE 0  Consumer-AMD residuals     ← DONE in this session
   │     (whitebox sweep, J/WB-O ports)
   ▼
PHASE 1  Datacenter AMD             ← ~$150,  MOST PAPER VALUE
   │     (MI300X + MI325X)
   │     Result: consumer↔datacenter AMD transfer answer
   ▼
PHASE 2  TPU + Trainium probes      ← ~$25,   DECIDE COLLECTOR DESIGN
   │     (2h v6e-1, 2h trn1.2xl)
   │     Result: telemetry_signal_matrix.md filled in
   ▼
PHASE 3  TPU + Trainium full corpus ← ~$300,  IF PHASE 2 IS PROMISING
   │     (v6e-8 + trn2.48xlarge)
   │     Result: parquets ready for cross-arch classifier
   ▼
PHASE 4  Cross-architecture eval    ← $0,     CPU-ONLY ANALYSIS
         (eval_cross_architecture.py + heatmap)
         Result: published-quality cross-vendor / cross-arch table
```

**Total cloud spend ~$475 happy path.** Within the user's stated
generous-budget envelope ($1500+).

## Decision points along the way

After each phase, reassess before authorizing the next.

### After Phase 1 (datacenter AMD)

Look at:
- Cross-tier AMD heatmap: is RX 9070 XT ↔ MI300X transfer weak (mirroring
  the paper's consumer↔datacenter NVIDIA finding)?
- Pooled CV with the expanded corpus: does it stay at ~96% binary?
- WB-L detection rate: with more training data, does it improve?

If pooled CV holds at 96% and the consumer↔datacenter gap appears on
AMD, you have a clean Figure 1 extension and probably enough for an
ICML-rebuttal-style addendum. You might decide TPU/Trainium are
incremental and stop. Or proceed if budget allows.

### After Phase 2 (telemetry probes)

Look at the filled-in signal matrix.

If TPU exposes ≥6 of the 9 signals at 1 Hz: proceed to Phase 3 TPU.
If TPU exposes <4 signals or no temporal granularity: **drop TPU**.
The Stage-2 shape-only head needs at least util/memory temporal signal
to work; without it there's nothing to classify.

Same logic for Trainium.

### After Phase 3 (corpora collected)

Run the cross-arch eval. Possible outcomes:

1. **Pooled training (NVIDIA + AMD + TPU + Trainium) ≥90% binary** —
   strong cross-architecture generalization, headline result.
2. **Pooled ~80% but Stage-2 shape-only head ≥90%** — interesting
   nuance, paper-worthy. "Vendor-invariant temporal patterns persist
   even when absolute levels don't."
3. **Pooled <70%** — the methodology is genuinely vendor-specific. Also
   publishable (negative result), recasts the paper's contribution as
   per-vendor monitoring rather than universal.

## What's already in place

- `setup_amd_datacenter.sh` — turnkey installer for MI300X/MI325X cloud
  nodes (Ubuntu 22.04/24.04/26.04, ROCm 6.4 via AMD's repo, PyTorch
  ROCm wheel, amdsmi-py matched to library version)
- `scripts/run_amd_datacenter_corpus.sh` — two-phase orchestrator
  (single-GPU sweep + multi-GPU DDP adversarial)
- 6 DDP adversarial entries in `workloads/registry_amd.py`
  (K_ddp, L_ddp_5/10, L_ddp_stagger_5, K_ddp_accum_4/8)
- `classifier/eval_cross_vendor.py` — already extensible to more vendors
- `classifier/classifier_oracle.py` — auto-detects pynvml vs amdsmi
  backend; will need a 3rd path for TPU and 4th for Trainium

## What's not yet written

- `scripts/collect_tpu_telemetry.py` — Phase 3 (extend
  `feat/tpu-adaptation`'s scaffold)
- `scripts/collect_trainium_telemetry.py` — Phase 3 (new)
- `workloads/registry_tpu.py`, `workloads/registry_trainium.py`
- `classifier/eval_cross_architecture.py` — Phase 4 (fork of
  `eval_cross_vendor.py`)
- Workload adaptations for XLA / Neuron (most workloads are pure
  PyTorch but a few need `torch_xla.core.xla_model` or
  `torch_neuronx.trace` ceremony)

Phases 2 and 3 will create these. Don't prebuild — let the probe results
inform the design.
