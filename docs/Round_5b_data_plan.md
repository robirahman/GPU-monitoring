# Round 5b data plan — matched-pair train+infer coverage

Date: 2026-04-19
Owners: Robi (local RTX 5080), cloud (5090 + A100-80 + H200 + B200)

## Goal

Build a train-vs-inference classifier that is robust across **GPU
type × model size**, by training on pooled telemetry from every GPU we
have access to. The classifier's primary output is P(training) for a
GPU telemetry window; secondary output is P(training | ML activity).

## What matters for this task

A classifier that learns train-vs-infer *robustly* needs **matched
pairs**: for each (GPU, model-size) cell, it should see both a training
run and an inference run at roughly the same memory footprint and compute
scale. Otherwise it learns "big memory + high util = training" as a
proxy, which is exactly what the Round 5 WB-O oracle exploited.

## Coverage audit (as of 2026-04-19, before this plan)

Matched pairs exist only at:

- **RTX 5080 / 5090**: TinyLlama 1.1B (whitebox_diluted_* / whitebox_lora_*
  training ↔ llm_infer_tinyllama_* inference).
- **H200**: FSDP-70B training ↔ Qwen 2.5 72B inference (fp16/fp8/TP2).
- **H200 / B200 / A100-80**: <0.1B training (ResNet, BERT, GPT-2) —
  matched inference is thin.

Every other (GPU, model-size) cell has training or inference but not
both. See `docs/Round_5b_audit.md` for the full cell-by-cell table.

## Collection plan

### Phase 4 — Consumer-tier matched-pair training (Robi, local)

Add LoRA training on 5080 + 5090 for every model size where we already
have inference. 300 s per run, 3 reps per workload, `--no-dcgm`. Uses
the new `workloads/lora_training.py` harness.

| Registry name | Model | 5080 | 5090 | Matches inference |
|---|---|:-:|:-:|---|
| `llm_train_qwen25_1p5b_lora` | Qwen 2.5 1.5B | ✓ | ✓ | `llm_infer_qwen25_1p5b*` |
| `llm_train_smollm2_1p7b_lora` | SmolLM2 1.7B | ✓ | ✓ | `llm_infer_smollm2_1p7b` |
| `llm_train_qwen25_3b_lora` | Qwen 2.5 3B | ✓ | ✓ | `llm_infer_qwen25_3b*` |
| `llm_train_phi3_mini_lora` | Phi-3-Mini 3.8B | ✓ (grad-ckpt) | ✓ | `llm_infer_phi3_mini*` |
| `llm_train_mistral_7b_lora` | Mistral 7B | ✗ (OOM on 16 GB) | ✓ | `llm_infer_mistral_7b_q4` |

Expected wall time: 4 workloads × 3 reps × ~5 min ≈ **1 h on 5080**,
plus 5 workloads × 3 reps ≈ ~1.5 h on 5090.

Launch:
```
.venv/Scripts/python.exe scripts/run_round5_defender_inference.py \
  --workloads llm_train_qwen25_1p5b_lora llm_train_smollm2_1p7b_lora \
              llm_train_qwen25_3b_lora llm_train_phi3_mini_lora \
  --reps 3 --no-dcgm
```

(`run_round5_defender_inference.py` is misnamed now that it also runs
training — leave as-is, it's just an orchestrator.)

### Phase 5 — Datacenter inference at matched FSDP sizes (cloud)

Add dense 7 B / 14 B / 32 B single-GPU inference on A100-80, H200, B200
to match the existing FSDP-7B / 13B / 30B training we have on those GPUs.
Run via `workloads/llm_inference_vllm.py`. Registry entries to add
(pending):

| Registry name | Model | HF id | Footprint | Runs on |
|---|---|---|---|---|
| `llm_infer_qwen25_7b_fp16` | Qwen 2.5 7B | `Qwen/Qwen2.5-7B-Instruct` | ~14 GB | A100-80, H200, B200 |
| `llm_infer_qwen25_14b_fp16` | Qwen 2.5 14B | `Qwen/Qwen2.5-14B-Instruct` | ~28 GB | A100-80, H200, B200 |
| `llm_infer_qwen25_32b_fp16` | Qwen 2.5 32B | `Qwen/Qwen2.5-32B-Instruct` | ~64 GB | A100-80, H200, B200 |

3 reps × 3 models × 3 GPUs = **27 runs** (~135 min, minus first-time HF
download time per GPU ≈ ~20-30 GB per model).

Reset progress JSON first on each new cloud host (the progress key
`(label, rep)` is not host-aware — see commit `c51b852`).

### Phase 6 — Datacenter small-model inference & single-GPU LoRA training (cloud, optional)

We have large FSDP training on H200/B200 (7B/13B/30B/70B) but no
matching *single-GPU* training on those boxes. This is not strictly
required for the phase-5 matched pairs — FSDP training is a reasonable
proxy for "this GPU is training" — but single-GPU LoRA on H200 would
give the cleanest possible comparison.

- `llm_train_qwen25_3b_lora` on H200: single-GPU training at the same
  scale as existing Qwen 2.5 3B inference already collected on H200.
- `llm_train_mistral_7b_lora` on A100-80 / H200: dense 7 B LoRA to
  match the phase-5 inference cell.

3 reps × 2 workloads × 2 GPUs = **12 runs**. Lower priority; schedule
after phases 4+5 are committed.

### Phase 7 — Other-class breadth on datacenter (cloud, optional)

The datacenter "other" class is thin (idle, nbody, gromacs, and a couple
of standalone synthetics on A100). Before finalising the pooled
classifier, it's worth collecting:

- `idle` on H200 / B200 (3 reps each, 5 min each).
- `cufft_benchmark` on H200 / B200.
- `mining_ethash_proxy` on H200 / B200 — same-host non-ML compute.

~12 runs. Cheap.

## Classifier structure — what changed

- `classifier/twostage.py` now uses the simpler two-binary-classifier
  structure: Stage 1 = `ml_training vs not-training`, Stage 2 =
  `ml_training vs ml_inference` (conditional on ML activity).
- `classifier/cross_gpu_generalization.py` and
  `classifier/results/cross_gpu_results.csv` removed.
- `classifier/consumer_to_datacenter_transfer.py` and
  `docs/normalized_features_ablation.md` (on `normalized-features`
  branch) retired with the branch.
- Pre-allocation features (`first_30s_mem_delta_mb/frac`,
  `first_30s_mem_ratio`, `time_to_mem_plateau_samples`) cherry-picked
  from the normalized-features branch to `main`. These target the WB-O
  memory-pre-allocation startup signature directly and were +7.4 pp on
  the retired cross-tier eval — should continue to help under pooled
  training. The normalized capacity features (`mem_used_frac_*`,
  `power_frac_*`) did not cherry-pick — their motivation was cross-tier
  generalisation.

## Open TODOs

- [x] Add phase-5 registry entries (`llm_infer_qwen25_7b_fp16`,
      `llm_infer_qwen25_14b_fp16`, `llm_infer_qwen25_32b_fp16`) and
      corresponding `INFERENCE_LABELS` entries.
- [x] Update `DATACENTER_LARGE_WORKLOADS` in
      `scripts/run_round5_defender_inference.py` to include phase-5
      workloads.
- [ ] Host-aware progress JSON: rename to include `hostname` so
      multi-host runs don't stomp each other. Currently mitigated by
      `--reset` on each new host.
- [ ] After phase-4 + phase-5 land, retrain the two-stage classifier
      on pooled data and re-run `classifier/compare_twostage_vs_threeway.py`.
      Target: Stage 1 P(training) on `whitebox_inference` parquet
      should be < 0.5 (was 0.995 under the Round 5a three-way).
- [ ] Leave-one-strategy-out with the new two-stage classifier on the
      full 18-family adversarial suite.

## Owner split

- **Robi / local 5080**: phase 4 (consumer LoRA) on 5080. Running now.
- **Cloud 5090**: phase 4 (all 5 LoRA workloads).
- **Cloud A100-80 / H200 / B200**: phase 5 (Qwen 7B/14B/32B dense inference).
- **Cloud H200 (optional)**: phase 6 single-GPU LoRA training.
- **Cloud H200 / B200 (optional)**: phase 7 other-class breadth.

Everything is reproducible via the `run_round5_defender_inference.py`
orchestrator with `--workloads` flag overrides; no new scripts needed
for phases 4-7 apart from the phase-5 registry entries.
