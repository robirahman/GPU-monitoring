# W-id → file / script map

Review order: open **impl** first, then **registry** label block, then **launcher**.  
Registry index: [`workloads/registry.py`](../workloads/registry.py).  
Shared launchers:

| Launcher | Path | Use |
|---|---|---|
| Single / generic | [`scripts/run_workload.py`](../scripts/run_workload.py) | Most registry labels |
| Multi-GPU + telemetry | [`scripts/run_workload_multi_gpu.py`](../scripts/run_workload_multi_gpu.py) | DDP / FSDP / MoE serve / W03–W05, W10, W15–W19, W23 (`--num-gpus` = the entry's `nproc_per_node`) |
| DiLoCo (multi-node) | [`scripts/run_diloco_vcf.sh`](../scripts/run_diloco_vcf.sh) | W13 / W14 (single-node registry: `diloco_1b_H500`, `diloco_streaming_1b_H512`) |
| MoE serve helper | [`scripts/run_moe_serving_vcf.sh`](../scripts/run_moe_serving_vcf.sh) | W11 / W12 |
| Trace multi-user | [`scripts/run_serving_trace.py`](../scripts/run_serving_trace.py) | W11 / W21 |
| Real disagg (vLLM+NIXL) | [`scripts/run_disagg_kv_vcf.sh`](../scripts/run_disagg_kv_vcf.sh) | W10 primary; builds the W18 profile via [`scripts/build_kv_profile.py`](../scripts/build_kv_profile.py) |
| Real training (torchtitan) | [`scripts/run_titan_job.py`](../scripts/run_titan_job.py) (+ [`scripts/titan_jobs.py`](../scripts/titan_jobs.py), helper [`scripts/run_moe_ep_train_vcf.sh`](../scripts/run_moe_ep_train_vcf.sh)) | W03 / W04 / W05 primaries |
| NCCL debug → JSONL | [`scripts/parse_nccl_coll_log.py`](../scripts/parse_nccl_coll_log.py) | truthful per-collective record of any real framework (NCCL's own COLL log) |
| IB-quiet harness | [`scripts/run_nvl_only_ibquiet.py`](../scripts/run_nvl_only_ibquiet.py) | W23 (wraps DDP/FSDP; registry `train_*_nvl_only_ibquiet`) |
| Shared helpers | [`workloads/interconnect_common.py`](../workloads/interconnect_common.py) | dist setup (nccl/gloo), JSONL comm log, KV geometry, sized LM — used by W03–W05, W10, W15–W19 |
| Planning note | [`docs/SPAR_Rahman_critique_and_interconnect_redteam.md`](SPAR_Rahman_critique_and_interconnect_redteam.md) | Family definitions |

Status legend: **smoke** · **code** · **partial** · **missing**

> **Primary vs microbench (2026-09-23 upgrade).** *Synthetic workloads are unit tests; the main SPAR tables
> use real frameworks only* — vLLM (serving, disaggregation), torchtitan (TP/PP/MoE-EP training), PyTorch
> DDP/FSDP, HF+PEFT. Hand-rolled collective gyms (`disagg_kv_synth`, `moe_ep_*_synthetic`, toy TP/PP) are kept,
> registered with a `[MICROBENCH ONLY — not SPAR primary corpus]` (or `[PARALLELISM UNIT / ABLATION]` /
> `[SCHEDULE MICROBENCH]`) description and `microbench_*` aliases. The per-ID list is
> [`REAL_WORKLOAD_BAR.md`](REAL_WORKLOAD_BAR.md). Real-stack workloads emit **no invented collective JSONL**:
> serve/traffic logs + NVML/DCGM/IB counters, plus (opt-in) NCCL's own debug log via
> [`scripts/parse_nccl_coll_log.py`](../scripts/parse_nccl_coll_log.py). All primaries are **code (real), not yet
> run on GPUs from this repo** — flags/protocols were checked against vLLM / torchtitan / NCCL source, and our
> orchestration against mock engines; nothing was launched on real hardware.

> **2026-09-23:** W03–W05, W10, W15–W19, W23 implemented and DiLoCo registered. Verified on
> **CPU / gloo only** (no GPU / NCCL run yet for those) → status **code**, not **smoke**.
>
> **2026-09-23 (later):** Idle/non-ML is **W24**. Temporal serve-cover + spatial colocation were
> moved to the **end** as **W25 / W26 classifier behavior probes** (not clean train\|infer labels;
> registry: `train_serve_cover_*`, `colocate_*`). Use them last to study how a trained detector
> behaves on mixed / ambiguous jobs — do **not** put them in the primary SPAR train-vs-infer corpus
> as ground-truth classes.
>
> **Comm-log convention** (all of the above): rank0 JSONL, one collective per line, `bytes_est` = per-rank send-side
> estimate (AllReduce ring `2(N-1)/N·payload`; A2A / P2P = payload), `payload_bytes` = raw buffer, `phase` = ground
> truth of *what the traffic really was* (a disguise is labelled as the training/inference it is), plus a
> `<stem>.meta.json` sidecar. Regimes with rare/gated sync also write `<stem>.steps.jsonl` (per-step windows).

---

## W01 — Dense DP / DDP train · smoke

| Kind | Path |
|---|---|
| **Impl** | [`workloads/pytorch_training_ddp.py`](../workloads/pytorch_training_ddp.py) |
| Registry | `ddp_resnet18_cifar10`, `ddp_resnet50_cifar10`, `ddp_resnet18_cifar10_amp` |
| Launch | `python scripts/run_workload_multi_gpu.py --workload ddp_resnet18_cifar10 --num-gpus 8` |
| Related FN/DDP | [`workloads/adversarial_ddp.py`](../workloads/adversarial_ddp.py) (`adversarial_K_ddp`, `adversarial_L_ddp_*`) |

---

## W02 — FSDP / ZeRO-3 train · smoke

| Kind | Path |
|---|---|
| **Impl** | [`workloads/fsdp_training.py`](../workloads/fsdp_training.py) |
| Registry | `fsdp_7b_nvlink`, `fsdp_13b_nvlink`, `fsdp_30b_nvlink`, `fsdp_30b_adamw_nvlink`, `fsdp_70b_nvlink` |
| Launch | `python scripts/run_workload_multi_gpu.py --workload fsdp_7b_nvlink --num-gpus 8` |

---

## W03 — TP + DP train · code (real primary)

| Kind | Path |
|---|---|
| **Primary (real)** | torchtitan via [`scripts/run_titan_job.py`](../scripts/run_titan_job.py) + job table [`scripts/titan_jobs.py`](../scripts/titan_jobs.py) |
| Registry (primary) | `train_tp2_dp4_qwen3_1p7b_titan` (TP=2, sequence-parallel OFF, replicated DP=4 → AllReduce on activations + grads, Megatron-style), `train_tp2_fsdp4_qwen3_14b_titan` (TP=2 SP-on × FSDP2=4 → AG/RS) |
| Launch | `python scripts/run_workload_multi_gpu.py --workload train_tp2_dp4_qwen3_1p7b_titan --num-gpus 8` |
| Logs | torchtitan step logs + `job.json` manifest; `--nccl-collective-log` (set in the registry) → NCCL COLL debug log → `scripts/parse_nccl_coll_log.py`. No comm JSONL is invented |
| Checked | every `--section.key` validated against the installed torchtitan 0.3.0 source (`titan_jobs.validate_job`, with negative controls); not launched |
| Unverified | whether torchtitan accepts `data_parallel_replicate × TP` without FSDP (fall back to the FSDP variant); memory fit of 14B; weights are random-init unless `--load-hf-weights` |
| Microbench twin | [`workloads/tp_dp_train.py`](../workloads/tp_dp_train.py) → `train_tp2_dp4_1b`, `train_tp2_dp2_150m` — **[PARALLELISM UNIT / ABLATION]**: replicated vocab/lm_head understates TP bytes; numerically verified vs an unsharded reference (rel. err ~1e-6 for TP×DP 2×2, 2×1, 4×1, 1×2, 1×3) |

---

## W04 — Pipeline 1F1B train · code (real primary)

| Kind | Path |
|---|---|
| **Primary (real)** | torchtitan pipeline parallel (`torch.distributed.pipelining`, `Schedule1F1B`) via [`scripts/run_titan_job.py`](../scripts/run_titan_job.py) |
| Registry (primary) | `train_pp4_1f1b_qwen3_1p7b_titan` — Qwen3-1.7B, 4 stages, 8 microbatches, no DP; **run with `--num-gpus 4`** |
| Logs | as W03 (step logs, manifest, optional NCCL COLL log; P2P shows up as `Send`/`Recv`) |
| Checked | flags validated against installed torchtitan source; not launched |
| Microbench twin | [`workloads/pp_1f1b_train.py`](../workloads/pp_1f1b_train.py) → `train_pp2_1f1b_150m`, `train_pp4_1f1b_1b` — **[SCHEDULE MICROBENCH]**: hand-rolled 1F1B, exact vs reference (PP=2,3,4; M<P; uneven split), P2P + bubble markers in its own comm log |

---

## W05 — MoE EP train · code (real primary)

| Kind | Path |
|---|---|
| **Primary (real)** | torchtitan MoE training, expert-parallel A2A issued by the framework, via [`scripts/run_titan_job.py`](../scripts/run_titan_job.py) / helper [`scripts/run_moe_ep_train_vcf.sh`](../scripts/run_moe_ep_train_vcf.sh) |
| Registry (primary) | `moe_ep_train_dsv3_16b_ep8` (DeepSeek-MoE-16B, 64 routed experts, EP=8), `moe_ep_train_qwen3_30b_a3b_ep8` (Qwen3-30B-A3B, EP=8, memory fit unverified) |
| Why not OLMoE/Mixtral | not in torchtitan 0.3.0; DeepSeek-MoE-16B and Qwen3-30B-A3B are real MoE architectures it does ship (weights random-init, c4 streamed) |
| Launch | `python scripts/run_workload_multi_gpu.py --workload moe_ep_train_dsv3_16b_ep8 --num-gpus 8`, or `bash scripts/run_moe_ep_train_vcf.sh dsv3` |
| Logs | torchtitan step logs (loss / tok/s / MFU), `job.json`, NCCL COLL debug log → `parse_nccl_coll_log.py` (real per-collective record incl. AllToAll/Send/Recv) |
| Checked | flags validated against installed torchtitan source; not launched |
| Microbench twin | [`workloads/moe_ep_train_synthetic.py`](../workloads/moe_ep_train_synthetic.py) → `moe_ep_train_synthetic`, `moe_ep_train_synthetic_dp` (aliases `microbench_*`) — **[MICROBENCH ONLY — not SPAR primary corpus]**: hand-written A2A gym, shared expert weights, verified fwd+bwd A2A order |

---

## W06 — DP / replicate infer · smoke

| Kind | Path |
|---|---|
| **Impl (LLM)** | [`workloads/llm_inference.py`](../workloads/llm_inference.py), [`workloads/llm_inference_vllm.py`](../workloads/llm_inference_vllm.py) |
| **Impl (vision)** | [`workloads/inference.py`](../workloads/inference.py) (`resnet50_inference`) |
| Registry (examples) | `llm_infer_phi3_mini`, `llm_infer_qwen25_7b_fp16`, `resnet50_inference`, many `llm_infer_*` |
| Launch | `python scripts/run_workload.py --workload llm_infer_qwen25_7b_fp16` |
| N independent replicas | `run_workload_multi_gpu.py --parallel --workload …` |

---

## W07 — TP infer prefill-heavy · smoke

| Kind | Path |
|---|---|
| **Impl** | [`workloads/llm_inference_vllm.py`](../workloads/llm_inference_vllm.py) |
| Registry | `llm_infer_qwen25_7b_prefill_heavy`, `llm_infer_mixtral_tp2_prefill_heavy` |
| Launch | `run_workload.py` / `run_workload_multi_gpu.py` (TP2 needs ≥2 GPUs) |

---

## W08 — TP infer decode-heavy · smoke

| Kind | Path |
|---|---|
| **Impl** | [`workloads/llm_inference_vllm.py`](../workloads/llm_inference_vllm.py) |
| Registry | `llm_infer_qwen25_7b_decode_heavy`, `llm_infer_mixtral_tp2_decode_heavy`, also `llm_infer_qwen25_7b_balanced` |
| Launch | same as W07 |

---

## W09 — TP infer midfill / agentic · smoke

| Kind | Path |
|---|---|
| **Impl** | [`workloads/llm_inference_vllm.py`](../workloads/llm_inference_vllm.py) (`--mode midfill`) |
| Registry | `llm_midfill_qwen25_7b`, `llm_midfill_mixtral_tp2` |
| Serve midfill traffic | [`workloads/llm_moe_serving.py`](../workloads/llm_moe_serving.py) → `moe_serve_mixtral_ep8_midfill` |

---

## W10 — Disagg prefill↔decode KV · code (real primary)

| Kind | Path |
|---|---|
| **Primary (real)** | [`workloads/disagg_kv_vllm.py`](../workloads/disagg_kv_vllm.py) — real vLLM engines, `NixlConnector`, `kv_producer` prefillers + `kv_consumer` decoders, real Qwen2.5-7B-Instruct, real KV cache moved GPU→GPU (NIXL/UCX; NVLink in-node) |
| Proxy | [`workloads/disagg_proxy.py`](../workloads/disagg_proxy.py) — stdlib re-implementation of vLLM's reference two-step flow (`do_remote_decode` → prefiller `max_tokens=1` → forward `kv_transfer_params` to decoder) |
| Registry (primary) | `disagg_kv_vllm_qwen7b` (4P+4D, mixed 0.5k–8k prompts), `disagg_kv_vllm_qwen7b_longctx` (16k–24k prompts, ~1–1.4 GB KV/handoff) |
| Helper | [`scripts/run_disagg_kv_vcf.sh`](../scripts/run_disagg_kv_vcf.sh) (`check` / `qwen7b` / `qwen7b_longctx`; also builds the W18 profile) |
| Needs | `pip install vllm nixl` on the cluster; vLLM documents NixlConnector as **experimental**. Checked against vLLM `main` source (`KVTransferConfig`, `VLLM_NIXL_SIDE_CHANNEL_*`, factory) — the cluster's version is unknown; `--dry-run` prints the exact commands |
| Logs | `--handoff-log`: per-request proxy JSONL (timestamps, prompt tokens, **KV blocks the prefiller reported**, `kv_bytes_est` = blocks × block × KV bytes/token — an **estimate**, labelled as such); `--traffic-log`; `<stem>.meta.json`. Fabric truth = NVLink/IB counters from the telemetry launcher. **No NCCL JSONL** (NIXL is not NCCL) |
| Feeds | `scripts/build_kv_profile.py` → `data/profiles/kv_handoff_<label>.json` → W18 `--schedule-from` |
| Verified locally | engine command/env construction; proxy protocol vs mock engines (asserts the reference contract, round-robin, failures); full orchestrator lifecycle vs a mock vLLM server; profile chain. **Not run against real vLLM/NIXL** |
| Microbench twin | [`workloads/disagg_kv_synth.py`](../workloads/disagg_kv_synth.py) → `disagg_kv_synth_1gb`, `disagg_kv_synth_256mb_burst` (aliases `microbench_*`) — **[MICROBENCH ONLY — not SPAR primary corpus]**: random uint8 P2P, useful as a detector unit test |

---

## W11 — Heavy continuous-batch TP serve (**multi-user**) · code

| Kind | Path |
|---|---|
| **Impl (server + traffic)** | [`workloads/llm_moe_serving.py`](../workloads/llm_moe_serving.py) |
| Registry (dense TP multi-user-ish) | **`moe_serve_qwen7b_tp4dp2_mixed`** (canonical). Alias `moe_serve_qwen7b_tp8_mixed` → same TP4×DP2 (TP=8 invalid: 28 heads) |
| Registry (MoE+mixed; also W12) | `moe_serve_olmoe_ep8_mixed`, `moe_serve_mixtral_ep8_mixed` |
| Trace-driven multi-user | [`workloads/serving_trace_replay.py`](../workloads/serving_trace_replay.py) + [`scripts/run_serving_trace.py`](../scripts/run_serving_trace.py) |
| Traces | [`data/external/llm_serving_traces/`](../data/external/llm_serving_traces/) (Azure / BurstGPT) |
| Helper | [`scripts/run_moe_serving_vcf.sh`](../scripts/run_moe_serving_vcf.sh) `m5_dense_ctl` / `m5_mixed` |
| Example | `python scripts/run_serving_trace.py --workload serving_azure_conv_2023 --trace data/external/llm_serving_traces/azure/AzureLLMInferenceTrace_conv_2023.csv --model Qwen/Qwen2.5-7B-Instruct --tp 2 --max-concurrency 32 --start-server` |

> Multi-user inference maps here (and W21 for arrival process; W12 if EP).

---

## W12 — MoE EP infer (AllToAll) · code

| Kind | Path |
|---|---|
| **Primary (real)** | [`workloads/llm_moe_serving.py`](../workloads/llm_moe_serving.py) — real vLLM `--enable-expert-parallel` serving |
| Registry (primary) | `moe_serve_olmoe_ep8_decode`, `moe_serve_mixtral_ep8_decode`, `moe_serve_mixtral_tp2dp4_ep_decode`, `*_mixed` |
| Helper | [`scripts/run_moe_serving_vcf.sh`](../scripts/run_moe_serving_vcf.sh) `m4_olmoe` / `m4_mixtral` |
| Microbench twin | [`workloads/moe_ep_synthetic.py`](../workloads/moe_ep_synthetic.py) → `moe_ep_synthetic`, `moe_ep_synthetic_fake_ar` (aliases `microbench_*`) — **[MICROBENCH ONLY — not SPAR primary corpus]** |
| Corpus rule | Any "SPAR default corpus" list uses only the real `moe_serve_*` labels above |

---

## W13 — DiLoCo / local SGD · code

| Kind | Path |
|---|---|
| **Impl** | [`workloads/diloco_production.py`](../workloads/diloco_production.py) |
| **Core** | [`workloads/diloco_core.py`](../workloads/diloco_core.py) |
| Launch | [`scripts/run_diloco_vcf.sh`](../scripts/run_diloco_vcf.sh) |
| Registry | `diloco_1b_H500` (single node, 3 outer syncs) — multi-node still via torchrun / sbatch |
| Example | `MODEL_SIZE=1b LOCAL_STEPS=500 sbatch scripts/run_diloco_vcf.sh` |

---

## W14 — Streaming DiLoCo · code

| Kind | Path |
|---|---|
| **Impl** | same as W13: [`workloads/diloco_production.py`](../workloads/diloco_production.py) |
| Flags | `--streaming --num-fragments N --fragment-sync-delay …` (see file docstring) |
| Registry | `diloco_streaming_1b_H512` (8 fragments, delay 4) |
| Launch | `STREAMING=1 … scripts/run_diloco_vcf.sh` |

---

## W15 — Grad-accum / rare sync · code (interconnect-grade logs)

| Kind | Path |
|---|---|
| **DDP rare sync** | [`workloads/adversarial_ddp.py`](../workloads/adversarial_ddp.py) → `adversarial_K_ddp_accum_{4,8}`, `adversarial_L_ddp_{5,10}`, `adversarial_L_ddp_stagger_5` — now accepts **`--comm-log`** |
| Interconnect aliases | `train_ddp_rare_sync_accum{4,8}`, `train_ddp_dilute_1of{5,10}` (same code + `--comm-log data/…`) |
| Comm log | one `all_reduce` per DDP bucket via a Python comm hook: `{t, op, phase, bytes_est, step, microstep, is_sync_step, batch_phase}`; `no_sync()` micro-steps log nothing |
| Window truth | `<stem>.steps.jsonl`: one record per batch `{t, t_end, step, microstep, is_sync_step, phase = train_sync \| train_accum \| train_local_nosync \| infer_forward}` |
| Single-GPU N | [`workloads/adversarial_round4.py`](../workloads/adversarial_round4.py) → `adversarial_N_grad_accum_{4,8,16}` (unchanged, no comm) |
| Catalog | [`docs/workloads_and_evasion_strategies.md`](workloads_and_evasion_strategies.md) |
| Behaviour | Without `--comm-log` the strategies are unchanged. With it, all ranks also stop on the same batch (rank0 clock) — the old per-rank `time.time()` stop can leave one rank blocked in an AllReduce (observed as a hang on fast CPU loops) |
| Finding | `L-stagger` logs **zero** AllReduces by construction (each rank trains alone under `no_sync()`; replicas never average) |
| Verified (CPU/gloo, `--synthetic-data`) | K-accum 25% sync windows (accum 4), L 20% (1/5), K every step |

---

## W16 — LoRA / PEFT-only sync · code

| Kind | Path |
|---|---|
| **Cross-rank sync** | [`workloads/lora_ddp.py`](../workloads/lora_ddp.py) — DDP where only LoRA adapters are AllReduced; base frozen, identical per rank, never synced (`init_sync=False`; `--init-sync` = broadcast base once) |
| Registry | `train_lora_only_ddp_qwen15` (~9 MB/step vs ~3 GB full-FT), `train_lora_only_ddp_tinyllama` |
| Launch | `python scripts/run_workload_multi_gpu.py --workload train_lora_only_ddp_qwen15 --num-gpus 8` |
| Comm log | `all_reduce` `phase=grad_sync_lora` per bucket + `<stem>.steps.jsonl` (`bytes_est` vs `dense_bytes_est`) |
| Verified (CPU/gloo) | random tiny Llama (`--model random-tiny`) **and real TinyLlama-1.1B weights** (2 ranks, fp32): 2.25M adapter params = 9.01 MB/step vs 4.4 GB full-FT (488×). `init_sync=False` measured: DDP construction 0.00 s vs 1.78 s with `--init-sync` (the 4.4 GB base broadcast). Adapter sizes from configs: Qwen2.5-1.5B 2.18M params = 8.7 MB, TinyLlama 9.0 MB (PEFT keeps adapters fp32). Not yet run: Qwen weights, bf16 / GPU path |
| **Matched-pair LoRA FT** | [`workloads/lora_training.py`](../workloads/lora_training.py) → `llm_train_qwen25_*_lora`, `llm_train_phi3_mini_lora`, … (`apply_lora` is reused by `lora_ddp.py`) |
| **Small LoRA** | [`workloads/lora_finetune.py`](../workloads/lora_finetune.py) (AMD registry mostly) |
| **White-box diluted LoRA** | [`workloads/adversarial_whitebox.py`](../workloads/adversarial_whitebox.py) → `whitebox_lora_N{5,10,20}` (single GPU) |

---

## W17 — PowerSGD / Top-k · code

| Kind | Path |
|---|---|
| **Impl** | [`workloads/powersgd_ddp.py`](../workloads/powersgd_ddp.py) — DDP + `powerSGD_hook` / `batched_powerSGD_hook` / **Top-k** (`--hook topk`) |
| Registry | `ddp_powersgd_resnet18`, `ddp_powersgd_150m`, `ddp_topk_resnet18`, `ddp_topk_150m` |
| Flags | PowerSGD: `--powersgd-rank`, `--start-powerSGD-iter`, `--min-compression-rate`. Top-k: `--topk-ratio`, `--topk-start-iter`. `--hook allreduce` = dense baseline through the same path |
| Comm log | every Python-level collective with exact size; `phase = grad_sync_dense_warmup \| grad_sync_powersgd \| grad_sync_topk`; Top-k logs `op=all_gather` (`k`, `numel`, `ratio`) — **no AllReduce** in its sparse phase, so the op mix changes too. `<stem>.steps.jsonl` per step: `bytes_est` vs `dense_bytes_est`, `compression_ratio` |
| Verified (CPU/gloo) | PowerSGD ResNet-18 rank 4: 44.7 MB → 0.62 MB/step (72×). Top-k: ratio 1.0 reproduces dense DDP losses exactly (Δ=0); ratio 0.01 → 50× at N=2 (7.35 MB → 147 KB tiny; 44.7 → 0.89 MB ResNet-18) |
| Caveat | Top-k uses all_gather, so bytes per rank grow ∝ (N−1): 50× at N=2 but ~5× at N=4 with ratio 0.05 — a small-N disguise. PowerSGD's cost does not grow with N |
---

## W18 — KV-shaped gradient disguise · code (calibrated to real W10)

| Kind | Path |
|---|---|
| **Impl** | [`workloads/train_kv_disguise.py`](../workloads/train_kv_disguise.py) — real DDP training, gradient sync re-cut to look like KV handoffs |
| Registry | `train_kv_disguise_ddp` / `_p2p` (hand `--block-mb`, runnable now); `*_realprof` after W10 (`--schedule-from data/profiles/kv_handoff_disagg_kv_vllm_qwen7b.json`, falls back to hand if missing) |
| Profile | [`scripts/build_kv_profile.py`](../scripts/build_kv_profile.py) + [`workloads/kv_profile.py`](../workloads/kv_profile.py). No profile committed — realprof warns + falls back until a W10 run produces one |
| Comm log | ground truth stays honest: `phase=grad_kv_block` / `grad_sync_p2p`, `mimics=kv_handoff`; meta records `schedule_source` + `schedule_provenance` |
| Verified (CPU/gloo) | `--verify`: disguised sync == `dist.all_reduce` (≤4e-9) with a profile-driven schedule in both modes; per-step schedules differ but always sum to \|g\|; p2p needs power-of-two world |
| External | ICTD `kv_disguise`, `kv_diloco_composite` (composite not ported) |

---

## W19 — Hybrid duty-cycle · code (on the wire)

| Kind | Path |
|---|---|
| **Wire-level hybrid** | [`workloads/hybrid_wire.py`](../workloads/hybrid_wire.py) — real training; decode-shaped small TP-like AllReduces every step (W08 scale) + full \|θ\| sync every H steps (`--sync-mode grad` accumulate-then-AllReduce, or `diloco` via `diloco_core`) |
| Registry | `train_hybrid_wire_H50`, `train_hybrid_wire_H500` |
| Launch | `python scripts/run_workload_multi_gpu.py --workload train_hybrid_wire_H50 --num-gpus 8` |
| Window truth | `<stem>.steps.jsonl` `phase = decode_shape \| train_sync`; comm log has aggregated `decode_shape` lines (`count`, `msg_bytes`) + `train_sync` buckets (DiLoCo mode: `outer_sync` events with diloco_core's own 1-based step counter) |
| Cover locality | `--cover-tp N` = GPUs/node keeps the cover intra-node; only the rare sync crosses nodes |
| Verified (CPU/gloo) | grad + diloco modes, 4 ranks with TP-like subgroups of 2 |
| **Interleave** | [`workloads/adversarial_advanced.py`](../workloads/adversarial_advanced.py) → `adversarial_F_interleave_{30,50}` (compute/mode hybrid, NVML-era) |
| **Online / diluted** | [`workloads/adversarial_round3.py`](../workloads/adversarial_round3.py) → `adversarial_K_online_learning`, `adversarial_L_diluted_*` |
| **Extreme serve-cover (≥95% infer duty)** | see **W25** (classifier behavior probe — temporal mix), not a primary train\|infer label |
| **Multi-GPU serve∥train** | see **W26** (classifier behavior probe — spatial colocation) |

---

## W20 — Fake collectives on serving · code (real serve)

| Kind | Path |
|---|---|
| **Primary (real serve)** | [`workloads/llm_moe_serving.py`](../workloads/llm_moe_serving.py) `--cotenant-fake-ar` + co-tenant [`workloads/fake_ar_cotenant.py`](../workloads/fake_ar_cotenant.py) (one NCCL rank per GPU, periodic `all_reduce(zeros)`, wall-clock aligned) |
| Registry (primary) | `moe_serve_qwen7b_tp4dp2_fake_ar`, `moe_serve_mixtral_ep8_fake_ar` (vLLM `--gpu-memory-utilization 0.85` to leave headroom) |
| **Not `tp8`** | Qwen2.5-7B has 28 heads → TP=8 invalid. Dense layout is TP4×DP2: `moe_serve_qwen7b_tp4dp2_mixed` / `*_fake_ar`. Old label `moe_serve_qwen7b_tp8_mixed` is an alias to tp4dp2 |
| Logs | `--traffic-log` (real requests), `--fake-ar-log` (the co-tenant's own collectives, `source=cotenant_fake_ar`, `phase=fake_ar`), `<stem>.window.json` (server-healthy / traffic start / end). vLLM is untouched; nothing is injected into it. Real-serve collectives are not in the co-tenant log (capture via NCCL COLL debug + parser if needed) |
| Verified locally | orchestrator lifecycle vs a mock vLLM server (co-tenant events fall inside the traffic window); co-tenant on gloo. **Not run with real vLLM/NCCL** |
| Microbench twin | `moe_ep_synthetic_fake_ar` (aliases `microbench_*`) — **[MICROBENCH ONLY — not SPAR primary corpus]**; [`week7/edge_cases/run_edge_cases.py`](../week7/edge_cases/run_edge_cases.py) EC1 remains a historical NVLink toy |

---

## W21 — Trace-replay serving · code

| Kind | Path |
|---|---|
| **Loadgen** | [`workloads/serving_trace_replay.py`](../workloads/serving_trace_replay.py) |
| **Orchestrator** | [`scripts/run_serving_trace.py`](../scripts/run_serving_trace.py) |
| **Traces + README** | [`data/external/llm_serving_traces/README.md`](../data/external/llm_serving_traces/README.md) |
| Registry | **not registered** — pass `--workload serving_azure_conv_2023` (label only) to orchestrator |
| Overlap | With high `--tp` + concurrency → also exercises **W11** |

---

## W22 — Fine-tune / short SFT · smoke

| Kind | Path |
|---|---|
| **Impl** | [`workloads/gpt2_finetune.py`](../workloads/gpt2_finetune.py), [`workloads/bert_finetune.py`](../workloads/bert_finetune.py), [`workloads/lora_training.py`](../workloads/lora_training.py) |
| Registry | `gpt2_wikitext2`, `bert_sst2`, `llm_train_*_lora` |
| Reveal context | [`data/external/Reveal_ib0_train_vs_infer.md`](../data/external/Reveal_ib0_train_vs_infer.md) (fine-tune quiet on ib0) |

---

## W23 — Intra-node-only train (IB quiet) · code

| Kind | Path |
|---|---|
| **Harness** | [`scripts/run_nvl_only_ibquiet.py`](../scripts/run_nvl_only_ibquiet.py) — wraps an unmodified single-node DDP/FSDP command |
| Registry | `train_ddp_nvl_only_ibquiet` (DDP ResNet-50), `train_fsdp7b_nvl_only_ibquiet` (FSDP 6.7B) |
| Launch | `python scripts/run_workload_multi_gpu.py --workload train_ddp_nvl_only_ibquiet --num-gpus 8` or `python scripts/run_nvl_only_ibquiet.py --sidecar data/x.json --nccl-evidence -- torchrun --nproc_per_node=8 …` |
| NCCL env | `NCCL_IB_DISABLE=1` (no IB/RoCE transport), `NCCL_P2P_LEVEL=NVL` (P2P only over NVLink), `NCCL_NET_GDR_LEVEL=LOC` (no GPUDirect RDMA); `--no-pin` measures without pinning; `--nccl-env K=V` overrides |
| Sidecar JSON | `{nvlink_expected: true, ib_expected: false, cmd, env_applied, t_start/t_end, returncode, ib_counters{per_port, total_bytes}, ib_quiet_verdict: quiet\|NOT_QUIET\|unknown, nccl_transport_evidence?}` |
| IB proof | samples `/sys/class/infiniband/*/ports/*/counters/{port_xmit_data,port_rcv_data}` (×4 octets, 32-bit wrap-corrected) before/during/after; no IB devices → `unknown`, never a false `quiet`; threshold 10 MiB default (mgmt/NFS noise) |
| Guard | refuses `--nnodes > 1` unless `--allow-multinode` |
| Verified (local) | fake sysfs tree: quiet / noisy+wraparound / no-IB / multi-node refusal. NCCL log patterns (`via NET/IB`, `via P2P`, `via SHM`) match published NCCL INFO examples (`[send] via NET/IB/0`, `via P2P/CUMEM/read`) but were **never run against a live NCCL job**; real IB counters need IB hardware. Note `via P2P` alone does not distinguish NVLink from PCIe P2P — `NCCL_P2P_LEVEL=NVL` is what constrains that, the log count is only corroboration |
| Dual collector | **built-in**: [`scripts/collect_telemetry.py`](../scripts/collect_telemetry.py) fuses NVLink (NVML) + IB/eth (sysfs) into every row via [`scripts/interconnect_collect.py`](../scripts/interconnect_collect.py); multi-GPU shares one `SharedFabricCache`. Sidecar still provides the IB-quiet *verdict* + NCCL evidence. |

---

## W24 — Idle / non-ML · smoke

| Kind | Path |
|---|---|
| Idle | [`workloads/idle.py`](../workloads/idle.py) → `idle` |
| HPC | [`workloads/scientific_hpc.py`](../workloads/scientific_hpc.py) → `cufft_benchmark`, `nbody_sim` |
| Mining | [`workloads/mining_proxy.py`](../workloads/mining_proxy.py), [`workloads/ethash_cuda.py`](../workloads/ethash_cuda.py) |
| Render | [`workloads/rendering_proxy.py`](../workloads/rendering_proxy.py), [`workloads/blender_render.py`](../workloads/blender_render.py) |

---

## Classifier behavior probes (run **last** — not primary train\|infer labels)

These two are **intentionally ambiguous**. They contain both training and inference activity
(time-multiplexed or space-multiplexed). They are **not** clean `y=train` / `y=infer` corpus
cells for fitting the detector.

**How to use them**

1. Train / calibrate the interconnect classifier on **W01–W23** (+ W24 idle/non-ML as negative).
2. Then run **W25 / W26** as held-out **behavior probes**: record P(train), window vs run-level,
   per-GPU vs node-aggregate scores, false alarms / misses.
3. Report them as *stress / deployment realism* results, not as another labeled class that
   dilutes the train-vs-infer taxonomy.

Window / role truth (`steps.jsonl`, `workload_role`) still matters — for analysis, not for
forcing a single hard label on the whole job.

---

## W25 — PROBE: temporal mix (~95% infer-shaped steps + rare train sync) · code

**Same process / GPU set**, time-multiplex. Neither pure train nor pure infer. Asks: if most
steps look like decode cover and only rare steps sync grads, how does the classifier behave
(windowed vs run-level vs volume-only)?

| Kind | Path |
|---|---|
| **Role in SPAR** | **Behavior probe only** — do not use as primary train\|infer training label |
| **Impl** | [`workloads/hybrid_wire.py`](../workloads/hybrid_wire.py) at large H (composes W19). Cover = [`interconnect_common.CoverTraffic`](../workloads/interconnect_common.py) (W08-scale). LoRA variant: [`workloads/lora_ddp.py`](../workloads/lora_ddp.py) `--sync-every H --cover` |
| **Registry** | `train_serve_cover_H20`, `train_serve_cover_H100`, `train_serve_cover_lora_H20` — `run_workload_multi_gpu.py --workload train_serve_cover_H20 --num-gpus 8` |
| **Window truth** | `<stem>.steps.jsonl`: `phase = decode_shape \| train_sync` per step; meta has step/byte fractions |
| **What to measure** | P(train) on cover windows vs sync windows; run-level CUSUM; whether volume-only fires (usually **only** `_lora_H20` lands mean bytes in the serve band — see REAL_WORKLOAD_BAR §5) |
| **Not** | A third ground-truth class; a substitute for W11/W13/W19 in the main table |

---

## W26 — PROBE: spatial colocation (most GPUs serve ∥ few train) · code

**Space-multiplex.** Real vLLM on most GPUs + real train on a disjoint set. Node-aggregate
traffic looks serve-heavy; train may be a small clique or a single GPU with **no** fabric
traffic. Asks: does a node-level detector miss the train slice? Does per-GPU labeling save it?

| Kind | Path |
|---|---|
| **Role in SPAR** | **Behavior probe only** — breaks “one job = one label”; not a clean class |
| **Orchestrator** | [`scripts/run_colocate_serve_train.py`](../scripts/run_colocate_serve_train.py) |
| **Registry** | `colocate_serve7_train1_qwen`, `colocate_serve_tp4_train_ddp2` (script owns telemetry; not via untagged `run_workload_multi_gpu`). 2-node variant **not built** |
| **Role tags** | `workload_role=serve\|train`, `role_label`, shared `run_id`; `<label>_<run>_roles.json` |
| **What to measure** | (a) node-aggregate P(train) vs (b) per-GPU / per-role scores; IB/eth are node-level only |
| **Prefer for fabric signal** | `colocate_serve_tp4_train_ddp2` (TP serve + DDP train both make NVLink traffic). `serve7_train1` has **zero** train interconnect by construction |
| **Scaffold** | [`week7/collect_edge_cases.py`](../week7/collect_edge_cases.py) |

---

## Prior corpus (Robi) — not a W-id, optional interconnect re-collect

These are **Rahman’s earlier NVML-era workloads / LOO evasion families**. They are **not** new SPAR interconnect designs and are **not** assigned W01–W26 ids. Several evasion ideas *do* appear as Related under W15 / W16 / W19 (F, K, L, N, DDP, whitebox_lora); everything below is what that map **does not** already cover.

Still useful: same registry labels can be re-run under the interconnect collectors (`collect_telemetry.py` + `interconnect_collect.py` fabric fusion, multi-GPU harness where applicable) to see whether NVLink / IB / eth counters add anything on top of the original 1 Hz NVML story. No new impl required unless you want `--comm-log` / steps sidecars (most of these are single-GPU and have nothing to log on the wire).

Catalog: [`docs/workloads_and_evasion_strategies.md`](workloads_and_evasion_strategies.md).  
LOO families: [`classifier/leave_one_strategy_out.py`](../classifier/leave_one_strategy_out.py) `STRATEGY_FAMILIES`.

### Clean gaps (in Robi’s registry, only partly under W22 / W24)

| What | Impl | Registry (examples) | Note vs W-map |
|---|---|---|---|
| Single-GPU CNN/MLP train | [`workloads/pytorch_training.py`](../workloads/pytorch_training.py) | `pytorch_resnet_cifar10`, `*_amp`, `pytorch_mlp_cifar10` | W01 is DDP only; W22 is GPT-2 / BERT / LoRA |
| Video encode | [`workloads/ffmpeg_nvenc.py`](../workloads/ffmpeg_nvenc.py) | `ffmpeg_nvenc` | Not listed under W24 |
| Real MD | [`workloads/gromacs_sim.py`](../workloads/gromacs_sim.py) | `gromacs_adh` | W24 only lists cuFFT / N-body |
| Modern non-vLLM infer | [`workloads/modern_inference.py`](../workloads/modern_inference.py) | `deepseek_r1_inference`, `qwen25_inference`, `whisper_inference` | Adjacent to W06; different entrypoint than `llm_inference*.py` |

### Evasion families with **no** W Related pointer

| Family | Impl | Registry labels | Evasion idea (one line) |
|---|---|---|---|
| **A_util_modulation** | [`adversarial_training.py`](../workloads/adversarial_training.py) | `adversarial_A_util_modulation` | Sync + random sleep → bursty util like render |
| **B_low_util** | same | `adversarial_B_low_util` | Tiny batch + CPU aug starves GPU |
| **D_temporal** | same | `adversarial_D_temporal_disruption` | Random bs / noise / sleeps kill ACF |
| **composite** | [`adversarial_composite.py`](../workloads/adversarial_composite.py) | `adversarial_composite_{00…95}` | Idle-fraction F + prealloc + noise (evasion-tax curve) |
| **E_memory_minimal** | [`adversarial_advanced.py`](../workloads/adversarial_advanced.py) | `adversarial_E_memory_minimal` | Checkpoint + tiny bs + GC → memory ≈ infer |
| **G_clock_throttled** | same | `adversarial_G_clock_throttled` | Power/SM clock lock + light sleep |
| **H_mimicry** | [`adversarial_round2.py`](../workloads/adversarial_round2.py) | `adversarial_H_mimicry_{mining,cufft}` | Feedback mix train vs target workload compute |
| **I_stochastic** | same | `adversarial_I_stochastic_30` | Stochastic interleave (kills F’s period) |
| **J_pid** | same | `adversarial_J_pid_75` | PID sleep holds ~75% util |
| **M_composite_memmin** | [`adversarial_round4.py`](../workloads/adversarial_round4.py) | `adversarial_M_composite_memmin_{10…40}` | Composite + E’s memmin stack |
| **whitebox_full** | [`adversarial_whitebox.py`](../workloads/adversarial_whitebox.py) | `whitebox_diluted_N*`, `*_ckpt`, `whitebox_oracle` | Diluted full FT / oracle at ~1B (W16 only lists LoRA) |
| **symgemm_L4_disguise** | [`adversarial_symgemm.py`](../workloads/adversarial_symgemm.py) | `symgemm_L4_sgdfused_*` | Forward-shaped backprop + fused SGD (Round 6) |
| **symgemm_L5_adam** | same | `symgemm_L5_adam_*` | Same shapes, real AdamW (positive control) |
| **symgemm_ppo_fused** | same | `symgemm_ppo_fused_r{8,32}_*` | Diluted PPO-shaped fused update |

### Already on the W-map (do **not** re-list as gaps)

| Family | Where |
|---|---|
| F interleave, K online, L diluted | W19 Related |
| N grad_accum; K_ddp / L_ddp / stagger / accum | W15 (+ W01 Related for K/L_ddp) |
| whitebox_lora | W16 |
| gpt2 / bert / matched-pair LoRA; idle / HPC / mine / render | W22 / W24 |
| L diluted / F interleave (temporal hybrid ancestors) | W19 Related; extreme mix → **W25 probe** (not a primary label) |

### Suggested interconnect re-collect (if you bother)

| Priority | What | Why |
|---|---|---|
| Low | A–J, composite, M (single-GPU) | Wire almost idle; mainly a negative control that fabric features stay flat while NVML still moves |
| Medium | `whitebox_full` vs W16 LoRA | Same diluted loop, full vs adapter memory — fabric still quiet, but pairs with W16 |
| Medium | Round-6 symgemm (+ PPO) | Kernel-level disguise; fabric null hypothesis again |
| High only if multi-GPU | Anything you later wrap with `torchrun` / `--comm-log` | Then it stops being “prior only” and should get a W Related or a new W |

Launch for single-GPU priors: `python scripts/run_workload.py --workload <label>` (same as original). No DiLoCo / IB-quiet / serving-trace launcher needed.

---

## Unique files to open (deduped review queue)

Suggested click-through order for interconnect-relevant code:

1. [`workloads/pytorch_training_ddp.py`](../workloads/pytorch_training_ddp.py) — W01  
2. [`workloads/fsdp_training.py`](../workloads/fsdp_training.py) — W02  
3. [`workloads/interconnect_common.py`](../workloads/interconnect_common.py) — shared helpers for the files below  
4. [`workloads/tp_dp_train.py`](../workloads/tp_dp_train.py) — W03 microbench  
5. [`workloads/pp_1f1b_train.py`](../workloads/pp_1f1b_train.py) — W04 microbench  
6. [`workloads/moe_ep_train_synthetic.py`](../workloads/moe_ep_train_synthetic.py) — W05 microbench  
7. [`workloads/llm_inference_vllm.py`](../workloads/llm_inference_vllm.py) — W07–W09  
8. [`workloads/disagg_kv_synth.py`](../workloads/disagg_kv_synth.py) — W10 microbench  
9. [`workloads/llm_moe_serving.py`](../workloads/llm_moe_serving.py) — W11 / W12  
10. [`workloads/serving_trace_replay.py`](../workloads/serving_trace_replay.py) + [`scripts/run_serving_trace.py`](../scripts/run_serving_trace.py) — W21  
11. [`workloads/moe_ep_synthetic.py`](../workloads/moe_ep_synthetic.py) — W12 / W20 microbench  
12. [`workloads/diloco_production.py`](../workloads/diloco_production.py) + [`workloads/diloco_core.py`](../workloads/diloco_core.py) — W13 / W14  
13. [`workloads/adversarial_ddp.py`](../workloads/adversarial_ddp.py) — W15  
14. [`workloads/lora_ddp.py`](../workloads/lora_ddp.py) — W16  
15. [`workloads/powersgd_ddp.py`](../workloads/powersgd_ddp.py) — W17  
16. [`workloads/train_kv_disguise.py`](../workloads/train_kv_disguise.py) — W18  
17. [`workloads/hybrid_wire.py`](../workloads/hybrid_wire.py) — W19 (+ compose **W25 probe**)  
18. [`week7/edge_cases/run_edge_cases.py`](../week7/edge_cases/run_edge_cases.py) — W20 EC1 microbench  
19. [`scripts/run_nvl_only_ibquiet.py`](../scripts/run_nvl_only_ibquiet.py) — W23  
20. [`scripts/interconnect_collect.py`](../scripts/interconnect_collect.py) + [`scripts/collect_telemetry.py`](../scripts/collect_telemetry.py) — fabric fusion  
21. [`scripts/run_workload_multi_gpu.py`](../scripts/run_workload_multi_gpu.py) — harness  
22. [`workloads/registry.py`](../workloads/registry.py) — label index  
23. [`workloads/disagg_kv_vllm.py`](../workloads/disagg_kv_vllm.py) + [`workloads/disagg_proxy.py`](../workloads/disagg_proxy.py) — W10 primary  
24. [`workloads/fake_ar_cotenant.py`](../workloads/fake_ar_cotenant.py) — W20 primary  
25. [`scripts/titan_jobs.py`](../scripts/titan_jobs.py) + [`scripts/run_titan_job.py`](../scripts/run_titan_job.py) — W03–W05 primaries  
26. [`scripts/run_colocate_serve_train.py`](../scripts/run_colocate_serve_train.py) — **W26 probe**  
27. [`docs/REAL_WORKLOAD_BAR.md`](REAL_WORKLOAD_BAR.md) — primary-corpus bar  

**Primary corpus:** W01–W24 (train / infer / non-ML with clean labels). **Behavior probes (run last):** **W25** (temporal mix), **W26** (spatial colocate) — not train\|infer ground truth. **Not built:** W26 2-node variant; real-vLLM time-multiplexed cover for W25. **Not yet GPU-smoked:** most new primaries/probes. **Microbench only:** synth / hand-rolled twins. **Prior (Robi):** section above.
