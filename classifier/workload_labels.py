"""Versioned job-objective labels. Raw workload names are never rewritten."""
import re
from functools import lru_cache

LABEL_SCHEMA_VERSION = "2026-10-11.3"

TRAINING_LABELS = {
    # Core training workloads
    "pytorch_resnet_cifar10", "pytorch_resnet_cifar10_amp",
    "pytorch_mlp_cifar10",
    "gpt2_wikitext2", "gpt2_wikitext2_amp",
    "bert_sst2", "bert_sst2_amp",
    # Edge cases that are still training
    "training_dual_gpu_dp", "training_single_gpu",
    "resnet18_small_batch_batch8", "resnet18_large_batch_batch2048",
    "resnet18_short_run_short",
    "mlp_small_batch_batch16", "mlp_large_batch_batch4096",
    # Week 4 edge case labels
    "BASELINE_TRAIN", "EC2", "EC3", "EC5", "EC6",
    # Adversarial strategies (still training, just disguised)
    "adversarial_A_util_modulation",
    "adversarial_B_low_util",
    "adversarial_D_temporal_disruption",
    # Composite adversarial at all idle fractions
    "adversarial_composite_00", "adversarial_composite_10",
    "adversarial_composite_20", "adversarial_composite_30",
    "adversarial_composite_40", "adversarial_composite_50",
    "adversarial_composite_60", "adversarial_composite_70",
    "adversarial_composite_80", "adversarial_composite_90",
    "adversarial_composite_95",
    # Advanced evasion strategies (still training, just better disguised)
    "adversarial_E_memory_minimal",
    "adversarial_F_interleave_30", "adversarial_F_interleave_50",
    "adversarial_G_clock_throttled",
    # Round 2 adversarial (counter-hardened classifier)
    "adversarial_H_mimicry_mining", "adversarial_H_mimicry_cufft",
    "adversarial_I_stochastic_30",
    "adversarial_J_pid_75",
    # Round 3 adversarial (training-as-inference)
    "adversarial_K_online_learning",
    "adversarial_L_diluted_2", "adversarial_L_diluted_5",
    "adversarial_L_diluted_10", "adversarial_L_diluted_20",
    # Controlled dilution sweep (consistent batch size = 128)
    "dilution_sweep_N1", "dilution_sweep_N2", "dilution_sweep_N3",
    "dilution_sweep_N5", "dilution_sweep_N10", "dilution_sweep_N20",
    # Round 4: composite + memory minimal
    "adversarial_M_composite_memmin_10", "adversarial_M_composite_memmin_20",
    "adversarial_M_composite_memmin_30", "adversarial_M_composite_memmin_40",
    # Round 4: gradient accumulation dilution
    "adversarial_N_grad_accum_4", "adversarial_N_grad_accum_8",
    "adversarial_N_grad_accum_16",
    # DDP baseline training (datacenter multi-GPU runs)
    "ddp_resnet18_cifar10", "ddp_resnet18_cifar10_amp", "ddp_resnet50_cifar10",
    # FSDP training (datacenter multi-GPU runs)
    "fsdp_7b_nvlink", "fsdp_13b_nvlink", "fsdp_30b_nvlink", "fsdp_30b_adamw_nvlink",
    "fsdp_70b_nvlink", "fsdp_70b_seq512_nvlink", "fsdp_70b_seqlen512",
    "fsdp_70b_adamw", "fsdp_70b_sgd_momentum",
    # DDP adversarial (datacenter multi-GPU evasion)
    "adversarial_K_ddp", "adversarial_K_ddp_accum_4", "adversarial_K_ddp_accum_8",
    "adversarial_L_ddp_5", "adversarial_L_ddp_10", "adversarial_L_ddp_stagger_5",
    # Round 5: White-box evasion at relevant model scale (full fine-tuning)
    "whitebox_diluted_N5", "whitebox_diluted_N10", "whitebox_diluted_N20", "whitebox_diluted_N50",
    "whitebox_diluted_N5_ckpt", "whitebox_diluted_N10_ckpt", "whitebox_diluted_N20_ckpt",
    "whitebox_oracle",
    # Round 5: White-box LoRA fine-tuning
    "whitebox_lora_N5", "whitebox_lora_N10", "whitebox_lora_N20",
    # Round 5 — AMD registry names (Tier 3 collection)
    "whitebox_WB-D_diluted", "whitebox_WB-L_lora", "whitebox_WB-O_oracle",
    # Tier 4 LoRA fine-tunes (AMD registry)
    "lora_finetune_qwen25_0p5b", "lora_finetune_smollm2_135m",
    # YOLO training (object detection)
    "yolo_train_coco128",
    # Generic training labels seen in datacenter runs
    "adversarial_training",
    "adversarial_training_2gpu_gpu0", "adversarial_training_2gpu_gpu1",
    "bert_finetune",
    "bert_finetune_2gpu_gpu0", "bert_finetune_2gpu_gpu1",
    "gpt2_finetune",
    "gpt2_finetune_2gpu_gpu0", "gpt2_finetune_2gpu_gpu1",
    "pytorch_training",
    "pytorch_training_2gpu_gpu0", "pytorch_training_2gpu_gpu1",
    # Round 5b (phase 4): matched-pair LoRA training on consumer tier
    "llm_train_qwen25_1p5b_lora",
    "llm_train_smollm2_1p7b_lora",
    "llm_train_qwen25_3b_lora",
    "llm_train_phi3_mini_lora",
    "llm_train_mistral_7b_lora",
    # Round 6: symmetric-GEMM backprop disguise. Weights genuinely change in
    # these, so they are training regardless of how the compute is shaped.
    # L3 is deliberately NOT here: it runs the dW GEMM but discards the result,
    # so no learning occurs and it belongs in INFERENCE_LABELS as a probe.
    "symgemm_L4_sgdfused", "symgemm_L4_sgdfused_m8192",
    "symgemm_L4_sgdfused_7b", "symgemm_L4_triton",
    "symgemm_L5_adam",
    # Round 6 evasion: DISGUISED PPO (fused-SGD policy update) -- stacks
    # temporal dilution on write minimisation.
    "symgemm_ppo_fused_r8", "symgemm_ppo_fused_r32",
    # BASELINE corpus: undisguised PPO (ordinary autograd + AdamW). A real RLHF
    # training regime, added so the corpus covers RL; not an evasion.
    "ppo_rlhf_r8", "ppo_rlhf_r32",
}

INFERENCE_LABELS = {
    "resnet50_inference",
    "resnet50_inference_standalone",
    "resnet50_inference_large_bs1024", "resnet50_inference_small_bs32",
    "resnet50_concurrent_concurrent",
    "resnet50_inference_2gpu_gpu0", "resnet50_inference_2gpu_gpu1",
    "BASELINE_INFER", "EC1", "EC4",
    # Modern inference models
    "deepseek_r1_inference",
    "qwen25_inference",
    "whisper_inference",
    "modern_inference",
    "modern_inference_2gpu_gpu0", "modern_inference_2gpu_gpu1",
    # Controlled dilution sweep: pure inference baseline
    "dilution_sweep_inference",
    # Round 5: White-box inference baseline (control)
    "whitebox_inference",
    "whitebox_WB-I_inference",
    # Round 7 organic negative: best-of-n / rejection sampling against a reward
    # model. Generation plus a scoring forward, no weight ever updated. This is
    # the non-training null pair for symgemm_ppo_fused, and unlike a
    # "PPO-with-the-update-removed" ablation it is a workload people actually
    # run (STaR/RFT data generation, inference-time scaling), so it belongs in
    # the baseline corpus rather than in the evasion ladder.
    "bestofn_rm_n8", "bestofn_rm_n32",
    # Round 6 negatives. L1 runs three forward passes per iteration so it is
    # FLOP- and throughput-matched to a training step; L3 additionally runs the
    # dW GEMM but discards the output, so it has identical compulsory DRAM
    # traffic to L1 with no weight ever changing. Both are genuinely
    # not-training: L1 vs L4 is the decisive comparison, L1 vs L3 the null pair.
    "symgemm_L0_fwd1", "symgemm_L1_fwd3", "symgemm_L1_fwd3_m8192",
    "symgemm_L1_fwd3_7b", "symgemm_L1_triton",
    "symgemm_L3_graddiscard",
    # Tier 4 image-generation inference (SDXL)
    "sdxl_inference",
    # Round 5b: new large-LLM inference anchors (single-GPU, open-license)
    "llm_infer_phi3_mini", "llm_infer_qwen25_1p5b", "llm_infer_qwen25_3b",
    "llm_infer_smollm2_1p7b", "llm_infer_mistral_7b_q4",
    "llm_infer_tinyllama_bs1", "llm_infer_tinyllama_bs2", "llm_infer_tinyllama_bs4",
    "llm_infer_tinyllama_bs8", "llm_infer_tinyllama_seq1024",
    "llm_infer_tinyllama_synthetic_bs1",
    "llm_infer_qwen25_3b_bs4", "llm_infer_qwen25_1p5b_streaming",
    "llm_infer_sdxl", "llm_infer_clip_embed_batched",
    # Round 5b phase 3: long-context, large-batch, encoder-only inference
    "llm_infer_phi3_mini_seq2048",
    "llm_infer_tinyllama_seq2048", "llm_infer_tinyllama_seq4096",
    "llm_infer_qwen25_1p5b_seq4096",
    "llm_infer_qwen25_3b_bs16",
    "llm_infer_clip_embed_bs256", "llm_infer_bge_embed_bs128",
    "llm_infer_vit_l_bs64",
    # Round 5b phase 5: matched-pair dense inference at FSDP sizes
    "llm_infer_qwen25_7b_fp16", "llm_infer_qwen25_14b_fp16",
    "llm_infer_qwen25_32b_fp16",
    # Round 5b: datacenter-only large-LLM inference (>80 GB, open-license)
    "llm_infer_qwen25_72b_fp16", "llm_infer_mixtral_8x7b",
    "llm_infer_qwen25_72b_fp8",
}

# Multi-GPU TP inference emits per-GPU parquets with _NgpuN suffix;
# enumerate them explicitly so the label sets cover every file.
for _n in (0, 1):
    INFERENCE_LABELS.add(f"llm_infer_qwen25_72b_tp2_2gpu_gpu{_n}")
for _n in range(8):
    INFERENCE_LABELS.add(f"llm_infer_deepseek_v3_tp8_8gpu_gpu{_n}")

OTHER_LABELS = {
    "idle", "idle_h100",
    "idle_2gpu_gpu0", "idle_2gpu_gpu1",
    "cufft_benchmark", "cufft_benchmark_standalone",
    "cufft_short_dur120", "cufft_concurrent_concurrent",
    "cufft_benchmark_2gpu_gpu0", "cufft_benchmark_2gpu_gpu1",
    "nbody_sim", "nbody_sim_standalone", "nbody_short_dur120",
    "nbody_sim_2gpu_gpu0", "nbody_sim_2gpu_gpu1",
    "mining_ethash_proxy", "ethash_cuda",
    "rendering_proxy", "blender_bmw",
    "ffmpeg_nvenc",
    "gromacs_adh",
    "nvlink_bandwidth", "nvlink_latency",
    "nvlink_p2p", "nvlink_p2p_2gpu_gpu0", "nvlink_p2p_2gpu_gpu1",
    "scientific_hpc",
    "scientific_hpc_2gpu_gpu0", "scientific_hpc_2gpu_gpu1",
}


LEGACY_TRAINING_LABELS = frozenset(TRAINING_LABELS)
LEGACY_INFERENCE_LABELS = frozenset(INFERENCE_LABELS)
LEGACY_OTHER_LABELS = frozenset(OTHER_LABELS)


def _legacy_label(label: str) -> str:
    """Map workload label to three-way category."""
    if label in LEGACY_TRAINING_LABELS:
        return "ml_training"
    if label in LEGACY_INFERENCE_LABELS:
        return "ml_inference"
    if label in LEGACY_OTHER_LABELS:
        return "other"
    # Heuristic fallback. Strip the per-stream rank token first so a multi-node
    # filename label (e.g. s16_train_fsdp7b_node1_gpu3, or a TPU s16 analogue
    # ..._w2c1) routes on its base label exactly as the single-stream label
    # would -- otherwise the trailing token defeats the endswith/keyword checks.
    lc = re.sub(r"_node\d+_gpu\d+$", "", label.lower())
    lc = re.sub(r"_w\d+c\d+$", "", lc)
    lc = re.sub(r"_gpu\d+$", "", lc)
    # White-box variants carry a model/duration suffix (whitebox_lora_7b_long,
    # whitebox_diluted_7b_N10_1h, ...) that the explicit lists above do not
    # enumerate, and none of them contain "train"/"adversarial_". Without these
    # keys they fall through to "other", which silently puts LoRA/full-FT
    # training runs into the negative class — both polluting the corpus and, if
    # they are left in a classifier's training set, teaching it that this
    # telemetry is NOT training. Checked before the "infer" rule so that only
    # whitebox_inference_* is treated as inference.
    if any(k in lc for k in ("whitebox_lora", "whitebox_diluted", "whitebox_oracle",
                             "wb-l", "wb-d", "wb-o")):
        return "ml_training"
    # Round 6 symmetric-GEMM ladder. Collection appends a _rep<N> suffix, and
    # cells carry scale/backend suffixes (_m8192, _7b, _triton), so the
    # explicit sets above cannot enumerate every run label. Classify by mode
    # instead: L4/L5 change weights (training), L0/L1/L3 never do. Without
    # this, a suffixed label falls through to "other" and the corpus quietly
    # learns that disguised training is not training -- the same failure the
    # whitebox rule above exists to prevent.
    # Undisguised PPO (mode P0: rollout + ordinary autograd/AdamW update). This
    # is a legitimate RL *training regime* for the baseline corpus, not an
    # evasion, so it is deliberately NOT symgemm_*-prefixed (that prefix is held
    # out of the clean training pool). Without this rule "ppo_rlhf_*" matches
    # none of the keywords below and would land in "other" -- teaching the
    # corpus that RL training is not training.
    # Best-of-n / rejection sampling: generation + reward-model scoring, no
    # optimiser anywhere. Checked BEFORE the "ppo_"/train keyword rules so a
    # future bestofn_ppo_* naming cannot be swept into training.
    if lc.startswith("bestofn_"):
        return "ml_inference"
    if lc.startswith("ppo_"):
        return "ml_training"
    if lc.startswith("symgemm_"):
        # PPO shape: the rollout is generation, but the policy update is an
        # ordinary gradient step that changes every trained weight, so the run
        # as a whole is training at a duty cycle set by rollout:update.
        # (symgemm_ppo_fused_* = the DISGUISED variant, an evasion cell.)
        if "_ppo" in lc:
            return "ml_training"
        # Pure autoregressive decode -- no weight ever changes.
        if "_decode" in lc:
            return "ml_inference"
        if "_l4" in lc or "_l5" in lc:
            return "ml_training"
        if "_l0" in lc or "_l1" in lc or "_l3" in lc:
            return "ml_inference"
        return "other"        # symgemm_idle
    if any(k in lc for k in ("train", "bert", "gpt", "resnet_cifar", "mlp_cifar",
                              "fsdp_", "ddp_", "adversarial_")):
        return "ml_training"
    if any(k in lc for k in ("infer",)):
        return "ml_inference"
    return "other"


# Collection aliases documented in docs/physical_sensor_detection_and_evasion.md
# §§6–7; each program performs optimizer steps. Keep exact names so an unknown
# future probe is not silently declared training because its name looks similar.
PHYSICAL_TRAINING = {
    "dil_v1", "dil_v2", "dil_v3",
    "ev2b_aperiodic", "ev2b_baseline", "ev2b_cofill", "ev2b_combo",
    "ev2b_gapfill0004", "ev2b_gapfill0010", "ev2b_gradaccum",
    "evasion2_aperiodic", "evasion2_baseline", "evasion2_cofill",
    "evasion2_combo", "evasion2_nobubble",
    "s2_fb16", "s2_fb24", "s3_tiny8", "s3_tp1", "s3_tp2", "s3_tp3",
    "stealth_dilution", "ev2_combo",
}
PHYSICAL_CONTROLS = {"ev2b_baseline", "evasion2_baseline"}
# Jobs that never take a gradient step are not training, whatever apparatus they
# keep resident: none of their windows contains a weight update. The MN-I filler
# at train fraction 0 is listed by name; every stalled_* cell (stalled_training.py
# holds a model and optimizer but never steps it) is matched by prefix.
ZERO_UPDATE_CONTROLS = {"advtrain_filler_only", "advtrain_satfiller_only"}
OTHER_LABELS.update({"ffmpeg_amf_av1", "ffmpeg_amf_h264", "ffmpeg_amf_hevc_4k",
                     "ffmpeg_vaapi", "scientific_fft", "scientific_nbody", "symgemm_idle",
                     "cufft_hpc", "mining_proxy", "rendering"})
TRAINING_LABELS.update({"resnet_amp", "resnet_fp32", "batch_size_sweep_dp", "width_sweep_dp"})
SCOPE = re.compile(r"^(?:smoke_)?(s16|s8|s1|tpuTMS\d*|tpuT\d+)_", re.I)
ADV = re.compile(r"advtrain|adversarial|whitebox|symgemm|typeconf|stalled|_mw8", re.I)


def canonical_workload(label):
    """Remove collector rank/repetition suffixes, retaining experiment parameters."""
    s = str(label)
    for pattern in (r"_node\d+(?:_gpu\d+)?$", r"_w\d+c\d+$", r"_gpu\d+$", r"_rep\d+$"):
        s = re.sub(pattern, "", s, flags=re.I)
    return s


@lru_cache(maxsize=16384)
def label_evidence(label):
    """Return objective and routing provenance; objective is NOT per-window activity.

    Training means the job performs gradient updates. A job that never updates
    weights is not training, even when it is named, built or shaped like a
    training job.
    """
    cell = canonical_workload(label)
    lc = SCOPE.sub("", cell).lower()
    # Week 7 uses descriptive EC names; unlike model-name substrings these
    # routes follow the programs in week7/edge_cases/run_edge_cases.py.
    edge = re.match(r"^ec([1-8])_(?:phantom|silent|sparse|mining|frozen|low_int|amp_mask|mem_idle)(?:_|$)", lc)
    if edge:
        n = int(edge[1])
        return ("ml_inference" if n in (1, 4) else "other" if n == 8 else "ml_training"), "week7_edge_case"
    if lc in PHYSICAL_TRAINING:
        return "ml_training", "physical_optimizer_steps"
    if lc in ZERO_UPDATE_CONTROLS:
        return "other", "zero_training_filler_control"
    if lc.startswith("stalled_"):
        return "other", "zero_update_stall_control"
    # Explicit collection intent precedes model-name substrings (infer_gpt,
    # nonml_ddp, etc.).
    for prefix, target in (("nonml_", "other"), ("infer", "ml_inference"),
                           ("train_", "ml_training"), ("advtrain_", "ml_training")):
        if lc.startswith(prefix):
            return target, "collection_intent"
    for labels, target in ((TRAINING_LABELS, "ml_training"),
                           (INFERENCE_LABELS, "ml_inference"), (OTHER_LABELS, "other")):
        # Legacy Week 4 exports append hardware / multi-device suffixes to
        # explicitly registered names (e.g. resnet18_small_batch_batch8_H100).
        registered = {x.lower() for x in labels}
        stripped = re.sub(r"_(?:h100|\d+gpu)$", "", cell.lower())
        if cell.lower() in registered or stripped in registered:
            return target, "explicit_registry"
    if lc.startswith(("idle", "other_")):
        return "other", "collection_control"
    if re.match(r"^nvlink_(?:bandwidth|latency|p2p)(?:_|$)", lc):
        return "other", "communication_benchmark"
    target = _legacy_label(cell)
    return target, "legacy_rule" if target != "other" else "unreviewed_other"


def threeway_label(label):
    return label_evidence(label)[0]


def validate_labels(labels):
    """Prevent an unseen fallback label from silently becoming a negative."""
    unknown = sorted({str(label) for label in labels
                      if label_evidence(label)[1] == "unreviewed_other"})
    if unknown:
        raise ValueError(f"Unreviewed ground truth for {unknown[:20]}; add source-backed taxonomy entries before fitting")


def workload_kind(label):
    lc = SCOPE.sub("", canonical_workload(label)).lower()
    if lc.startswith("stalled_") or lc in PHYSICAL_CONTROLS or lc in ZERO_UPDATE_CONTROLS:
        return "control"
    if lc in PHYSICAL_TRAINING or ADV.search(lc) or lc.startswith("stealth"):
        return "adversarial"
    return "honest"


# Honest programs whose work switches kind from slot to slot: the distributed RL
# jobs (actors serve rollouts while the learner trains) and single-GPU PPO.
MIXED_WORK_FAMILIES = {"train_impala", "train_apexrl", "train_ppodist", "ppo_rlhf"}


def program_work(label):
    """The only kind of work a program ever does, or None when it mixes kinds.

    Only for windows with no STEP evidence. An inference server, a non-ML
    benchmark or a plain trainer can only do its one kind of work. A disguise,
    an RL job or a diluted sweep changes kind between slots, so its windows
    cannot be labeled without the launcher log.
    """
    lc = SCOPE.sub("", canonical_workload(label)).lower()
    if workload_kind(label) == "adversarial" or workload_family(label) in MIXED_WORK_FAMILIES:
        return None
    sweep = re.fullmatch(r"dilution_sweep_n(\d+)", lc)
    if sweep and int(sweep[1]) > 1:
        return None
    return threeway_label(label)


FAMILY_SCHEMA_VERSION = "2026-10-11.1"
MERGE = {"ddpkaccum": "ddp", "ddpstagger": "ddp", "lp": "lowpower",
         "filler": "satfiller", "vllm": "qwen", "flatbubble": "flatarena"}


def workload_family(label, level="mechanism"):
    """Hold out parameter siblings, plus an optional shared-model backbone test.

    Mechanism groups merge honest/evasive pipeline and async serving schedules.
    Backbone groups additionally merge programs using fsdp_training.RandomTransformerLM.
    These are complementary generalization claims, not independent model families.
    """
    if level not in ("mechanism", "backbone"):
        raise ValueError(level)
    lc = SCOPE.sub("", canonical_workload(label)).lower()
    if lc in PHYSICAL_TRAINING or lc.startswith("adversarial_physical_"):
        return "physical_resnet_evasion"
    parts = lc.split("_")
    cls = parts[0]
    base = re.sub(r"\d+(?:_\d+)?[bkmg]?$", "", parts[1]) if len(parts) > 1 else ""
    base = MERGE.get(base, base)
    if cls == "trainmem":
        cls = "train"
    if cls == "infermem":
        cls = "infershard"
    if cls == "train" and base in {"accum", "accumsync"} and "fsdp" in lc:
        base = "fsdp"
    if level == "backbone" and (
        (cls == "train" and base in {"fsdp", "lora", "pipe", "localsgd", "streamdiloco", "decoupdiloco", "impala", "apexrl", "ppodist"})
        or (cls == "advtrain" and base in {"pipe", "interleave", "satint", "satfiller", "oracle", "flatarena", "bubble"})
        or (cls == "stalled" and "fsdp" in lc)
    ):
        return "random_transformer_lm"
    if cls == "advtrain" and base in {"interleave", "satint", "satfiller"}:
        return "multinode_interleave"
    if cls == "inferserve":
        return "inferserve_async"
    if cls in ("train", "advtrain") and base == "pipe":
        return "train_pipe"
    return f"{cls}_{base}" if base else cls
