"""
Improved Three-Way GPU Workload Classifier
===========================================
Classifies GPU telemetry windows as: ml_training / ml_inference / other

Key improvements over Week 5 baseline (48-61% → target 80%+):
1. Training-specific temporal features (epoch periodicity, memory growth rate)
2. Inference-distinguishing features (no memory growth + high util, batch-pause ratio)
3. Loads ALL data sources (data/, week4/, week5/, data/h100_nvl/)
4. XGBoost with balanced class weights and proper cross-validation
5. Better label mapping for edge-case workloads

Usage:
    python -m classifier.threeway_improved
    python classifier/threeway_improved.py
"""

import json
import logging
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Label taxonomy ───────────────────────────────────────────────────────────

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


def threeway_label(label: str) -> str:
    """Map workload label to three-way category."""
    if label in TRAINING_LABELS:
        return "ml_training"
    if label in INFERENCE_LABELS:
        return "ml_inference"
    if label in OTHER_LABELS:
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


# ── Raw signal columns ───────────────────────────────────────────────────────

RAW_SIGNALS = [
    "gpu_utilization_pct", "mem_utilization_pct", "mem_used_mb",
    "power_draw_w", "temperature_c", "sm_clock_mhz", "mem_clock_mhz",
    "pcie_tx_mbps", "pcie_rx_mbps",
]

STAT_NAMES = ["mean", "std", "min", "max", "p25", "p50", "p75", "p95",
              "iqr", "range", "cv", "skew", "kurt"]


# ── Data loading ─────────────────────────────────────────────────────────────

# GPU name aliases: SKUs NVML reports differently that we want to treat as
# one class. H200 SXM and H200 NVL are the same silicon / 141 GB HBM3e —
# only the form factor differs, so we merge them for classifier training.
# A100-SXM4-40GB vs 80GB stay separate (different memory capacity matters),
# H100-80GB HBM3 vs H100-NVL stay separate (different memory + TDP class).
GPU_NAME_ALIASES: dict[str, str] = {
    "NVIDIA H200 NVL": "NVIDIA H200",
}


def _normalize_gpu_name(name):
    if name is None:
        return None
    s = str(name).strip()
    return GPU_NAME_ALIASES.get(s, s)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename variant column names to the standard schema."""
    rename = {
        "timestamp_epoch": "ts",
        "timestamp":       "ts",
        "gpu_util":        "gpu_utilization_pct",
        "mem_util":        "mem_utilization_pct",
        "power_w":         "power_draw_w",
        "temp_c":          "temperature_c",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    if "ts" not in df.columns and "timestamp_utc" in df.columns:
        df["ts"] = pd.to_datetime(df["timestamp_utc"]).astype("int64") / 1e9

    if "mem_used_gb" in df.columns and "mem_used_mb" not in df.columns:
        df["mem_used_mb"] = df["mem_used_gb"] * 1024

    for sig in RAW_SIGNALS:
        if sig not in df.columns:
            df[sig] = 0.0

    if "gpu_name" in df.columns:
        df["gpu_name"] = df["gpu_name"].map(_normalize_gpu_name)

    return df


def load_all_data() -> pd.DataFrame:
    """Load telemetry from all data sources."""
    frames = []

    # 1. Main data/ directory
    _skipped: list[tuple[str, str]] = []
    for p in sorted((PROJECT_ROOT / "data").rglob("*.parquet")):
        # nic_*.parquet are host-NIC sidecar series, not per-GPU telemetry
        # streams. multinode_features.attach_nic_sidecar joins them on
        # (node_id, epoch second) from this same tree; loading them here as well
        # injects rows with no `ts`/`gpu_name`, and the NaN run_start that
        # follows crashes assign_stream_and_job_ids with IntCastingNaNError.
        if p.name.startswith("nic_"):
            continue
        # Quarantined streams are NOT corpus. ingest_session.sh moves idle
        # streams (median utilisation <= 1%) into a _quarantine_idle/ directory
        # so a multi-node cell whose peer died cannot enter as labelled
        # workload data -- but that directory sits under data/, and this loader
        # rglobs data/**, so the quarantine was cosmetic: all 78 streams it
        # caught on 2026-09-08 were being loaded anyway, including into a
        # retrain. Excluding by PATH here is the invariant that actually holds,
        # wherever a future quarantine directory ends up.
        if any("_quarantine" in part for part in p.parts):
            continue
        try:
            df = pd.read_parquet(p)
            df = _normalize_columns(df)
            if "workload_label" not in df.columns:
                # Parse from filename
                stem = p.stem
                # TPU parquets end `..._w<W>c<C>_Google_TPU_<gen>_<run_id>`, and the
                # generic rsplit("_", 3) fallback strips only the last three fields,
                # leaving a label of `tpuTMS_nonml_duty30_rep2_w0c0_Google` -- a
                # truncated device name glued to the chip token. That label then
                # defeats workload_group(), whose `_rep\d+$` anchor no longer
                # matches, so every rep and every chip of every TPU cell becomes its
                # own "workload" and held-out-by-workload CV silently degenerates
                # into run-grouped CV -- exactly the leak that criterion exists to
                # remove. Split on the accelerator name as the NVIDIA path does.
                if "_NVIDIA_" in stem:
                    parts = stem.split("_NVIDIA_")[0]
                elif "_Google_TPU_" in stem:
                    parts = stem.split("_Google_TPU_")[0]
                else:
                    parts = stem.rsplit("_", 3)[0]
                df["workload_label"] = parts
            if "run_id" not in df.columns:
                df["run_id"] = p.stem
            frames.append(df)
        except Exception as e:
            # Counted and reported at WARNING below. Six of nine legacy *tpu*
            # parquets were being dropped here without a single visible line,
            # and a schema the loader does not expect will do the same to real
            # TPU telemetry.
            _skipped.append((p.name, str(e)[:120]))
            log.debug(f"skip {p.name}: {e}")

    # 2. Week 4 data
    week4_data = PROJECT_ROOT / "week4" / "data"
    if week4_data.exists():
        for p in sorted(week4_data.glob("*.parquet")):
            try:
                df = pd.read_parquet(p)
                df = _normalize_columns(df)
                if "workload_label" not in df.columns:
                    df["workload_label"] = p.stem.split("_NVIDIA_")[0] if "_NVIDIA_" in p.stem else p.stem
                if "run_id" not in df.columns:
                    df["run_id"] = f"w4_{p.stem}"
                frames.append(df)
            except Exception as e:
                log.debug(f"skip {p.name}: {e}")

    # 3. Week 4 edge cases (JSON)
    ec_dir = PROJECT_ROOT / "week4" / "results" / "edge_cases"
    if ec_dir.exists():
        for p in sorted(ec_dir.glob("*_telemetry.json")):
            try:
                with open(p) as f:
                    records = json.load(f)
                if not records:
                    continue
                df = pd.DataFrame(records)
                df = _normalize_columns(df)
                case_name = p.stem.replace("_telemetry", "")
                df["workload_label"] = case_name
                df["run_id"] = f"ec_{case_name}"
                frames.append(df)
            except Exception as e:
                log.debug(f"skip {p.name}: {e}")

    # 4. Week 4 DDP traces
    ddp_dir = PROJECT_ROOT / "week4" / "results" / "ddp"
    if ddp_dir.exists():
        for p in sorted(ddp_dir.glob("*.parquet")):
            try:
                df = pd.read_parquet(p)
                df = _normalize_columns(df)
                if "workload_label" not in df.columns:
                    df["workload_label"] = "training_dual_gpu_dp"
                if "run_id" not in df.columns:
                    df["run_id"] = f"ddp_{p.stem}"
                frames.append(df)
            except Exception as e:
                log.debug(f"skip {p.name}: {e}")

    # 5. Week 4 dataset-scale traces
    dscale_dir = PROJECT_ROOT / "week4" / "results" / "dataset_scale"
    if dscale_dir.exists():
        for p in sorted(dscale_dir.rglob("telemetry.parquet")):
            try:
                df = pd.read_parquet(p)
                df = _normalize_columns(df)
                if "workload_label" not in df.columns:
                    df["workload_label"] = "training_dual_gpu_dp"
                if "run_id" not in df.columns:
                    df["run_id"] = f"dscale_{p.parent.name}"
                frames.append(df)
            except Exception as e:
                log.debug(f"skip {p.name}: {e}")

    if not frames:
        raise RuntimeError("No telemetry data found")

    combined = pd.concat(frames, ignore_index=True)

    # Drop rows with no timestamp. A few parquets carry an all-null padding row
    # (every metric NaN, including ts/gpu_name/hostname) -- most likely a
    # *.parquet.partial checkpoint flushed with a header and no sample. Such a
    # row cannot be windowed, and it propagates a NaN run_start into
    # assign_stream_and_job_ids, which then dies with IntCastingNaNError.
    # NON-FINITE counts as "no timestamp" too. `.isna()` is False for +/-inf, and
    # assign_stream_and_job_ids casts (run_start / BUCKET) to int64 -- so a single
    # infinite ts anywhere in the corpus raises the same IntCastingNaNError this
    # guard exists to prevent, taking every other run down with it. Observed on
    # the full 3.6 M-row corpus, where it aborted the binary train-vs-rest CV.
    if "ts" in combined.columns:
        _ts = pd.to_numeric(combined["ts"], errors="coerce")
        bad = ~np.isfinite(_ts.to_numpy(dtype="float64", na_value=np.nan))
        n_bad = int(bad.sum())
        if n_bad:
            log.warning(f"dropping {n_bad} row(s) with a missing or non-finite "
                        f"timestamp")
            combined = combined[~bad].reset_index(drop=True)

    # Deduplicate: some files may appear in multiple directories.
    # gpu_uuid must be in the key. NVIDIA runs survive on (ts, run_id) only by
    # luck: one collector process per GPU means each has its own datetime.now(),
    # so the float epochs differ. The TPU collector samples every chip in ONE
    # poll and stamps them with a single `now`
    # (scripts/collect_tpu_telemetry.py:428-429), and the queue mandates a shared
    # run_id across workers -- so a 16-chip slice would silently collapse to one
    # chip. Adding the per-chip uuid keeps the original intent (identical rows
    # from duplicated files still dedup, since a copy shares all three keys).
    if "ts" in combined.columns and "run_id" in combined.columns:
        _key = ["ts", "run_id"]
        if "gpu_uuid" in combined.columns:
            _key.append("gpu_uuid")
        _before = len(combined)
        combined = combined.drop_duplicates(subset=_key, keep="first")
        log.info(f"dedup on {_key}: {_before:,} -> {len(combined):,} rows")
        combined = combined.sort_values(["run_id", "ts"]).reset_index(drop=True)

    # Add three-way labels
    combined["threeway_label"] = combined["workload_label"].apply(threeway_label)

    label_dist = combined.groupby("threeway_label")["run_id"].nunique()
    if _skipped:
        log.warning(f"skipped {len(_skipped)} unreadable/incompatible parquet(s) "
                    f"under data/: " + ", ".join(n for n, _ in _skipped[:5])
                    + (" ..." if len(_skipped) > 5 else ""))
        for _n, _e in _skipped[:5]:
            log.warning(f"  {_n}: {_e}")
    log.info(f"Total rows: {len(combined):,} | runs: {combined['run_id'].nunique()}")
    log.info(f"Three-way distribution (runs):\n{label_dist.to_string()}")

    return combined


# ── Enhanced feature extraction ──────────────────────────────────────────────

def acf_lag(x: np.ndarray, lag: int) -> float:
    """Autocorrelation at given lag."""
    if len(x) < lag + 3:
        return 0.0
    mu = x.mean()
    var = ((x - mu) ** 2).mean()
    if var < 1e-12:
        return 0.0
    return float(((x[:-lag] - mu) * (x[lag:] - mu)).mean() / var)


def fft_dominant_period(x: np.ndarray) -> float:
    """Dominant period (in samples) from FFT. Captures epoch periodicity."""
    if len(x) < 10:
        return 0.0
    x_centered = x - x.mean()
    fft_vals = np.abs(np.fft.rfft(x_centered))
    # Skip DC component (index 0) and very low frequencies
    if len(fft_vals) < 3:
        return 0.0
    fft_vals[0] = 0
    peak_idx = np.argmax(fft_vals[1:]) + 1
    if fft_vals[peak_idx] < 1e-6:
        return 0.0
    period = len(x) / peak_idx
    return float(period)


def fft_peak_power(x: np.ndarray) -> float:
    """Ratio of peak FFT magnitude to total — measures periodicity strength."""
    if len(x) < 10:
        return 0.0
    x_centered = x - x.mean()
    fft_vals = np.abs(np.fft.rfft(x_centered))
    if len(fft_vals) < 3:
        return 0.0
    fft_vals[0] = 0
    total = fft_vals.sum()
    if total < 1e-12:
        return 0.0
    return float(fft_vals.max() / total)


def extract_enhanced_features(chunk: pd.DataFrame) -> dict:
    """Extract ~155 features per window, including training-specific temporals."""
    feats = {}

    # ── 1. Base statistics (9 signals × 13 stats = 117 features) ─────────
    # BATCHED ACROSS SIGNALS, not because it is tidier but because the per-call
    # overhead dominated. Profiled on 2026-09-18: windowing the corpus cost
    # 23.4 ms/window, of which scipy.stats' axis_nan_policy wrapper was 56% --
    # 18 calls per window (9 signals x skew+kurtosis), and most of that time was
    # the wrapper re-deriving each function's signature with inspect and
    # re-broadcasting its arguments, not computing moments. scipy guarantees
    # f(M, axis=0)[i] == f(M[:, i]), verified here including NaN placement on
    # constant columns, so batching is numerically identical and turns 18 calls
    # into 2 (and 9 np.percentile calls into 1).
    cols = [chunk[s].fillna(0).values.astype(float) if s in chunk.columns
            else np.zeros(max(len(chunk), 1)) for s in RAW_SIGNALS]
    M = np.column_stack(cols)
    n = M.shape[0]
    mus = M.mean(axis=0)
    sds = M.std(axis=0)
    vmins = M.min(axis=0)
    vmaxs = M.max(axis=0)
    P = (np.percentile(M, [25, 50, 75, 95], axis=0) if n > 1
         else np.tile(mus, (4, 1)))
    # Constant columns (an idle GPU, or a TPU's hard-wired -1 sentinel) make
    # scipy return NaN here -- that is the EXISTING behaviour, preserved: the
    # hot paths nan_to_num it to 0 downstream.
    skews = (sp_stats.skew(M, axis=0) if n > 2 else np.zeros(len(RAW_SIGNALS)))
    kurts = (sp_stats.kurtosis(M, axis=0) if n > 3 else np.zeros(len(RAW_SIGNALS)))

    for i, sig in enumerate(RAW_SIGNALS):
        mu, sd = mus[i], sds[i]
        p25, p50, p75, p95 = P[0, i], P[1, i], P[2, i], P[3, i]
        feats[f"{sig}_mean"] = mu
        feats[f"{sig}_std"] = sd
        feats[f"{sig}_min"] = vmins[i]
        feats[f"{sig}_max"] = vmaxs[i]
        feats[f"{sig}_p25"] = p25
        feats[f"{sig}_p50"] = p50
        feats[f"{sig}_p75"] = p75
        feats[f"{sig}_p95"] = p95
        feats[f"{sig}_iqr"] = p75 - p25
        feats[f"{sig}_range"] = vmaxs[i] - vmins[i]
        feats[f"{sig}_cv"] = sd / (abs(mu) + 1e-9)
        feats[f"{sig}_skew"] = float(skews[i])
        feats[f"{sig}_kurt"] = float(kurts[i])

    # ── 2. Cross-signal derived features (5) ─────────────────────────────
    util = chunk["gpu_utilization_pct"].fillna(0).values.astype(float)
    power = chunk["power_draw_w"].fillna(0).values.astype(float)
    pcie_tx = chunk["pcie_tx_mbps"].fillna(0).values.astype(float)
    pcie_rx = chunk["pcie_rx_mbps"].fillna(0).values.astype(float)
    sm_clk = chunk["sm_clock_mhz"].fillna(0).values.astype(float)
    mem_used = chunk["mem_used_mb"].fillna(0).values.astype(float)

    feats["power_per_util"] = power.mean() / (util.mean() + 1.0)
    feats["pcie_total_mean"] = (pcie_tx + pcie_rx).mean()
    # Clamp the clock mean at >= 0 before the +1 guard. Non-NVIDIA collectors
    # write -1 for channels the hardware never exposes (TPU: all of power,
    # temperature and both clocks -- scripts/collect_tpu_telemetry.py:74-78), and
    # -1 + 1 = 0 made this exactly +inf. Every hot path uses .fillna(0), which
    # does NOT touch infinities, so model_store --fit, twostage --train,
    # evasion_cost_eval and scope_ladder_eval all raised
    # "Input X contains infinity"; worse, the cross-vendor evaluators call
    # np.nan_to_num without posinf= and silently turned it into 3.4e38.
    # Clamping makes a -1 sentinel behave exactly like an absent column, which
    # _normalize_columns already zero-fills.
    _sm_mean = max(float(sm_clk.mean()), 0.0)
    feats["util_per_sm_pct"] = util.mean() / (_sm_mean + 1.0) * 1000.0

    # DRAM-activity cross-signal ratios (Round 6 monitor response). mem_util is
    # NVML's fraction of the sample period spent reading/writing device memory —
    # the one channel a weight write must touch and inference never does. These
    # ratios make the DRAM-per-compute direction explicit, which the marginal
    # stats alone leave diagonal. See classifier/rebuttal_analysis/symgemm_eval.py.
    mem_util = chunk["mem_utilization_pct"].fillna(0).values.astype(float) \
        if "mem_utilization_pct" in chunk.columns else np.zeros(max(len(chunk), 1))
    mu_mean, gu_mean = mem_util.mean(), util.mean()
    mu_p95 = np.percentile(mem_util, 95) if len(mem_util) > 1 else mu_mean
    gu_p95 = np.percentile(util, 95) if len(util) > 1 else gu_mean
    feats["mem_util_per_gpu_util"] = mu_mean / (gu_mean + 1.0)
    feats["mem_util_minus_gpu_util"] = mu_mean - gu_mean
    feats["mem_util_per_gpu_util_p95"] = mu_p95 / (gu_p95 + 1.0)
    feats["power_per_mem_util"] = power.mean() / (mu_mean + 1.0)

    # Peak / conditional DRAM-write features (Round 6, PPO response). A diluted
    # trainer (PPO's sparse policy update, gradient accumulation) writes weights
    # in a brief periodic BURST, so its window MEAN mem_util is indistinguishable
    # from decode inference (both ~6%) while the update SPIKE is not (mem_util_max
    # ~11 vs ~8). Mean-based ratios above miss that; these target the spike and
    # its rhythm — the one channel a weight write must touch.
    if len(util) > 1:
        gu_hi = np.percentile(util, 75)
        hi = util > gu_hi
        if hi.sum() >= 1:
            mem_hi = mem_util[hi].mean()          # DRAM during the compute bursts
            gpu_hi = util[hi].mean()
        else:                                     # flat window: fall back to means
            mem_hi, gpu_hi = mu_mean, gu_mean
    else:
        mem_hi, gpu_hi = mu_mean, gu_mean
    mu_p50 = np.percentile(mem_util, 50) if len(mem_util) > 1 else mu_mean
    feats["mem_util_at_high_util"] = mem_hi
    feats["mem_util_spike"] = mu_p95 - mu_p50
    feats["mem_util_burst_ratio"] = mem_hi / (gpu_hi + 1.0)
    feats["mem_util_fft_peak_power"] = fft_peak_power(mem_util)

    # ── 3. Autocorrelation at multiple lags (key for periodicity) ────────
    for lag in [1, 2, 5, 10, 20]:
        feats[f"acf{lag}_gpu_util"] = acf_lag(util, lag)
        feats[f"acf{lag}_power"] = acf_lag(power, lag)
        feats[f"acf{lag}_mem_used"] = acf_lag(mem_used, lag)

    # ── 4. TRAINING-SPECIFIC: Memory growth features ─────────────────────
    # Training shows memory growth during model/optimizer init, then plateau
    # Inference: flat memory. Other: varies but no ramp-then-plateau.
    if len(mem_used) >= 5:
        # Memory slope (linear fit)
        x_t = np.arange(len(mem_used), dtype=float)
        coeffs = np.polyfit(x_t, mem_used, 1)
        feats["mem_slope"] = float(coeffs[0])

        # Memory slope in first half vs second half (training: positive then flat)
        mid = len(mem_used) // 2
        if mid >= 3:
            slope_first = np.polyfit(np.arange(mid, dtype=float), mem_used[:mid], 1)[0]
            slope_second = np.polyfit(np.arange(len(mem_used) - mid, dtype=float), mem_used[mid:], 1)[0]
            feats["mem_slope_first_half"] = float(slope_first)
            feats["mem_slope_second_half"] = float(slope_second)
            feats["mem_slope_ratio"] = float(abs(slope_first) / (abs(slope_second) + 1e-6))
        else:
            feats["mem_slope_first_half"] = 0.0
            feats["mem_slope_second_half"] = 0.0
            feats["mem_slope_ratio"] = 0.0

        # Memory stability: std of second half (training should be stable after init)
        if mid >= 3:
            feats["mem_second_half_cv"] = float(mem_used[mid:].std() / (mem_used[mid:].mean() + 1e-9))
        else:
            feats["mem_second_half_cv"] = 0.0
    else:
        feats["mem_slope"] = 0.0
        feats["mem_slope_first_half"] = 0.0
        feats["mem_slope_second_half"] = 0.0
        feats["mem_slope_ratio"] = 0.0
        feats["mem_second_half_cv"] = 0.0

    # ── 5. TRAINING-SPECIFIC: Epoch periodicity via FFT ──────────────────
    feats["gpu_util_fft_period"] = fft_dominant_period(util)
    feats["gpu_util_fft_peak_power"] = fft_peak_power(util)
    feats["power_fft_period"] = fft_dominant_period(power)
    feats["power_fft_peak_power"] = fft_peak_power(power)
    feats["mem_used_fft_peak_power"] = fft_peak_power(mem_used)

    # ── 6. INFERENCE-SPECIFIC: Batch-pause pattern ───────────────────────
    # Inference has bursts of high util then drops; training is more sustained
    if len(util) >= 5:
        # Fraction of time at high utilization (>80%)
        feats["duty_cycle_80"] = float((util >= 80).mean())
        # Fraction of time at very low utilization (<10%) — idle gaps
        feats["idle_frac"] = float((util < 10).mean())
        # Number of transitions between high and low util per second
        high = util >= 50
        transitions = np.sum(np.abs(np.diff(high.astype(int))))
        feats["util_transitions_per_sec"] = float(transitions / (len(util) + 1e-9))
        # Burstiness: ratio of max to mean utilization
        feats["util_burstiness"] = float(util.max() / (util.mean() + 1e-9))
    else:
        feats["duty_cycle_80"] = 0.0
        feats["idle_frac"] = 0.0
        feats["util_transitions_per_sec"] = 0.0
        feats["util_burstiness"] = 0.0

    # ── 7. Power stability features ──────────────────────────────────────
    if len(power) >= 5:
        feats["power_cv"] = float(power.std() / (power.mean() + 1e-9))
        # Rolling variance of power (30-sample window or all if shorter)
        w = min(30, len(power))
        rolling_mean = pd.Series(power).rolling(window=w, min_periods=1).mean().values
        feats["power_rolling_var"] = float(rolling_mean.var())
        # Power range as fraction of mean
        feats["power_range_frac"] = float((power.max() - power.min()) / (power.mean() + 1e-9))
    else:
        feats["power_cv"] = 0.0
        feats["power_rolling_var"] = 0.0
        feats["power_range_frac"] = 0.0

    # ── 8. PCIe pattern features (training has periodic data loading) ────
    pcie_total = pcie_tx + pcie_rx
    if len(pcie_total) >= 5:
        feats["pcie_cv"] = float(pcie_total.std() / (pcie_total.mean() + 1e-9))
        feats["pcie_fft_peak_power"] = fft_peak_power(pcie_total)
        feats["pcie_nonzero_frac"] = float((pcie_total > 1.0).mean())
    else:
        feats["pcie_cv"] = 0.0
        feats["pcie_fft_peak_power"] = 0.0
        feats["pcie_nonzero_frac"] = 0.0

    # ── 9. Encoder/decoder (video encoding signature) ────────────────────
    for col in ("encoder_util_pct", "decoder_util_pct"):
        if col in chunk.columns:
            s = chunk[col].fillna(0).values.astype(float)
            feats[f"{col}_mean"] = float(s.mean())
            feats[f"{col}_nonzero_frac"] = float((s > 0).mean())
        else:
            feats[f"{col}_mean"] = 0.0
            feats[f"{col}_nonzero_frac"] = 0.0

    # ── 10. DCGM features if available ───────────────────────────────────
    if "dcgm_tensor_active" in chunk.columns:
        s = chunk["dcgm_tensor_active"].fillna(0).values.astype(float)
        feats["dcgm_tensor_active_mean"] = float(s.mean())
        feats["dcgm_tensor_active_max"] = float(s.max())
    if "dcgm_fp16_pipe_active" in chunk.columns and "dcgm_fp32_pipe_active" in chunk.columns:
        fp16 = chunk["dcgm_fp16_pipe_active"].fillna(0).mean()
        fp32 = chunk["dcgm_fp32_pipe_active"].fillna(0).mean()
        total = fp16 + fp32
        feats["dcgm_fp16_ratio"] = float(fp16 / total) if total > 0 else 0.0

    return feats


# ── Run-level features (pre-allocation detection) ───────────────────────────

def run_level_features(run_df: pd.DataFrame) -> dict:
    """Features that need the whole run, not just a window.

    Designed to catch the WB-O startup signature: Round 5 showed the oracle
    runs a dummy backward+step at init to materialize AdamW buffers (~6.3 GB
    of optimizer state) before trying to look like inference. No legitimate
    inference workload allocates optimizer state at startup, so a sharp
    first-30s memory jump is a strong pre-training tell.
    """
    feats = {}
    mem = run_df["mem_used_mb"].fillna(0).values.astype(float)
    mem_total = float(run_df["mem_total_mb"].iloc[0]) if "mem_total_mb" in run_df.columns else 1.0

    # Telemetry is ~1 Hz; first 30 samples ≈ first 30 s of the run.
    first_n = min(30, len(mem))
    if first_n >= 5 and len(mem) >= 5:
        mem_start = float(np.median(mem[:3]))
        mem_peak_first = float(np.max(mem[:first_n]))
        delta = mem_peak_first - mem_start
        feats["first_30s_mem_delta_mb"] = delta
        feats["first_30s_mem_delta_frac"] = delta / max(mem_total, 1.0)
        feats["first_30s_mem_ratio"] = mem_peak_first / max(mem_start, 1.0)
    else:
        feats["first_30s_mem_delta_mb"] = 0.0
        feats["first_30s_mem_delta_frac"] = 0.0
        feats["first_30s_mem_ratio"] = 1.0

    # Time to 90 % of plateau memory — slow ramp = true training init,
    # near-zero samples = pre-allocation trick.
    if len(mem) >= 10:
        plateau = float(np.median(mem[-min(30, len(mem)):]))
        mem0 = float(mem[0])
        span = plateau - mem0
        if abs(span) < 1e-6:
            feats["time_to_mem_plateau_samples"] = 0.0
        else:
            threshold = mem0 + 0.9 * span
            target = mem >= threshold if span > 0 else mem <= threshold
            hits = np.where(target)[0]
            feats["time_to_mem_plateau_samples"] = float(hits[0]) if len(hits) > 0 else float(len(mem))
    else:
        feats["time_to_mem_plateau_samples"] = 0.0

    return feats


# ── Sliding window extraction ────────────────────────────────────────────────

def sliding_windows(
    df: pd.DataFrame,
    window_sec: float,
    stride_sec: float,
    run_level_mode: str = "causal",
    step_log_map: dict | None = None,
) -> pd.DataFrame:
    """Extract sliding-window feature vectors per run.

    ``run_level_mode`` controls the whole-run "pre-allocation" features
    (``first_30s_mem_*``, ``time_to_mem_plateau_samples``):

      - ``"causal"`` – (DEFAULT) run-level features recomputed per window from
        run start up to that window's end only, so no window sees its own future
        while still exploiting the genuinely-observable startup signature. This
        is what a real-time monitor computes and is the reported/deployed
        configuration (see report/neurips_rebuttal.md §1).
      - ``"none"``   – run-level features omitted entirely. Leak-free floor.
      - ``"orig"``   – computed once over the entire run and attached to every
        window. The plateau term reads ``mem[-30:]`` (the end of the run), so
        early windows carry information from their own future. LEAKY: retained
        only to reproduce archived pre-fix numbers, never for training/serving.
    """
    if run_level_mode not in ("orig", "none", "causal"):
        raise ValueError(
            f"run_level_mode must be 'orig', 'none', or 'causal', got {run_level_mode!r}")

    results = []
    for run_id, run_df in df.groupby("run_id"):
        run_df = run_df.sort_values("ts").reset_index(drop=True)
        if len(run_df) < 5:
            continue

        t_start = run_df["ts"].iloc[0]
        t_end = run_df["ts"].iloc[-1]
        duration = t_end - t_start
        label = run_df["threeway_label"].iloc[0]
        wl_label = run_df["workload_label"].iloc[0]
        gpu_name = run_df["gpu_name"].iloc[0] if "gpu_name" in run_df.columns else ""
        run_feats_full = run_level_features(run_df) if run_level_mode == "orig" else None
        # Per-window ground truth (train step timestamps), if a STEP log exists
        # for this run. None => fall back to the run-level label downstream.
        run_train_ts = step_log_map.get(run_id, {}).get("train_ts") if step_log_map else None

        def emit(chunk, wstart, wend):
            if len(chunk) < 5:
                return
            feats = extract_enhanced_features(chunk)
            if run_level_mode == "orig":
                feats.update(run_feats_full)
            elif run_level_mode == "causal":
                # Only samples observable by the window's end — no look-ahead.
                feats.update(run_level_features(run_df[run_df["ts"] < wend]))
            feats["run_id"] = run_id
            feats["threeway_label"] = label
            feats["workload_label"] = wl_label
            feats["gpu_name"] = gpu_name
            feats["window_start"] = wstart
            feats["window_end"] = wend
            if step_log_map is not None:
                # 1.0 = a real training step occurred in this window, 0.0 = none,
                # NaN = no STEP log for this run (use the run-level label instead).
                if run_train_ts is not None:
                    active = bool(((run_train_ts >= wstart) & (run_train_ts < wend)).any())
                    feats["window_train_active"] = 1.0 if active else 0.0
                else:
                    feats["window_train_active"] = np.nan
            results.append(feats)

        if duration < window_sec:
            emit(run_df, t_start, t_end)
        else:
            wstart = t_start
            while wstart + window_sec <= t_end + 1:
                chunk = run_df[(run_df["ts"] >= wstart) & (run_df["ts"] < wstart + window_sec)]
                emit(chunk, wstart, wstart + window_sec)
                wstart += stride_sec

    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results)


# ── Classification ───────────────────────────────────────────────────────────

def workload_group(labels) -> np.ndarray:
    """Group key for held-out-by-WORKLOAD validation.

    Grouping CV by run_id leaks workload identity. There are ~2100 runs over
    ~170 distinct workloads, so every held-out run has siblings of the SAME
    workload in the training fold, and the score measures "recognise a workload
    I have already seen" rather than "generalise to one I have not". Measured
    consequence: run-grouped CV endorsed tau=0.10, which turns out to fire on
    >50% of windows of honest non-ML workloads held out by workload.

    The deployment question is whether an HPC job the detector has never seen
    trips it, so validation must hold out whole workloads.
    """
    s = pd.Series(labels).astype(str)
    s = s.str.replace(r"_node\d+_gpu\d+$", "", regex=True)
    # TPU per-chip rank token, e.g. `..._rep2_w0c0` (worker 0 chip 0; the worker
    # index is slice*1000 + host index on a multislice group). It sits AFTER the
    # rep token, so it must be stripped first or the `_rep\d+$` anchor below never
    # matches and each chip stream is treated as a separate workload.
    s = s.str.replace(r"_w\d+c\d+$", "", regex=True)
    s = s.str.replace(r"_rep\d+$", "", regex=True)
    return s.values


def get_feature_cols(df: pd.DataFrame) -> list:
    """Return all numeric feature columns (exclude metadata)."""
    exclude = {"run_id", "threeway_label", "workload_label", "gpu_name",
               "window_start", "window_end", "window_sec", "n_samples",
               "window_train_active"}   # per-window ground truth, never a feature
    return [c for c in df.columns if c not in exclude and df[c].dtype in (np.float64, np.float32, np.int64, np.int32, float, int)]


def compute_sample_weights(y, groups=None, beta=None, per_run=None):
    """Compute per-sample weights with class + (optional) per-run reweighting.

    Default behaviour (groups=None, beta=None) matches the original
    sklearn-"balanced" scheme exactly, so existing callers are unaffected.

    When `groups` is provided, two additional corrections kick in:

    1. **Effective-number-of-samples class weighting** (Cui et al. 2019).
       Replaces inverse frequency with `(1 - beta) / (1 - beta ** n_c)`.
       With beta close to 1 (default 0.9999) this flattens the diminishing
       returns of near-duplicate samples — useful when one class has many
       files from a single workload and another has fewer files from
       several workloads.

    2. **Per-run normalisation.** Each sample's weight is scaled by
       `1 / n_windows_in_its_run`, so a 300s run producing ~9 windows does
       not dominate a 30s run producing ~1 window. Activated by
       `per_run=True` (the default when groups is given).

    Final weights are renormalised so their mean is 1.0.

    Parameters
    ----------
    y : array-like — class labels (encoded integers or strings).
    groups : array-like, optional — run_id per sample. Required for per-run
        normalisation.
    beta : float, optional — effective-number-of-samples smoothing (in [0,1)).
        When None, falls back to the classic inverse-frequency formula.
    per_run : bool, optional — if True, divide by windows-per-run. Defaults
        to True when `groups` is given, False otherwise.

    Returns
    -------
    weights : np.ndarray of shape (len(y),), dtype float32, mean ≈ 1.
    """
    y_arr = np.asarray(y)
    n = len(y_arr)

    if per_run is None:
        per_run = groups is not None
    if per_run and groups is None:
        raise ValueError("per_run=True requires groups to be provided")

    # Step 1: per-run weight — each sample gets 1/n_samples_in_run, so every
    # run contributes a total weight of 1 (a 10-window run no longer
    # outweighs a 1-window run).
    if per_run:
        groups_arr = np.asarray(groups)
        if len(groups_arr) != n:
            raise ValueError("groups must have the same length as y")
        _, inverse_g = np.unique(groups_arr, return_inverse=True)
        counts_g = np.bincount(inverse_g)
        w_run = 1.0 / counts_g[inverse_g]
    else:
        w_run = np.ones(n, dtype=np.float64)

    # Step 2: class weights computed on *effective* class counts
    # (i.e., counts after per-run normalisation). This lets a class with
    # many runs-per-class still balance against one with few runs,
    # independent of how long any single run happens to be.
    classes, inverse_c = np.unique(y_arr, return_inverse=True)
    n_classes = len(classes)
    eff_counts = np.bincount(inverse_c, weights=w_run)

    if beta is None:
        # Classic inverse-frequency, on effective counts.
        # (Reduces to sklearn's "balanced" when per_run is False.)
        total_eff = eff_counts.sum()
        class_w = total_eff / (n_classes * eff_counts)
    else:
        if not (0.0 <= beta < 1.0):
            raise ValueError(f"beta must be in [0, 1); got {beta}")
        # Round effective counts for the Cui formula's integer-n assumption;
        # with beta=0.9999 this is not sensitive.
        eff_n_cui = (1.0 - np.power(beta, eff_counts)) / (1.0 - beta)
        raw = 1.0 / eff_n_cui
        class_w = raw / raw.mean()

    w = w_run * class_w[inverse_c]

    mean_w = w.mean()
    if mean_w > 0:
        w = w / mean_w
    return w.astype(np.float32)


# Keep balanced_subsample for backward compatibility but prefer compute_sample_weights
def balanced_subsample(X, y, groups=None, random_state=42):
    """Downsample to equal class counts, sampling by group (run_id).

    DEPRECATED: prefer compute_sample_weights() which retains all data.
    """
    rng = np.random.RandomState(random_state)
    classes = np.unique(y)

    if groups is None:
        class_indices = {cls: np.where(y == cls)[0] for cls in classes}
        target = min(len(idxs) for idxs in class_indices.values())
        selected = []
        for cls in classes:
            idxs = class_indices[cls]
            chosen = rng.choice(idxs, size=target, replace=False)
            selected.extend(chosen)
        selected = sorted(selected)
        return X[selected], y[selected]

    class_groups = {cls: [] for cls in classes}
    group_to_indices = {}
    for i, (g, c) in enumerate(zip(groups, y)):
        key = (g, c)
        if key not in group_to_indices:
            group_to_indices[key] = []
        group_to_indices[key].append(i)

    for (g, c), idxs in group_to_indices.items():
        class_groups[c].append((g, idxs))

    class_counts = {cls: sum(len(idxs) for _, idxs in glist)
                    for cls, glist in class_groups.items()}
    target = min(class_counts.values())

    selected = []
    for cls in classes:
        glist = class_groups[cls].copy()
        rng.shuffle(glist)
        count = 0
        for g, idxs in glist:
            if count >= target:
                break
            selected.extend(idxs)
            count += len(idxs)

    selected = sorted(selected)
    return X[selected], y[selected]


def train_and_evaluate():
    """Main training and evaluation pipeline."""
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import classification_report, accuracy_score, f1_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    import xgboost as xgb

    # Load data
    log.info("Loading all telemetry data...")
    df_raw = load_all_data()

    # Extract sliding windows at multiple sizes
    best_results = {}

    for win_sec, stride_sec in [(15, 7), (30, 15), (60, 30)]:
        log.info(f"\n{'='*60}")
        log.info(f"Window: {win_sec}s, stride: {stride_sec}s")
        log.info(f"{'='*60}")

        df_win = sliding_windows(df_raw, window_sec=win_sec, stride_sec=stride_sec,
                                 run_level_mode="causal")
        if df_win.empty:
            log.warning(f"No windows for {win_sec}s, skipping")
            continue

        feature_cols = get_feature_cols(df_win)
        X = df_win[feature_cols].fillna(0).values.astype(np.float32)
        y = df_win["threeway_label"].values
        groups = df_win["run_id"].values

        log.info(f"Windows: {len(df_win)} | Features: {len(feature_cols)}")
        log.info(f"Class distribution: {pd.Series(y).value_counts().to_dict()}")

        # Check if we have enough classes and samples
        unique_classes = np.unique(y)
        if len(unique_classes) < 3:
            log.warning(f"Only {len(unique_classes)} classes found, need 3. Skipping.")
            continue

        # Check minimum samples per class per run for cross-validation
        run_class_counts = df_win.groupby("run_id")["threeway_label"].first().value_counts()
        min_runs = run_class_counts.min()
        n_splits = min(5, min_runs)
        if n_splits < 2:
            log.warning(f"Not enough runs per class for cross-validation (min={min_runs})")
            # Fall back to single train/test split
            n_splits = 2

        log.info(f"Using {n_splits}-fold cross-validation")

        # Define classifiers
        classifiers = {
            "XGBoost": xgb.XGBClassifier(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                use_label_encoder=False, eval_metric="mlogloss",
                random_state=42, n_jobs=-1, verbosity=0,
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=400, max_depth=None, min_samples_leaf=2,
                max_features="sqrt", class_weight="balanced",
                random_state=42, n_jobs=-1,
            ),
            "GradientBoosting": GradientBoostingClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, random_state=42,
            ),
        }

        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

        for clf_name, clf in classifiers.items():
            accs, f1s = [], []
            all_y_true, all_y_pred = [], []

            for fold_idx, (tr_idx, te_idx) in enumerate(cv.split(X, y, groups)):
                X_tr, X_te = X[tr_idx], X[te_idx]
                y_tr, y_te = y[tr_idx], y[te_idx]

                # Handle XGBoost label encoding + class weights
                if clf_name == "XGBoost":
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    y_tr_enc = le.fit_transform(y_tr)
                    w_tr = compute_sample_weights(y_tr_enc)
                    clf.fit(X_tr, y_tr_enc, sample_weight=w_tr)
                    y_pred_enc = clf.predict(X_te)
                    y_pred = le.inverse_transform(y_pred_enc)
                else:
                    clf.fit(X_tr, y_tr)
                    y_pred = clf.predict(X_te)

                acc = accuracy_score(y_te, y_pred)
                f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)
                accs.append(acc)
                f1s.append(f1)
                all_y_true.extend(y_te)
                all_y_pred.extend(y_pred)

            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            mean_f1 = np.mean(f1s)

            log.info(f"  [{clf_name:20s}] acc={mean_acc:.4f}±{std_acc:.4f}  f1_macro={mean_f1:.4f}")

            # Print per-class report for best window
            if clf_name == "XGBoost":
                report = classification_report(all_y_true, all_y_pred, zero_division=0)
                log.info(f"\n{report}")

            key = f"{clf_name}_{win_sec}s"
            best_results[key] = {
                "classifier": clf_name,
                "window_sec": win_sec,
                "acc_mean": mean_acc,
                "acc_std": std_acc,
                "f1_macro": mean_f1,
            }

        # Feature importance for best classifier (XGBoost)
        if "XGBoost" in classifiers:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_enc = le.fit_transform(y)
            xgb_clf = classifiers["XGBoost"]
            xgb_clf.fit(X, y_enc)
            importances = xgb_clf.feature_importances_
            top_idx = np.argsort(importances)[::-1][:25]
            log.info(f"\nTop-25 features (XGBoost, {win_sec}s window):")
            for i in top_idx:
                log.info(f"  {feature_cols[i]:45s} {importances[i]:.4f}")

    # Summary
    log.info("\n" + "=" * 60)
    log.info("SUMMARY OF ALL RESULTS")
    log.info("=" * 60)
    for key, res in sorted(best_results.items(), key=lambda x: -x[1]["acc_mean"]):
        log.info(f"  {key:35s}  acc={res['acc_mean']:.4f}±{res['acc_std']:.4f}  f1={res['f1_macro']:.4f}")

    # Save results
    results_dir = PROJECT_ROOT / "classifier" / "results"
    results_dir.mkdir(exist_ok=True)
    pd.DataFrame(best_results.values()).to_csv(results_dir / "threeway_results.csv", index=False)
    log.info(f"\nResults saved to {results_dir / 'threeway_results.csv'}")


if __name__ == "__main__":
    train_and_evaluate()
