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
    # Generic training labels seen in datacenter runs
    "adversarial_training",
    "adversarial_training_2gpu_gpu0", "adversarial_training_2gpu_gpu1",
    "bert_finetune",
    "bert_finetune_2gpu_gpu0", "bert_finetune_2gpu_gpu1",
    "gpt2_finetune",
    "gpt2_finetune_2gpu_gpu0", "gpt2_finetune_2gpu_gpu1",
    "pytorch_training",
    "pytorch_training_2gpu_gpu0", "pytorch_training_2gpu_gpu1",
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
    # Round 5b: new large-LLM inference anchors (single-GPU, open-license)
    "llm_infer_phi3_mini", "llm_infer_qwen25_1p5b", "llm_infer_qwen25_3b",
    "llm_infer_smollm2_1p7b", "llm_infer_mistral_7b_q4",
    "llm_infer_tinyllama_bs8", "llm_infer_tinyllama_seq1024",
    "llm_infer_qwen25_3b_bs4", "llm_infer_qwen25_1p5b_streaming",
    "llm_infer_sdxl", "llm_infer_clip_embed_batched",
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
    # Heuristic fallback
    lc = label.lower()
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

    return df


def load_all_data() -> pd.DataFrame:
    """Load telemetry from all data sources."""
    frames = []

    # 1. Main data/ directory
    for p in sorted((PROJECT_ROOT / "data").rglob("*.parquet")):
        try:
            df = pd.read_parquet(p)
            df = _normalize_columns(df)
            if "workload_label" not in df.columns:
                # Parse from filename
                stem = p.stem
                parts = stem.split("_NVIDIA_")[0] if "_NVIDIA_" in stem else stem.rsplit("_", 3)[0]
                df["workload_label"] = parts
            if "run_id" not in df.columns:
                df["run_id"] = p.stem
            frames.append(df)
        except Exception as e:
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

    # Deduplicate: some files may appear in multiple directories
    if "ts" in combined.columns and "run_id" in combined.columns:
        combined = combined.drop_duplicates(subset=["ts", "run_id"], keep="first")
        combined = combined.sort_values(["run_id", "ts"]).reset_index(drop=True)

    # Add three-way labels
    combined["threeway_label"] = combined["workload_label"].apply(threeway_label)

    label_dist = combined.groupby("threeway_label")["run_id"].nunique()
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
    for sig in RAW_SIGNALS:
        vals = chunk[sig].fillna(0).values.astype(float) if sig in chunk.columns else np.zeros(max(len(chunk), 1))
        n = len(vals)
        mu = vals.mean()
        sd = vals.std()
        vmin, vmax = vals.min(), vals.max()
        p25, p50, p75, p95 = np.percentile(vals, [25, 50, 75, 95]) if n > 1 else (mu,) * 4

        feats[f"{sig}_mean"] = mu
        feats[f"{sig}_std"] = sd
        feats[f"{sig}_min"] = vmin
        feats[f"{sig}_max"] = vmax
        feats[f"{sig}_p25"] = p25
        feats[f"{sig}_p50"] = p50
        feats[f"{sig}_p75"] = p75
        feats[f"{sig}_p95"] = p95
        feats[f"{sig}_iqr"] = p75 - p25
        feats[f"{sig}_range"] = vmax - vmin
        feats[f"{sig}_cv"] = sd / (abs(mu) + 1e-9)
        feats[f"{sig}_skew"] = float(sp_stats.skew(vals)) if n > 2 else 0.0
        feats[f"{sig}_kurt"] = float(sp_stats.kurtosis(vals)) if n > 3 else 0.0

    # ── 2. Cross-signal derived features (5) ─────────────────────────────
    util = chunk["gpu_utilization_pct"].fillna(0).values.astype(float)
    power = chunk["power_draw_w"].fillna(0).values.astype(float)
    pcie_tx = chunk["pcie_tx_mbps"].fillna(0).values.astype(float)
    pcie_rx = chunk["pcie_rx_mbps"].fillna(0).values.astype(float)
    sm_clk = chunk["sm_clock_mhz"].fillna(0).values.astype(float)
    mem_used = chunk["mem_used_mb"].fillna(0).values.astype(float)

    feats["power_per_util"] = power.mean() / (util.mean() + 1.0)
    feats["pcie_total_mean"] = (pcie_tx + pcie_rx).mean()
    feats["util_per_sm_pct"] = util.mean() / (sm_clk.mean() + 1.0) * 1000.0

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


# ── Sliding window extraction ────────────────────────────────────────────────

def sliding_windows(df: pd.DataFrame, window_sec: float, stride_sec: float) -> pd.DataFrame:
    """Extract sliding-window feature vectors per run."""
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

        def emit(chunk):
            if len(chunk) < 5:
                return
            feats = extract_enhanced_features(chunk)
            feats["run_id"] = run_id
            feats["threeway_label"] = label
            feats["workload_label"] = wl_label
            results.append(feats)

        if duration < window_sec:
            emit(run_df)
        else:
            wstart = t_start
            while wstart + window_sec <= t_end + 1:
                chunk = run_df[(run_df["ts"] >= wstart) & (run_df["ts"] < wstart + window_sec)]
                emit(chunk)
                wstart += stride_sec

    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results)


# ── Classification ───────────────────────────────────────────────────────────

def get_feature_cols(df: pd.DataFrame) -> list:
    """Return all numeric feature columns (exclude metadata)."""
    exclude = {"run_id", "threeway_label", "workload_label", "window_start", "window_sec", "n_samples"}
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

        df_win = sliding_windows(df_raw, window_sec=win_sec, stride_sec=stride_sec)
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
