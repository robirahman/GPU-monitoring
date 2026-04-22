"""
Energy-Based Theoretical Detection Bound + Wall-Power Correlation
==================================================================

Computes the minimum detectable training fraction based on the physics
of GPU computation:
- Training requires ~3x the FLOPS of inference (forward + backward + update)
- Energy consumption is proportional to FLOPS on the same hardware
- The backward pass stores activations, increasing memory

Also validates NVML power_draw_w against Tapo P115 wall-power measurements:
- GPU power vs wall power correlation (R²)
- System overhead (PSU + CPU + RAM)
- Wall-power energy profiles by workload category

This script:
1. Measures energy-per-sample for inference vs training from RTX 5080 data
2. Computes the theoretical minimum detectable training fraction
3. Compares with empirical detection from the dilution sweep (Strategy L)
4. Validates GPU power against wall power from Tapo P115 smart plug

Usage:
    python -m classifier.energy_bound
"""

import glob
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "classifier" / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_wall_power_runs(label_pattern: str) -> list[pd.DataFrame]:
    """Load wall-power parquets matching a label pattern."""
    files = sorted(glob.glob(str(PROJECT_ROOT / "data" / "wall_power" / f"{label_pattern}*5080*_wall_power.parquet")))
    runs = []
    for f in files:
        df = pd.read_parquet(f)
        runs.append(df)
    return runs


def load_matched_pairs(label_pattern: str) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Load matched GPU + wall-power parquet pairs by run_id."""
    gpu_files = sorted(glob.glob(str(PROJECT_ROOT / "data" / f"{label_pattern}*5080*.parquet")))
    wall_dir = PROJECT_ROOT / "data" / "wall_power"
    pairs = []
    for gf in gpu_files:
        gdf = pd.read_parquet(gf)
        if "run_id" not in gdf.columns:
            continue
        run_id = gdf["run_id"].iloc[0]
        # Find matching wall-power file
        wall_files = list(wall_dir.glob(f"{label_pattern}*{run_id[:8]}*_wall_power.parquet"))
        if wall_files:
            wdf = pd.read_parquet(wall_files[0])
            pairs.append((gdf, wdf))
    return pairs


def discover_all_wall_power_pairs() -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """Discover every (label, gpu_df, wall_df) triple by scanning data/wall_power/.

    Each wall-power parquet contains `workload_label` and `run_id` columns;
    we match the GPU parquet that shares the same run_id (first 8 hex chars).
    This replaces the earlier hard-coded label dictionary and captures every
    Tapo P115-instrumented run in the corpus.
    """
    wall_dir = PROJECT_ROOT / "data" / "wall_power"
    data_dir = PROJECT_ROOT / "data"
    triples = []
    for wf in sorted(wall_dir.glob("*_wall_power.parquet")):
        wdf = pd.read_parquet(wf)
        if "run_id" not in wdf.columns or "workload_label" not in wdf.columns:
            continue
        if wdf.empty:
            continue
        run_id = str(wdf["run_id"].iloc[0])
        label = str(wdf["workload_label"].iloc[0])
        # Find GPU parquet with the same run_id prefix (exclude the wall_power file itself).
        candidates = [
            p for p in data_dir.glob(f"*{run_id[:8]}*.parquet")
            if "wall_power" not in p.name
        ]
        if not candidates:
            continue
        gdf = pd.read_parquet(candidates[0])
        triples.append((label, gdf, wdf))
    return triples


def load_rtx5080_runs(label_pattern: str) -> list[pd.DataFrame]:
    """Load all RTX 5080 parquet files matching a label pattern."""
    files = sorted(glob.glob(str(PROJECT_ROOT / "data" / f"{label_pattern}*5080*.parquet")))
    runs = []
    for f in files:
        df = pd.read_parquet(f)
        if "timestamp_epoch" not in df.columns and "ts" in df.columns:
            df["timestamp_epoch"] = df["ts"]
        runs.append(df)
    return runs


def compute_energy_stats(runs: list[pd.DataFrame], label: str) -> dict:
    """Compute energy, power, memory stats from a set of runs."""
    all_power = []
    all_mem = []
    all_util = []
    total_energy_j = 0
    total_duration = 0

    for df in runs:
        power = df["power_draw_w"].values
        duration = df["timestamp_epoch"].iloc[-1] - df["timestamp_epoch"].iloc[0]
        energy = power.sum()  # Approximate: 1 Hz sampling → power(W) ≈ energy(J) per sample
        total_energy_j += energy
        total_duration += duration
        all_power.extend(power)
        all_mem.extend(df["mem_used_mb"].values)
        all_util.extend(df["gpu_utilization_pct"].values)

    return {
        "label": label,
        "n_runs": len(runs),
        "total_duration_s": total_duration,
        "power_mean": np.mean(all_power),
        "power_std": np.std(all_power),
        "power_p5": np.percentile(all_power, 5),
        "power_p95": np.percentile(all_power, 95),
        "mem_mean": np.mean(all_mem),
        "mem_std": np.std(all_mem),
        "util_mean": np.mean(all_util),
        "util_std": np.std(all_util),
        "energy_per_second": total_energy_j / max(total_duration, 1),
    }


def main():
    log.info("=" * 70)
    log.info("ENERGY-BASED THEORETICAL DETECTION BOUND")
    log.info("=" * 70)

    # ── Measure energy profiles ──────────────────────────────────────────
    workloads = {
        "inference": "resnet50_inference",
        "training": "pytorch_resnet_cifar10",
        "K_online_learning": "adversarial_K_online_learning",
        "L_diluted_2": "adversarial_L_diluted_2",
        "L_diluted_5": "adversarial_L_diluted_5",
        "L_diluted_10": "adversarial_L_diluted_10",
        "L_diluted_20": "adversarial_L_diluted_20",
        "idle": "idle",
        "mining": "mining_ethash_proxy",
        "cufft": "cufft_benchmark",
    }

    stats = {}
    for name, pattern in workloads.items():
        runs = load_rtx5080_runs(pattern)
        if runs:
            stats[name] = compute_energy_stats(runs, name)

    if "inference" not in stats or "training" not in stats:
        log.error("Need both inference and training data on RTX 5080")
        return

    log.info("\n--- Energy Profiles (RTX 5080) ---")
    log.info(f"{'Workload':25s} {'Power (W)':>12s} {'Memory (MB)':>12s} {'Util (%)':>10s}")
    log.info("-" * 65)
    for name, s in sorted(stats.items()):
        log.info(f"{name:25s} {s['power_mean']:8.1f}±{s['power_std']:4.1f}"
                 f" {s['mem_mean']:8.0f}±{s['mem_std']:4.0f}"
                 f" {s['util_mean']:6.1f}±{s['util_std']:4.1f}")

    # ── Compute energy ratio ─────────────────────────────────────────────
    E_inf = stats["inference"]["power_mean"]
    E_train = stats["training"]["power_mean"]
    ratio = E_train / E_inf if E_inf > 0 else float("inf")

    log.info(f"\n--- Energy Ratio ---")
    log.info(f"  Inference power:  {E_inf:.1f} W")
    log.info(f"  Training power:   {E_train:.1f} W")
    log.info(f"  Ratio (train/inf): {ratio:.2f}x")
    log.info(f"  Backward pass overhead: {ratio - 1:.2f}x ({(ratio-1)*100:.0f}% extra)")

    # ── Memory ratio ─────────────────────────────────────────────────────
    M_inf = stats["inference"]["mem_mean"]
    M_train = stats["training"]["mem_mean"]
    mem_ratio = M_train / M_inf if M_inf > 0 else float("inf")
    log.info(f"\n--- Memory Ratio ---")
    log.info(f"  Inference memory:  {M_inf:.0f} MB")
    log.info(f"  Training memory:   {M_train:.0f} MB")
    log.info(f"  Ratio (train/inf): {mem_ratio:.2f}x")

    if "K_online_learning" in stats:
        M_K = stats["K_online_learning"]["mem_mean"]
        log.info(f"  Strategy K memory: {M_K:.0f} MB (ratio vs inf: {M_K/M_inf:.2f}x)")

    # ── Theoretical minimum detectable training fraction ─────────────────
    log.info(f"\n--- Theoretical Detection Bound (Power-Based) ---")

    # The noise floor of the power signal
    sigma_inf = stats["inference"]["power_std"]
    # In a 60s window with 1 Hz sampling, we have N=60 samples
    # Standard error of the mean = sigma / sqrt(N)
    N_window = 60
    se = sigma_inf / np.sqrt(N_window)

    # Mixed power: P_mix = (1-f)*P_inf + f*P_train
    # Detectable when: P_mix - P_inf > z * se (z=1.96 for 95% confidence)
    # → f * (P_train - P_inf) > z * se
    # → f > z * se / (P_train - P_inf)
    delta_P = E_train - E_inf
    if delta_P > 0:
        for z, conf in [(1.96, "95%"), (2.58, "99%"), (1.0, "84%")]:
            f_min = z * se / delta_P
            log.info(f"  At {conf} confidence: min detectable training fraction f = {f_min:.4f} "
                     f"({f_min*100:.2f}%)")
            log.info(f"    → Throughput loss: {f_min*100:.2f}% of samples must be training")
            log.info(f"    → Dilution ratio N = {1/f_min:.1f} (train 1/{1/f_min:.0f} batches)")
    else:
        log.info("  Training power <= inference power — power-based bound not applicable")

    # ── Memory-based bound (harder to evade) ─────────────────────────────
    log.info(f"\n--- Theoretical Detection Bound (Memory-Based) ---")
    sigma_mem = stats["inference"]["mem_std"]
    se_mem = sigma_mem / np.sqrt(N_window)
    delta_M = M_train - M_inf

    if delta_M > 0:
        f_min_mem = 1.96 * se_mem / delta_M
        log.info(f"  Memory delta: {delta_M:.0f} MB (training - inference)")
        log.info(f"  Memory noise SE: {se_mem:.1f} MB")
        log.info(f"  Min detectable fraction (95%): f = {f_min_mem:.6f}")
        log.info(f"  Memory is much more discriminative than power")
    else:
        log.info(f"  Training memory ({M_train:.0f}) <= inference memory ({M_inf:.0f})")
        log.info(f"  Note: this may be due to different batch sizes")

    # ── Empirical dilution curve ─────────────────────────────────────────
    log.info(f"\n--- Empirical Dilution Curve (Strategy L) ---")
    dilution_data = []
    for d in [2, 5, 10, 20]:
        name = f"L_diluted_{d}"
        if name in stats:
            s = stats[name]
            dilution_data.append({
                "dilution_N": d,
                "train_fraction": 1.0 / d,
                "power_mean": s["power_mean"],
                "mem_mean": s["mem_mean"],
                "util_mean": s["util_mean"],
                "power_delta_from_inf": s["power_mean"] - E_inf,
                "mem_delta_from_inf": s["mem_mean"] - M_inf,
            })
            log.info(f"  N={d:2d} (train {1/d:5.1%}): "
                     f"power={s['power_mean']:.1f}W (Δ={s['power_mean']-E_inf:+.1f}), "
                     f"mem={s['mem_mean']:.0f}MB (Δ={s['mem_mean']-M_inf:+.0f})")

    if dilution_data:
        df_dilution = pd.DataFrame(dilution_data)
        df_dilution.to_csv(RESULTS_DIR / "dilution_energy_curve.csv", index=False)
        log.info(f"\n  Saved: {RESULTS_DIR / 'dilution_energy_curve.csv'}")

    # ── Save full energy stats ───────────────────────────────────────────
    df_stats = pd.DataFrame(list(stats.values()))
    df_stats.to_csv(RESULTS_DIR / "energy_profiles_5080.csv", index=False)
    log.info(f"  Saved: {RESULTS_DIR / 'energy_profiles_5080.csv'}")

    # ── Wall-Power Correlation Analysis ─────────────────────────────────
    log.info(f"\n{'='*70}")
    log.info("WALL-POWER CORRELATION ANALYSIS (Tapo P115)")
    log.info(f"{'='*70}")

    triples = discover_all_wall_power_pairs()
    log.info(f"  Discovered {len(triples)} (label, gpu, wall) triples in data/wall_power/")

    # Aggregate paired seconds by workload label. Multiple runs for the same
    # label accumulate into one summary row; if only one run exists per label,
    # the aggregate is just that run's paired samples.
    wl_gpu_by_label: dict[str, list[float]] = {}
    wl_wall_by_label: dict[str, list[float]] = {}
    wl_runs_by_label: dict[str, int] = {}

    for label, gdf, wdf in triples:
        gdf = gdf.copy()
        wdf = wdf.copy()
        gdf["ts_round"] = gdf["timestamp_epoch"].round(0)
        wdf["ts_round"] = wdf["timestamp_epoch"].round(0)

        merged = pd.merge(gdf[["ts_round", "power_draw_w"]],
                          wdf[["ts_round", "wall_power_w"]],
                          on="ts_round", how="inner")

        if len(merged) < 10:
            continue

        wl_gpu_by_label.setdefault(label, []).extend(merged["power_draw_w"].values)
        wl_wall_by_label.setdefault(label, []).extend(merged["wall_power_w"].values)
        wl_runs_by_label[label] = wl_runs_by_label.get(label, 0) + 1

    all_gpu_power = []
    all_wall_power = []
    per_workload_summary = []

    for label in sorted(wl_gpu_by_label):
        wl_gpu = wl_gpu_by_label[label]
        wl_wall = wl_wall_by_label[label]
        if not wl_gpu:
            continue

        gpu_mean = float(np.mean(wl_gpu))
        wall_mean = float(np.mean(wl_wall))
        overhead = wall_mean - gpu_mean

        all_gpu_power.extend(wl_gpu)
        all_wall_power.extend(wl_wall)

        per_workload_summary.append({
            "workload": label,
            "n_runs": wl_runs_by_label[label],
            "gpu_power_mean_w": gpu_mean,
            "wall_power_mean_w": wall_mean,
            "overhead_w": overhead,
            "n_samples": len(wl_gpu),
        })

    if not per_workload_summary:
        log.info("  No matched GPU + wall-power pairs found. Skipping.")
    else:
        gpu_arr = np.array(all_gpu_power)
        wall_arr = np.array(all_wall_power)

        # Overall correlation
        corr = np.corrcoef(gpu_arr, wall_arr)[0, 1]
        r_squared = corr ** 2

        # Linear fit: wall = slope * gpu + intercept
        slope, intercept = np.polyfit(gpu_arr, wall_arr, 1)
        residuals = wall_arr - (slope * gpu_arr + intercept)
        rmse = np.sqrt(np.mean(residuals ** 2))

        log.info(f"\n--- Sample-Level Correlation ({len(gpu_arr)} paired samples) ---")
        log.info(f"  Pearson r:     {corr:.4f}")
        log.info(f"  R²:            {r_squared:.4f}")
        log.info(f"  Linear fit:    wall_power = {slope:.3f} * gpu_power + {intercept:.1f}")
        log.info(f"  RMSE:          {rmse:.1f} W")
        log.info(f"  Interpretation: slope={slope:.3f} accounts for PSU efficiency (~{1/slope:.0%})")

        # Per-workload summary
        df_summary = pd.DataFrame(per_workload_summary)
        log.info(f"\n--- Per-Workload Power Summary ---")
        log.info(f"{'Workload':35s} {'GPU (W)':>10s} {'Wall (W)':>10s} {'Overhead':>10s}")
        log.info("-" * 70)

        # Categorize workloads
        categories = {
            "idle": "Idle",
            "resnet50_inference": "Inference",
            "cufft_benchmark": "HPC",
            "nbody_sim": "HPC",
            "mining_ethash_proxy": "Mining",
            "rendering_proxy": "Rendering",
        }
        training_prefixes = ["pytorch_", "gpt2_", "bert_"]
        adversarial_prefixes = ["adversarial_"]

        for _, row in df_summary.sort_values("gpu_power_mean_w").iterrows():
            wl = row["workload"]
            cat = categories.get(wl)
            if cat is None:
                if any(wl.startswith(p) for p in training_prefixes):
                    cat = "Training"
                elif any(wl.startswith(p) for p in adversarial_prefixes):
                    cat = "Adversarial"
                else:
                    cat = "Other"
            log.info(f"  {wl:35s} {row['gpu_power_mean_w']:8.1f}  {row['wall_power_mean_w']:8.1f}  "
                     f"{row['overhead_w']:+8.1f}  [{cat}]")

        # System overhead analysis
        overheads = df_summary["overhead_w"].values
        log.info(f"\n--- System Overhead (Wall - GPU) ---")
        log.info(f"  Mean:   {np.mean(overheads):.1f} W")
        log.info(f"  Std:    {np.std(overheads):.1f} W")
        log.info(f"  Min:    {np.min(overheads):.1f} W (at lowest GPU load)")
        log.info(f"  Max:    {np.max(overheads):.1f} W (at highest GPU load)")
        log.info(f"  This overhead = CPU + RAM + PSU losses + motherboard + fans")

        # Re-compute energy bound using wall power
        wall_stats = {}
        for name, pattern in {"inference": "resnet50_inference", "training": "pytorch_resnet_cifar10"}.items():
            sub = df_summary[df_summary["workload"] == pattern]
            if not sub.empty:
                wall_stats[name] = sub.iloc[0]["wall_power_mean_w"]

        if "inference" in wall_stats and "training" in wall_stats:
            W_inf = wall_stats["inference"]
            W_train = wall_stats["training"]
            wall_ratio = W_train / W_inf

            log.info(f"\n--- Wall-Power Energy Bound ---")
            log.info(f"  Inference wall power:  {W_inf:.1f} W")
            log.info(f"  Training wall power:   {W_train:.1f} W")
            log.info(f"  Wall ratio (train/inf): {wall_ratio:.2f}x")
            log.info(f"  (GPU-only ratio was {ratio:.2f}x)")

            # Wall power noise for inference
            inf_pairs = load_matched_pairs("resnet50_inference")
            if inf_pairs:
                wall_samples = []
                for _, wdf in inf_pairs:
                    wall_samples.extend(wdf["wall_power_w"].values)
                sigma_wall = np.std(wall_samples)
                se_wall = sigma_wall / np.sqrt(N_window)
                delta_W = W_train - W_inf
                if delta_W > 0:
                    f_min_wall = 1.96 * se_wall / delta_W
                    log.info(f"  Wall power noise (σ):  {sigma_wall:.1f} W")
                    log.info(f"  Wall power SE (60s):   {se_wall:.1f} W")
                    log.info(f"  Min detectable f (95%): {f_min_wall:.4f} ({f_min_wall*100:.2f}%)")
                    log.info(f"  → Dilution ratio N = {1/f_min_wall:.1f}")

        # Save results
        df_summary.to_csv(RESULTS_DIR / "wall_power_correlation.csv", index=False)
        log.info(f"\n  Saved: {RESULTS_DIR / 'wall_power_correlation.csv'}")

        # Save correlation stats
        corr_stats = {
            "pearson_r": corr,
            "r_squared": r_squared,
            "slope": slope,
            "intercept": intercept,
            "rmse": rmse,
            "n_paired_samples": len(gpu_arr),
            "n_workloads": len(df_summary),
            "overhead_mean_w": np.mean(overheads),
            "overhead_std_w": np.std(overheads),
        }
        pd.DataFrame([corr_stats]).to_csv(RESULTS_DIR / "wall_power_fit.csv", index=False)
        log.info(f"  Saved: {RESULTS_DIR / 'wall_power_fit.csv'}")

    log.info(f"\n{'='*70}")
    log.info("ENERGY BOUND ANALYSIS COMPLETE")
    log.info(f"{'='*70}")


if __name__ == "__main__":
    main()
