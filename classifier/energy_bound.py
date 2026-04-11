"""
Energy-Based Theoretical Detection Bound
=========================================

Computes the minimum detectable training fraction based on the physics
of GPU computation:
- Training requires ~3x the FLOPS of inference (forward + backward + update)
- Energy consumption is proportional to FLOPS on the same hardware
- The backward pass stores activations, increasing memory

This script:
1. Measures energy-per-sample for inference vs training from RTX 5080 data
2. Computes the theoretical minimum detectable training fraction
3. Compares with empirical detection from the dilution sweep (Strategy L)

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

    log.info(f"\n{'='*70}")
    log.info("ENERGY BOUND ANALYSIS COMPLETE")
    log.info(f"{'='*70}")


if __name__ == "__main__":
    main()
