"""Figure 5: 4-panel telemetry traces (training / inference / evasion / mining)."""
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "report" / "figures"
DATA = ROOT / "data"

PANELS = [
    ("Training — GPT-2 on WikiText-2",
     "gpt2_wikitext2_NVIDIA_GeForce_RTX_5080_b52ba3de_20260410_114004.parquet",
     "#1f77b4"),
    ("Inference — Qwen2.5-1.5B",
     "llm_infer_qwen25_1p5b_NVIDIA_GeForce_RTX_5080_2fd67231_20260417_212729.parquet",
     "#2ca02c"),
    ("Evasion — training disguised as mining (H_mimicry)",
     "adversarial_H_mimicry_mining_NVIDIA_GeForce_RTX_5080_0500acd3_20260411_115809.parquet",
     "#d62728"),
    ("Non-ML — ethash mining",
     "mining_ethash_proxy_NVIDIA_GeForce_RTX_5080_211503e3_20260410_130851.parquet",
     "#7f7f7f"),
]

MAX_SECONDS = 120  # truncate all panels to the first 120s for readability

fig, axes = plt.subplots(4, 1, figsize=(7.5, 8.0), sharex=True)
for ax, (title, fname, color) in zip(axes, PANELS):
    df = pd.read_parquet(DATA / fname).reset_index(drop=True)
    if "ts" in df.columns:
        t = df["ts"].values.astype(float)
        t = t - t[0]
    else:
        t = np.arange(len(df), dtype=float)
    mask = t <= MAX_SECONDS
    t = t[mask]
    util = df["gpu_utilization_pct"].values[mask]
    power = df["power_draw_w"].values[mask]
    mem = df["mem_used_mb"].values[mask] / 1024  # to GB

    # Left axis: util (%) and power (W/3 scale)
    ax.plot(t, util, color=color, linewidth=1.5, label="GPU util (%)")
    ax.plot(t, power, color=color, linewidth=1.2, linestyle="--",
            alpha=0.7, label="Power (W)")
    ax.set_ylabel("util (%) / power (W)", fontsize=9)
    ax.set_ylim(0, max(110, power.max() * 1.1))
    ax.set_title(title, fontsize=10, loc="left")
    ax.grid(alpha=0.3)

    # Right axis: memory (GB)
    axR = ax.twinx()
    axR.fill_between(t, 0, mem, color="tab:orange", alpha=0.15, label="Memory (GB)")
    axR.plot(t, mem, color="tab:orange", linewidth=1.0, alpha=0.7)
    axR.set_ylabel("memory (GB)", color="tab:orange", fontsize=9)
    axR.tick_params(axis="y", colors="tab:orange")
    axR.set_ylim(0, max(16, mem.max() * 1.2))

    # Legend once, in the first panel
    if ax is axes[0]:
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = axR.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8, ncol=3)

axes[-1].set_xlabel("Time (s)")
fig.suptitle("NVML telemetry traces across workload types (RTX 5080, 1 Hz)",
             fontsize=11, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.98))
fig.savefig(FIG / "telemetry_traces_4panel.png", dpi=150)
fig.savefig(FIG / "telemetry_traces_4panel.pdf")
log.info(f"Saved telemetry_traces_4panel.png/.pdf")
