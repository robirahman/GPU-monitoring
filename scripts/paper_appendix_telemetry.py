"""Appendix figure: raw NVML telemetry for a selected cross-section of the corpus.

Twelve 1 Hz traces on a common time axis, chosen to carry an argument rather
than to sample the catalogue evenly:

  row 1  the three classes the classifier must separate, undisguised
  row 2  Round 1-2 evasion, where the disguise is visible to the eye
  row 3  Round 3-6 evasion, where it is not

The last two panels are the point of the figure. symgemm_L4_sgdfused is real
training with the optimiser step folded into a matmul ladder; symgemm_L1_fwd3
is its FLOP-matched forward-only control, which contains no training at all.
At the same matrix size their utilisation and power traces differ by well under
a point, and only the memory channel separates them.

Each panel is annotated with that family's baseline detection rate from Table 7
(XGBoost, job-level holdout), so the trace and the number sit together.

Window length: WINDOW_S seconds from run start. See --window. 120 s is the
default because it is the longest span that fits inside essentially every run
in the corpus (O_composite_idle_pad runs are exactly 120 s; the short
supervised-training cells are 30-75 s) while still clearing the median
memory-allocation transient of ~19 s with ~100 s of steady state behind it.

Usage:
  python3 scripts/paper_appendix_telemetry.py --data-root data \
      --out report/figures/telemetry_appendix/telemetry_catalog
"""
from __future__ import annotations
import argparse, glob, os, re, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

UC, PC, MC = "gpu_utilization_pct", "power_draw_w", "mem_used_mb"
PRE_S = 8        # seconds of lead-in kept before activity onset
MIN_POST_S = 40  # shortest acceptable post-onset trace

# (label glob, panel title, class tag, Table 7 baseline detection, colour)
PANELS = [
    # (regex, GPU substring|None, title, class tag, Table 7 baseline det., colour, mem-axis group)
    # Pinned to the RTX 5080. The March-2026 A100-SXM4-40GB gpt2 runs report
    # ~41 W at 84% utilisation, which is not credible for a loaded A100 and is
    # not representative: other A100-40GB runs in the corpus reach 391 W.
    (r"^gpt2_wikitext2_NVIDIA",            "RTX_5080", "GPT-2 / WikiText-2",         "training",  None, "#1f77b4", "a"),
    (r"^fsdp_70b_nvlink_gpu0_",            "H200",     "FSDP 70B (8xH200, gpu0)",    "training",  None, "#1f77b4", "b"),
    (r"^llm_infer_qwen25_7b_fp16_",        None,       "Qwen2.5-7B inference",       "inference", None, "#2ca02c", "c"),
    (r"^mining_ethash_proxy_",             "B200",     "Ethash mining",              "non-ML",    None, "#7f7f7f", "mim"),

    (r"^adversarial_A_util_modulation_",   None,       "A: utilisation modulation",  "evasion", 30.5, "#d62728", "e"),
    (r"^adversarial_B_low_util_",          None,       "B: low-utilisation",         "evasion", 30.0, "#d62728", "f"),
    (r"^adversarial_H_mimicry_mining_",    "B200",     "H: mimicry of mining",       "evasion", 57.5, "#d62728", "mim"),
    (r"^adversarial_F_interleave_50_",     None,       "F: interleave (10 s cycle)", "evasion", 14.7, "#d62728", "g"),

    (r"^adversarial_K_online_learning_",   None,       "K: online learning",         "evasion", 88.8, "#ff7f0e", "h"),
    (r"^adversarial_L_diluted_20_",        None,       "L: diluted 1-in-20",         "evasion", 98.7, "#ff7f0e", "i"),
    (r"^symgemm_L4_sgdfused_m8192_",       "A100",     "symgemm L4: fused SGD",      "evasion", 12.0, "#d62728", "sym"),
    (r"^symgemm_L1_fwd3_m8192_",           "A100",     "symgemm L1: fwd-only CTRL",  "control",  0.0, "#7f7f7f", "sym"),
]


def onset(d, want):
    """Index where the run actually starts doing work.

    Aligning every panel on t=0 wastes half the axis on process start-up and
    makes the comparison unfair: the GPT-2 run idles for 47 s while the model
    and dataset load, whereas the evasion runs are busy by t=5 s. Onset is the
    first sample where memory has reached 95% of its run maximum or
    utilisation first exceeds 5%, whichever comes first. PRE_S seconds of
    lead-in are kept so the memory-allocation ramp -- itself a feature the
    classifier uses -- stays visible.
    """
    u = d[UC].to_numpy(float)
    iu = np.flatnonzero(np.nan_to_num(u) > 5.0)
    i_u = int(iu[0]) if len(iu) else 0
    i_m = i_u
    if MC in d.columns:
        m = np.nan_to_num(d[MC].to_numpy(float))
        if m.max() > 0:
            im = np.flatnonzero(m >= 0.95 * m.max())
            if len(im):
                i_m = int(im[0])
    return max(0, min(i_u, i_m) - PRE_S)


def load_trace(root: str, rx: str, gpu: str | None, want: int):
    """The MEDIAN-representative usable run for a label, not the longest.

    Picking the longest run selects stalled or idle outliers: the longest
    gpt2_wikitext2 trace is a 4050 Laptop run sitting at 0.1% utilisation, and
    the longest qwen2.5-7B inference trace spends its first 120 s loading the
    model. Ranking the candidate runs by mean utilisation over the plotted
    window and taking the median one gives a panel that represents its cell.
    """
    pat = re.compile(rx)
    cands = []
    for f in sorted(glob.glob(os.path.join(root, "**", "*.parquet"),
                              recursive=True)):
        b = os.path.basename(f)
        if not pat.search(b):
            continue
        if any(k in b for k in ("AMD", "Instinct", "TPU", "_dcgm_", "_bmc_",
                                "_scope_", "nic_")):
            continue
        if gpu and gpu not in b:
            continue
        try:
            d = pd.read_parquet(f)
        except Exception:
            continue
        if UC not in d.columns:
            continue
        if "gpu_index" in d.columns and d["gpu_index"].nunique() > 1:
            d = d[d["gpu_index"] == sorted(d["gpu_index"].unique())[0]]
        d = d.reset_index(drop=True)
        o = onset(d, want)
        # Some real training cells are simply shorter than the window: the
        # gpt2_wikitext2 and bert_sst2 runs finish in 45-75 s. Those panels end
        # early on a shared axis rather than being dropped, which is itself
        # worth seeing -- a 120 s snapshot is longer than some genuine runs.
        if len(d) - o < min(want * 0.85, MIN_POST_S):
            continue
        d = d.iloc[o:o + want].reset_index(drop=True)
        u = d[UC].to_numpy(float)[:want]
        if np.isnan(u).all():
            continue
        # Drop runs carrying the -1 "power unavailable" sentinel, which would
        # otherwise be annotated as a negative wattage.
        if PC in d.columns and np.nanmean(d[PC].to_numpy(float)[:want]) < 0:
            continue
        cands.append((float(np.nanmean(u)), b, d))
    if not cands:
        return None
    # Prefer runs that fill the window. Without this the median-by-utilisation
    # pick can land on a 42-sample FSDP trace even though that cell has runs
    # of 480-840 s. Cells whose runs are all short (gpt2_wikitext2, bert_sst2)
    # fall through and plot at their true length.
    full = [c for c in cands if len(c[2]) >= want * 0.85]
    pool = full or cands
    pool.sort(key=lambda c: c[0])
    _, b, d = pool[len(pool) // 2]
    return b, d, len(pool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=120,
                    help="seconds of each run to plot, from run start")
    a = ap.parse_args()
    W = a.window

    fig, axes = plt.subplots(3, 4, figsize=(15.0, 8.4), sharex=True)
    used = []
    memmax: dict[str, float] = {}
    prescan = {}
    for pat, gpu, title, tag, det, col, grp in PANELS:
        got = load_trace(a.data_root, pat, gpu, W)
        prescan[title] = got
        if got and MC in got[1].columns:
            mm = float(np.nanmax(got[1][MC].to_numpy(float)[:W] / 1024.0))
            memmax[grp] = max(memmax.get(grp, 0.0), mm)
    for ax, (pat, gpu, title, tag, det, col, grp) in zip(axes.ravel(), PANELS):
        got = prescan[title]
        if got is None:
            ax.text(.5, .5, f"no trace for\n{pat}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="crimson")
            ax.set_title(title, fontsize=9.5, loc="left")
            used.append((title, "MISSING", 0, 0, np.nan, np.nan, np.nan))
            continue
        fn, d, ncand = got
        u = d[UC].to_numpy(float)[:W]
        t = np.arange(len(u), dtype=float)
        ax.fill_between(t, 0, u, color=col, alpha=.28, linewidth=0)
        ax.plot(t, u, color=col, lw=1.25)
        ax.set_ylim(0, 105)
        ax.set_yticks([0, 50, 100])
        ax.grid(alpha=.25)

        mem = np.full(len(u), np.nan)
        if MC in d.columns:
            mem = d[MC].to_numpy(float)[:W] / 1024.0
            axR = ax.twinx()
            axR.plot(t, mem, color="black", lw=1.0, ls=":")
            axR.set_ylim(0, max(1.0, memmax.get(grp, float(np.nanmax(mem))) * 1.35))
            axR.tick_params(labelsize=7)
            axR.set_ylabel("mem (GB)", fontsize=7)

        pw = d[PC].to_numpy(float)[:W] if PC in d.columns else np.full(len(u), np.nan)
        lab = f"{title}"
        ax.set_title(lab, fontsize=9.5, loc="left")
        note = f"{tag}"
        if det is not None:
            note += f" · baseline det. {det:.1f}%"
        ax.text(.02, .035, note, transform=ax.transAxes, fontsize=7.4,
                color="#333", bbox=dict(fc="white", ec="none", alpha=.78, pad=1.4))
        ax.text(.98, .93, f"util {np.nanmean(u):.0f}%  ·  {np.nanmean(pw):.0f} W",
                transform=ax.transAxes, fontsize=7.2, ha="right", va="top",
                color="#333")
        used.append((title, fn, ncand, len(u), float(np.nanmean(u)),
                     float(np.nanmean(pw)), float(np.nanmean(mem))))

    for ax in axes[-1]:
        ax.set_xlabel(f"seconds from activity onset (1 Hz, {W} s)", fontsize=8.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("GPU utilisation (%)", fontsize=8.5)
    fig.suptitle("Raw NVML telemetry across the corpus — "
                 f"{W} s of one representative run per workload, aligned on activity onset "
                 "(shaded: utilisation; dotted: memory)", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, .965))

    out = a.out
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out + ".png", dpi=150)
    fig.savefig(out + ".pdf")
    plt.close(fig)
    print(f"wrote {out}.png and {out}.pdf  (window {W} s)\n")
    print(f"{'panel':30s} {'cand':>4} {'n':>4} {'util':>6} {'W':>7} {'memGB':>7}  file")
    for ttl, fn, nc, n, um, pm, mm in used:
        print(f"{ttl:30s} {nc:4d} {n:4d} {um:6.1f} {pm:7.1f} {mm:7.2f}  {fn[:56]}")


if __name__ == "__main__":
    main()
