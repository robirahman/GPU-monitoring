# -*- coding: utf-8 -*-
"""Regenerate the two conference-poster charts from the camera-ready results.

Both charts are drawn to the pixel aspect of the region they replace on the
Canva poster, in the poster's own palette (mint/blue sampled from the design's
solid fills), so they drop in without restyling.

Bar chart. Leave-one-family-out detection, from paper_table7_loo_xgb.json --
NOT the arms-race `unseen` column. The two differ: `unseen` fits on corpus_N
minus the family, so a Round 1 family is scored by a detector that has seen
nothing past Round 1, whereas this fits on everything except the held-out
family. The poster's caption describes the latter ("train on every other
evasion family"), and so does Table 7's header, so this is the one to show.

Line chart. Accuracy vs window size for both classifiers, from
paper_figure3_job_id{,_xgb}.json.

Usage:
  python3 scripts/poster_charts.py --out report/poster
"""
import argparse, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "DejaVu Sans"
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from scipy.interpolate import PchipInterpolator

RES = Path("docs/camera_ready_results_20261022")
INK, MUTED, GRID = "#17263c", "#3d4f68", "#d6dce4"
BLUE, MINT, ORANGE, RED = "#2d6fb8", "#61c79b", "#e09020", "#c03020"
DDP = {"K_ddp", "K_ddp_accum", "L_ddp", "L_ddp_stagger"}


def bar_chart(out):
    J = json.loads((RES / "paper_table7_loo_xgb.json").read_text())
    rows = [r for r in J["rows"] if r.get("is_evasion")]
    rows.sort(key=lambda r: r["loo_rate"])          # worst at bottom of a barh
    names = [r["family"] for r in rows]
    vals = [r["loo_rate"] * 100 for r in rows]
    cols = [BLUE if n in DDP else RED if v < 50 else ORANGE if v < 80 else MINT
            for n, v in zip(names, vals)]

    fig, ax = plt.subplots(figsize=(6.51, 6.02))
    y = range(len(rows))
    ax.barh(y, vals, color=cols, height=0.72, zorder=3)
    ax.axvline(90, color=MUTED, ls=":", lw=1.1, zorder=4)
    ax.text(90, len(rows) - 0.1, "90%", color=MUTED, fontsize=8.5,
            ha="center", va="bottom")
    for i, v in enumerate(vals):
        ax.text(v + 1.2, i, f"{v:.1f}", va="center", ha="left",
                fontsize=8.2, color=INK)
    ax.set_yticks(list(y)); ax.set_yticklabels(names, fontsize=8.6, color=INK)
    ax.set_xlim(0, 118); ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(axis="x", labelsize=9, colors=MUTED, length=3)
    ax.set_xlabel("Leave-one-strategy-out detection rate (%)",
                  fontsize=9.5, color=MUTED, labelpad=6)
    ax.set_ylim(-0.7, len(rows) - 0.2)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    # Legend below the plot, as on the original poster: inside the axes it
    # covers the value label on the shortest 50-80% bar.
    ax.legend(handles=[Patch(facecolor=BLUE, label="Multi-GPU (DDP) — all 100%"),
                       Patch(facecolor=MINT, label="Single-GPU, $\\geq$80%"),
                       Patch(facecolor=ORANGE, label="Single-GPU, 50–80%"),
                       Patch(facecolor=RED, label="Single-GPU, <50%")],
              loc="upper center", bbox_to_anchor=(0.5, -0.075), ncol=2,
              fontsize=8.4, frameon=False, handlelength=1.1, handleheight=1.0,
              labelcolor=INK, columnspacing=1.6, handletextpad=0.6)
    fig.tight_layout(pad=0.4)
    fig.savefig(out / "poster_loo_bars.png", dpi=400, transparent=True)
    plt.close(fig)
    below90 = [n for n, v in zip(names, vals) if v < 90]
    sg = [v for n, v in zip(names, vals) if n not in DDP and not n.startswith("whitebox")]
    print(f"bar chart: {len(rows)} families, mean {sum(vals)/len(vals):.2f}%")
    print(f"  below 90: {len(below90)} -> {below90}")
    print(f"  single-GPU non-white-box mean detection {sum(sg)/len(sg):.2f}% "
          f"(evasion {100-sum(sg)/len(sg):.1f}%)")


def line_chart(out):
    def rows(f):
        d = json.loads((RES / f).read_text())["rows"]
        d.sort(key=lambda r: r["window_sec"])
        return [r["window_sec"] for r in d], [r["accuracy"] * 100 for r in d]
    wx, xgb = rows("paper_figure3_job_id_xgb.json")
    wr, rf = rows("paper_figure3_job_id.json")

    # Smooth with PCHIP rather than a cubic spline. PCHIP is shape-preserving:
    # it will not overshoot between knots, so the curve never implies an
    # accuracy higher (or lower) than anything actually measured. Markers stay
    # on the nine measured sizes so the curve is not mistaken for data.
    def smooth(x, y):
        xs = np.linspace(min(x), max(x), 400)
        return xs, PchipInterpolator(np.asarray(x, float), np.asarray(y, float))(xs)

    fig, ax = plt.subplots(figsize=(7.63, 3.10))
    ax.axvspan(0, 15, color=MINT, alpha=0.18, zorder=0)
    ax.text(7.5, 96.62, "$\\leq$15 s alert latency", fontsize=8.4,
            color=MUTED, ha="center", va="bottom")
    sr = smooth(wr, rf); sx = smooth(wx, xgb)
    ax.plot(*sr, lw=1.9, color="#9db0c2", label="RandomForest", zorder=3,
            solid_capstyle="round")
    ax.plot(wr, rf, ls="none", marker="o", ms=4.5, color="#9db0c2", zorder=3)
    ax.plot(*sx, lw=2.4, color=BLUE, label="XGBoost", zorder=4,
            solid_capstyle="round")
    ax.plot(wx, xgb, ls="none", marker="o", ms=5.0, color=BLUE, zorder=5)
    ax.set_xlim(0, 126); ax.set_ylim(96.5, 99.6)
    ax.set_xticks([0, 20, 40, 60, 80, 100, 120])
    ax.set_yticks([96.5, 97.0, 97.5, 98.0, 98.5, 99.0, 99.5])
    ax.set_yticklabels([f"{v:.3f}".rstrip("0").rstrip(".") if False else f"{v/100:.3f}"
                        for v in [96.5, 97.0, 97.5, 98.0, 98.5, 99.0, 99.5]])
    ax.tick_params(labelsize=9, colors=MUTED, length=3)
    ax.set_xlabel("Window size (s)", fontsize=9.5, color=MUTED, labelpad=4)
    ax.set_ylabel("Accuracy (5-fold CV)", fontsize=9.5, color=MUTED, labelpad=6)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    h, l = ax.get_legend_handles_labels()
    ax.legend(h, l, fontsize=8.8, frameon=False, loc="lower right",
              labelcolor=INK, ncol=2, handlelength=1.8)
    fig.tight_layout(pad=0.4)
    fig.savefig(out / "poster_window_accuracy.png", dpi=400, transparent=True)
    plt.close(fig)
    print(f"line chart: XGB {min(xgb):.2f}-{max(xgb):.2f}, RF {min(rf):.2f}-{max(rf):.2f}")
    print("  XGB >=15s range %.2f-%.2f" % (min(xgb[2:]), max(xgb[2:])))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="report/poster")
    a = ap.parse_args()
    o = Path(a.out); o.mkdir(parents=True, exist_ok=True)
    bar_chart(o); line_chart(o)
    print("wrote", o / "poster_loo_bars.png", "and", o / "poster_window_accuracy.png")
