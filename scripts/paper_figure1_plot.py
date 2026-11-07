#!/usr/bin/env python3
"""Figure 1: cross-GPU transfer matrix with a true leave-one-GPU-out row.

Reads the regenerated JSON rather than classifier/results/cross_gpu_transfer_
binary_rf.csv. That CSV carries a row labelled "Leave-one-GPU-out", but the
string appears in no .py file in any commit and the sibling generator
(regen_crossgpu_heatmap.py) labels the analogous row "Pooled (5-fold CV)", so
it is very likely pooled CV mislabelled. The LOGO row here is computed by
scripts/paper_figure1.py: for each test GPU, train on the other eight only.

    PYTHONPATH=. python3 scripts/paper_figure1_plot.py \
        --json docs/camera_ready_results_20261022/paper_figure1_30s_xgb.json \
        --out report/figures/cross_gpu_heatmap_xgb
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42   # TrueType, avoid Type-3 for camera-ready
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", required=True, help="path stem; .png and .pdf are written")
    ap.add_argument("--title", default=None)
    # Sizing. The default geometry is generous and is meant to be viewed on
    # screen; at the width a NeurIPS column allows it is downscaled ~2.3x, which
    # drops the 9 pt tick labels to about 4 pt. --figsize lets the figure be
    # generated at the size it will actually be printed, so the font arguments
    # below are true point sizes on the page rather than nominal ones.
    ap.add_argument("--figsize", default=None, metavar="W,H",
                    help="inches, e.g. 5.4,2.6 (default: scales with GPU count)")
    ap.add_argument("--font-tick", type=float, default=9.0)
    ap.add_argument("--font-cell", type=float, default=8.0)
    ap.add_argument("--font-title", type=float, default=10.0)
    ap.add_argument("--font-axis", type=float, default=None,
                    help="axis labels and colorbar; defaults to --font-tick")
    # The column labels repeat the row labels verbatim, and rotating the full
    # names eats roughly a third of the figure height. Abbreviating just the
    # columns keeps the full names on the rows, where they cost nothing.
    ap.add_argument("--short-xticks", action="store_true")
    ap.add_argument("--xtick-rotation", type=float, default=35.0)
    a = ap.parse_args()
    f_axis = a.font_axis if a.font_axis is not None else a.font_tick

    d = json.loads(Path(a.json).read_text())
    gpus = d["gpus"]; tr = d["transfer"]; logo = d["logo"]
    n = len(gpus)
    M = np.full((n + 1, n), np.nan)
    for i, g in enumerate(gpus):
        for j, h in enumerate(gpus):
            v = tr.get(g, {}).get(h)
            if v is not None:
                M[i, j] = v
    for j, h in enumerate(gpus):
        if h in logo:
            M[n, j] = logo[h]

    rows = list(gpus) + ["Leave-one-GPU-out"]
    lo = float(np.nanmin(M))
    vmin = 0.2 if lo < 0.5 else 0.5

    if a.figsize:
        fs = tuple(float(x) for x in a.figsize.split(","))
    else:
        fs = (0.95 * n + 3.0, 0.58 * (n + 1) + 2.0)
    fig, ax = plt.subplots(figsize=fs)
    im = ax.imshow(M, cmap="RdYlGn", vmin=vmin, vmax=1.0, aspect="auto")
    ax.set_xticks(range(n)); ax.set_yticks(range(n + 1))
    def _short(g):
        g = (g.replace("A100 SXM4 ", "A100-").replace("H100 80GB", "H100-80")
              .replace("H100 NVL", "H100-NVL").replace("H200 / H200 NVL", "H200")
              .replace("RTX 4050 Laptop", "RTX4050").replace("RTX ", "RTX"))
        return g
    xlab = [_short(g) for g in gpus] if a.short_xticks else gpus
    ax.set_xticklabels(xlab, rotation=a.xtick_rotation,
                       ha="right" if a.xtick_rotation else "center",
                       fontsize=a.font_tick)
    ax.set_yticklabels(rows, fontsize=a.font_tick)
    ax.set_xlabel("Test GPU", fontsize=f_axis)
    ax.set_ylabel("Train GPU", fontsize=f_axis)
    for i in range(n + 1):
        for j in range(n):
            v = M[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v*100:.1f}", ha="center", va="center",
                    color="white" if v < 0.55 else "black", fontsize=a.font_cell)
    # The diagonal trains and tests on the same GPU WITHOUT grouping, so it is an
    # upper bound rather than a generalisation estimate. Outline it so no reader
    # mistakes those cells for the honest within-GPU number (Table 4's grouped CV).
    for i in range(n):
        ax.add_patch(plt.Rectangle((i - .5, i - .5), 1, 1, fill=False,
                                   edgecolor="black", lw=1.4, ls=":"))
    # Separate the LOGO row from the pairwise block.
    ax.axhline(n - 0.5, color="black", lw=2.0)
    if a.title:
        ax.set_title(a.title, fontsize=a.font_title)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cb.set_label("binary accuracy", fontsize=f_axis)
    cb.ax.tick_params(labelsize=f_axis)
    fig.tight_layout()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=400)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"wrote {out.with_suffix('.png')} and {out.with_suffix('.pdf')}")
    print(f"  matrix {M.shape}, range {np.nanmin(M)*100:.1f}-{np.nanmax(M)*100:.1f}%")
    off = M[:n, :n][~np.eye(n, dtype=bool)]
    print(f"  off-diagonal transfer: mean {np.nanmean(off)*100:.1f}%  "
          f"min {np.nanmin(off)*100:.1f}%  below 90%: {int((off < .9).sum())}/{off.size}")
    lg = np.array([logo[g] for g in gpus])
    print(f"  LOGO row: mean {lg.mean()*100:.1f}%  min {lg.min()*100:.1f}%  max {lg.max()*100:.1f}%")

if __name__ == "__main__":
    main()
