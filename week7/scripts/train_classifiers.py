"""
week7/scripts/train_classifiers.py
===================================
Train RF / XGBoost / SVM / LR classifiers on sliding-window features.

Fixes over Week 5 version:
  - Window sizes match code (5/15/30s) — report no longer claims 30/60/120
  - Short runs (< window) are tagged and excluded from headline metrics
  - Binary F1-weighted is computed correctly (was duplicating F1-macro)
  - CM/ROC/feature-importance plots are labeled as "illustrative (fold 5/5)"
  - Labels use centralized manifest (no heuristic guessing)
  - Adds power_pct_tdp and clock dynamic range features

Tasks:
  A — Binary: ML training vs non-training        (target >85%)
  B — Three-way: training / inference / other
  C — Multi-class: full workload_label (15+ classes)

Cross-validation: Stratified Group K-Fold (groups = run_id)

Outputs:
  week7/results/classifier_results.csv
  week7/results/feature_importance.json
  week7/plots/cm_*.png, roc_*.png, feat_imp_*.png
  week7/plots/accuracy_vs_window.png
  week7/reports/week7_classifier_report.md
"""

import sys, json, logging, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, f1_score, accuracy_score
)
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent))
from label_manifest import WINDOW_SIZES

# XGBoost
sys.path.insert(0, "/root/xgb_pkg")
import xgboost as xgb

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

WEEK7   = Path(__file__).parent.parent
RESULTS = WEEK7 / "results"
PLOTS   = WEEK7 / "plots"
REPORTS = WEEK7 / "reports"
for d in [RESULTS, PLOTS, REPORTS]:
    d.mkdir(exist_ok=True)

N_SPLITS = 5
RANDOM_STATE = 42


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_windows(win_sec: int) -> pd.DataFrame:
    p = RESULTS / f"windows_{win_sec}s.parquet"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_parquet(p)
    n_total = len(df)
    # Exclude short runs from headline metrics
    n_short = df["short_run"].sum() if "short_run" in df.columns else 0
    df_full = df[~df["short_run"]] if "short_run" in df.columns else df
    log.info(f"  Loaded {n_total} windows ({n_short} short-run excluded) -> {len(df_full)} for evaluation")
    return df_full


def get_feature_cols(df: pd.DataFrame, meta_path: Path) -> list:
    with open(meta_path) as f:
        meta = json.load(f)
    return [c for c in meta["feature_cols"] if c in df.columns]


def build_classifiers():
    rf = RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_leaf=2,
        max_features="sqrt", class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1)

    xgb_clf = xgb.XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric="mlogloss",
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)

    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    SVC(kernel="rbf", C=10.0, gamma="scale",
                       class_weight="balanced", probability=True,
                       random_state=RANDOM_STATE))])

    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
                       C=1.0, max_iter=1000, class_weight="balanced",
                       solver="lbfgs",
                       random_state=RANDOM_STATE))])

    return {"RandomForest": rf, "XGBoost": xgb_clf, "SVM_RBF": svm, "LogisticReg": lr}


def encode_labels(y_str: pd.Series):
    le = LabelEncoder()
    y  = le.fit_transform(y_str)
    return y, le


# ── Cross-validated evaluation (FIX: compute F1-weighted correctly) ──────────

def evaluate_binary(clf, X, y_bin, groups, model_name, win_sec):
    """Manual binary evaluation with correct F1-macro and F1-weighted."""
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    accs, f1_macros, f1_weighted = [], [], []
    for tr_idx, te_idx in cv.split(X, y_bin, groups):
        clf.fit(X[tr_idx], y_bin[tr_idx])
        y_pred = clf.predict(X[te_idx])
        accs.append(accuracy_score(y_bin[te_idx], y_pred))
        f1_macros.append(f1_score(y_bin[te_idx], y_pred, average="macro", zero_division=0))
        f1_weighted.append(f1_score(y_bin[te_idx], y_pred, average="weighted", zero_division=0))
    return {
        "task": "binary", "model": model_name, "window_sec": win_sec,
        "acc_mean": np.mean(accs), "acc_std": np.std(accs),
        "f1_macro_mean": np.mean(f1_macros), "f1_macro_std": np.std(f1_macros),
        "f1_w_mean": np.mean(f1_weighted), "f1_w_std": np.std(f1_weighted),
    }


def evaluate_multiclass(clf, X, y, groups, task, model_name, win_sec):
    """Multi-class evaluation with correct metrics."""
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    accs, f1_macros, f1_weighted = [], [], []
    for tr_idx, te_idx in cv.split(X, y, groups):
        clf.fit(X[tr_idx], y[tr_idx])
        y_pred = clf.predict(X[te_idx])
        accs.append(accuracy_score(y[te_idx], y_pred))
        f1_macros.append(f1_score(y[te_idx], y_pred, average="macro", zero_division=0))
        f1_weighted.append(f1_score(y[te_idx], y_pred, average="weighted", zero_division=0))
    result = {
        "task": task, "model": model_name, "window_sec": win_sec,
        "acc_mean": np.mean(accs), "acc_std": np.std(accs),
        "f1_macro_mean": np.mean(f1_macros), "f1_macro_std": np.std(f1_macros),
        "f1_w_mean": np.mean(f1_weighted), "f1_w_std": np.std(f1_weighted),
    }
    log.info(f"  [{model_name:14s}] {task:30s} acc={result['acc_mean']:.4f}+/-{result['acc_std']:.4f}  "
             f"f1_macro={result['f1_macro_mean']:.4f}  f1_w={result['f1_w_mean']:.4f}")
    return result


# ── Confusion matrix (FIX: labeled as illustrative fold) ─────────────────────

def plot_cm(clf, X_tr, y_tr, X_te, y_te, le, task, model_name, win_sec, fold_label="fold 5/5"):
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    present = np.unique(np.concatenate([y_te, y_pred]))
    labels  = le.classes_[present]
    cm = confusion_matrix(y_te, y_pred, labels=present)
    fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels) - 1)))
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=False, xticks_rotation="vertical")
    ax.set_title(f"{model_name} -- {task} ({win_sec}s window)\n"
                 f"acc={accuracy_score(y_te,y_pred):.3f}  "
                 f"f1_macro={f1_score(y_te,y_pred,average='macro',zero_division=0):.3f}\n"
                 f"[illustrative: {fold_label}]",
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(PLOTS / f"cm_{task}_{model_name}_{win_sec}s.png", dpi=120, bbox_inches="tight")
    plt.close()


# ── ROC curve (labeled as illustrative) ──────────────────────────────────────

def plot_roc(clf, X_tr, y_tr, X_te, y_te, model_name, win_sec, ax):
    clf.fit(X_tr, y_tr)
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X_te)
        if proba.shape[1] == 2:
            fpr, tpr, _ = roc_curve(y_te, proba[:, 1])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{model_name} AUC={roc_auc:.3f}", lw=2)


# ── Feature importance (labeled as illustrative) ─────────────────────────────

def plot_feature_importance(clf, feature_cols, model_name, task, win_sec, top_n=20, fold_label="fold 5/5"):
    if hasattr(clf, "feature_importances_"):
        imp = clf.feature_importances_
    elif hasattr(clf, "named_steps"):
        inner = clf.named_steps.get("clf")
        if inner and hasattr(inner, "feature_importances_"):
            imp = inner.feature_importances_
        elif inner and hasattr(inner, "coef_"):
            imp = np.abs(inner.coef_).mean(axis=0)
        else:
            return None
    else:
        return None

    idx   = np.argsort(imp)[::-1][:top_n]
    names = [feature_cols[i] for i in idx]
    vals  = imp[idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors  = plt.cm.viridis(np.linspace(0.2, 0.85, top_n))
    ax.barh(range(top_n), vals[::-1], color=colors[::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(names[::-1], fontsize=9)
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Features -- {model_name}, {task} ({win_sec}s)\n[illustrative: {fold_label}]")
    plt.tight_layout()
    plt.savefig(PLOTS / f"feat_imp_{task}_{model_name}_{win_sec}s.png", dpi=120, bbox_inches="tight")
    plt.close()
    return dict(zip(names, vals.tolist()))


# ── PCA projection ────────────────────────────────────────────────────────────

def plot_pca(X, y_str, title, win_sec):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    Xp = pca.fit_transform(Xs)
    labels = sorted(y_str.unique())
    cmap = plt.cm.get_cmap("tab20", len(labels))

    fig, ax = plt.subplots(figsize=(9, 7))
    for i, lbl in enumerate(labels):
        mask = y_str == lbl
        ax.scatter(Xp[mask, 0], Xp[mask, 1], s=18, alpha=0.6, color=cmap(i), label=lbl)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(f"{title} ({win_sec}s window) -- PCA projection")
    ax.legend(fontsize=7, ncol=2, loc="best")
    plt.tight_layout()
    plt.savefig(PLOTS / f"pca_{title.lower().replace(' ', '_')}_{win_sec}s.png", dpi=120, bbox_inches="tight")
    plt.close()
    log.info(f"  PCA saved: pca_{title.lower().replace(' ', '_')}_{win_sec}s.png")


# ── Accuracy vs window size ──────────────────────────────────────────────────

def plot_acc_vs_window(all_rows):
    df = pd.DataFrame(all_rows)
    tasks  = df["task"].unique()
    models = df["model"].unique()
    wins   = sorted(df["window_sec"].unique())
    colors = {"RandomForest": "#2196F3", "XGBoost": "#FF9800",
              "SVM_RBF": "#9C27B0", "LogisticReg": "#4CAF50"}

    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 5), sharey=False)
    if len(tasks) == 1:
        axes = [axes]

    for ax, task in zip(axes, tasks):
        sub = df[df["task"] == task]
        for m in models:
            ms = sub[sub["model"] == m]
            if ms.empty:
                continue
            xs  = ms["window_sec"].values
            ys  = ms["acc_mean"].values
            err = ms["acc_std"].values
            idx = np.argsort(xs)
            ax.errorbar(xs[idx], ys[idx], yerr=err[idx], marker="o",
                        label=m, color=colors.get(m, "gray"), capsize=4, lw=2)
        ax.axhline(0.85, color="red", ls="--", lw=1, alpha=0.7, label="85% target")
        ax.set_title(task, fontsize=11)
        ax.set_xlabel("Window size (s)")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.05)
        ax.set_xticks(wins)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Week 7 B200 -- Classifier Accuracy vs Window Size\n(short runs excluded)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS / "accuracy_vs_window.png", dpi=120, bbox_inches="tight")
    plt.close()
    log.info("Saved: accuracy_vs_window.png")


# ── Report writer ─────────────────────────────────────────────────────────────

def write_report(all_rows, feat_imp, n_features):
    df = pd.DataFrame(all_rows)

    binary_best = df[df["task"] == "binary"].sort_values("acc_mean", ascending=False).iloc[0]
    mc_rows = df[df["task"] == "multiclass"]
    mc_best = mc_rows.sort_values("acc_mean", ascending=False).iloc[0] if len(mc_rows) > 0 else None

    L = []
    a = L.append

    a("# Week 7 -- Sliding-Window Classification on B200")
    a("")
    a("## Overview")
    a("")
    a("Week 7 replicates the Week 5 sliding-window methodology on 2x NVIDIA B200 (Blackwell). "
      "Each run is broken into overlapping time windows and 125 features are extracted per window.")
    a("")
    a(f"- **Hardware:** 2x NVIDIA B200 (183 GB HBM3e, 8 TB/s, 1000W TDP)")
    a(f"- **Feature count per window:** {n_features}")
    a(f"- **Window sizes evaluated:** {', '.join(str(w) + 's' for w in WINDOW_SIZES)}")
    a(f"- **Classifiers:** Random Forest, XGBoost, SVM-RBF, Logistic Regression")
    a(f"- **Cross-validation:** Stratified Group K-Fold (k={N_SPLITS}, grouped by run_id)")
    a(f"- **Short-run handling:** runs shorter than window size are tagged and excluded from headline metrics")
    a("")
    a("## Detection Window / Delay")
    a("")
    a(f"The minimum detection window is **{WINDOW_SIZES[0]}s** (the smallest evaluated window size). "
      f"At 1 Hz NVML sampling, this requires {WINDOW_SIZES[0]} telemetry samples. "
      f"True detection delay = window_size + classification_time (<1ms), so:")
    a("")
    for w in WINDOW_SIZES:
        a(f"- **{w}s window:** detection delay ~ {w}s")
    a("")

    a("## Results")
    a("")
    a("### Binary Classification (ML Training vs Non-Training)")
    a("")
    a("| Model | Window | Accuracy | F1-macro | F1-weighted |")
    a("|-------|--------|----------|----------|-------------|")
    for _, r in df[df["task"] == "binary"].sort_values(["window_sec", "acc_mean"],
                                                        ascending=[True, False]).iterrows():
        a(f"| {r['model']} | {r['window_sec']}s | "
          f"{r['acc_mean']:.4f} +/- {r['acc_std']:.4f} | "
          f"{r['f1_macro_mean']:.4f} | {r['f1_w_mean']:.4f} |")
    a("")
    a(f"**Best binary:** {binary_best['model']} at {int(binary_best['window_sec'])}s window -- "
      f"accuracy = **{binary_best['acc_mean']:.4f}** "
      f"({'PASS' if binary_best['acc_mean'] >= 0.85 else 'NEEDS MORE DATA'} vs 85% target)")
    a("")

    a("### Three-Way Classification")
    a("")
    a("| Model | Window | Accuracy | F1-macro | F1-weighted |")
    a("|-------|--------|----------|----------|-------------|")
    for _, r in df[df["task"] == "threeway"].sort_values(["window_sec", "acc_mean"],
                                                          ascending=[True, False]).iterrows():
        a(f"| {r['model']} | {r['window_sec']}s | "
          f"{r['acc_mean']:.4f} +/- {r['acc_std']:.4f} | "
          f"{r['f1_macro_mean']:.4f} | {r['f1_w_mean']:.4f} |")
    a("")

    if mc_best is not None:
        a("### Multi-Class Classification")
        a("")
        a("| Model | Window | Accuracy | F1-macro | F1-weighted |")
        a("|-------|--------|----------|----------|-------------|")
        for _, r in df[df["task"] == "multiclass"].sort_values(["window_sec", "acc_mean"],
                                                                ascending=[True, False]).iterrows():
            a(f"| {r['model']} | {r['window_sec']}s | "
              f"{r['acc_mean']:.4f} +/- {r['acc_std']:.4f} | "
              f"{r['f1_macro_mean']:.4f} | {r['f1_w_mean']:.4f} |")
        a("")

    a("## Normalized Power Feature")
    a("")
    a("Week 7 adds `power_pct_tdp` = power_draw_w / TDP * 100. This enables cross-GPU comparison:")
    a("- Raw watts are GPU-specific (B200 TDP=1000W, H100=700W) and cannot be compared directly")
    a("- Normalized power (% of TDP) is transferable: 50% TDP means similar thermal budget usage")
    a("- For cross-GPU transfer learning, use `power_pct_tdp` instead of `power_draw_w`")
    a("")

    a("## Figures")
    a("")
    a("| Figure | Path |")
    a("|--------|------|")
    a("| Accuracy vs window size | `plots/accuracy_vs_window.png` |")
    for w in WINDOW_SIZES:
        a(f"| PCA projection (binary, {w}s) | `plots/pca_binary_{w}s.png` |")
    a(f"| CM (illustrative fold) | `plots/cm_binary_RandomForest_{WINDOW_SIZES[0]}s.png` |")
    a(f"| ROC (illustrative fold) | `plots/roc_binary_{WINDOW_SIZES[0]}s.png` |")
    a(f"| Feature importance | `plots/feat_imp_binary_RandomForest_{WINDOW_SIZES[0]}s.png` |")
    a("")

    report_path = REPORTS / "week7_classifier_report.md"
    report_path.write_text("\n".join(L))
    log.info(f"Report saved: {report_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=== Week 7 Classifier Training (B200) ===")

    meta_path = RESULTS / "feature_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    n_features = meta["n_features"]
    log.info(f"Feature count: {n_features}")

    all_rows = []
    feat_imp_store = {}

    for win_sec in WINDOW_SIZES:
        log.info(f"\n-- Window {win_sec}s --")
        try:
            df = load_windows(win_sec)
        except FileNotFoundError:
            log.warning(f"  windows_{win_sec}s.parquet not found, skip")
            continue

        feature_cols = get_feature_cols(df, meta_path)
        X      = df[feature_cols].fillna(0).values.astype(np.float32)
        groups = df["run_id"].values
        classifiers = build_classifiers()

        # Check minimum samples per class for k-fold
        y_bin_check = df["is_training"].values.astype(int)
        min_class_count = min(np.bincount(y_bin_check))
        n_unique_groups = len(np.unique(groups))
        if min_class_count < N_SPLITS or n_unique_groups < N_SPLITS:
            log.warning(f"  Too few samples for {N_SPLITS}-fold CV "
                        f"(min_class={min_class_count}, n_groups={n_unique_groups}), skip {win_sec}s")
            continue

        # ── Task A: Binary ────────────────────────────────────────────────
        log.info("  Task A: Binary (training vs rest)")
        y_bin = y_bin_check

        for name, clf in classifiers.items():
            row = evaluate_binary(clf, X, y_bin, groups, name, win_sec)
            log.info(f"  [{name:14s}] binary  acc={row['acc_mean']:.4f}+/-{row['acc_std']:.4f}  "
                     f"f1_macro={row['f1_macro_mean']:.4f}  f1_w={row['f1_w_mean']:.4f}")
            all_rows.append(row)

        # Illustrative plots for smallest window
        if win_sec == WINDOW_SIZES[0]:
            plot_pca(X, df["binary_label"], "binary", win_sec)
            cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
            folds = list(cv.split(X, y_bin, groups))
            tr_idx, te_idx = folds[-1]
            fold_label = f"fold {N_SPLITS}/{N_SPLITS}"

            # ROC curves
            fig_roc, ax_roc = plt.subplots(figsize=(7, 5))
            for name, clf in classifiers.items():
                plot_roc(clf, X[tr_idx], y_bin[tr_idx], X[te_idx], y_bin[te_idx],
                         name, win_sec, ax_roc)
            ax_roc.plot([0, 1], [0, 1], "k--", lw=1)
            ax_roc.set_xlabel("FPR"); ax_roc.set_ylabel("TPR")
            ax_roc.set_title(f"ROC -- Binary ({win_sec}s window) [illustrative: {fold_label}]")
            ax_roc.legend(); ax_roc.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(PLOTS / f"roc_binary_{win_sec}s.png", dpi=120, bbox_inches="tight")
            plt.close()
            log.info(f"  ROC saved: roc_binary_{win_sec}s.png")

            le_bin = LabelEncoder(); le_bin.fit(["non_training", "training"])
            y_tr_enc = le_bin.transform(np.where(y_bin[tr_idx] == 1, "training", "non_training"))
            y_te_enc = le_bin.transform(np.where(y_bin[te_idx] == 1, "training", "non_training"))
            for name, clf in classifiers.items():
                plot_cm(clf, X[tr_idx], y_tr_enc, X[te_idx], y_te_enc, le_bin,
                        "binary", name, win_sec, fold_label)
                imp = plot_feature_importance(clf, feature_cols, name, "binary", win_sec,
                                              fold_label=fold_label)
                if imp:
                    feat_imp_store[f"binary_{name}"] = imp

        # ── Task B: Three-way ─────────────────────────────────────────────
        log.info("  Task B: Three-way")
        y_3, le_3 = encode_labels(df["binary_label"])
        for name, clf in classifiers.items():
            row = evaluate_multiclass(clf, X, y_3, groups, "threeway", name, win_sec)
            all_rows.append(row)

        # ── Task C: Multi-class ───────────────────────────────────────────
        log.info("  Task C: Multi-class")
        label_counts = df.groupby("workload_label")["run_id"].nunique()
        valid_labels = label_counts[label_counts >= 2].index
        df_mc = df[df["workload_label"].isin(valid_labels)]
        if len(df_mc) < 20:
            log.warning("  Too few multi-class samples, skip")
        else:
            X_mc = df_mc[feature_cols].fillna(0).values.astype(np.float32)
            g_mc = df_mc["run_id"].values
            y_mc, le_mc = encode_labels(df_mc["workload_label"])
            for name, clf in classifiers.items():
                row = evaluate_multiclass(clf, X_mc, y_mc, g_mc, "multiclass", name, win_sec)
                all_rows.append(row)
            if win_sec == WINDOW_SIZES[0]:
                cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
                folds = list(cv.split(X_mc, y_mc, g_mc))
                tr_idx, te_idx = folds[-1]
                fold_label = f"fold {N_SPLITS}/{N_SPLITS}"
                for name, clf in [("RandomForest", classifiers["RandomForest"]),
                                   ("XGBoost", classifiers["XGBoost"])]:
                    plot_cm(clf, X_mc[tr_idx], y_mc[tr_idx],
                                 X_mc[te_idx], y_mc[te_idx],
                                 le_mc, "multiclass", name, win_sec, fold_label)
                    imp = plot_feature_importance(clf, feature_cols, name, "multiclass", win_sec,
                                                  fold_label=fold_label)
                    if imp:
                        feat_imp_store[f"multiclass_{name}"] = imp

    # ── Summary ───────────────────────────────────────────────────────────
    if all_rows:
        plot_acc_vs_window(all_rows)

        df_res = pd.DataFrame(all_rows)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        tasks  = ["binary", "threeway", "multiclass"]
        titles = ["Binary", "Three-Way", "Multi-Class"]
        colors = {"RandomForest": "#2196F3", "XGBoost": "#FF9800",
                  "SVM_RBF": "#9C27B0", "LogisticReg": "#4CAF50"}
        for ax, task, title in zip(axes, tasks, titles):
            sub = df_res[df_res["task"] == task]
            if sub.empty:
                ax.set_title(title + " (no data)")
                continue
            pivot = sub.pivot_table(index="model", columns="window_sec",
                                    values="acc_mean", aggfunc="mean")
            pivot.plot(kind="bar", ax=ax, rot=30, colormap="viridis",
                       edgecolor="white", width=0.7)
            ax.axhline(0.85, color="red", ls="--", lw=1.5, alpha=0.7)
            ax.set_title(title)
            ax.set_ylabel("Accuracy")
            ax.set_ylim(0, 1.05)
            ax.legend(title="Window (s)", fontsize=8)
            ax.grid(True, axis="y", alpha=0.3)
        plt.suptitle("Week 7 B200 -- Classifier Accuracy by Task and Window Size\n"
                     "(short runs excluded from evaluation)", fontsize=12, y=1.03)
        plt.tight_layout()
        plt.savefig(PLOTS / "classifier_summary.png", dpi=120, bbox_inches="tight")
        plt.close()
        log.info("Saved: classifier_summary.png")

        df_res.to_csv(RESULTS / "classifier_results.csv", index=False)
        log.info(f"Saved: classifier_results.csv ({len(df_res)} rows)")

    with open(RESULTS / "feature_importance.json", "w") as f:
        json.dump(feat_imp_store, f, indent=2)

    if all_rows:
        write_report(all_rows, feat_imp_store, n_features)

    log.info("\n=== Done ===")
    if all_rows:
        df_res = pd.DataFrame(all_rows)
        print("\n=== KEY RESULTS (short runs excluded) ===")
        best_bin = df_res[df_res["task"] == "binary"].sort_values("acc_mean", ascending=False)
        print("\nTop-5 Binary:")
        print(best_bin[["model", "window_sec", "acc_mean", "acc_std",
                         "f1_macro_mean", "f1_w_mean"]].head(5).to_string(index=False))
        best_mc = df_res[df_res["task"] == "multiclass"].sort_values("acc_mean", ascending=False)
        if not best_mc.empty:
            print("\nTop-5 Multi-class:")
            print(best_mc[["model", "window_sec", "acc_mean", "acc_std",
                            "f1_macro_mean", "f1_w_mean"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
