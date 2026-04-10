#!/usr/bin/env python3
"""
Week 7 Step 3: Baseline Classifier Suite
- Feature extraction: statistical, FFT, memory growth
- Train: Random Forest, SVM, Logistic Regression (+ XGBoost if available)
- Evaluate: binary (ML training vs rest), 3-way (training/inference/other),
            multi-class (full label set)
- Sliding window evaluation at 30s, 60s, 120s
- Save all metrics and confusion matrices

Outputs to week7/results/
"""

import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).parent.parent
DATA_DIRS = [REPO_ROOT / "data", REPO_ROOT / "week7" / "data"]
RESULTS_DIR = REPO_ROOT / "week7" / "results"
PLOTS_DIR  = REPO_ROOT / "week7" / "plots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

METRICS = [
    "gpu_utilization_pct",
    "mem_utilization_pct",
    "power_draw_w",
    "temperature_c",
    "sm_clock_mhz",
    "mem_clock_mhz",
    "mem_used_mb",
    "pcie_tx_mbps",
    "pcie_rx_mbps",
]

CATEGORY_MAP = {
    "idle": "baseline",
    "pytorch_resnet_cifar10": "ml_training",
    "pytorch_resnet_cifar10_amp": "ml_training",
    "pytorch_mlp_cifar10": "ml_training",
    "gpt2_wikitext2": "ml_training",
    "gpt2_wikitext2_amp": "ml_training",
    "bert_sst2": "ml_training",
    "bert_sst2_amp": "ml_training",
    "resnet50_inference": "ml_inference",
    "resnet50_inference_small": "ml_inference",
    "resnet50_inference_large": "ml_inference",
    "resnet50_concurrent": "ml_inference",
    "cufft_benchmark": "scientific_hpc",
    "cufft_short": "scientific_hpc",
    "cufft_concurrent": "scientific_hpc",
    "nbody_sim": "scientific_hpc",
    "nbody_short": "scientific_hpc",
    "mining_ethash_proxy": "crypto_mining",
    "rendering_proxy": "rendering",
    "resnet18_small_batch": "ml_training",
    "resnet18_large_batch": "ml_training",
    "resnet18_short_run": "ml_training",
    "mlp_small_batch": "ml_training",
    "mlp_large_batch": "ml_training",
    "idle_h100": "baseline",
}


def load_data(dirs):
    dfs = []
    for d in dirs:
        d = Path(d)
        if not d.exists():
            continue
        for f in sorted(d.glob("*.parquet")):
            try:
                df = pd.read_parquet(f)
                if "workload_label" not in df.columns:
                    continue
                df["workload_label"] = df["workload_label"].str.split("_NVIDIA").str[0]
                dfs.append(df)
            except Exception:
                pass
    if not dfs:
        return pd.DataFrame()
    full = pd.concat(dfs, ignore_index=True)
    full["category"] = full["workload_label"].map(lambda x: CATEGORY_MAP.get(x, "unknown"))
    if "timestamp_epoch" in full.columns:
        full["elapsed_sec"] = full.groupby("run_id")["timestamp_epoch"].transform(
            lambda s: s - s.min()
        )
    return full


def extract_features_per_run(df, window_duration=None):
    """
    Extract aggregate features per run (or per window if window_duration given).
    window_duration: if set, slide a fixed window (seconds) over each run.
    """
    numeric_cols = [c for c in METRICS if c in df.columns]
    records = []

    for run_id, grp in df.groupby("run_id"):
        grp = grp.sort_values("elapsed_sec" if "elapsed_sec" in grp.columns else grp.index.name or "index")
        label = grp["workload_label"].iloc[0]
        cat   = grp["category"].iloc[0]

        if window_duration is not None and "elapsed_sec" in grp.columns:
            t_max = grp["elapsed_sec"].max()
            step = window_duration
            windows = [(t, t + window_duration) for t in np.arange(0, t_max - window_duration + 1, step)]
        else:
            windows = [(None, None)]

        for (t_start, t_end) in windows:
            if t_start is not None:
                w = grp[(grp["elapsed_sec"] >= t_start) & (grp["elapsed_sec"] < t_end)]
            else:
                w = grp

            if len(w) < 5:
                continue

            feat = {"run_id": run_id, "workload_label": label, "category": cat}
            if t_start is not None:
                feat["window_start"] = t_start

            for col in numeric_cols:
                vals = w[col].dropna()
                if len(vals) == 0:
                    continue
                feat[f"{col}_mean"]  = vals.mean()
                feat[f"{col}_std"]   = vals.std()
                feat[f"{col}_cv"]    = vals.std() / (vals.mean() + 1e-9)
                feat[f"{col}_p25"]   = vals.quantile(0.25)
                feat[f"{col}_p75"]   = vals.quantile(0.75)
                feat[f"{col}_iqr"]   = vals.quantile(0.75) - vals.quantile(0.25)
                feat[f"{col}_range"] = vals.max() - vals.min()
                # FFT energy in 0.1-5 Hz band
                if len(vals) >= 16:
                    fft_amp = np.abs(np.fft.rfft(vals.values - vals.mean()))
                    freqs   = np.fft.rfftfreq(len(vals), d=1.0)
                    band    = (freqs >= 0.1) & (freqs <= 5.0)
                    feat[f"{col}_fft_energy"] = float(fft_amp[band].sum()) if band.any() else 0.0

            # Memory growth rate
            if "mem_used_mb" in w.columns:
                mem = w["mem_used_mb"].dropna()
                if len(mem) >= 4:
                    slope = np.polyfit(np.arange(len(mem)), mem.values, 1)[0]
                    feat["mem_growth_rate"] = slope

            records.append(feat)

    return pd.DataFrame(records)


def prepare_Xy(feature_df, target_col):
    """Return X (feature matrix) and y (label array)."""
    meta_cols = {"run_id", "workload_label", "category", "window_start"}
    feat_cols = [c for c in feature_df.columns
                 if c not in meta_cols
                 and feature_df[c].dtype in (np.float64, np.float32, float, np.int64)]
    X = feature_df[feat_cols].fillna(0).values
    y = feature_df[target_col].values
    return X, y, feat_cols


def evaluate_classifier(name, clf, X, y, cv=5):
    """Cross-validate and return metrics dict."""
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.metrics import make_scorer, f1_score
    from sklearn.preprocessing import LabelEncoder

    # Ensure X is a clean float64 numpy array
    X = np.asarray(X, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Encode string labels to integers
    le = LabelEncoder()
    y_enc = le.fit_transform(np.asarray(y))

    counts = np.bincount(y_enc)
    n_splits = min(cv, int(counts.min()))
    n_splits = max(2, n_splits)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scorers = {
        "accuracy": "accuracy",
        "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
        "f1_weighted": make_scorer(f1_score, average="weighted", zero_division=0),
    }
    cv_results = cross_validate(clf, X, y_enc, cv=skf, scoring=scorers,
                                 return_train_score=False, n_jobs=1)
    return {
        "model": name,
        "accuracy_mean": cv_results["test_accuracy"].mean(),
        "accuracy_std":  cv_results["test_accuracy"].std(),
        "f1_macro_mean": cv_results["test_f1_macro"].mean(),
        "f1_macro_std":  cv_results["test_f1_macro"].std(),
        "f1_weighted_mean": cv_results["test_f1_weighted"].mean(),
        "f1_weighted_std":  cv_results["test_f1_weighted"].std(),
    }


def plot_confusion_matrix(clf, X, y, classes, title, out_path):
    """Train on full set and plot confusion matrix."""
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.metrics import confusion_matrix
    from sklearn.preprocessing import LabelEncoder

    if len(set(y)) < 2:
        return
    try:
        X = np.asarray(X, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        le = LabelEncoder()
        y_enc = le.fit_transform(np.asarray(y))
        classes_enc = le.transform(classes) if set(classes) <= set(le.classes_) else le.classes_
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
        train_idx, test_idx = next(sss.split(X, y_enc))
        clf.fit(X[train_idx], y_enc[train_idx])
        y_pred = clf.predict(X[test_idx])
        cm = confusion_matrix(y_enc[test_idx], y_pred, labels=le.transform(le.classes_))

        fig, ax = plt.subplots(figsize=(max(6, len(classes) * 1.2),
                                        max(5, len(classes) * 1.0)))
        cmap = LinearSegmentedColormap.from_list("blue_white", ["white", "#1565C0"])
        im = ax.imshow(cm, cmap=cmap)
        plt.colorbar(im, ax=ax)
        display_classes = list(le.classes_)
        ax.set_xticks(range(len(display_classes)))
        ax.set_yticks(range(len(display_classes)))
        ax.set_xticklabels(display_classes, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(display_classes, fontsize=8)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() * 0.5 else "black",
                        fontsize=9)
        plt.tight_layout()
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"    Confusion matrix skipped: {e}")


def build_classifiers():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    clfs = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
        ),
        "SVM_RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=10, gamma="scale",
                        class_weight="balanced", probability=True, random_state=42)),
        ]),
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, class_weight="balanced",
                                      random_state=42, C=1.0)),
        ]),
    }
    try:
        import xgboost as xgb
        clfs["XGBoost"] = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            use_label_encoder=False, eval_metric="mlogloss",
            random_state=42, n_jobs=-1,
        )
    except ImportError:
        pass
    return clfs


def run_task(task_name, feature_df, target_col, clfs, label_col=None):
    """Run all classifiers on a task. Returns a list of metric dicts."""
    if label_col is None:
        label_col = target_col
    valid = feature_df.dropna(subset=[target_col])
    counts = valid[target_col].value_counts()
    # Need at least 2 classes with ≥2 samples each
    valid_classes = counts[counts >= 2].index
    valid = valid[valid[target_col].isin(valid_classes)]
    if len(valid) < 10 or valid[target_col].nunique() < 2:
        print(f"  [{task_name}] insufficient data (n={len(valid)}, "
              f"classes={valid[target_col].nunique()}), skipping")
        return []

    X, y, feat_cols = prepare_Xy(valid, target_col)
    classes = sorted(valid[target_col].unique())
    print(f"  [{task_name}] n={len(valid)}, classes={classes}")

    results = []
    for clf_name, clf in clfs.items():
        try:
            metrics = evaluate_classifier(clf_name, clf, X, y)
            metrics["task"] = task_name
            results.append(metrics)
            print(f"    {clf_name:20s}  acc={metrics['accuracy_mean']:.3f}±{metrics['accuracy_std']:.3f}"
                  f"  f1_macro={metrics['f1_macro_mean']:.3f}")
            # Confusion matrix for best model
            cm_path = PLOTS_DIR / f"cm_{task_name}_{clf_name.lower().replace(' ', '_')}.png"
            plot_confusion_matrix(clf, X, y, classes,
                                  f"{task_name} — {clf_name}", cm_path)
        except Exception as e:
            print(f"    {clf_name} FAILED: {e}")

    # Feature importance (RandomForest)
    if "RandomForest" in clfs:
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import LabelEncoder
            rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
            X_np = np.asarray(X, dtype=np.float64)
            X_np = np.nan_to_num(X_np, nan=0.0, posinf=0.0, neginf=0.0)
            le_fi = LabelEncoder()
            y_enc_fi = le_fi.fit_transform(np.asarray(y))
            rf.fit(X_np, y_enc_fi)
            imp = pd.Series(rf.feature_importances_, index=feat_cols).nlargest(20)
            fig, ax = plt.subplots(figsize=(8, 6))
            imp.sort_values().plot.barh(ax=ax, color="#1565C0")
            ax.set_title(f"Top-20 Feature Importance — {task_name}", fontsize=11)
            ax.set_xlabel("Importance")
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / f"feat_importance_{task_name}.png",
                        dpi=120, bbox_inches="tight")
            plt.close()
        except Exception as e:
            print(f"    Feature importance skipped: {e}")

    return results


def run_sliding_window_eval(df, window_secs_list, clfs):
    """Evaluate classifiers across sliding window sizes."""
    summary = []
    for ws in window_secs_list:
        print(f"\n  Sliding window: {ws}s")
        feat_df = extract_features_per_run(df, window_duration=ws)
        if feat_df.empty or feat_df["category"].nunique() < 2:
            print("    insufficient data")
            continue
        # Binary only for speed
        feat_df["is_training"] = (feat_df["category"] == "ml_training").astype(int)
        valid = feat_df.dropna()
        if len(valid) < 10 or valid["is_training"].nunique() < 2:
            continue
        X, y, _ = prepare_Xy(valid, "is_training")
        for clf_name, clf in clfs.items():
            try:
                m = evaluate_classifier(clf_name, clf, X, y)
                m["window_sec"] = ws
                m["task"] = "binary_training_vs_rest"
                summary.append(m)
                print(f"    {clf_name:20s}  acc={m['accuracy_mean']:.3f}  "
                      f"f1_macro={m['f1_macro_mean']:.3f}")
            except Exception as e:
                print(f"    {clf_name} failed: {e}")
    return summary


def main():
    print("=" * 60)
    print("Week 7 Classifier Suite")
    print("=" * 60)

    print("\n[1/5] Loading data...")
    df = load_data(DATA_DIRS)
    if df.empty:
        print("No data found. Run collect_edge_cases.py first.")
        sys.exit(0)
    df["category"] = df["workload_label"].map(lambda x: CATEGORY_MAP.get(x, "unknown"))
    print(f"  {len(df):,} rows, {df['run_id'].nunique()} runs")

    print("\n[2/5] Extracting per-run features...")
    feat_df = extract_features_per_run(df)
    print(f"  {len(feat_df)} feature vectors, {len([c for c in feat_df.columns if c not in ('run_id','workload_label','category')])} features")
    feat_df.to_csv(RESULTS_DIR / "classifier_features.csv", index=False)

    # Add derived labels
    feat_df["binary_training"] = (feat_df["category"] == "ml_training").astype(int).map(
        {0: "non_training", 1: "ml_training"}
    )
    feat_df["three_way"] = feat_df["category"].map(
        lambda x: x if x in ("ml_training", "ml_inference") else "other"
    )

    print("\n[3/5] Building classifiers...")
    clfs = build_classifiers()
    print(f"  Models: {list(clfs.keys())}")

    all_results = []

    print("\n--- Task A: Binary (ML Training vs. Rest) ---")
    all_results.extend(run_task("binary_training_vs_rest", feat_df, "binary_training", clfs))

    print("\n--- Task B: Three-Way (training / inference / other) ---")
    all_results.extend(run_task("threeway", feat_df, "three_way", clfs))

    print("\n--- Task C: Multi-class (full workload label) ---")
    all_results.extend(run_task("multiclass_label", feat_df, "workload_label", clfs))

    print("\n--- Task D: Multi-class (category) ---")
    all_results.extend(run_task("multiclass_category", feat_df, "category", clfs))

    print("\n[4/5] Sliding-window evaluation (30s / 60s / 120s)...")
    window_results = run_sliding_window_eval(df, [30, 60, 120], clfs)
    all_results.extend(window_results)

    print("\n[5/5] Saving results...")
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(RESULTS_DIR / "classifier_results.csv", index=False)
        print(f"  Saved {len(results_df)} result rows to classifier_results.csv")

        # Summary plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, metric, ylabel in zip(
            axes,
            ["accuracy_mean", "f1_macro_mean"],
            ["Accuracy", "F1 Macro"],
        ):
            pivot = results_df[results_df["window_sec"].isna()].pivot_table(
                values=metric, index="model", columns="task", aggfunc="mean"
            )
            if not pivot.empty:
                pivot.plot.bar(ax=ax, colormap="Set2")
                ax.set_title(f"{ylabel} by Model and Task", fontsize=11)
                ax.set_ylabel(ylabel)
                ax.set_ylim(0, 1.1)
                ax.tick_params(axis="x", rotation=20)
                ax.legend(fontsize=7, ncol=2)
                ax.grid(True, alpha=0.3, axis="y")
        plt.suptitle("Classifier Performance Summary", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "classifier_summary.png", dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  Saved classifier_summary.png")

        # Sliding window plot
        win_df = results_df[results_df["window_sec"].notna()]
        if not win_df.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            for model, grp in win_df.groupby("model"):
                grp = grp.sort_values("window_sec")
                ax.plot(grp["window_sec"], grp["accuracy_mean"], marker="o", label=model)
                ax.fill_between(grp["window_sec"],
                                grp["accuracy_mean"] - grp["accuracy_std"],
                                grp["accuracy_mean"] + grp["accuracy_std"],
                                alpha=0.15)
            ax.set_xlabel("Window size (s)")
            ax.set_ylabel("Accuracy")
            ax.set_title("Binary Classifier Accuracy vs. Window Size", fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / "sliding_window_accuracy.png", dpi=120, bbox_inches="tight")
            plt.close()
            print(f"  Saved sliding_window_accuracy.png")

    print("\n" + "=" * 60)
    print(f"Classifier suite complete. Results in: {RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
