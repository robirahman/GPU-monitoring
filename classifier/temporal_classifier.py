"""
Temporal GPU Workload Classifier
================================

Classifies GPU workloads from telemetry time series data using temporal
patterns (epoch periodicity, forward/backward pass asymmetry, memory growth
patterns) rather than just aggregate statistics.

Workload categories: ml_training, ml_inference, hpc, crypto_mining,
                     rendering, video_encoding, idle

Usage:
    python -m classifier.temporal_classifier          # from project root
    python classifier/temporal_classifier.py          # from project root
"""

import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Sklearn is intentionally NOT in requirements.txt — give a clear message.
# ---------------------------------------------------------------------------
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import StratifiedShuffleSplit
except ImportError:
    print(
        "scikit-learn is required but not installed.\n"
        "Install it with:  pip install scikit-learn\n"
        "(It is intentionally omitted from requirements.txt to keep the "
        "core telemetry stack lightweight.)"
    )
    sys.exit(1)

# ===================================================================
# Workload category definitions
# ===================================================================

WORKLOAD_CATEGORIES: Dict[str, List[str]] = {
    "ml_training": [
        "pytorch_resnet_cifar10",
        "pytorch_resnet_cifar10_amp",
        "pytorch_mlp_cifar10",
        "gpt2_wikitext2",
        "gpt2_wikitext2_amp",
        "bert_sst2",
        "bert_sst2_amp",
    ],
    "ml_inference": [
        "resnet50_inference",
    ],
    "hpc": [
        "cufft_benchmark",
        "nbody_sim",
        "gromacs_adh",
    ],
    "crypto_mining": [
        "ethash_cuda",
    ],
    "rendering": [
        "blender_bmw",
    ],
    "video_encoding": [
        "ffmpeg_nvenc",
    ],
    "idle": [
        "idle",
    ],
}

# Reverse lookup: workload_label -> category
LABEL_TO_CATEGORY: Dict[str, str] = {
    label: cat
    for cat, labels in WORKLOAD_CATEGORIES.items()
    for label in labels
}

# ===================================================================
# Core telemetry columns (always expected) and optional DCGM columns
# ===================================================================

CORE_METRICS = [
    "gpu_utilization_pct",
    "mem_utilization_pct",
    "mem_used_mb",
    "power_draw_w",
    "temperature_c",
    "sm_clock_mhz",
    "mem_clock_mhz",
    "pcie_tx_mbps",
    "pcie_rx_mbps",
    "encoder_util_pct",
    "decoder_util_pct",
]

DCGM_PROFILING_COLS = [
    "dcgm_tensor_active",
    "dcgm_fp16_pipe_active",
    "dcgm_fp32_pipe_active",
]


# ===================================================================
# Feature extraction
# ===================================================================

class TemporalFeatureExtractor:
    """Extract aggregate and temporal features from a single-run time series.

    The key insight of this project is that *temporal dynamics* (epoch
    periodicity, memory growth slopes, utilization duty cycles) are far
    more discriminative than snapshot-level aggregate statistics.
    """

    # Lags (in sample steps) at which to compute autocorrelation.
    # These capture periodicity at different time-scales.
    AUTOCORR_LAGS = [1, 2, 5, 10, 20, 50]

    # Rolling window size (in samples) for rolling-variance features.
    ROLLING_WINDOW = 30

    def __init__(self) -> None:
        self._feature_names: Optional[List[str]] = None

    @property
    def feature_names(self) -> Optional[List[str]]:
        """Names of features produced by the last call to extract()."""
        return self._feature_names

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, df: pd.DataFrame) -> Dict[str, float]:
        """Return a flat dict of features for one run (one DataFrame)."""
        features: Dict[str, float] = {}

        # --- 1. Aggregate statistics for every core metric -------------
        for col in CORE_METRICS:
            if col not in df.columns:
                continue
            series = df[col].dropna()
            if series.empty:
                continue
            features[f"{col}_mean"] = float(series.mean())
            features[f"{col}_std"] = float(series.std())
            features[f"{col}_min"] = float(series.min())
            features[f"{col}_max"] = float(series.max())

        # --- 2. Temporal: autocorrelation at multiple lags -------------
        self._add_autocorrelation(df, features)

        # --- 3. Temporal: rolling-window variance ----------------------
        self._add_rolling_variance(df, features)

        # --- 4. Temporal: memory trajectory slope ----------------------
        self._add_memory_slope(df, features)

        # --- 5. Temporal: power coefficient of variation ---------------
        self._add_power_cv(df, features)

        # --- 6. Temporal: utilization duty cycle -----------------------
        self._add_duty_cycle(df, features)

        # --- 7. Encoder/decoder utilisation ----------------------------
        self._add_encoder_decoder(df, features)

        # --- 8. DCGM profiling ratios (when available) -----------------
        self._add_dcgm_ratios(df, features)

        self._feature_names = sorted(features.keys())
        return features

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_autocorr(series: pd.Series, lag: int) -> float:
        """Autocorrelation that returns 0.0 on degenerate input."""
        if len(series) <= lag:
            return 0.0
        try:
            val = series.autocorr(lag=lag)
            return float(val) if np.isfinite(val) else 0.0
        except Exception:
            return 0.0

    def _add_autocorrelation(
        self, df: pd.DataFrame, features: Dict[str, float]
    ) -> None:
        """Autocorrelation of gpu_util and mem_used at several lags.

        Epoch periodicity in ML training shows up as positive autocorrelation
        at a lag corresponding to the epoch length (in sample steps).
        """
        for col in ("gpu_utilization_pct", "mem_used_mb"):
            if col not in df.columns:
                continue
            series = df[col].dropna().reset_index(drop=True)
            for lag in self.AUTOCORR_LAGS:
                features[f"{col}_autocorr_lag{lag}"] = self._safe_autocorr(
                    series, lag
                )

    def _add_rolling_variance(
        self, df: pd.DataFrame, features: Dict[str, float]
    ) -> None:
        """Variance of the rolling mean captures training dynamics.

        During ML training the rolling mean of GPU util shifts between
        high (forward+backward pass) and lower (data loading/checkpointing),
        producing high rolling-mean variance.  Steady workloads (mining,
        inference) have near-zero rolling-mean variance.
        """
        w = self.ROLLING_WINDOW
        for col in ("gpu_utilization_pct", "mem_used_mb", "power_draw_w"):
            if col not in df.columns:
                continue
            series = df[col].dropna()
            if len(series) < w:
                features[f"{col}_rolling_var"] = 0.0
                continue
            rolling_mean = series.rolling(window=w, min_periods=1).mean()
            features[f"{col}_rolling_var"] = float(rolling_mean.var())

    def _add_memory_slope(
        self, df: pd.DataFrame, features: Dict[str, float]
    ) -> None:
        """Linear slope of mem_used_mb over the run.

        ML training typically shows memory growth then stabilisation;
        inference is flat; idle is near-zero.
        """
        if "mem_used_mb" not in df.columns:
            return
        series = df["mem_used_mb"].dropna().reset_index(drop=True)
        if len(series) < 2:
            features["mem_used_mb_slope"] = 0.0
            return
        x = np.arange(len(series), dtype=np.float64)
        # Simple least-squares slope (fast, no extra deps)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            coeffs = np.polyfit(x, series.values, 1)
        features["mem_used_mb_slope"] = float(coeffs[0])

    @staticmethod
    def _add_power_cv(df: pd.DataFrame, features: Dict[str, float]) -> None:
        """Coefficient of variation of power draw.

        Crypto mining maintains extremely steady power draw (low CV);
        ML training fluctuates (high CV).
        """
        if "power_draw_w" not in df.columns:
            return
        series = df["power_draw_w"].dropna()
        mean = series.mean()
        if mean == 0 or len(series) < 2:
            features["power_draw_w_cv"] = 0.0
        else:
            features["power_draw_w_cv"] = float(series.std() / mean)

    @staticmethod
    def _add_duty_cycle(
        df: pd.DataFrame, features: Dict[str, float]
    ) -> None:
        """Fraction of time GPU utilisation is above 80%.

        Different workload types have very different duty cycles:
        mining/HPC ≈ 1.0, training 0.6–0.9, inference bursty, idle ≈ 0.
        """
        if "gpu_utilization_pct" not in df.columns:
            return
        series = df["gpu_utilization_pct"].dropna()
        if series.empty:
            features["gpu_util_duty_cycle_80"] = 0.0
            return
        features["gpu_util_duty_cycle_80"] = float((series >= 80).mean())

    @staticmethod
    def _add_encoder_decoder(
        df: pd.DataFrame, features: Dict[str, float]
    ) -> None:
        """Encoder/decoder utilisation — nonzero essentially only for
        video encoding workloads (ffmpeg_nvenc).
        """
        for col in ("encoder_util_pct", "decoder_util_pct"):
            if col not in df.columns:
                continue
            series = df[col].dropna()
            features[f"{col}_mean"] = float(series.mean()) if not series.empty else 0.0
            features[f"{col}_nonzero_frac"] = (
                float((series > 0).mean()) if not series.empty else 0.0
            )

    @staticmethod
    def _add_dcgm_ratios(
        df: pd.DataFrame, features: Dict[str, float]
    ) -> None:
        """When DCGM profiling columns are present, compute tensor-core
        and fp16/fp32 ratios — very discriminative for AMP training.
        """
        has_tensor = "dcgm_tensor_active" in df.columns
        has_fp16 = "dcgm_fp16_pipe_active" in df.columns
        has_fp32 = "dcgm_fp32_pipe_active" in df.columns

        if has_tensor:
            series = df["dcgm_tensor_active"].dropna()
            features["dcgm_tensor_active_mean"] = (
                float(series.mean()) if not series.empty else 0.0
            )

        if has_fp16 and has_fp32:
            fp16 = df["dcgm_fp16_pipe_active"].dropna()
            fp32 = df["dcgm_fp32_pipe_active"].dropna()
            fp16_mean = float(fp16.mean()) if not fp16.empty else 0.0
            fp32_mean = float(fp32.mean()) if not fp32.empty else 0.0
            total = fp16_mean + fp32_mean
            features["dcgm_fp16_ratio"] = (
                fp16_mean / total if total > 0 else 0.0
            )


# ===================================================================
# Dataset loading
# ===================================================================

def _find_parquet_files(data_dir: str) -> List[Path]:
    """Recursively find all .parquet files under *data_dir*."""
    return sorted(Path(data_dir).rglob("*.parquet"))


def load_dataset(
    data_dir: str = "data",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[
    np.ndarray, np.ndarray, Dict, Optional[np.ndarray], Optional[np.ndarray]
]:
    """Load parquet files, extract features, and optionally split.

    Returns
    -------
    X_train, y_train, metadata, X_test, y_test
        If fewer than 2 samples exist in any category (making stratified
        split impossible), X_test and y_test are None and all data is
        returned in X_train / y_train.
    """
    parquet_files = _find_parquet_files(data_dir)
    if not parquet_files:
        raise FileNotFoundError(
            f"No .parquet files found under '{data_dir}'. "
            "Run the data collection scripts first."
        )

    extractor = TemporalFeatureExtractor()

    rows: List[Dict[str, float]] = []
    categories: List[str] = []
    run_ids: List[str] = []
    workload_labels: List[str] = []

    for pf in parquet_files:
        df = pd.read_parquet(pf)

        # Determine the workload label ---------------------------------
        if "workload_label" in df.columns:
            wl = df["workload_label"].iloc[0]
        else:
            # Fall back to parsing the filename
            wl = pf.stem.rsplit("_", 3)[0]  # strip uuid, date, time

        cat = LABEL_TO_CATEGORY.get(wl)
        if cat is None:
            print(f"  [skip] unknown workload label '{wl}' in {pf.name}")
            continue

        # Determine run_id ---------------------------------------------
        if "run_id" in df.columns:
            rid = str(df["run_id"].iloc[0])
        else:
            rid = pf.stem

        # Extract features ---------------------------------------------
        feat = extractor.extract(df)
        rows.append(feat)
        categories.append(cat)
        run_ids.append(rid)
        workload_labels.append(wl)

    if not rows:
        raise ValueError("No valid runs found after scanning parquet files.")

    # Build aligned feature matrix (fill missing features with 0) ------
    all_keys = sorted({k for row in rows for k in row})
    X = np.array(
        [[row.get(k, 0.0) for k in all_keys] for row in rows],
        dtype=np.float64,
    )
    # Replace any NaN/inf that slipped through
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    y = np.array(categories)

    metadata = {
        "feature_names": all_keys,
        "run_ids": run_ids,
        "workload_labels": workload_labels,
    }

    # Stratified train/test split if possible --------------------------
    label_counts = pd.Series(y).value_counts()
    can_split = (label_counts >= 2).all() and len(y) >= 4

    if can_split and test_size > 0:
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
        train_idx, test_idx = next(splitter.split(X, y))
        return (
            X[train_idx],
            y[train_idx],
            metadata,
            X[test_idx],
            y[test_idx],
        )

    print(
        "  [info] Not enough samples per category for stratified split; "
        "returning all data as training set."
    )
    return X, y, metadata, None, None


# ===================================================================
# Classifier wrapper
# ===================================================================

class WorkloadClassifier:
    """Thin wrapper around sklearn RandomForestClassifier.

    Random Forest is a solid baseline: handles mixed feature scales,
    provides feature importances, and is robust to overfitting on
    small datasets.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        random_state: int = 42,
        **rf_kwargs,
    ) -> None:
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight="balanced",
            **rf_kwargs,
        )
        self._feature_names: Optional[List[str]] = None

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "WorkloadClassifier":
        """Fit the model. Returns *self* for chaining."""
        self.model.fit(X, y)
        self._feature_names = feature_names
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> str:
        """Print and return a classification report + confusion matrix."""
        y_pred = self.predict(X)
        report = classification_report(y, y_pred, zero_division=0)
        cm = confusion_matrix(y, y_pred, labels=self.model.classes_)

        output_lines = [
            "=" * 60,
            "Classification Report",
            "=" * 60,
            report,
            "Confusion Matrix",
            "-" * 60,
        ]
        # Pretty-print confusion matrix with class labels
        header = "  ".join(f"{c:>12s}" for c in self.model.classes_)
        output_lines.append(f"{'':>14s}{header}")
        for i, row_label in enumerate(self.model.classes_):
            row_vals = "  ".join(f"{v:>12d}" for v in cm[i])
            output_lines.append(f"{row_label:>14s}{row_vals}")
        output_lines.append("=" * 60)

        text = "\n".join(output_lines)
        print(text)
        return text

    def feature_importance(
        self, top_n: int = 20
    ) -> List[Tuple[str, float]]:
        """Return (feature_name, importance) pairs sorted descending."""
        importances = self.model.feature_importances_
        names = self._feature_names or [
            f"feature_{i}" for i in range(len(importances))
        ]
        pairs = sorted(
            zip(names, importances), key=lambda x: x[1], reverse=True
        )
        return pairs[:top_n]


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    # Resolve data directory relative to the project root
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_dir = project_root / "data"

    print(f"Looking for parquet files in: {data_dir}")
    print()

    # --- Load dataset -------------------------------------------------
    X_train, y_train, metadata, X_test, y_test = load_dataset(
        data_dir=str(data_dir),
        test_size=0.2,
    )
    feature_names = metadata["feature_names"]

    n_total = len(y_train) + (len(y_test) if y_test is not None else 0)
    unique_cats = sorted(set(y_train) | (set(y_test) if y_test is not None else set()))
    print(f"Loaded {n_total} runs across {len(unique_cats)} categories: {unique_cats}")
    print(f"Feature dimensionality: {X_train.shape[1]}")
    print()

    # --- Train --------------------------------------------------------
    clf = WorkloadClassifier()
    clf.train(X_train, y_train, feature_names=feature_names)

    # --- Evaluate on training data (always available) -----------------
    print("=== Training set performance ===")
    clf.evaluate(X_train, y_train)
    print()

    # --- Evaluate on held-out test set (when available) ---------------
    if X_test is not None and y_test is not None:
        print(f"=== Test set performance ({len(y_test)} samples) ===")
        clf.evaluate(X_test, y_test)
        print()
    else:
        print(
            "[info] No test set available (too few samples per category). "
            "Showing training performance only.\n"
        )

    # --- Feature importances ------------------------------------------
    print("Top-20 most important features:")
    print("-" * 50)
    for name, imp in clf.feature_importance(top_n=20):
        bar = "#" * int(imp * 200)
        print(f"  {name:>40s}  {imp:.4f}  {bar}")
    print()


if __name__ == "__main__":
    main()
