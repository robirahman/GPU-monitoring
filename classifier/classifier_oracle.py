"""
Classifier Oracle — Real-Time White-Box Classifier Query
=========================================================

Provides the white-box adversary with the ability to query the exact same
XGBoost classifier used by the defender, in real time, using live GPU
telemetry from pynvml. This is the strongest possible attack: the adversary
knows every feature, every tree split, and every decision threshold.

Usage from adversarial_whitebox.py:
    oracle = ClassifierOracle(gpu_index=0)
    oracle.start_polling()       # starts background telemetry collection
    p = oracle.query_p_training(window_sec=15)  # P(ml_training)
    oracle.stop_polling()

Standalone test:
    python -m classifier.classifier_oracle --duration 30
"""

import logging
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ClassifierOracle:
    """Query the trained XGBoost classifier using live GPU telemetry.

    The oracle:
    1. Polls pynvml at 1 Hz in a background thread
    2. Maintains a rolling buffer of telemetry samples
    3. On query, extracts the same features as threeway_improved.py
    4. Returns P(ml_training) from the same XGBoost model
    """

    def __init__(self, gpu_index=0, max_buffer_sec=120):
        self.gpu_index = gpu_index
        self.max_buffer_sec = max_buffer_sec
        self._buffer = []       # list of dicts (one per 1Hz sample)
        self._lock = threading.Lock()
        self._polling = False
        self._poll_thread = None

        # Initialize pynvml
        if not HAS_PYNVML:
            raise RuntimeError("pynvml required for ClassifierOracle")
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

        # Load or train the classifier
        self._clf = None
        self._le = None
        self._feature_cols = None
        self._build_classifier()

    def _build_classifier(self):
        """Train the same XGBoost classifier used by the defender."""
        try:
            from sklearn.preprocessing import LabelEncoder
            import xgboost as xgb
        except ImportError:
            log.warning("sklearn/xgboost not available; oracle will return None")
            return

        from classifier.threeway_improved import (
            load_all_data, sliding_windows, get_feature_cols,
            balanced_subsample,
        )

        log.info("Oracle: training classifier from existing data...")
        df_raw = load_all_data()

        # Use 60s windows (same as LOO evaluation)
        df_win = sliding_windows(df_raw, window_sec=60, stride_sec=30)
        if df_win.empty:
            log.warning("No windows; oracle disabled")
            return

        feature_cols = get_feature_cols(df_win)
        X = df_win[feature_cols].fillna(0).values.astype(np.float32)
        y = df_win["threeway_label"].values
        groups = df_win["run_id"].values

        le = LabelEncoder()
        y_enc = le.fit_transform(y)

        # Balanced training (same as defender)
        X_bal, y_bal = balanced_subsample(X, y_enc, groups, random_state=42)

        clf = xgb.XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="mlogloss", random_state=42, n_jobs=-1, verbosity=0,
        )
        clf.fit(X_bal, y_bal)

        self._clf = clf
        self._le = le
        self._feature_cols = feature_cols
        log.info(f"Oracle: classifier ready ({len(feature_cols)} features, "
                 f"{len(X_bal)} balanced training windows)")

    # ── Telemetry polling ──────────────────────────────────────────────

    def _read_sample(self):
        """Read one telemetry sample from pynvml."""
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            power = pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
            temp = pynvml.nvmlDeviceGetTemperature(
                self._handle, pynvml.NVML_TEMPERATURE_GPU)
            sm_clk = pynvml.nvmlDeviceGetClockInfo(
                self._handle, pynvml.NVML_CLOCK_SM)
            mem_clk = pynvml.nvmlDeviceGetClockInfo(
                self._handle, pynvml.NVML_CLOCK_MEM)

            # PCIe throughput
            try:
                pcie_tx = pynvml.nvmlDeviceGetPcieThroughput(
                    self._handle, pynvml.NVML_PCIE_UTIL_TX_BYTES) / 1024.0
                pcie_rx = pynvml.nvmlDeviceGetPcieThroughput(
                    self._handle, pynvml.NVML_PCIE_UTIL_RX_BYTES) / 1024.0
            except pynvml.NVMLError:
                pcie_tx = 0.0
                pcie_rx = 0.0

            return {
                "ts": time.time(),
                "gpu_utilization_pct": float(util.gpu),
                "mem_utilization_pct": float(util.memory),
                "mem_used_mb": mem_info.used / (1024 * 1024),
                "power_draw_w": power,
                "temperature_c": float(temp),
                "sm_clock_mhz": float(sm_clk),
                "mem_clock_mhz": float(mem_clk),
                "pcie_tx_mbps": pcie_tx,
                "pcie_rx_mbps": pcie_rx,
            }
        except pynvml.NVMLError:
            return None

    def _poll_loop(self):
        """Background thread: poll telemetry at 1 Hz."""
        while self._polling:
            sample = self._read_sample()
            if sample is not None:
                with self._lock:
                    self._buffer.append(sample)
                    # Trim buffer to max size
                    if len(self._buffer) > self.max_buffer_sec:
                        self._buffer = self._buffer[-self.max_buffer_sec:]
            time.sleep(1.0)

    def start_polling(self):
        """Start background telemetry collection."""
        if self._polling:
            return
        self._polling = True
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()

    def stop_polling(self):
        """Stop background telemetry collection."""
        self._polling = False
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=3)

    # ── Classifier query ───────────────────────────────────────────────

    def query_p_training(self, window_sec=15):
        """Return P(ml_training) for the last `window_sec` seconds of telemetry.

        Returns None if the classifier isn't ready or insufficient data.
        """
        if self._clf is None:
            return None

        with self._lock:
            if len(self._buffer) < max(5, window_sec):
                return None
            recent = self._buffer[-window_sec:]

        # Build a DataFrame matching the format extract_enhanced_features expects
        df = pd.DataFrame(recent)

        # Extract features using the same function as the classifier
        from classifier.threeway_improved import extract_enhanced_features
        feats = extract_enhanced_features(df)

        # Build feature vector in the same column order
        x = np.array(
            [feats.get(col, 0.0) for col in self._feature_cols],
            dtype=np.float32,
        ).reshape(1, -1)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # Query classifier
        proba = self._clf.predict_proba(x)[0]
        training_idx = list(self._le.classes_).index(
            self._le.transform(["ml_training"])[0])
        return float(proba[training_idx])


# ── Standalone test ────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test classifier oracle")
    parser.add_argument("--duration", type=int, default=30)
    parser.add_argument("--gpu-index", type=int, default=0)
    args = parser.parse_args()

    oracle = ClassifierOracle(gpu_index=args.gpu_index)
    oracle.start_polling()

    print(f"Polling GPU {args.gpu_index} for {args.duration}s...")
    print("Waiting 5s for buffer warmup...")
    time.sleep(5)

    start = time.time()
    while time.time() - start < args.duration:
        p = oracle.query_p_training(window_sec=15)
        if p is not None:
            label = "TRAINING" if p > 0.5 else "safe"
            print(f"  P(training) = {p:.4f}  [{label}]  "
                  f"buffer={len(oracle._buffer)}s")
        else:
            print(f"  P(training) = N/A  buffer={len(oracle._buffer)}s")
        time.sleep(2)

    oracle.stop_polling()
    print("Done.")


if __name__ == "__main__":
    main()
