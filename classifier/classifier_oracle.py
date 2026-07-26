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

# AMD backend: amdsmi ships with ROCm under /opt/rocm/share/amd_smi or as a
# pip package after `pip install amdsmi`. We import lazily here; collectors
# use the same module so the import surface is well-known.
import os, sys
_AMDSMI_PATH = "/opt/rocm/share/amd_smi"
if os.path.isdir(_AMDSMI_PATH) and _AMDSMI_PATH not in sys.path:
    sys.path.insert(0, _AMDSMI_PATH)
try:
    import amdsmi as _amdsmi
    HAS_AMDSMI = True
except ImportError:
    HAS_AMDSMI = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ClassifierOracle:
    """Query the trained XGBoost classifier using live GPU telemetry.

    The oracle:
    1. Polls pynvml at 1 Hz in a background thread
    2. Maintains a rolling buffer of telemetry samples
    3. On query, extracts the same features as threeway_improved.py
    4. Returns P(ml_training) from the same XGBoost model
    """

    def __init__(self, gpu_index=0, max_buffer_sec=120, backend="auto"):
        """
        backend:
          "nvidia" — force pynvml/NVML
          "amd"    — force amdsmi/ROCm
          "auto"   — try pynvml first, then amdsmi
        """
        self.gpu_index = gpu_index
        self.max_buffer_sec = max_buffer_sec
        self._buffer = []       # list of dicts (one per 1Hz sample)
        self._lock = threading.Lock()
        self._polling = False
        self._poll_thread = None
        self._backend = None
        self._handle = None

        if backend == "auto":
            if HAS_PYNVML:
                self._init_nvidia()
            elif HAS_AMDSMI:
                self._init_amd()
            else:
                raise RuntimeError(
                    "Neither pynvml nor amdsmi is installed — "
                    "ClassifierOracle needs at least one GPU telemetry backend"
                )
        elif backend == "nvidia":
            if not HAS_PYNVML:
                raise RuntimeError("pynvml not installed but backend='nvidia' requested")
            self._init_nvidia()
        elif backend == "amd":
            if not HAS_AMDSMI:
                raise RuntimeError("amdsmi not installed but backend='amd' requested")
            self._init_amd()
        else:
            raise ValueError(f"Unknown backend: {backend!r}")

        # Load or train the classifier. This must live here, after the backend
        # dispatch above, NOT inside one of the _init_* methods — it is
        # backend-independent, and having it in _init_amd only meant _clf was
        # never defined on NVIDIA, so the first query_p_training() raised
        # AttributeError and every WB-O run died ~14 s in.
        self._clf = None
        self._le = None
        self._feature_cols = None
        self._build_classifier()

    def _init_nvidia(self):
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        self._backend = "nvidia"
        log.info(f"ClassifierOracle: pynvml backend, gpu_index={self.gpu_index}")

    def _init_amd(self):
        _amdsmi.amdsmi_init()
        handles = _amdsmi.amdsmi_get_processor_handles()
        # Filter out CPU sockets — amdsmi reports them as AMD_GPU too on some
        # systems; the actual graphics device is the one where
        # amdsmi_get_gpu_activity succeeds.
        gpu_handles = []
        for h in handles:
            try:
                _amdsmi.amdsmi_get_gpu_activity(h)
                gpu_handles.append(h)
            except Exception:
                continue
        if not gpu_handles:
            raise RuntimeError("amdsmi found no GPU-capable processors")
        if self.gpu_index >= len(gpu_handles):
            raise RuntimeError(
                f"gpu_index={self.gpu_index} but only {len(gpu_handles)} GPUs visible"
            )
        self._handle = gpu_handles[self.gpu_index]
        self._backend = "amd"
        asic = _amdsmi.amdsmi_get_gpu_asic_info(self._handle) or {}
        log.info(f"ClassifierOracle: amdsmi backend, "
                 f"gpu_index={self.gpu_index}, {asic.get('market_name', '?')}")

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
            compute_sample_weights,
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

        # Class-weighted training (same as defender)
        w = compute_sample_weights(y_enc)

        clf = xgb.XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="mlogloss", random_state=42, n_jobs=-1, verbosity=0,
        )
        clf.fit(X, y_enc, sample_weight=w)

        self._clf = clf
        self._le = le
        self._feature_cols = feature_cols
        log.info(f"Oracle: classifier ready ({len(feature_cols)} features, "
                 f"{len(X)} class-weighted training windows)")

    # ── Telemetry polling ──────────────────────────────────────────────

    def _read_sample(self):
        """Read one telemetry sample from the active backend."""
        if self._backend == "nvidia":
            return self._read_sample_nvidia()
        if self._backend == "amd":
            return self._read_sample_amd()
        return None

    def _read_sample_nvidia(self):
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

    def _read_sample_amd(self):
        """Mirror of _read_sample_nvidia using amdsmi (matches the 9-signal
        schema used by scripts/collect_amdsmi_telemetry.py)."""
        try:
            activity = _amdsmi.amdsmi_get_gpu_activity(self._handle) or {}
            mem_used = _amdsmi.amdsmi_get_gpu_memory_usage(
                self._handle, _amdsmi.AmdSmiMemoryType.VRAM)
            power_info = _amdsmi.amdsmi_get_power_info(self._handle) or {}
            # current_socket_power can be sentinel "N/A"; fall back to socket_power
            raw_power = power_info.get("current_socket_power")
            if not isinstance(raw_power, (int, float)):
                raw_power = power_info.get("socket_power") or \
                            power_info.get("average_socket_power") or 0.0
            metrics = _amdsmi.amdsmi_get_gpu_metrics_info(self._handle) or {}
            edge_temp = metrics.get("temperature_edge", 0)
            gfx_clk_info = _amdsmi.amdsmi_get_clock_info(
                self._handle, _amdsmi.AmdSmiClkType.GFX) or {}
            mem_clk_info = _amdsmi.amdsmi_get_clock_info(
                self._handle, _amdsmi.AmdSmiClkType.MEM) or {}

            # amdsmi doesn't expose PCIe TX/RX on consumer cards; use -1
            # sentinel (matches the collector's convention).
            return {
                "ts": time.time(),
                "gpu_utilization_pct": float(activity.get("gfx_activity", 0)),
                "mem_utilization_pct": float(activity.get("umc_activity", 0)),
                "mem_used_mb": float(mem_used) / (1024 * 1024),
                "power_draw_w": float(raw_power),
                "temperature_c": float(edge_temp),
                "sm_clock_mhz": float(gfx_clk_info.get("clk", 0)),
                "mem_clock_mhz": float(mem_clk_info.get("clk", 0)),
                "pcie_tx_mbps": -1.0,
                "pcie_rx_mbps": -1.0,
            }
        except Exception:
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
        training_idx = int(self._le.transform(["ml_training"])[0])
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
