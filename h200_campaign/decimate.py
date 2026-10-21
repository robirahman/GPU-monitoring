"""Streaming, sample-index-exact FIR decimation for the scope stream.

The 5000E's narrowest hardware bandwidth limit is 20 MHz, so streaming straight
at ~1 MS/s would fold everything between 0.5 and 20 MHz into the analysed band.
Instead the scope streams at 19.53 MS/s and this module decimates by 20 in two
linear-phase FIR stages. Each output sample k is CENTRED on input sample k*M
(group delay compensated), so decimated index k maps to raw index 20*k exactly.
"""
from __future__ import annotations

import numpy as np
from scipy import signal


def design_stage(fs_in: float, m: int, pass_hz: float, stop_hz: float, atten_db: float = 80.0):
    """Kaiser-window lowpass with an odd tap count (integer group delay)."""
    width = (stop_hz - pass_hz) / (fs_in / 2)
    n, beta = signal.kaiserord(atten_db, width)
    n |= 1
    h = signal.firwin(n, (pass_hz + stop_hz) / 2, window=("kaiser", beta), fs=fs_in)
    return h.astype(np.float64)


class StageDecimator:
    """y[k] = sum_j h[j] x[k*M + c - j], c = (L-1)/2, with x[n<0] = 0."""

    def __init__(self, h: np.ndarray, m: int):
        assert len(h) % 2 == 1
        self.h, self.m, self.c = h, m, (len(h) - 1) // 2
        self.buf = np.zeros(self.c, dtype=np.float64)   # absolute index -c .. -1
        self.buf_start = -self.c
        self.k_next = 0
        # Prepend p zeros so that (2c + p) is a multiple of M; then upfirdn's
        # down-sampling phase lands exactly on the centred outputs.
        self.p = (-2 * self.c) % m

    def feed(self, x: np.ndarray) -> np.ndarray:
        self.buf = np.concatenate([self.buf, x.astype(np.float64, copy=False)])
        buf_end = self.buf_start + self.buf.size
        k_max = (buf_end - 1 - self.c) // self.m
        if k_max < self.k_next:
            return np.empty(0)
        s = self.k_next * self.m - self.c - self.buf_start
        e = k_max * self.m + self.c - self.buf_start + 1
        seg = self.buf[s:e]
        if self.p:
            seg = np.concatenate([np.zeros(self.p), seg])
        full = signal.upfirdn(self.h, seg, up=1, down=self.m)
        first = (2 * self.c + self.p) // self.m
        y = full[first:first + (k_max - self.k_next + 1)]
        # keep only what the next output still needs
        keep_from = (k_max + 1) * self.m - self.c - self.buf_start
        self.buf = self.buf[keep_from:]
        self.buf_start += keep_from
        self.k_next = k_max + 1
        return y


class Decimator:
    """Two stages: x4 then x5, alias-protected passband 0..pass_hz."""

    def __init__(self, fs_in: float, pass_hz: float = 400e3):
        fs1 = fs_in / 4
        fs2 = fs1 / 5
        self.fs_in, self.fs_out, self.factor = fs_in, fs2, 20
        # Stage 1 only has to protect the final passband from images of fs1.
        h1 = design_stage(fs_in, 4, pass_hz, fs1 - pass_hz)
        # Stage 2: everything above fs2 - pass_hz would fold into 0..pass_hz.
        h2 = design_stage(fs1, 5, pass_hz, fs2 - pass_hz)
        self.stages = [StageDecimator(h1, 4), StageDecimator(h2, 5)]
        self.pass_hz = pass_hz
        self.taps = [len(h1), len(h2)]

    def feed(self, x: np.ndarray) -> np.ndarray:
        for st in self.stages:
            x = st.feed(x)
        return x

    def describe(self) -> dict:
        return {"fs_in_hz": self.fs_in, "fs_out_hz": self.fs_out, "factor": self.factor,
                "alias_protected_band_hz": [0.0, self.pass_hz], "stopband_atten_db": 80.0,
                "stage_taps": self.taps, "stage_factors": [4, 5], "design": "kaiser firwin, linear phase",
                "index_map": "decimated index k is centred on raw index 20*k (group delay compensated)"}
