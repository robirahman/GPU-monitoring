"""Empirical KV-handoff profile: build it from a real disagg run, sample W18's schedule from it.

Pure Python (no torch). Flow::

    disagg_kv_vllm_qwen7b  --> data/disagg_kv_vllm_qwen7b_handoffs.jsonl   (proxy log, real run)
    scripts/build_kv_profile.py  --> data/profiles/kv_handoff_<label>.json
    train_kv_disguise.py --schedule-from <profile.json>   (block sizes + gaps follow it)

The profile stores an ordered, paired sample of (handoff bytes, gap-before-handoff) so burst
structure survives sampling. Sizes are the proxy's KV-block ESTIMATES (see disagg_proxy.py);
provenance is recorded in the profile so it cannot be mistaken for a wire capture.
"""
from __future__ import annotations

import json
import random
import statistics
from typing import Iterable

PROFILE_VERSION = 1
MAX_SAMPLE = 2048


def _pct(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    i = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return float(sorted_vals[i])


def _summ(vals: list[float]) -> dict:
    s = sorted(vals)
    return {"min": _pct(s, 0), "p10": _pct(s, 0.1), "p50": _pct(s, 0.5), "p90": _pct(s, 0.9),
            "max": _pct(s, 1.0), "mean": (statistics.fmean(s) if s else 0.0)}


def build_profile(records: Iterable[dict], *, source: str, min_bytes: int = 1) -> dict:
    """Profile from proxy handoff records (status ok, ``kv_bytes_est`` > 0)."""
    recs = [r for r in records if r.get("status") == "ok" and (r.get("kv_bytes_est") or 0) >= min_bytes]
    if len(recs) < 2:
        raise ValueError(f"need >= 2 successful handoffs with kv_bytes_est, got {len(recs)}")
    recs.sort(key=lambda r: r.get("t_decode_start") or r["t"])
    starts = [r.get("t_decode_start") or r["t"] for r in recs]
    gaps = [0.0] + [max(0.0, b - a) for a, b in zip(starts, starts[1:])]
    sizes = [int(r["kv_bytes_est"]) for r in recs]
    span = max(starts[-1] - starts[0], 1e-9)
    stride = max(1, len(recs) // MAX_SAMPLE)
    sample = list(range(0, len(recs), stride))[:MAX_SAMPLE]
    return {
        "version": PROFILE_VERSION,
        "source": source,
        "provenance": "KV bytes are proxy ESTIMATES (kv_blocks x block_size x kv_bytes_per_token) "
                      "from a real disaggregated vLLM run; not a wire capture",
        "n_handoffs": len(recs),
        "kv_bytes_per_token": recs[0].get("kv_bytes_per_token"),
        "block_size": recs[0].get("block_size"),
        "prompt_tokens": _summ([float(r.get("prompt_tokens") or 0) for r in recs]),
        "size_bytes": _summ([float(x) for x in sizes]),
        "interarrival_s": _summ(gaps[1:]),
        "duty": {"handoffs_per_s": len(recs) / span, "span_s": span},
        "sizes_sample": [sizes[i] for i in sample],
        "gaps_sample_s": [round(gaps[i], 6) for i in sample],
    }


def load_profile(path: str) -> dict:
    with open(path) as f:
        p = json.load(f)
    if p.get("version") != PROFILE_VERSION or not p.get("sizes_sample"):
        raise ValueError(f"{path}: not a v{PROFILE_VERSION} KV handoff profile")
    return p


def sample_schedule(profile: dict, total_bytes: int, *, itemsize: int = 1, seed: int = 0,
                    gap_scale: float = 1.0, max_gap_s: float = 2.0) -> tuple[list[int], list[float]]:
    """Blocks (in bytes, multiples of ``itemsize``) covering ``total_bytes`` + gap before each.

    Walks the profile's ordered (size, gap) sample from a seeded random offset, wrapping,
    so the block-size distribution and burst structure follow the real handoffs. The last
    block is truncated to fit. Gaps are ``gap_scale``-scaled and capped at ``max_gap_s``.
    """
    sizes, gaps = profile["sizes_sample"], profile["gaps_sample_s"]
    rng = random.Random(seed)
    i = rng.randrange(len(sizes))
    out_sizes: list[int] = []
    out_gaps: list[float] = []
    left = total_bytes
    while left > 0:
        sz = max(itemsize, (int(sizes[i % len(sizes)]) // itemsize) * itemsize)
        sz = min(sz, left)
        out_sizes.append(sz)
        out_gaps.append(min(max_gap_s, max(0.0, gaps[i % len(gaps)] * gap_scale)))
        left -= sz
        i += 1
    out_gaps[0] = 0.0                       # no wait before the first block of a sync
    return out_sizes, out_gaps
