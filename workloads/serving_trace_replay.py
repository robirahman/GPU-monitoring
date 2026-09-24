#!/usr/bin/env python3
"""Replay Azure / BurstGPT serving traces against a live vLLM OpenAI server.

This is the *loadgen only* — it does NOT collect GPU telemetry and does NOT
start vLLM. Pipeline:

    CSV (arrival + in/out lengths)
      → pad dummy prompts to ContextTokens / Request tokens
      → POST /v1/completions  (max_tokens = GeneratedTokens, ignore_eos)
      → vLLM continuous-batches on the GPU
      → (external) TelemetryCollector / run_workload.py records NVML

Examples:
    # vLLM already running on :8000
    python workloads/serving_trace_replay.py \\
        --trace data/external/llm_serving_traces/azure/AzureLLMInferenceTrace_conv_2023.csv \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --duration 300 --speed-factor 10 --max-concurrency 32

    # BurstGPT API slice, wall-clock capped
    python workloads/serving_trace_replay.py \\
        --trace data/external/llm_serving_traces/burstgpt/BurstGPT_without_fails_1.csv \\
        --dataset burstgpt --log-type api \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --duration 600 --speed-factor 50 --max-concurrency 64
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional


# ---------------------------------------------------------------------------
# Trace rows
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TraceRequest:
    """One production request, stripped to what replay needs."""
    t_sec: float          # seconds from first request in the loaded slice
    in_tokens: int
    out_tokens: int
    meta: str = ""        # free-form tag (model / log type / …)


def _parse_ts(value: str) -> float:
    """Azure wall-clock or BurstGPT relative-seconds → epoch-ish float."""
    value = value.strip()
    try:
        return float(value)
    except ValueError:
        pass
    # Azure: "2023-11-16 18:15:46.6805900" (up to 7 fractional digits)
    for fmt in (
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            # trim to 6 fractional digits for datetime
            if "." in value:
                head, frac = value.split(".", 1)
                value_norm = f"{head}.{frac[:6]}"
            else:
                value_norm = value
            return datetime.strptime(value_norm, fmt).timestamp()
        except ValueError:
            continue
    raise ValueError(f"unparseable timestamp: {value!r}")


def _detect_dataset(path: Path, explicit: Optional[str]) -> str:
    if explicit and explicit != "auto":
        return explicit
    with path.open(newline="") as f:
        header = next(csv.reader(f))
    cols = {c.strip().lower() for c in header}
    if "contexttokens" in cols and "generatedtokens" in cols:
        return "azure"
    if "request tokens" in cols and "response tokens" in cols:
        return "burstgpt"
    raise ValueError(
        f"cannot auto-detect dataset from columns {header}; "
        f"pass --dataset azure|burstgpt"
    )


def load_trace(
    path: Path,
    dataset: str = "auto",
    log_type: Optional[str] = None,
    model_filter: Optional[str] = None,
    max_requests: Optional[int] = None,
    start_offset_sec: float = 0.0,
    window_sec: Optional[float] = None,
) -> list[TraceRequest]:
    """Load and normalize a trace slice.

    Timestamps are rebased so the first kept request is t=0.
    `start_offset_sec` / `window_sec` select a wall-clock window *after*
    rebasing to the file's first row (BurstGPT) or absolute time (Azure).
    """
    dataset = _detect_dataset(path, dataset)
    rows: list[tuple[float, int, int, str]] = []

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            # normalize keys (BurstGPT has spaces)
            row = {k.strip().lower(): (v.strip() if isinstance(v, str) else v)
                   for k, v in raw.items() if k}

            if dataset == "azure":
                t = _parse_ts(row["timestamp"])
                inn = int(float(row["contexttokens"]))
                out = int(float(row["generatedtokens"]))
                meta = "azure"
            else:
                t = _parse_ts(row["timestamp"])
                inn = int(float(row["request tokens"]))
                out = int(float(row["response tokens"]))
                lt = row.get("log type", "")
                model = row.get("model", "")
                if log_type:
                    want = log_type.lower().replace("_", " ")
                    # accept "api" / "conversation" shorthand
                    if want in ("api", "api log") and "api" not in lt.lower():
                        continue
                    if want in ("conversation", "conversation log", "conv") and \
                            "conversation" not in lt.lower():
                        continue
                if model_filter and model_filter.lower() not in model.lower():
                    continue
                if out <= 0:
                    continue
                meta = f"{model}|{lt}"

            if inn <= 0 or out <= 0:
                continue
            rows.append((t, inn, out, meta))

    if not rows:
        raise RuntimeError(f"no rows kept from {path}")

    rows.sort(key=lambda r: r[0])
    t0 = rows[0][0]
    # absolute → relative
    rel = [(t - t0, inn, out, meta) for t, inn, out, meta in rows]

    if start_offset_sec > 0 or window_sec is not None:
        lo = start_offset_sec
        hi = start_offset_sec + window_sec if window_sec is not None else float("inf")
        rel = [(t, inn, out, meta) for t, inn, out, meta in rel if lo <= t < hi]
        if not rel:
            raise RuntimeError(
                f"window [{lo}, {hi}) empty after filter; "
                f"file span is 0..{rows[-1][0] - t0:.0f}s"
            )
        t0w = rel[0][0]
        rel = [(t - t0w, inn, out, meta) for t, inn, out, meta in rel]

    if max_requests is not None:
        rel = rel[:max_requests]

    return [TraceRequest(t, inn, out, meta) for t, inn, out, meta in rel]


# ---------------------------------------------------------------------------
# Prompt padding
# ---------------------------------------------------------------------------

_tokenizer = None
_tokenizer_failed = False


def _get_tokenizer(model_id: str):
    global _tokenizer, _tokenizer_failed
    if _tokenizer is not None or _tokenizer_failed:
        return _tokenizer
    try:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True, use_fast=True,
        )
        print(f"[loadgen] tokenizer loaded for {model_id}", flush=True)
    except Exception as e:
        print(f"[loadgen] tokenizer unavailable ({e}); using char heuristic",
              flush=True)
        _tokenizer_failed = True
        _tokenizer = None
    return _tokenizer


def pad_prompt(n_tokens: int, model_id: str, seed_text: str = "hello") -> str:
    """Build a prompt whose tokenized length is ≈ n_tokens."""
    n_tokens = max(1, n_tokens)
    tok = _get_tokenizer(model_id)
    if tok is None:
        # ~1 token ≈ 4 chars for English BPE; overshoot then trim by chars
        return (seed_text + " ") * max(1, n_tokens)

    # Grow by repetition, then trim with tokenizer
    unit = seed_text + " "
    text = unit
    # exponential grow
    while len(tok.encode(text, add_special_tokens=False)) < n_tokens:
        text = text + text
        if len(text) > n_tokens * 20:
            break
    ids = tok.encode(text, add_special_tokens=False)[:n_tokens]
    return tok.decode(ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# HTTP client (stdlib — no openai/httpx dependency)
# ---------------------------------------------------------------------------

@dataclass
class RequestResult:
    ok: bool
    latency_s: float
    in_tokens: int
    out_tokens: int
    error: str = ""


def post_completion(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout_s: float,
    temperature: float = 0.0,
) -> RequestResult:
    url = base_url.rstrip("/") + "/v1/completions"
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": int(max_tokens),
        "temperature": temperature,
        # vLLM extension: force full generation length (ignore EOS)
        "ignore_eos": True,
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            _ = resp.read()
        return RequestResult(
            ok=True,
            latency_s=time.perf_counter() - t0,
            in_tokens=0,  # filled by caller
            out_tokens=max_tokens,
        )
    except Exception as e:
        return RequestResult(
            ok=False,
            latency_s=time.perf_counter() - t0,
            in_tokens=0,
            out_tokens=max_tokens,
            error=str(e),
        )


def wait_server(base_url: str, timeout_s: float = 300.0) -> None:
    """Block until GET /v1/models succeeds."""
    url = base_url.rstrip("/") + "/v1/models"
    deadline = time.time() + timeout_s
    last_err = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    print(f"[loadgen] server ready at {base_url}", flush=True)
                    return
        except Exception as e:
            last_err = str(e)
        time.sleep(2)
    raise RuntimeError(f"vLLM not ready at {base_url} after {timeout_s}s: {last_err}")


# ---------------------------------------------------------------------------
# Replay engine
# ---------------------------------------------------------------------------

class Stats:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.submitted = 0
        self.ok = 0
        self.fail = 0
        self.latencies: list[float] = []

    def record(self, r: RequestResult) -> None:
        with self.lock:
            self.submitted += 1
            if r.ok:
                self.ok += 1
                self.latencies.append(r.latency_s)
            else:
                self.fail += 1

    def snapshot(self) -> str:
        with self.lock:
            n = self.ok
            p50 = sorted(self.latencies)[n // 2] if n else 0.0
            return (f"ok={self.ok} fail={self.fail} "
                    f"in_flight_done={self.submitted} p50_lat={p50:.2f}s")


def replay(
    requests: list[TraceRequest],
    *,
    base_url: str,
    model: str,
    speed_factor: float,
    max_concurrency: int,
    duration_s: Optional[float],
    request_timeout_s: float,
    prompt_cache: dict[int, str],
) -> Stats:
    """Fire requests on a scaled timeline. speed_factor>1 = faster than realtime."""
    stats = Stats()
    wall0 = time.perf_counter()
    stop_at = wall0 + duration_s if duration_s else None

    def _prompt_for(n: int) -> str:
        if n not in prompt_cache:
            prompt_cache[n] = pad_prompt(n, model)
        return prompt_cache[n]

    def _send(tr: TraceRequest) -> RequestResult:
        prompt = _prompt_for(tr.in_tokens)
        r = post_completion(
            base_url, model, prompt, tr.out_tokens, request_timeout_s,
        )
        r.in_tokens = tr.in_tokens
        stats.record(r)
        return r

    print(
        f"[loadgen] replaying {len(requests)} reqs  "
        f"speed_factor={speed_factor}  max_concurrency={max_concurrency}  "
        f"duration={duration_s or 'trace'}",
        flush=True,
    )
    print(
        f"[loadgen] trace span={requests[-1].t_sec:.1f}s → "
        f"wall≈{requests[-1].t_sec / speed_factor:.1f}s",
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        inflight: list[Future] = []
        last_report = wall0

        for i, tr in enumerate(requests):
            now = time.perf_counter()
            if stop_at is not None and now >= stop_at:
                print(f"[loadgen] duration cap hit after {i} submits", flush=True)
                break

            target = wall0 + (tr.t_sec / speed_factor)
            sleep_for = target - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)

            # backpressure: don't pile unbounded futures
            while True:
                inflight = [f for f in inflight if not f.done()]
                if len(inflight) < max_concurrency:
                    break
                time.sleep(0.01)

            inflight.append(pool.submit(_send, tr))

            if time.perf_counter() - last_report >= 5.0:
                print(f"[loadgen] {stats.snapshot()}  submitted={i+1}", flush=True)
                last_report = time.perf_counter()

        # drain
        for f in inflight:
            f.result()

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Replay Azure/BurstGPT traces against vLLM (loadgen only)",
    )
    ap.add_argument("--trace", type=str, required=True, help="Path to CSV trace")
    ap.add_argument("--dataset", choices=["auto", "azure", "burstgpt"], default="auto")
    ap.add_argument("--log-type", type=str, default=None,
                    help="BurstGPT only: api | conversation")
    ap.add_argument("--model-filter", type=str, default=None,
                    help="BurstGPT only: substring match on Model column")
    ap.add_argument("--base-url", type=str, default="http://127.0.0.1:8000")
    ap.add_argument("--model", type=str, required=True,
                    help="Model id as registered in vLLM (usually HF id)")
    ap.add_argument("--speed-factor", type=float, default=1.0,
                    help=">1 replays faster than realtime (e.g. 10 = 10×)")
    ap.add_argument("--max-concurrency", type=int, default=32)
    ap.add_argument("--duration", type=float, default=None,
                    help="Wall-clock seconds to run (truncate replay)")
    ap.add_argument("--max-requests", type=int, default=None)
    ap.add_argument("--start-offset-sec", type=float, default=0.0,
                    help="Skip this many seconds into the trace (pre-rebase window)")
    ap.add_argument("--window-sec", type=float, default=None,
                    help="Keep only this many seconds of trace after offset")
    ap.add_argument("--request-timeout", type=float, default=300.0)
    ap.add_argument("--wait-server", action="store_true",
                    help="Block until /v1/models is up before replaying")
    ap.add_argument("--wait-server-timeout", type=float, default=600.0)
    ap.add_argument("--no-tokenizer", action="store_true",
                    help="Skip HF tokenizer; use char heuristic for padding")
    args = ap.parse_args()

    trace_path = Path(args.trace)
    if not trace_path.is_file():
        # allow repo-relative
        alt = Path(__file__).resolve().parents[1] / args.trace
        if alt.is_file():
            trace_path = alt
        else:
            print(f"ERROR: trace not found: {args.trace}", file=sys.stderr)
            sys.exit(1)

    if args.no_tokenizer:
        global _tokenizer_failed
        _tokenizer_failed = True

    reqs = load_trace(
        trace_path,
        dataset=args.dataset,
        log_type=args.log_type,
        model_filter=args.model_filter,
        max_requests=args.max_requests,
        start_offset_sec=args.start_offset_sec,
        window_sec=args.window_sec,
    )
    print(
        f"[loadgen] loaded {len(reqs)} requests from {trace_path.name}  "
        f"in=[{min(r.in_tokens for r in reqs)},{max(r.in_tokens for r in reqs)}]  "
        f"out=[{min(r.out_tokens for r in reqs)},{max(r.out_tokens for r in reqs)}]",
        flush=True,
    )

    if args.wait_server:
        wait_server(args.base_url, args.wait_server_timeout)

    prompt_cache: dict[int, str] = {}
    t0 = time.perf_counter()
    stats = replay(
        reqs,
        base_url=args.base_url,
        model=args.model,
        speed_factor=max(args.speed_factor, 1e-6),
        max_concurrency=args.max_concurrency,
        duration_s=args.duration,
        request_timeout_s=args.request_timeout,
        prompt_cache=prompt_cache,
    )
    elapsed = time.perf_counter() - t0
    print(
        f"[loadgen] done in {elapsed:.1f}s  {stats.snapshot()}  "
        f"unique_prompt_lens={len(prompt_cache)}",
        flush=True,
    )
    if stats.fail and stats.ok == 0:
        sys.exit(2)


if __name__ == "__main__":
    main()
