#!/usr/bin/env python3
"""LLM serving with a REAL arrival process -- the inference shape the corpus lacks.

WHY THIS EXISTS. Every LLM inference cell we have runs `llm_inference_vllm.py`,
whose loop is:

    while time.time() - start < duration:
        batch = [pad(p, prompt_tokens) for p in PROMPTS[...]][:batch_size]
        outputs = llm.generate(batch, SamplingParams(max_tokens=max_tokens))

Every iteration has the same batch size, the same padded prompt length, the same
output length, and a hard barrier at the end of `generate()`. The GPU therefore
goes: prefill burst -> decode ramp -> drain to idle -> repeat, at a FIXED period.
That is structurally a training loop with the backward pass deleted, and it is
almost certainly why held-out long-context inference reads as training
(section 32b: the infershard_qwen family scores 0.0185 when held out).

Real serving has none of that structure. Requests arrive asynchronously, prompt
and output lengths are heterogeneous, and continuous batching means the GPU never
drains between "steps" -- there is no period to lock onto. This script drives
vLLM's OpenAI-compatible server over HTTP with a controllable arrival process, so
the telemetry is aperiodic by construction.

Modes, chosen to span the axis the corpus is missing:

  poisson   exponential inter-arrival at --qps, lognormal prompt and output
            lengths. The realistic case.
  saturate  keep --concurrency requests in flight at all times. The queue never
            drains, so utilisation is high and FLAT -- the hardest negative,
            because it is busy like training without being periodic like it.
  bursty    alternating heavy and light phases of --burst-sec. Irregular duty
            cycle at a period far longer than a training step.
  prefill   very long prompts, max_tokens=1. Pure prefill: the single most
            training-like thing inference does, isolated.
  decode    short prompts, long generations. Memory-bandwidth-bound decode with
            almost no GEMM burst -- the opposite pole from prefill.

Uses only the stdlib for the load generator (urllib + threads), because the
node's venv-vllm is built by `pip install vllm ray` and nothing else is
guaranteed to be there.

    python workloads/llm_serving_async.py --model qwen25_7b_fp16 --mode poisson \
        --qps 4 --duration 600 --tensor-parallel-size 8 --pipeline-parallel-size 2
"""
import argparse
import json
import math
import os
import random
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from workloads.llm_inference_vllm import VLLM_MODELS, PROMPTS  # noqa: E402

MODES = ("poisson", "saturate", "bursty", "prefill", "decode")


def _server_accepts(flag, _cache={}):
    """Is `flag` in this vLLM's api_server --help? Probed once, cached."""
    if not _cache:
        try:
            h = subprocess.run(
                [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--help"],
                capture_output=True, text=True, timeout=180)
            _cache["text"] = (h.stdout or "") + (h.stderr or "")
        except Exception:
            _cache["text"] = ""          # probe failed: pass no optional flags
    return flag in _cache["text"]


def wait_for_server(base, timeout):
    """Poll /health until the engine has finished loading weights."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"{base}/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(5)
    return False


def make_prompt(rng, target_tokens):
    """Roughly `target_tokens` of text, assembled from the shared prompt pool.

    Deliberately NOT padded to an exact length: uniform prompt length is half of
    what makes the existing cells periodic. ~0.75 tokens/word is close enough for
    a load generator -- the engine reports the true count back to us.
    """
    words = max(8, int(target_tokens * 0.75))
    out = []
    while sum(len(p.split()) for p in out) < words:
        out.append(rng.choice(PROMPTS))
    return " ".join(out)


def sample_lengths(rng, mode, args):
    if mode == "prefill":
        return int(rng.uniform(args.prompt_lo, args.prompt_hi)), 1
    if mode == "decode":
        return int(rng.uniform(64, 256)), int(rng.uniform(args.out_lo, args.out_hi))
    # Lognormal in both, which is what production request mixes look like: many
    # short requests, a long tail of big ones.
    p = int(min(args.prompt_hi, max(args.prompt_lo,
                rng.lognormvariate(math.log(args.prompt_mu), args.prompt_sigma))))
    o = int(min(args.out_hi, max(args.out_lo,
                rng.lognormvariate(math.log(args.out_mu), args.out_sigma))))
    return p, o


class Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.done = self.err = self.ptok = self.otok = 0

    def add(self, ptok, otok, err=False):
        with self.lock:
            if err:
                self.err += 1
            else:
                self.done += 1
                self.ptok += ptok
                self.otok += otok


def fire(base, model_id, prompt, max_tokens, stats, timeout):
    body = json.dumps({"model": model_id, "prompt": prompt,
                       "max_tokens": max_tokens, "temperature": 0.7,
                       "top_p": 0.9}).encode()
    req = urllib.request.Request(f"{base}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read().decode())
        u = d.get("usage", {})
        stats.add(u.get("prompt_tokens", 0), u.get("completion_tokens", 0))
    except Exception:
        stats.add(0, 0, err=True)


def main():
    ap = argparse.ArgumentParser(description="vLLM serving under a real arrival process")
    ap.add_argument("--model", required=True, choices=list(VLLM_MODELS.keys()))
    ap.add_argument("--mode", default="poisson", choices=MODES)
    ap.add_argument("--duration", type=int, default=600)
    ap.add_argument("--qps", type=float, default=4.0, help="poisson/bursty arrival rate")
    ap.add_argument("--concurrency", type=int, default=64,
                    help="in-flight requests for --mode saturate, and the worker "
                         "pool size for every other mode")
    ap.add_argument("--burst-sec", type=float, default=20.0)
    ap.add_argument("--burst-ratio", type=float, default=6.0,
                    help="heavy-phase qps multiplier for --mode bursty")
    ap.add_argument("--prompt-mu", type=float, default=1200)
    ap.add_argument("--prompt-sigma", type=float, default=0.9)
    ap.add_argument("--prompt-lo", type=int, default=128)
    ap.add_argument("--prompt-hi", type=int, default=16000)
    ap.add_argument("--out-mu", type=float, default=128)
    ap.add_argument("--out-sigma", type=float, default=0.8)
    ap.add_argument("--out-lo", type=int, default=8)
    ap.add_argument("--out-hi", type=int, default=1024)
    ap.add_argument("--tensor-parallel-size", type=int, default=0)
    ap.add_argument("--pipeline-parallel-size", type=int, default=1)
    ap.add_argument("--distributed-executor-backend", default=None)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=0,
                    help="0 = let vLLM take the model's own limit. Setting this by "
                         "hand above max_position_embeddings is what killed "
                         "kv32k_out2k on every rep.")
    ap.add_argument("--enable-chunked-prefill", action="store_true")
    ap.add_argument("--enable-prefix-caching", action="store_true")
    ap.add_argument("--port", type=int, default=8077)
    ap.add_argument("--startup-timeout", type=int, default=1800)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    hf_id, dtype, _min_gpus, default_tp = VLLM_MODELS[args.model]
    tp = args.tensor_parallel_size or default_tp
    base = f"http://127.0.0.1:{args.port}"

    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
           "--model", hf_id, "--dtype", dtype, "--port", str(args.port),
           "--tensor-parallel-size", str(tp),
           "--pipeline-parallel-size", str(args.pipeline_parallel_size),
           "--gpu-memory-utilization", str(args.gpu_memory_utilization)]
    # vLLM dropped --disable-log-requests (request logging is now off unless you
    # pass --enable-log-requests), and argparse rejects unknown flags outright:
    #
    #   api_server.py: error: unrecognized arguments: --disable-log-requests
    #   ERROR: vLLM server never became healthy
    #
    # That one flag failed ALL SEVEN inferserve cells on the 2026-09-21 A100-40GB
    # pair -- the entire async-serving family, which exists specifically to stop
    # long-context inference reading as training. Each cell still burned its full
    # 1800 s startup budget before giving up. Ask the installed server which
    # spelling it has instead of guessing, and accept neither if it has neither.
    if _server_accepts("--disable-log-requests"):
        cmd += ["--disable-log-requests"]
    elif not _server_accepts("--enable-log-requests"):
        print("note: this vLLM exposes no request-logging switch", flush=True)
    if args.max_model_len:
        cmd += ["--max-model-len", str(args.max_model_len)]
    if args.distributed_executor_backend:
        cmd += ["--distributed-executor-backend", args.distributed_executor_backend]
    if args.enable_chunked_prefill:
        cmd += ["--enable-chunked-prefill"]
    if args.enable_prefix_caching:
        cmd += ["--enable-prefix-caching"]

    print(f"serving {args.model} ({hf_id}) tp={tp} pp={args.pipeline_parallel_size} "
          f"mode={args.mode}", flush=True)
    env = dict(os.environ, VLLM_USE_FLASHINFER_SAMPLER="0")
    srv = subprocess.Popen(cmd, env=env)
    try:
        if not wait_for_server(base, args.startup_timeout):
            print("ERROR: vLLM server never became healthy", file=sys.stderr)
            return 2
        print(f"server healthy after {args.startup_timeout}s budget; starting load",
              flush=True)

        rng = random.Random(args.seed)
        stats = Stats()
        sem = threading.Semaphore(args.concurrency)
        threads = []
        t0 = time.time()
        last = t0

        def launch(prompt, out_tokens):
            def run():
                try:
                    fire(base, hf_id, prompt, out_tokens, stats, timeout=600)
                finally:
                    sem.release()
            sem.acquire()
            th = threading.Thread(target=run, daemon=True)
            th.start()
            threads.append(th)

        while time.time() - t0 < args.duration:
            ptok, otok = sample_lengths(rng, args.mode, args)
            launch(make_prompt(rng, ptok), otok)

            if args.mode == "saturate":
                pass                      # the semaphore alone paces submission
            elif args.mode == "bursty":
                phase = int((time.time() - t0) / args.burst_sec) % 2
                q = args.qps * (args.burst_ratio if phase == 0 else 1.0 / args.burst_ratio)
                time.sleep(rng.expovariate(max(q, 0.05)))
            else:
                time.sleep(rng.expovariate(max(args.qps, 0.05)))

            now = time.time()
            if now - last >= 15:
                el = now - t0
                print(f"  t={el:6.0f}s  done={stats.done:6d} err={stats.err:4d}  "
                      f"in={stats.ptok/max(el,1):8.1f} tok/s  "
                      f"out={stats.otok/max(el,1):7.1f} tok/s  "
                      f"inflight~{args.concurrency - sem._value}", flush=True)
                last = now

        for th in threads:
            th.join(timeout=120)
        el = time.time() - t0
        print(f"{args.model} {args.mode} done: {stats.done} requests "
              f"({stats.err} failed), {stats.ptok} prompt tok, {stats.otok} output tok, "
              f"{el:.0f}s wall", flush=True)
        return 0
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=120)
        except subprocess.TimeoutExpired:
            srv.kill()


if __name__ == "__main__":
    sys.exit(main())
