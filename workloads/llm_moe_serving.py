#!/usr/bin/env python3
"""SPAR M4/M5 — real MoE serving via vLLM OpenAI server + traffic client.

M4: start ``vllm`` with ``--enable-expert-parallel`` (EP = TP × DP AllToAll).
M5: keep the server up and drive a mixed prefill / midfill / decode traffic mix.

Orchestrator owns the server subprocess lifecycle so the outer telemetry launcher
(``run_workload.py`` / ``run_workload_multi_gpu.py``) only needs one command.

Examples::

    # M4 smoke — small MoE, 8-way EP (TP=1, DP=8)
    python workloads/llm_moe_serving.py \\
        --model olmoe_1b7b --tp 1 --dp 8 --enable-expert-parallel \\
        --traffic mixed --duration 300 --port 8000

    # M4 Mixtral EP on 8×H100
    python workloads/llm_moe_serving.py \\
        --model mixtral_8x7b --tp 1 --dp 8 --enable-expert-parallel \\
        --traffic decode --duration 300

    # M5 production-ish mix (prefix-caching on)
    python workloads/llm_moe_serving.py \\
        --model mixtral_8x7b --tp 2 --dp 4 --enable-expert-parallel \\
        --enable-prefix-caching --traffic mixed --concurrency 8 \\
        --duration 600 --traffic-log data/serving_traffic.jsonl

    # W20: the same REAL serve + a co-tenant issuing periodic fake AllReduce (padding traffic)
    python workloads/llm_moe_serving.py \\
        --model mixtral_8x7b --tp 2 --dp 4 --enable-expert-parallel --gpu-memory-utilization 0.85 \\
        --traffic mixed --duration 300 --traffic-log data/serving.jsonl \\
        --cotenant-fake-ar --fake-ar-every-s 2 --fake-ar-mb 64 --fake-ar-log data/fake_ar.jsonl

W20 (``--cotenant-fake-ar``): after the server is healthy, ``workloads/fake_ar_cotenant.py`` is
launched (one NCCL rank per GPU) next to vLLM and AllReduces a zero buffer every
``--fake-ar-every-s`` seconds for the duration of the traffic. vLLM is untouched; the co-tenant's
own collectives are logged (``--fake-ar-log``) on the same wall clock as ``--traffic-log``, and
``<fake-ar-log stem>.window.json`` records the server-healthy / traffic start / end timestamps.
Leave GPU headroom: ``--gpu-memory-utilization 0.85``.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any


# short-name -> (HF id, default dtype string for --dtype)
SERVE_MODELS = {
    "olmoe_1b7b": ("allenai/OLMoE-1B-7B-0924", "bfloat16"),
    "qwen15_moe_a27b": ("Qwen/Qwen1.5-MoE-A2.7B-Chat", "bfloat16"),
    "mixtral_8x7b": ("mistralai/Mixtral-8x7B-Instruct-v0.1", "bfloat16"),
    "mixtral_8x22b": ("mistralai/Mixtral-8x22B-Instruct-v0.1", "bfloat16"),
    "qwen25_7b": ("Qwen/Qwen2.5-7B-Instruct", "bfloat16"),  # dense control
}

_FILLER = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
    "mike november oscar papa quebec romeo sierra tango uniform victor whiskey "
    "xray yankee zulu "
)

_BASE_PROMPTS = [
    "Summarise the plot and themes of War and Peace focusing on Pierre.",
    "Explain how a transformer processes a long context window with KV cache.",
    "Analyse the 2008 financial crisis and the role of MBS and CDS.",
    "Describe eukaryotic protein synthesis from transcription to trafficking.",
    "Give a technical overview of modern GPU architecture and HBM.",
    "Compare consequentialism, deontology, and virtue ethics briefly.",
]


def _pad_words(base: str, approx_tokens: int) -> str:
    """Rough pad: ~1.3 words/token. Good enough for regime shaping."""
    if approx_tokens <= 0:
        return base
    need = max(0, int(approx_tokens * 1.3) - len(base.split()))
    words = _FILLER.split()
    extra: list[str] = []
    while len(extra) < need:
        extra.extend(words)
    return base + " " + " ".join(extra[:need])


def _post_json(url: str, payload: dict, timeout: float = 600.0) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get(url: str, timeout: float = 5.0) -> int:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status
    except Exception:
        return -1


def wait_healthy(base: str, timeout_s: float) -> bool:
    """Poll /v1/models until ready or timeout."""
    deadline = time.time() + timeout_s
    url = f"{base}/v1/models"
    while time.time() < deadline:
        if _get(url) == 200:
            return True
        time.sleep(2.0)
    return False


def build_server_cmd(args: argparse.Namespace, model_id: str, dtype: str) -> list[str]:
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_id,
        "--dtype", dtype,
        "--host", args.host,
        "--port", str(args.port),
        "--tensor-parallel-size", str(args.tp),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--max-model-len", str(args.max_model_len),
        "--trust-remote-code",
    ]
    if args.dp > 1:
        cmd.extend(["--data-parallel-size", str(args.dp)])
    if args.enable_expert_parallel:
        cmd.append("--enable-expert-parallel")
    if args.enable_prefix_caching:
        cmd.append("--enable-prefix-caching")
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    if args.max_num_seqs is not None:
        cmd.extend(["--max-num-seqs", str(args.max_num_seqs)])
    if args.extra_vllm_arg:
        cmd.extend(args.extra_vllm_arg)
    return cmd


def _role_tags() -> dict[str, str]:
    """W25: tags from scripts/run_colocate_serve_train.py (empty when not colocated)."""
    m = {"workload_role": "SPAR_WORKLOAD_ROLE", "role_label": "SPAR_ROLE_LABEL", "run_id": "SPAR_RUN_ID"}
    return {k: os.environ[v] for k, v in m.items() if os.environ.get(v)}


def build_cotenant_cmd(args: argparse.Namespace, world: int) -> list[str]:
    """Command for the W20 fake-AllReduce co-tenant (one NCCL rank per GPU)."""
    here = os.path.dirname(os.path.abspath(__file__))
    with socket.socket() as sk:                    # explicit loopback rendezvous (single node): robust
        sk.bind(("127.0.0.1", 0))                  # where --standalone resolves the hostname badly
        port = sk.getsockname()[1]
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nnodes=1",
           f"--nproc_per_node={args.fake_ar_gpus or world}",
           "--master_addr=127.0.0.1", f"--master_port={port}",
           os.path.join(here, "fake_ar_cotenant.py"), "--backend", args.fake_ar_backend,
           "--every-s", str(args.fake_ar_every_s), "--mb", str(args.fake_ar_mb),
           "--burst", str(args.fake_ar_burst),
           "--duration", str(args.duration + 20)]
    if args.fake_ar_log:
        cmd += ["--comm-log", args.fake_ar_log]
    return cmd


def pick_request(traffic: str, args: argparse.Namespace, turn: int) -> tuple[str, dict[str, Any]]:
    """Return (regime_label, openai_chat_body_fields besides model)."""
    base = _BASE_PROMPTS[turn % len(_BASE_PROMPTS)]

    if traffic == "mixed":
        # Weighted toward decode (interactive) with some prefill + midfill.
        regime = random.choices(
            ["prefill", "decode", "midfill"],
            weights=[0.25, 0.45, 0.30],
            k=1,
        )[0]
    else:
        regime = traffic

    if regime == "prefill":
        prompt = _pad_words(base, args.prefill_input_len)
        max_tokens = args.prefill_max_tokens
    elif regime == "decode":
        prompt = _pad_words(base, args.decode_input_len)
        max_tokens = args.decode_max_tokens
    else:  # midfill — long shared-looking prefix + short new turn
        prefix = _pad_words("SYSTEM CONTEXT:\n" + base + "\n", args.midfill_prefix_len)
        append = _pad_words(
            f"\n[user turn {turn}] Please continue with a short answer:\n",
            args.midfill_append_len,
        )
        prompt = prefix + append
        max_tokens = args.midfill_max_tokens

    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": False,
    }
    return regime, body


def run_traffic(
    args: argparse.Namespace,
    model_id: str,
    stop_event: threading.Event,
    log_fh,
) -> dict[str, int]:
    base = f"http://{args.host}:{args.port}"
    url = f"{base}/v1/chat/completions"
    counts = {"prefill": 0, "decode": 0, "midfill": 0, "error": 0}
    lock = threading.Lock()
    turn = 0
    turn_lock = threading.Lock()

    def one_req(_: int) -> None:
        nonlocal turn
        if stop_event.is_set():
            return
        with turn_lock:
            my_turn = turn
            turn += 1
        regime, body = pick_request(args.traffic, args, my_turn)
        body["model"] = model_id
        t0 = time.time()
        try:
            resp = _post_json(url, body, timeout=args.request_timeout)
            latency = time.time() - t0
            usage = resp.get("usage") or {}
            out_tok = int(usage.get("completion_tokens") or 0)
            in_tok = int(usage.get("prompt_tokens") or 0)
            with lock:
                counts[regime] = counts.get(regime, 0) + 1
            if log_fh is not None:
                log_fh.write(json.dumps({
                    **_role_tags(),
                    "t": time.time(),
                    "regime": regime,
                    "prompt_tokens": in_tok,
                    "completion_tokens": out_tok,
                    "latency_s": round(latency, 4),
                    "turn": my_turn,
                }, separators=(",", ":")) + "\n")
        except Exception as e:
            with lock:
                counts["error"] += 1
            if log_fh is not None:
                log_fh.write(json.dumps({
                    **_role_tags(),
                    "t": time.time(),
                    "regime": regime,
                    "error": str(e)[:200],
                    "turn": my_turn,
                }, separators=(",", ":")) + "\n")

    deadline = time.time() + args.duration
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        while time.time() < deadline and not stop_event.is_set():
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            # Keep the pool saturated with a burst of concurrent requests.
            futs = [pool.submit(one_req, i) for i in range(args.concurrency)]
            for f in as_completed(futs, timeout=max(remaining, 1.0) + args.request_timeout):
                try:
                    f.result()
                except Exception:
                    pass
            if args.inter_burst_sleep > 0:
                time.sleep(args.inter_burst_sleep)

    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description="SPAR M4/M5 MoE vLLM serving + traffic")
    ap.add_argument("--model", required=True, choices=list(SERVE_MODELS.keys()))
    ap.add_argument("--tp", type=int, default=1, help="tensor parallel size")
    ap.add_argument("--dp", type=int, default=1, help="data parallel size (EP=TP*DP when EP on)")
    ap.add_argument("--enable-expert-parallel", action="store_true",
                    help="M4: shard MoE experts → AllToAll (EP=TP×DP)")
    ap.add_argument("--enable-prefix-caching", action="store_true",
                    help="Help midfill reuse prefix KV")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--duration", type=int, default=300,
                    help="Traffic duration seconds (after server healthy)")
    ap.add_argument("--server-ready-timeout", type=int, default=1800,
                    help="Max wait for /v1/models (weight download can be slow)")
    ap.add_argument("--traffic", choices=["prefill", "decode", "midfill", "mixed"],
                    default="mixed")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--inter-burst-sleep", type=float, default=0.0)
    ap.add_argument("--request-timeout", type=float, default=600.0)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--max-num-seqs", type=int, default=None)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--dtype", default="", help="Override model default dtype")
    # Regime shape knobs
    ap.add_argument("--prefill-input-len", type=int, default=4096)
    ap.add_argument("--prefill-max-tokens", type=int, default=16)
    ap.add_argument("--decode-input-len", type=int, default=512)
    ap.add_argument("--decode-max-tokens", type=int, default=1024)
    ap.add_argument("--midfill-prefix-len", type=int, default=8192)
    ap.add_argument("--midfill-append-len", type=int, default=256)
    ap.add_argument("--midfill-max-tokens", type=int, default=128)
    ap.add_argument("--traffic-log", default="",
                    help="JSONL path for per-request regime labels (STEP truth)")
    ap.add_argument("--extra-vllm-arg", action="append", default=[],
                    help="Passthrough raw arg to vLLM server (repeatable)")
    ap.add_argument("--cotenant-fake-ar", action="store_true",
                    help="W20: run a fake-AllReduce co-tenant (workloads/fake_ar_cotenant.py) next to the serve")
    ap.add_argument("--fake-ar-every-s", type=float, default=2.0)
    ap.add_argument("--fake-ar-mb", type=float, default=64.0)
    ap.add_argument("--fake-ar-burst", type=int, default=1)
    ap.add_argument("--fake-ar-log", default="", help="co-tenant collective JSONL (rank0)")
    ap.add_argument("--fake-ar-gpus", type=int, default=0, help="co-tenant ranks (default TP*DP)")
    ap.add_argument("--fake-ar-backend", choices=["auto", "nccl", "gloo"], default="auto",
                    help="gloo only for CPU smoke tests")
    ap.add_argument("--fake-ar-warmup-s", type=float, default=15.0,
                    help="wait after launching the co-tenant so its NCCL init finishes before traffic")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print server command and exit")
    args = ap.parse_args()

    model_id, default_dtype = SERVE_MODELS[args.model]
    dtype = args.dtype or default_dtype
    world = args.tp * args.dp
    ep = world if args.enable_expert_parallel else 0

    print(f"llm_moe_serving model={args.model} ({model_id})")
    print(f"  TP={args.tp} DP={args.dp} world_gpus={world} "
          f"EP={'on EP_size='+str(ep) if args.enable_expert_parallel else 'off'}")
    print(f"  traffic={args.traffic} concurrency={args.concurrency} duration={args.duration}s")

    cmd = build_server_cmd(args, model_id, dtype)
    print("  server cmd:", " ".join(cmd))
    cot_cmd = build_cotenant_cmd(args, world) if args.cotenant_fake_ar else None
    if cot_cmd:
        print("  co-tenant cmd:", " ".join(cot_cmd))
    if args.dry_run:
        return

    os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    log_fh = None
    if args.traffic_log:
        os.makedirs(os.path.dirname(args.traffic_log) or ".", exist_ok=True)
        log_fh = open(args.traffic_log, "a", buffering=1)

    # Start server
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    stop_log = threading.Event()
    cotenant: dict[str, Any] = {"proc": None}

    def _pump_stdout() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            if stop_log.is_set():
                break
            sys.stdout.write("[vllm] " + line)
            sys.stdout.flush()

    pump = threading.Thread(target=_pump_stdout, daemon=True)
    pump.start()

    def _cleanup(signum=None, frame=None) -> None:
        print(f"\nShutting down vLLM server (signal={signum}) …")
        cp = cotenant["proc"]
        if cp is not None and cp.poll() is None:
            cp.send_signal(signal.SIGTERM)
            try:
                cp.wait(timeout=20)
            except subprocess.TimeoutExpired:
                cp.kill()
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
        if log_fh is not None:
            log_fh.close()
        if signum is not None:
            sys.exit(128 + (signum if isinstance(signum, int) else 0))

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    base = f"http://{args.host}:{args.port}"
    print(f"Waiting for server at {base}/v1/models (timeout {args.server_ready_timeout}s) …")
    if not wait_healthy(base, args.server_ready_timeout):
        print("ERROR: server did not become healthy in time", file=sys.stderr)
        if proc.poll() is None:
            proc.kill()
        # Dump last chance: process already exited?
        rc = proc.poll()
        print(f"  server rc={rc}", file=sys.stderr)
        sys.exit(1)

    print("Server healthy — starting traffic")
    window = {"t_server_healthy": time.time()}
    if cot_cmd:
        cotenant["proc"] = subprocess.Popen(cot_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                            text=True, bufsize=1)
        threading.Thread(target=lambda: [sys.stdout.write("[fake_ar] " + ln)
                                         for ln in cotenant["proc"].stdout], daemon=True).start()
        window["t_cotenant_start"] = time.time()
        window["cotenant_cmd"] = cot_cmd
        time.sleep(args.fake_ar_warmup_s)   # let the co-tenant's NCCL init finish before traffic
    window["t_traffic_start"] = time.time()
    stop_traffic = threading.Event()
    try:
        counts = run_traffic(args, model_id, stop_traffic, log_fh)
    finally:
        window["t_traffic_end"] = time.time()
        stop_traffic.set()
        _cleanup()
        if cot_cmd and args.fake_ar_log:
            wpath = os.path.splitext(args.fake_ar_log)[0] + ".window.json"
            os.makedirs(os.path.dirname(wpath) or ".", exist_ok=True)
            with open(wpath, "w") as wf:
                json.dump({"model": args.model, "tp": args.tp, "dp": args.dp, "ep": args.enable_expert_parallel,
                           "fake_ar_every_s": args.fake_ar_every_s, "fake_ar_mb": args.fake_ar_mb,
                           "traffic_log": args.traffic_log, **window}, wf, indent=2)

    print(f"Traffic done: {counts}")
    print(
        f"llm_moe_serving complete: model={args.model} "
        f"TP={args.tp} DP={args.dp} EP={args.enable_expert_parallel} "
        f"traffic={args.traffic}"
    )


if __name__ == "__main__":
    main()
