#!/usr/bin/env python3
"""W10 (primary) -- REAL disaggregated prefill<->decode serving with vLLM + NIXL.

Two pools of real vLLM engines serving a real model (default Qwen2.5-7B-Instruct):
  * prefillers  ``kv_role=kv_producer``  -- compute the prompt KV cache
  * decoders    ``kv_role=kv_consumer``  -- pull that KV cache from the prefiller GPU over
    NIXL/UCX (NVLink inside a node) and generate
tied together by ``disagg_proxy.py`` (the documented vLLM two-step request flow). The KV
bytes on the wire are real KV cache; nothing is synthetic. Supersedes
``disagg_kv_synth.py`` (now a MICROBENCH). vLLM's docs mark this feature *experimental*.

Recipe (vLLM "NixlConnector usage", single node)::

    CUDA_VISIBLE_DEVICES=0 UCX_NET_DEVICES=all VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \\
      vllm serve <model> --port 8100 \\
        --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer",...}'
    (decoder: kv_role=kv_consumer, its own GPU / port 8200 / side-channel port)

This orchestrator builds exactly those commands per engine (``--num-prefill`` /
``--num-decode`` engines, ``--tp`` GPUs each), starts them, waits for health, runs the
proxy in-process, drives closed-loop traffic with a prompt-length mix (each prompt gets a
unique nonce so vLLM's prefix cache cannot hide the KV handoff), and shuts everything down.

What is and is not logged (deliberately no invented collective log):
  * ``--handoff-log``  per-request proxy JSONL: prefill/decode timestamps, prompt tokens,
    KV blocks the prefiller reported handing over, ``kv_bytes_est`` (ESTIMATE: blocks x block
    size x KV bytes/token). Not a wire measurement.
  * ``--traffic-log``  client-side per-request log (same shape as llm_moe_serving).
  * Fabric truth = the NVLink / IB counters the telemetry launcher records for the same
    window (``scripts/run_workload_multi_gpu.py``); the KV transfer is not NCCL so there is
    no NCCL JSONL. ``<handoff-log stem>.meta.json`` records commands, GPU map and versions.
  * ``scripts/build_kv_profile.py`` turns the handoff log into the profile W18 mimics.

Launch (8 GPUs: 4 prefill + 4 decode)::

    python scripts/run_workload_multi_gpu.py --workload disagg_kv_vllm_qwen7b --num-gpus 8
    bash scripts/run_disagg_kv_vcf.sh qwen7b            # also builds the W18 profile
    python workloads/disagg_kv_vllm.py --dry-run --num-prefill 1 --num-decode 1   # print commands

Status: code (real). Not run on GPU from this repo yet; needs ``pip install vllm nixl``.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import signal
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from importlib import metadata
from importlib.util import find_spec

from disagg_proxy import DisaggProxy
from kv_geometry import SERVE_MODEL_KV, kv_bytes_per_token
from llm_moe_serving import SERVE_MODELS, _BASE_PROMPTS, _get, _pad_words, _post_json


@dataclass
class EngineSpec:
    role: str                       # prefill | decode
    idx: int
    gpus: list[int]
    http_port: int
    side_port: int
    cmd: list[str]
    env: dict[str, str]
    name: str = ""
    proc: subprocess.Popen | None = field(default=None, repr=False)
    tail: collections.deque = field(default_factory=lambda: collections.deque(maxlen=60), repr=False)


def kv_transfer_config(role: str, load_failure_policy: str, buffer_device: str) -> str:
    kv_role = {"prefill": "kv_producer", "decode": "kv_consumer"}[role]
    cfg = {"kv_connector": "NixlConnector", "kv_role": kv_role,
           "kv_load_failure_policy": load_failure_policy}
    if buffer_device:
        cfg["kv_buffer_device"] = buffer_device
    return json.dumps(cfg, separators=(",", ":"))


def build_engine_specs(args: argparse.Namespace, model_id: str, dtype: str) -> list[EngineSpec]:
    """Pure function: the exact vLLM commands / env per engine (unit-tested, used by --dry-run)."""
    specs: list[EngineSpec] = []
    gpu = args.first_gpu
    plan = [("prefill", i, args.prefill_base_port + i) for i in range(args.num_prefill)] + \
           [("decode", j, args.decode_base_port + j) for j in range(args.num_decode)]
    for k, (role, idx, http_port) in enumerate(plan):
        gpus = list(range(gpu, gpu + args.tp))
        gpu += args.tp
        side = args.side_channel_base_port + 8 * k          # NIXL uses side_port + tp_rank
        cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
               "--model", model_id, "--dtype", dtype,
               "--host", args.host, "--port", str(http_port),
               "--tensor-parallel-size", str(args.tp),
               "--gpu-memory-utilization", str(args.gpu_memory_utilization),
               "--max-model-len", str(args.max_model_len),
               "--kv-transfer-config", kv_transfer_config(role, args.kv_load_failure_policy,
                                                          args.kv_buffer_device)]
        if args.enforce_eager:
            cmd.append("--enforce-eager")
        cmd.extend(args.extra_vllm_arg)
        env = {"CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus)),
               "UCX_NET_DEVICES": "all",
               "VLLM_NIXL_SIDE_CHANNEL_HOST": "localhost",
               "VLLM_NIXL_SIDE_CHANNEL_PORT": str(side),
               "VLLM_WORKER_MULTIPROC_METHOD": "spawn"}
        specs.append(EngineSpec(role, idx, gpus, http_port, side, cmd, env, name=f"{role}{idx}"))
    return specs


def parse_len_mix(spec: str) -> list[tuple[int, float]]:
    """"512:0.3,2048:0.5,8192:0.2" -> [(512, .3), (2048, .5), (8192, .2)]."""
    mix = []
    for part in spec.split(","):
        n, _, w = part.partition(":")
        mix.append((int(n), float(w or 1.0)))
    if not mix or any(n <= 0 or w <= 0 for n, w in mix):
        raise SystemExit(f"bad --prompt-len-mix {spec!r}")
    return mix


def make_body(model_id: str, prompt_tokens: int, max_tokens: int, turn: int) -> dict:
    nonce = uuid.uuid4().hex
    base = _BASE_PROMPTS[turn % len(_BASE_PROMPTS)]
    # nonce FIRST: vLLM prefix caching keys on the leading tokens, and identical prompts would
    # otherwise collapse the handoff to (almost) nothing.
    prompt = f"[session {nonce}] " + _pad_words(base, prompt_tokens)
    return {"model": model_id, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.7, "stream": False}


def run_traffic(args, model_id: str, proxy_url: str, log_fh, stop: threading.Event) -> dict:
    mix = parse_len_mix(args.prompt_len_mix)
    lens, weights = [m[0] for m in mix], [m[1] for m in mix]
    counts = {"ok": 0, "error": 0}
    lock = threading.Lock()
    turn = [0]
    deadline = time.time() + args.duration

    def worker(wid: int) -> None:
        rng = random.Random(args.seed * 1000 + wid)
        while time.time() < deadline and not stop.is_set():
            with lock:
                my_turn = turn[0]
                turn[0] += 1
            n = rng.choices(lens, weights)[0]
            body = make_body(model_id, n, args.max_tokens, my_turn)
            t0 = time.time()
            try:
                resp = _post_json(proxy_url, body, timeout=args.request_timeout)
                usage = resp.get("usage") or {}
                rec = {"t": time.time(), "regime": "disagg", "target_prompt_tokens": n,
                       "prompt_tokens": usage.get("prompt_tokens"),
                       "completion_tokens": usage.get("completion_tokens"),
                       "latency_s": round(time.time() - t0, 4), "turn": my_turn}
                with lock:
                    counts["ok"] += 1
            except Exception as e:                                # noqa: BLE001
                rec = {"t": time.time(), "regime": "disagg", "error": str(e)[:200], "turn": my_turn}
                with lock:
                    counts["error"] += 1
            if log_fh is not None:
                with lock:
                    log_fh.write(json.dumps(rec, separators=(",", ":")) + "\n")

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for w in range(args.concurrency):
            pool.submit(worker, w)
    return counts


def preflight() -> dict:
    """Fail fast (with fixes) if the serving stack is missing. Does not import vllm."""
    info = {}
    for mod, dist_name, fix in (("vllm", "vllm", "pip install vllm"),
                                ("nixl", "nixl", "pip install nixl   (vLLM NixlConnector dependency)")):
        if find_spec(mod) is None:
            raise SystemExit(f"preflight: python module '{mod}' not found -> {fix}")
        try:
            info[mod] = metadata.version(dist_name)
        except metadata.PackageNotFoundError:
            info[mod] = "unknown"
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description="W10 real disaggregated prefill/decode (vLLM NixlConnector)",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen25_7b", choices=list(SERVE_MODELS))
    ap.add_argument("--num-prefill", type=int, default=4)
    ap.add_argument("--num-decode", type=int, default=4)
    ap.add_argument("--tp", type=int, default=1, help="GPUs per engine")
    ap.add_argument("--first-gpu", type=int, default=0)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--proxy-port", type=int, default=8192)
    ap.add_argument("--prefill-base-port", type=int, default=8100)
    ap.add_argument("--decode-base-port", type=int, default=8200)
    ap.add_argument("--side-channel-base-port", type=int, default=5600)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--dtype", default="")
    ap.add_argument("--enforce-eager", action="store_true", help="the vLLM doc example uses it")
    ap.add_argument("--kv-load-failure-policy", choices=["fail", "recompute"], default="fail")
    ap.add_argument("--kv-buffer-device", default="", help="e.g. cuda (vLLM default otherwise)")
    ap.add_argument("--kv-geometry", default="", help="layers,kv_heads,head_dim (needed for models "
                    "not in kv_geometry.SERVE_MODEL_KV)")
    ap.add_argument("--kv-dtype-bytes", type=int, default=2)
    ap.add_argument("--block-size", type=int, default=16, help="vLLM KV block size (tokens)")
    ap.add_argument("--extra-vllm-arg", action="append", default=[])
    ap.add_argument("--duration", type=int, default=300, help="traffic seconds after all engines healthy")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--prompt-len-mix", default="512:0.3,2048:0.5,8192:0.2",
                    help="'tokens:weight,...' prompt-length mix")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--request-timeout", type=float, default=900.0)
    ap.add_argument("--server-ready-timeout", type=int, default=2400)
    ap.add_argument("--handoff-log", default="")
    ap.add_argument("--traffic-log", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-preflight", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print engine commands and exit")
    args = ap.parse_args()

    model_id, default_dtype = SERVE_MODELS[args.model]
    dtype = args.dtype or default_dtype
    if args.kv_geometry:
        geom = tuple(int(x) for x in args.kv_geometry.split(","))
    elif args.model in SERVE_MODEL_KV:
        geom = SERVE_MODEL_KV[args.model]
    else:
        raise SystemExit(f"no KV geometry for {args.model}: pass --kv-geometry layers,kv_heads,head_dim")
    kv_tok = kv_bytes_per_token(*geom, dtype_bytes=args.kv_dtype_bytes)
    specs = build_engine_specs(args, model_id, dtype)

    print(f"disagg_kv_vllm model={args.model} ({model_id}) {args.num_prefill} prefill + "
          f"{args.num_decode} decode engines, TP={args.tp}, {kv_tok / 1024:.1f} KiB KV/token")
    for s in specs:
        print(f"  [{s.name}] GPUs {s.gpus} http:{s.http_port} nixl-side:{s.side_port}")
        print("    env:", " ".join(f"{k}={v}" for k, v in s.env.items()))
        print("    cmd:", " ".join(s.cmd))
    print(f"  proxy :{args.proxy_port} -> prefill ports {[s.http_port for s in specs if s.role == 'prefill']} "
          f"decode ports {[s.http_port for s in specs if s.role == 'decode']}")
    if args.dry_run:
        return

    versions = {} if args.skip_preflight else preflight()
    os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    meta_path = os.path.splitext(args.handoff_log)[0] + ".meta.json" if args.handoff_log else ""
    if meta_path:
        os.makedirs(os.path.dirname(meta_path) or ".", exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump({"workload": "disagg_kv_vllm", "wid": "W10", "model": model_id, "versions": versions,
                       "kv_bytes_per_token": kv_tok, "block_size": args.block_size,
                       "engines": [{"name": s.name, "gpus": s.gpus, "http_port": s.http_port,
                                    "side_port": s.side_port, "cmd": s.cmd, "env": s.env} for s in specs],
                       "prompt_len_mix": args.prompt_len_mix, "concurrency": args.concurrency,
                       "handoff_bytes_basis": "ESTIMATE from reported KV blocks; fabric truth is "
                                              "NVLink/IB counters from the telemetry launcher",
                       "t_start": time.time()}, f, indent=2)

    stop_pump = threading.Event()
    for s in specs:
        s.proc = subprocess.Popen(s.cmd, env={**os.environ, **s.env}, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1)

        def pump(sp=s) -> None:
            for line in sp.proc.stdout:                       # type: ignore[union-attr]
                sp.tail.append(line.rstrip())
                if not stop_pump.is_set():
                    sys.stdout.write(f"[{sp.name}] {line}")
        threading.Thread(target=pump, daemon=True).start()

    proxy = None
    log_fh = None
    stop_traffic = threading.Event()

    def cleanup(signum=None, frame=None) -> None:
        stop_traffic.set()
        stop_pump.set()
        if proxy is not None:
            proxy.close()
        for sp in specs:
            if sp.proc is not None and sp.proc.poll() is None:
                sp.proc.send_signal(signal.SIGTERM)
        for sp in specs:
            if sp.proc is not None:
                try:
                    sp.proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    sp.proc.kill()
        if log_fh is not None:
            log_fh.close()
        if signum is not None:
            sys.exit(128 + signum)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        deadline = time.time() + args.server_ready_timeout
        pending = list(specs)
        print(f"Waiting for {len(pending)} engines (timeout {args.server_ready_timeout}s) ...")
        while pending and time.time() < deadline:
            for s in list(pending):
                if s.proc.poll() is not None:
                    print(f"ERROR: engine {s.name} exited rc={s.proc.returncode}; last output:", file=sys.stderr)
                    print("\n".join(s.tail), file=sys.stderr)
                    sys.exit(1)
                if _get(f"http://{args.host}:{s.http_port}/v1/models") == 200:
                    print(f"  {s.name} healthy")
                    pending.remove(s)
            time.sleep(2.0)
        if pending:
            print(f"ERROR: engines not healthy in time: {[s.name for s in pending]}", file=sys.stderr)
            sys.exit(1)

        proxy = DisaggProxy([(args.host, s.http_port) for s in specs if s.role == "prefill"],
                            [(args.host, s.http_port) for s in specs if s.role == "decode"],
                            args.handoff_log, kv_tok, args.block_size, args.request_timeout)
        proxy.serve(args.host, args.proxy_port)
        if args.traffic_log:
            os.makedirs(os.path.dirname(args.traffic_log) or ".", exist_ok=True)
            log_fh = open(args.traffic_log, "a", buffering=1)
        print(f"Proxy up on :{args.proxy_port}; traffic for {args.duration}s "
              f"(concurrency {args.concurrency}, mix {args.prompt_len_mix})")
        counts = run_traffic(args, model_id, f"http://{args.host}:{args.proxy_port}/v1/chat/completions",
                             log_fh, stop_traffic)
        print(f"Traffic done: {counts}; proxy: {proxy.counts}")
    finally:
        cleanup()


if __name__ == "__main__":
    main()
