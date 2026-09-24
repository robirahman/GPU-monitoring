#!/usr/bin/env python3
"""End-to-end: (optional) start vLLM → replay trace → collect NVML metrics.

What this script does
---------------------
1. Optionally spawn `vllm serve <model>` on --port (GPU sits under vLLM).
2. Start the existing SPAR TelemetryCollector (NVML ± DCGM @ 1 Hz).
3. Run workloads/serving_trace_replay.py — CSV arrival/length → HTTP traffic.
4. Stop collector → write labeled parquet under --output-dir.
5. Tear down vLLM if we started it.

The CSV is only a *traffic script*. Real metrics come from step 2/4.

Examples
--------
# vLLM already running elsewhere:
python scripts/run_serving_trace.py \\
  --workload serving_azure_conv_2023 \\
  --trace data/external/llm_serving_traces/azure/AzureLLMInferenceTrace_conv_2023.csv \\
  --model Qwen/Qwen2.5-7B-Instruct \\
  --duration 300 --speed-factor 10 --max-concurrency 32 \\
  --no-start-server

# One-shot on a free GPU (starts vLLM for you):
python scripts/run_serving_trace.py \\
  --workload serving_burstgpt_api \\
  --trace data/external/llm_serving_traces/burstgpt/BurstGPT_without_fails_1.csv \\
  --dataset burstgpt --log-type api \\
  --model Qwen/Qwen2.5-7B-Instruct \\
  --duration 600 --speed-factor 50 --max-concurrency 64 \\
  --start-server --tp 1 --gpu-index 0
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from scripts.collect_telemetry import TelemetryCollector  # noqa: E402


def _generate_filename(label: str, gpu_name: str, run_id: str, ext: str) -> str:
    gpu_short = gpu_name.replace(" ", "_").replace("-", "_")
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{label}_{gpu_short}_{run_id[:8]}_{date_str}.{ext}"


def _start_vllm(
    model: str,
    port: int,
    tp: int,
    gpu_index: int,
    max_model_len: int,
    gpu_mem_util: float,
    extra_args: list[str],
) -> subprocess.Popen:
    env = os.environ.copy()
    # Single-GPU: pin visibility. Multi-TP: caller should set CUDA_VISIBLE_DEVICES.
    if tp == 1 and "CUDA_VISIBLE_DEVICES" not in env:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--port", str(port),
        "--tensor-parallel-size", str(tp),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_mem_util),
        "--trust-remote-code",
    ] + extra_args

    print(f"[orch] starting vLLM: {' '.join(cmd)}", flush=True)
    log_path = _REPO_ROOT / "data" / f"vllm_serve_{port}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "w")
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
    )
    proc._spar_log_f = log_f  # type: ignore[attr-defined]
    proc._spar_log_path = log_path  # type: ignore[attr-defined]
    print(f"[orch] vLLM pid={proc.pid}  log={log_path}", flush=True)
    return proc


def _stop_vllm(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    print(f"[orch] stopping vLLM pid={proc.pid}", flush=True)
    try:
        if hasattr(os, "killpg"):
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.terminate()
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        try:
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                proc.kill()
        except ProcessLookupError:
            pass
    log_f = getattr(proc, "_spar_log_f", None)
    if log_f:
        log_f.close()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Orchestrate vLLM + trace replay + NVML collection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--workload", type=str, required=True,
                    help="Label written into telemetry parquet (e.g. serving_azure_conv)")
    ap.add_argument("--trace", type=str, required=True)
    ap.add_argument("--dataset", choices=["auto", "azure", "burstgpt"], default="auto")
    ap.add_argument("--log-type", type=str, default=None)
    ap.add_argument("--model-filter", type=str, default=None)
    ap.add_argument("--model", type=str, required=True,
                    help="HF model id (must match what vLLM serves)")
    ap.add_argument("--base-url", type=str, default="http://127.0.0.1:8000")
    ap.add_argument("--port", type=int, default=8000)

    ap.add_argument("--start-server", action="store_true",
                    help="Spawn vLLM in this process tree")
    ap.add_argument("--no-start-server", action="store_true",
                    help="Assume vLLM already running (default if neither flag)")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--vllm-extra", nargs="*", default=[],
                    help="Extra args forwarded to vLLM api_server")

    ap.add_argument("--speed-factor", type=float, default=10.0)
    ap.add_argument("--max-concurrency", type=int, default=32)
    ap.add_argument("--duration", type=float, default=300.0,
                    help="Wall-clock replay seconds (telemetry window ≈ this)")
    ap.add_argument("--max-requests", type=int, default=None)
    ap.add_argument("--start-offset-sec", type=float, default=0.0)
    ap.add_argument("--window-sec", type=float, default=None)
    ap.add_argument("--request-timeout", type=float, default=300.0)

    ap.add_argument("--gpu-index", type=int, default=0)
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--output-dir", type=str, default="data/")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--no-dcgm", action="store_true")
    ap.add_argument("--baseline-pre-sec", type=float, default=2.0)
    args = ap.parse_args()

    start_server = args.start_server and not args.no_start_server
    # default: don't start unless asked (safer on shared clusters)
    if not args.start_server and not args.no_start_server:
        start_server = False

    run_id = args.run_id or uuid.uuid4().hex[:12]
    output_dir = _REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve trace path
    trace_path = Path(args.trace)
    if not trace_path.is_file():
        alt = _REPO_ROOT / args.trace
        if alt.is_file():
            trace_path = alt
        else:
            print(f"ERROR: trace not found: {args.trace}", file=sys.stderr)
            sys.exit(1)

    loadgen = [
        sys.executable,
        str(_REPO_ROOT / "workloads" / "serving_trace_replay.py"),
        "--trace", str(trace_path),
        "--dataset", args.dataset,
        "--base-url", args.base_url,
        "--model", args.model,
        "--speed-factor", str(args.speed_factor),
        "--max-concurrency", str(args.max_concurrency),
        "--duration", str(args.duration),
        "--request-timeout", str(args.request_timeout),
        "--wait-server",
        "--wait-server-timeout", "900",
    ]
    if args.log_type:
        loadgen += ["--log-type", args.log_type]
    if args.model_filter:
        loadgen += ["--model-filter", args.model_filter]
    if args.max_requests is not None:
        loadgen += ["--max-requests", str(args.max_requests)]
    if args.start_offset_sec:
        loadgen += ["--start-offset-sec", str(args.start_offset_sec)]
    if args.window_sec is not None:
        loadgen += ["--window-sec", str(args.window_sec)]

    vllm_proc = None
    collector = None
    try:
        if start_server:
            vllm_proc = _start_vllm(
                model=args.model,
                port=args.port,
                tp=args.tp,
                gpu_index=args.gpu_index,
                max_model_len=args.max_model_len,
                gpu_mem_util=args.gpu_memory_utilization,
                extra_args=args.vllm_extra,
            )

        collector = TelemetryCollector(
            gpu_index=args.gpu_index,
            interval_sec=args.interval,
            workload_label=args.workload,
            run_id=run_id,
            enable_dcgm=not args.no_dcgm,
        )
        out_name = _generate_filename(
            args.workload, collector.metadata.gpu_name, run_id, "parquet",
        )
        output_path = output_dir / out_name

        print("=" * 60)
        print("SPAR serving-trace run")
        print("=" * 60)
        print(f"  label:      {args.workload}")
        print(f"  trace:      {trace_path}")
        print(f"  model:      {args.model}")
        print(f"  vLLM:       {'spawned' if start_server else 'external'} {args.base_url}")
        print(f"  speed:      {args.speed_factor}×  conc={args.max_concurrency}  dur={args.duration}s")
        print(f"  GPU:        {collector.metadata.gpu_name} (index {args.gpu_index})")
        print(f"  run_id:     {run_id}")
        print(f"  output:     {output_path}")
        print("=" * 60)

        collector.start()
        print(f"\n[orch] baseline {args.baseline_pre_sec}s...", flush=True)
        time.sleep(args.baseline_pre_sec)

        print(f"[orch] loadgen: {' '.join(loadgen)}", flush=True)
        # Hard cap: duration + server wait + slack for drain
        timeout = int(args.duration + 900 + 120)
        rc = subprocess.run(loadgen, timeout=timeout).returncode

        print(f"\n[orch] loadgen exit={rc}; post-cooldown 2s", flush=True)
        time.sleep(2)
        collector.stop()
        collector.save(str(output_path), fmt="parquet")
        print(f"[orch] wrote {output_path}  samples={collector.sample_count}", flush=True)
        collector.cleanup()
        collector = None

        if rc != 0:
            sys.exit(rc)

    except KeyboardInterrupt:
        print("\n[orch] interrupted", flush=True)
        if collector is not None:
            try:
                collector.stop()
                out = output_dir / _generate_filename(
                    args.workload, collector.metadata.gpu_name, run_id, "parquet",
                )
                collector.save(str(out), fmt="parquet")
                print(f"[orch] partial save → {out}", flush=True)
                collector.cleanup()
            except Exception as e:
                print(f"[orch] save-on-interrupt failed: {e}", flush=True)
        sys.exit(130)
    finally:
        _stop_vllm(vllm_proc)


if __name__ == "__main__":
    main()
