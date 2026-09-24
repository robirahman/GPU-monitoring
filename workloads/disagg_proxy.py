#!/usr/bin/env python3
"""Prefill -> decode routing proxy for vLLM disaggregated serving (stdlib only).

Implements the request flow of vLLM's reference ``toy_proxy_server.py``
(``tests/v1/kv_connector/nixl_integration/`` in the vLLM repo) so we do not depend on a
file that ships only in the source tree, and so the handoff can be logged:

  1. To a *prefiller* (``kv_role=kv_producer``): the client request with
     ``max_tokens=1``, ``stream=False`` and
     ``kv_transfer_params={"do_remote_decode": true, "do_remote_prefill": false,
     "remote_engine_id"/"remote_block_ids"/"remote_host"/"remote_port": null}``.
     The prefiller computes the prompt KV, keeps the blocks, and answers with its own
     ``kv_transfer_params`` (engine id, block ids, host/port of its NIXL side channel).
  2. To a *decoder* (``kv_role=kv_consumer``): the original request plus those
     ``kv_transfer_params``. The decoder pulls the KV blocks straight from the prefiller
     GPU (NIXL/UCX; NVLink inside a node) and generates.

Only non-streaming requests are supported (the SPAR traffic client is non-streaming).
Endpoints: ``POST /v1/completions``, ``POST /v1/chat/completions``, ``GET /healthcheck``,
``GET /v1/models`` (forwarded to a decoder so a plain "wait until /v1/models is 200"
health check works against the proxy).

Handoff log (JSONL, one record per request) -- this is a *serve/traffic log*, not a
collective log: nothing here sees NCCL/NIXL. Byte figures are ESTIMATES from the number
of KV blocks the prefiller reported handing over x block size x KV bytes/token of the
model geometry; the fabric-side truth is the NVLink/IB counters the telemetry launcher
records for the same wall-clock window.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PREFILL_KV_PARAMS = {
    "do_remote_decode": True,
    "do_remote_prefill": False,
    "remote_engine_id": None,
    "remote_block_ids": None,
    "remote_host": None,
    "remote_port": None,
}


def _post(url: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def count_blocks(remote_block_ids) -> int:
    """Number of KV blocks in a prefiller's ``remote_block_ids`` (flat list or list of lists)."""
    if not remote_block_ids:
        return 0
    if isinstance(remote_block_ids[0], (list, tuple)):
        return sum(len(g) for g in remote_block_ids)
    return len(remote_block_ids)


class DisaggProxy:
    def __init__(self, prefillers: list[tuple[str, int]], decoders: list[tuple[str, int]],
                 handoff_log: str = "", kv_bytes_per_token: int = 0, block_size: int = 16,
                 timeout: float = 900.0) -> None:
        if not prefillers or not decoders:
            raise ValueError("need at least one prefiller and one decoder")
        self.prefillers, self.decoders = prefillers, decoders
        self.kv_bytes_per_token, self.block_size, self.timeout = kv_bytes_per_token, block_size, timeout
        self._pf_rr, self._dc_rr = itertools.count(), itertools.count()
        self._lock = threading.Lock()
        self._fh = None
        if handoff_log:
            os.makedirs(os.path.dirname(handoff_log) or ".", exist_ok=True)
            self._fh = open(handoff_log, "a", buffering=1)
        self.server: ThreadingHTTPServer | None = None
        self.counts = {"ok": 0, "error": 0}

    # ── core two-step flow ───────────────────────────────────────────────────
    def handle(self, path: str, body: dict) -> tuple[int, dict]:
        with self._lock:
            pi = next(self._pf_rr) % len(self.prefillers)
            di = next(self._dc_rr) % len(self.decoders)
        pf, dc = self.prefillers[pi], self.decoders[di]
        rec = {"t": time.time(), "path": path, "prefill": f"{pf[0]}:{pf[1]}", "decode": f"{dc[0]}:{dc[1]}",
               "prefill_idx": pi, "decode_idx": di,
               "kv_bytes_basis": "ESTIMATE: kv_blocks x block_size x kv_bytes_per_token; not a wire measurement",
               "block_size": self.block_size, "kv_bytes_per_token": self.kv_bytes_per_token}

        pre = dict(body)
        pre["kv_transfer_params"] = dict(PREFILL_KV_PARAMS)
        pre["stream"] = False
        pre["max_tokens"] = 1
        pre["max_completion_tokens"] = 1
        for k in ("stream_options", "min_tokens", "min_completion_tokens"):
            pre.pop(k, None)
        try:
            rec["t_prefill_start"] = time.time()
            pf_resp = _post(f"http://{pf[0]}:{pf[1]}{path}", pre, self.timeout)
            rec["t_prefill_end"] = time.time()
            kv = pf_resp.get("kv_transfer_params") or {}
            usage = pf_resp.get("usage") or {}
            rec["prompt_tokens"] = int(usage.get("prompt_tokens") or 0)
            rec["kv_blocks"] = count_blocks(kv.get("remote_block_ids"))
            rec["kv_bytes_est"] = rec["kv_blocks"] * self.block_size * self.kv_bytes_per_token
            rec["remote_engine_id"] = kv.get("remote_engine_id")

            dec = dict(body)
            dec["stream"] = False
            dec["kv_transfer_params"] = kv
            rec["t_decode_start"] = time.time()
            dc_resp = _post(f"http://{dc[0]}:{dc[1]}{path}", dec, self.timeout)
            rec["t_decode_end"] = time.time()
            rec["completion_tokens"] = int((dc_resp.get("usage") or {}).get("completion_tokens") or 0)
            rec["status"] = "ok"
            self.counts["ok"] += 1
            self._log(rec)
            return 200, dc_resp
        except Exception as e:                                    # noqa: BLE001 - reported to client + log
            rec["status"], rec["error"] = "error", str(e)[:300]
            self.counts["error"] += 1
            self._log(rec)
            return 502, {"error": {"message": f"disagg proxy: {e}"[:500], "type": "bad_gateway"}}

    def _log(self, rec: dict) -> None:
        if self._fh is not None:
            with self._lock:
                self._fh.write(json.dumps(rec, separators=(",", ":")) + "\n")

    # ── HTTP server ──────────────────────────────────────────────────────────
    def serve(self, host: str = "127.0.0.1", port: int = 8192) -> ThreadingHTTPServer:
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):               # quiet
                pass

            def _send(self, code: int, obj: dict) -> None:
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/healthcheck":
                    self._send(200, {"status": "ok", "prefillers": len(proxy.prefillers),
                                     "decoders": len(proxy.decoders), **proxy.counts})
                elif self.path == "/v1/models":
                    dc = proxy.decoders[0]
                    try:
                        with urllib.request.urlopen(f"http://{dc[0]}:{dc[1]}/v1/models", timeout=5) as r:
                            self._send(r.status, json.loads(r.read().decode()))
                    except Exception as e:           # noqa: BLE001
                        self._send(503, {"error": str(e)[:200]})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):
                if self.path not in ("/v1/completions", "/v1/chat/completions"):
                    self._send(404, {"error": "not found"})
                    return
                try:
                    body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                except ValueError:
                    self._send(400, {"error": "invalid JSON"})
                    return
                code, obj = proxy.handle(self.path, body)
                self._send(code, obj)

        self.server = ThreadingHTTPServer((host, port), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self.server

    def close(self) -> None:
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def _hostports(hosts: list[str], ports: list[int]) -> list[tuple[str, int]]:
    if len(hosts) == 1:
        hosts = hosts * len(ports)
    if len(hosts) != len(ports):
        raise SystemExit("--*-hosts must have 1 entry or the same count as --*-ports")
    return list(zip(hosts, ports))


def main() -> None:
    ap = argparse.ArgumentParser(description="vLLM disaggregated prefill->decode proxy")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8192)
    ap.add_argument("--prefiller-hosts", nargs="+", default=["localhost"])
    ap.add_argument("--prefiller-ports", nargs="+", type=int, required=True)
    ap.add_argument("--decoder-hosts", nargs="+", default=["localhost"])
    ap.add_argument("--decoder-ports", nargs="+", type=int, required=True)
    ap.add_argument("--handoff-log", default="")
    ap.add_argument("--kv-bytes-per-token", type=int, default=0)
    ap.add_argument("--block-size", type=int, default=16)
    args = ap.parse_args()
    proxy = DisaggProxy(_hostports(args.prefiller_hosts, args.prefiller_ports),
                        _hostports(args.decoder_hosts, args.decoder_ports),
                        args.handoff_log, args.kv_bytes_per_token, args.block_size)
    srv = proxy.serve(args.host, args.port)
    print(f"disagg proxy on {args.host}:{args.port} -> prefill {proxy.prefillers} / decode {proxy.decoders}")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    main()
