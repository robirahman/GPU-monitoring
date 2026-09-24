"""Tests for the REAL-framework workload scaffolding (no GPU, no vLLM, no torch needed).

What these can and cannot show: they verify OUR code -- engine command construction, the
prefill->decode proxy protocol (against mock engines that assert the vLLM reference-proxy
contract), profile building/sampling, launcher argument tables. They do not prove that real
vLLM / torchtitan / NIXL behave as assumed; that needs a GPU run on the cluster.
"""

import http.server
import json
import subprocess
import sys
import threading
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "workloads"))

import disagg_kv_vllm as dkv  # noqa: E402
import kv_profile  # noqa: E402
from disagg_proxy import DisaggProxy, count_blocks  # noqa: E402


# ── mock engines ─────────────────────────────────────────────────────────────

class MockEngine:
    """Tiny OpenAI-ish server standing in for a vLLM engine (role prefill|decode)."""

    def __init__(self, role, fail=False, block_ids=None):
        self.role, self.fail, self.seen = role, fail, []
        block_ids = block_ids if block_ids is not None else [list(range(10, 22))]   # 12 blocks
        engine = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _json(self, code, obj):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._json(200, {"data": [{"id": "mock"}]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                engine.seen.append(body)
                if engine.fail:
                    self._json(500, {"error": "boom"})
                elif engine.role == "prefill":
                    self._json(200, {"usage": {"prompt_tokens": 190, "completion_tokens": 1},
                                     "kv_transfer_params": {"remote_engine_id": "eng-1", "remote_host": "h",
                                                            "remote_port": 5600, "remote_block_ids": block_ids,
                                                            "do_remote_prefill": True}})
                else:
                    self._json(200, {"usage": {"prompt_tokens": 190, "completion_tokens": 7}})

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def engines():
    made = []

    def make(role, **kw):
        e = MockEngine(role, **kw)
        made.append(e)
        return e
    yield make
    for e in made:
        e.close()


def _post(port, path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


# ── proxy protocol ───────────────────────────────────────────────────────────

def test_proxy_follows_reference_two_step_flow(engines, tmp_path):
    pf, dc = engines("prefill"), engines("decode")
    log = tmp_path / "handoffs.jsonl"
    proxy = DisaggProxy([("127.0.0.1", pf.port)], [("127.0.0.1", dc.port)], str(log),
                        kv_bytes_per_token=57344, block_size=16)
    srv = proxy.serve("127.0.0.1", 0)
    try:
        client_body = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 64,
                       "stream": True, "stream_options": {"include_usage": True}, "min_tokens": 5}
        code, resp = _post(srv.server_address[1], "/v1/chat/completions", client_body)
    finally:
        proxy.close()
    assert code == 200 and resp["usage"]["completion_tokens"] == 7

    (pre,), (dec,) = pf.seen, dc.seen
    assert pre["kv_transfer_params"] == {"do_remote_decode": True, "do_remote_prefill": False,
                                         "remote_engine_id": None, "remote_block_ids": None,
                                         "remote_host": None, "remote_port": None}
    assert pre["max_tokens"] == 1 and pre["max_completion_tokens"] == 1 and pre["stream"] is False
    assert "stream_options" not in pre and "min_tokens" not in pre
    # the decoder gets the ORIGINAL request (full max_tokens) + exactly what the prefiller returned
    assert dec["max_tokens"] == 64 and dec["kv_transfer_params"]["remote_engine_id"] == "eng-1"
    assert dec["kv_transfer_params"]["remote_block_ids"] == [list(range(10, 22))]

    (rec,) = [json.loads(line) for line in log.read_text().splitlines()]
    assert rec["status"] == "ok" and rec["kv_blocks"] == 12 and rec["prompt_tokens"] == 190
    assert rec["kv_bytes_est"] == 12 * 16 * 57344
    assert "ESTIMATE" in rec["kv_bytes_basis"]
    assert rec["t_prefill_start"] <= rec["t_prefill_end"] <= rec["t_decode_start"] <= rec["t_decode_end"]


def test_proxy_round_robins_and_reports_failures(engines, tmp_path):
    pfs, dcs = [engines("prefill"), engines("prefill")], [engines("decode"), engines("decode")]
    bad = engines("prefill", fail=True)
    proxy = DisaggProxy([("127.0.0.1", p.port) for p in pfs], [("127.0.0.1", d.port) for d in dcs],
                        str(tmp_path / "h.jsonl"), 1000, 16)
    srv = proxy.serve("127.0.0.1", 0)
    port = srv.server_address[1]
    try:
        for _ in range(4):
            assert _post(port, "/v1/completions", {"prompt": "x", "max_tokens": 4})[0] == 200
        assert [len(p.seen) for p in pfs] == [2, 2] and [len(d.seen) for d in dcs] == [2, 2]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthcheck") as r:
            assert json.loads(r.read())["ok"] == 4
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models") as r:     # forwarded to a decoder
            assert r.status == 200
    finally:
        proxy.close()

    proxy2 = DisaggProxy([("127.0.0.1", bad.port)], [("127.0.0.1", dcs[0].port)], str(tmp_path / "e.jsonl"), 1, 16)
    srv2 = proxy2.serve("127.0.0.1", 0)
    try:
        code, obj = _post(srv2.server_address[1], "/v1/completions", {"prompt": "x"})
    finally:
        proxy2.close()
    assert code == 502 and "bad_gateway" in json.dumps(obj)
    assert json.loads((tmp_path / "e.jsonl").read_text().splitlines()[0])["status"] == "error"


def test_count_blocks_handles_flat_and_grouped():
    assert count_blocks(None) == 0 and count_blocks([]) == 0
    assert count_blocks([1, 2, 3]) == 3
    assert count_blocks([[1, 2], [3]]) == 3


# ── orchestrator ─────────────────────────────────────────────────────────────

def _args(**over):
    base = dict(num_prefill=2, num_decode=2, tp=1, first_gpu=0, host="127.0.0.1", prefill_base_port=8100,
                decode_base_port=8200, side_channel_base_port=5600, max_model_len=16384,
                gpu_memory_utilization=0.9, kv_load_failure_policy="fail", kv_buffer_device="",
                enforce_eager=False, extra_vllm_arg=[])
    base.update(over)
    import argparse
    return argparse.Namespace(**base)


def test_engine_specs_match_documented_nixl_recipe():
    specs = dkv.build_engine_specs(_args(), "Qwen/Qwen2.5-7B-Instruct", "bfloat16")
    assert [s.name for s in specs] == ["prefill0", "prefill1", "decode0", "decode1"]
    assert [s.gpus for s in specs] == [[0], [1], [2], [3]]
    assert len({s.http_port for s in specs}) == 4 and len({s.side_port for s in specs}) == 4
    for s in specs:
        i = s.cmd.index("--kv-transfer-config")
        cfg = json.loads(s.cmd[i + 1])
        assert cfg["kv_connector"] == "NixlConnector"
        assert cfg["kv_role"] == ("kv_producer" if s.role == "prefill" else "kv_consumer")
        assert cfg["kv_load_failure_policy"] == "fail"
        assert s.env["VLLM_NIXL_SIDE_CHANNEL_PORT"] == str(s.side_port)
        assert s.env["CUDA_VISIBLE_DEVICES"] == ",".join(map(str, s.gpus))
        assert s.env["UCX_NET_DEVICES"] == "all"
        assert "--tensor-parallel-size" in s.cmd and "Qwen/Qwen2.5-7B-Instruct" in s.cmd


def test_engine_specs_tp2_spaces_gpus_and_side_channels():
    specs = dkv.build_engine_specs(_args(num_prefill=1, num_decode=1, tp=2, enforce_eager=True),
                                   "m", "bfloat16")
    assert [s.gpus for s in specs] == [[0, 1], [2, 3]]
    assert specs[1].side_port - specs[0].side_port >= 2          # NIXL uses side_port + tp_rank
    assert all("--enforce-eager" in s.cmd for s in specs)


def test_len_mix_and_unique_prompts():
    assert dkv.parse_len_mix("512:0.3,2048:0.7") == [(512, 0.3), (2048, 0.7)]
    with pytest.raises(SystemExit):
        dkv.parse_len_mix("0:1")
    a, b = (dkv.make_body("m", 300, 8, 0), dkv.make_body("m", 300, 8, 0))
    pa, pb = a["messages"][0]["content"], b["messages"][0]["content"]
    assert pa != pb and pa.split("]")[0] != pb.split("]")[0]     # nonce leads the prompt (defeats prefix cache)


def test_traffic_loop_through_proxy(engines, tmp_path):
    import argparse
    pf, dc = engines("prefill"), engines("decode")
    proxy = DisaggProxy([("127.0.0.1", pf.port)], [("127.0.0.1", dc.port)], str(tmp_path / "h.jsonl"), 1000, 16)
    srv = proxy.serve("127.0.0.1", 0)
    args = argparse.Namespace(prompt_len_mix="64:1", duration=1, concurrency=3, max_tokens=4, seed=0,
                              request_timeout=10)
    log = open(tmp_path / "traffic.jsonl", "w")
    try:
        counts = dkv.run_traffic(args, "m", f"http://127.0.0.1:{srv.server_address[1]}/v1/chat/completions",
                                 log, threading.Event())
    finally:
        log.close()
        proxy.close()
    assert counts["ok"] > 3 and counts["error"] == 0
    prompts = {b["messages"][0]["content"] for b in pf.seen}
    assert len(prompts) == len(pf.seen)                           # every request unique


def test_dry_run_prints_commands_without_vllm():
    r = subprocess.run([sys.executable, str(ROOT / "workloads" / "disagg_kv_vllm.py"), "--dry-run",
                        "--num-prefill", "1", "--num-decode", "1"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "kv_producer" in r.stdout and "kv_consumer" in r.stdout
    assert "56.0 KiB KV/token" in r.stdout                        # Qwen2.5-7B geometry


# ── profile (W10 -> W18) ─────────────────────────────────────────────────────

def _records(n=200):
    recs, t = [], 1000.0
    for i in range(n):
        t += 0.5 if i % 10 else 0.05                              # bursty: every 10th arrives fast
        toks = 256 if i % 3 else 4096
        recs.append({"t": t, "t_decode_start": t, "status": "ok", "prompt_tokens": toks,
                     "kv_bytes_est": toks * 57344, "kv_bytes_per_token": 57344, "block_size": 16})
    recs.append({"t": t + 1, "status": "error"})
    return recs


def test_profile_build_and_schedule_sampling(tmp_path):
    prof = kv_profile.build_profile(_records(), source="unit")
    assert prof["n_handoffs"] == 200 and "ESTIMATE" in prof["provenance"]
    assert prof["size_bytes"]["max"] == 4096 * 57344 and prof["size_bytes"]["min"] == 256 * 57344

    sizes, gaps = kv_profile.sample_schedule(prof, total_bytes=1_000_000_000 // 4 * 4, itemsize=4, seed=3)
    assert sum(sizes) == 1_000_000_000 // 4 * 4 and all(s % 4 == 0 and s > 0 for s in sizes)
    assert gaps[0] == 0.0 and len(gaps) == len(sizes) and max(gaps) <= 2.0
    assert set(sizes[:-1]) <= {256 * 57344, 4096 * 57344}         # blocks follow the empirical sizes
    assert kv_profile.sample_schedule(prof, 10**9, itemsize=4, seed=3) == kv_profile.sample_schedule(prof, 10**9, itemsize=4, seed=3)

    path = tmp_path / "p.json"
    path.write_text(json.dumps(prof))
    assert kv_profile.load_profile(str(path))["n_handoffs"] == 200
    with pytest.raises(ValueError):
        kv_profile.build_profile([{"status": "error"}], source="x")


def test_build_kv_profile_cli(tmp_path):
    log = tmp_path / "h.jsonl"
    log.write_text("\n".join(json.dumps(r) for r in _records(50)))
    out = tmp_path / "profiles" / "p.json"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_kv_profile.py"), "--handoffs", str(log),
                        "--out", str(out)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert json.loads(out.read_text())["n_handoffs"] == 50


# ── end-to-end lifecycles against the mock vLLM server (tests/mock_vllm) ─────

import os  # noqa: E402
import socket  # noqa: E402

MOCK_VLLM = str(ROOT / "tests" / "mock_vllm")


def _free_ports(n):
    socks = [socket.socket() for _ in range(n)]
    for s in socks:
        s.bind(("127.0.0.1", 0))
    ports = [s.getsockname()[1] for s in socks]
    for s in socks:
        s.close()
    return ports


def _mock_env():
    env = {**os.environ, "PYTHONPATH": MOCK_VLLM + os.pathsep + os.environ.get("PYTHONPATH", "")}
    if sys.platform == "darwin":
        env.setdefault("GLOO_SOCKET_IFNAME", "lo0")
    return env


def test_disagg_orchestrator_end_to_end_then_profile(tmp_path):
    pf_port, dc_port, proxy_port = _free_ports(3)
    handoffs, traffic = tmp_path / "handoffs.jsonl", tmp_path / "traffic.jsonl"
    r = subprocess.run(
        [sys.executable, str(ROOT / "workloads" / "disagg_kv_vllm.py"), "--skip-preflight",
         "--num-prefill", "1", "--num-decode", "1", "--prefill-base-port", str(pf_port),
         "--decode-base-port", str(dc_port), "--proxy-port", str(proxy_port),
         "--duration", "3", "--concurrency", "2", "--prompt-len-mix", "64:1", "--max-tokens", "4",
         "--server-ready-timeout", "60", "--handoff-log", str(handoffs), "--traffic-log", str(traffic)],
        capture_output=True, text=True, timeout=120, env=_mock_env())
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]

    recs = [json.loads(line) for line in handoffs.read_text().splitlines()]
    assert len(recs) >= 4 and all(x["status"] == "ok" for x in recs)
    assert all(x["kv_blocks"] == 7 and x["kv_bytes_est"] == 7 * 16 * 57344 for x in recs)
    tr = [json.loads(line) for line in traffic.read_text().splitlines()]
    assert tr and all("error" not in x for x in tr)
    meta = json.loads((tmp_path / "handoffs.meta.json").read_text())
    assert [e["name"] for e in meta["engines"]] == ["prefill0", "decode0"]
    assert "ESTIMATE" in meta["handoff_bytes_basis"]

    # chain: real-run handoff log -> profile -> loadable by W18's loader
    prof = tmp_path / "p.json"
    r2 = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_kv_profile.py"), "--handoffs",
                         str(handoffs), "--out", str(prof)], capture_output=True, text=True, timeout=60)
    assert r2.returncode == 0, r2.stderr
    assert kv_profile.load_profile(str(prof))["n_handoffs"] == len(recs)


def test_moe_serving_cotenant_fake_ar_lifecycle(tmp_path):
    pytest.importorskip("torch")
    (port,) = _free_ports(1)
    traffic, fake = tmp_path / "traffic.jsonl", tmp_path / "fake_ar.jsonl"
    r = subprocess.run(
        [sys.executable, str(ROOT / "workloads" / "llm_moe_serving.py"), "--model", "qwen25_7b",
         "--tp", "1", "--dp", "2", "--port", str(port), "--traffic", "decode", "--concurrency", "2",
         "--duration", "6", "--decode-max-tokens", "2", "--server-ready-timeout", "60",
         "--traffic-log", str(traffic), "--cotenant-fake-ar", "--fake-ar-every-s", "0.5",
         "--fake-ar-mb", "0.25", "--fake-ar-gpus", "2", "--fake-ar-backend", "gloo",
         "--fake-ar-warmup-s", "2", "--fake-ar-log", str(fake)],
        capture_output=True, text=True, timeout=180, env=_mock_env())
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]

    win = json.loads((tmp_path / "fake_ar.window.json").read_text())
    ev = [json.loads(line) for line in fake.read_text().splitlines()]
    assert len(ev) >= 4 and {e["source"] for e in ev} == {"cotenant_fake_ar"}
    assert {e["phase"] for e in ev} == {"fake_ar"} and all(e["payload_bytes"] == 262144 for e in ev)
    # co-tenant padding overlaps the real traffic window on the same wall clock
    inside = [e for e in ev if win["t_traffic_start"] <= e["t"] <= win["t_traffic_end"]]
    assert len(inside) >= 3
    tr = [json.loads(line) for line in traffic.read_text().splitlines()]
    assert tr and min(x["t"] for x in tr) >= win["t_traffic_start"] - 1
    assert win["cotenant_cmd"][0] == sys.executable


# ── registry: every new PRIMARY label's exact command dry-runs through its own launcher ──────────

PRIMARY_ORCHESTRATED = [
    "disagg_kv_vllm_qwen7b", "disagg_kv_vllm_qwen7b_longctx",
    "moe_serve_qwen7b_tp4dp2_fake_ar", "moe_serve_mixtral_ep8_fake_ar",
    "moe_ep_train_dsv3_16b_ep8", "moe_ep_train_qwen3_30b_a3b_ep8",
    "train_tp2_dp4_qwen3_1p7b_titan", "train_tp2_fsdp4_qwen3_14b_titan", "train_pp4_1f1b_qwen3_1p7b_titan",
]


@pytest.mark.parametrize("label", PRIMARY_ORCHESTRATED)
def test_primary_registry_commands_dry_run(label):
    from workloads.registry import WORKLOAD_REGISTRY

    cmd = list(WORKLOAD_REGISTRY[label]["command"]) + ["--dry-run"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=ROOT)
    assert r.returncode == 0, f"{label}: {r.stdout[-800:]}{r.stderr[-800:]}"


def test_primary_and_microbench_labelling():
    from workloads.registry import WORKLOAD_REGISTRY as R

    banner = "[MICROBENCH ONLY — not SPAR primary corpus]"
    for label in ("moe_ep_train_synthetic", "moe_ep_train_synthetic_dp", "disagg_kv_synth_1gb",
                  "disagg_kv_synth_256mb_burst", "moe_ep_synthetic", "moe_ep_synthetic_fake_ar"):
        assert R[label]["description"].startswith(banner), label
        assert R["microbench_" + label] is R[label]            # alias; old name keeps working
    for label in PRIMARY_ORCHESTRATED + ["train_kv_disguise_ddp_realprof", "train_kv_disguise_p2p_realprof"]:
        assert "MICROBENCH" not in R[label]["description"]
    # W18 realprof points at a path a real W10 run produces; falls back to --block-mb if missing
    cmd = R["train_kv_disguise_p2p_realprof"]["command"]
    assert cmd[cmd.index("--schedule-from") + 1].endswith("kv_handoff_disagg_kv_vllm_qwen7b.json")
    assert "--block-mb" in cmd
    # tp8 fake_ar was never registered (28 heads); dense control is tp4dp2 (+ legacy tp8 alias)
    assert "moe_serve_qwen7b_tp8_fake_ar" not in R
    assert "moe_serve_qwen7b_tp4dp2_mixed" in R
    assert "--tp" in R["moe_serve_qwen7b_tp8_mixed"]["command"]
    tp_idx = R["moe_serve_qwen7b_tp8_mixed"]["command"].index("--tp")
    assert R["moe_serve_qwen7b_tp8_mixed"]["command"][tp_idx + 1] == "4"  # alias → TP4, not 8


@pytest.mark.skipif(__import__("importlib.util").util.find_spec("torchtitan") is None,
                    reason="torchtitan not installed: schema validation needs its source")
def test_titan_jobs_validate_against_installed_torchtitan():
    sys.path.insert(0, str(ROOT / "scripts"))
    import titan_jobs

    for job in titan_jobs.JOBS.values():
        assert titan_jobs.validate_job(job) == [], job.name
    bad = titan_jobs.TitanJob("x", "W03", "qwen3", "qwen3_1_7b", 8, "r", ["--parallelism.tensor_parallel_degre", "2"])
    assert titan_jobs.validate_job(bad)                          # the validator does catch typos


def test_nccl_coll_log_parser_matches_nccl_source_format():
    sys.path.insert(0, str(ROOT / "scripts"))
    import parse_nccl_coll_log as pl

    line = ("[2026-09-23 20:01:02.123456] h100:4242:4300 [3] NCCL INFO AllReduce: opCount 3a sendbuff 0x7f1c "
            "recvbuff 0x7f1c count 1048576 datatype 9 op 0 root 0 comm 0x55d0 [nranks=8] stream 0x55d1")
    ev = pl.parse_line(line)
    assert ev["op"] == "AllReduce" and ev["datatype"] == "bf16" and ev["payload_bytes"] == 2 * 1048576
    assert ev["bytes_est"] == int(2 * 7 / 8 * 2 * 1048576) and ev["nranks"] == 8 and ev["cuda_dev"] == 3
    assert ev["op_count"] == 0x3a and ev["t"] is not None
    assert pl.parse_line("h:1:2 [0] NCCL INFO Channel 00/0 : 0[0] -> 1[1] via P2P/CUMEM/read") is None
    ag = pl.parse_line(line.replace("AllReduce", "AllGather"))
    assert ag["bytes_est"] == 7 * ag["payload_bytes"]             # AllGather count is per-rank
