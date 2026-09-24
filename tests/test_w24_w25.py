"""W24 (temporal serve-cover) + W25 (spatial serve||train colocation).

W24: gloo runs of lora_ddp (--sync-every/--cover) and hybrid_wire assert the step-window truth and
the duty-fraction claims. W25: the orchestrator is exercised with fake collectors and real child
processes (no GPU, no NVML, no vLLM): role tags on telemetry / env / manifest, the window overlap,
and every "refuse to mix silently" path.
"""

import json
import os
import socket
import subprocess
import sys
import types
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "workloads"))

import run_colocate_serve_train as rc  # noqa: E402


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def events(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def torchrun(nproc, script, *args, timeout=300):
    torch = pytest.importorskip("torch")   # noqa: F841
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nnodes=1", f"--nproc_per_node={nproc}",
           "--master_addr=127.0.0.1", f"--master_port={_free_port()}", str(ROOT / "workloads" / script),
           "--backend", "gloo", *map(str, args)]
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    if sys.platform == "darwin":
        env.setdefault("GLOO_SOCKET_IFNAME", "lo0")
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout, env=env)
    assert r.returncode == 0, f"{script}: {r.stdout[-1500:]}{r.stderr[-2500:]}"


# ══ W24 ══════════════════════════════════════════════════════════════════════

def test_duty_fractions_separate_step_share_from_byte_share():
    ic = pytest.importorskip("torch") and __import__("interconnect_common")
    d = ic.duty_fractions(20, sync_wire_bytes=9_394_000_000, cover_wire_bytes_per_step=11_200_000)
    assert d["train_step_fraction"] == pytest.approx(0.05) and d["infer_shaped_step_fraction"] == pytest.approx(0.95)
    # the finding: with full-|theta| sync, 95% of STEPS serve-shaped still leaves ~98% of BYTES as sync
    assert d["expected_train_wire_byte_fraction"] > 0.97
    lora = ic.duty_fractions(20, sync_wire_bytes=15_300_000, cover_wire_bytes_per_step=11_200_000)
    assert lora["expected_train_wire_byte_fraction"] < 0.08        # adapter sync: bytes really are serve-dominated


def test_w24_registry_labels():
    from workloads.registry import WORKLOAD_REGISTRY as R

    def flag(cmd, name):
        return cmd[cmd.index(name) + 1]

    assert flag(R["train_serve_cover_H20"]["command"], "--sync-every") == "20"
    assert flag(R["train_serve_cover_H100"]["command"], "--sync-every") == "100"
    lora = R["train_serve_cover_lora_H20"]["command"]
    assert lora[2].endswith("lora_ddp.py") and "--cover" in lora and flag(lora, "--sync-every") == "20"
    for label in ("train_serve_cover_H20", "train_serve_cover_H100", "train_serve_cover_lora_H20"):
        assert "--comm-log" in R[label]["command"]                # steps.jsonl is written next to it
        assert R[label]["description"].startswith("W24")
    # W08-scale cover: the hybrid_wire/lora_ddp defaults are not overridden in the registry
    assert "--decode-hidden" not in R["train_serve_cover_H20"]["command"]


def test_w24_lora_dilution_phases_and_cover(tmp_path):
    log = tmp_path / "lcov.jsonl"
    torchrun(2, "lora_ddp.py", "--model", "random-tiny", "--batch-size", 2, "--seq-len", 16, "--sync-every", 4,
             "--max-steps", 9, "--cover", "--decode-hidden", 64, "--decode-layers", 2, "--comm-log", log)
    steps = events(tmp_path / "lcov.steps.jsonl")
    assert "".join("S" if e["phase"] == "train_sync" else "d" for e in steps) == "dddSdddSd"
    assert [e["step"] for e in steps if e["is_sync_step"]] == [3, 7]
    ev = events(log)
    sync_steps = {e["step"] for e in ev if e["phase"] == "grad_sync_lora"}
    assert sync_steps == {3, 7}                                    # adapters AllReduced ONLY on sync steps
    assert {e["step"] for e in ev if e["phase"] == "decode_shape"} == set(range(9))    # cover every step
    assert all(e["bytes_est"] == 0 for e in steps if not e["is_sync_step"])            # cover not counted as grad sync
    meta = json.loads((tmp_path / "lcov.meta.json").read_text())
    assert meta["wid"] == "W24" and meta["train_step_fraction"] == pytest.approx(0.25)


def test_w24_hybrid_wire_meta_has_duty_fractions(tmp_path):
    log = tmp_path / "h20.jsonl"
    torchrun(4, "hybrid_wire.py", "--size", "tiny", "--batch-size", 2, "--seq-len", 16, "--sync-every", 20,
             "--max-steps", 3, "--cover-tp", 2, "--decode-hidden", 64, "--decode-layers", 2, "--comm-log", log)
    meta = json.loads((tmp_path / "h20.meta.json").read_text())
    assert meta["infer_shaped_step_fraction"] == pytest.approx(0.95) and meta["sync_period_steps"] == 20
    assert "expected_train_wire_byte_fraction" in meta
    assert {e["phase"] for e in events(tmp_path / "h20.steps.jsonl")} == {"decode_shape"}   # no sync in 3 steps


# ══ W25: planning helpers ════════════════════════════════════════════════════

def test_gpu_lists_and_nproc_rewrite():
    assert rc.parse_gpu_list("0-6") == [0, 1, 2, 3, 4, 5, 6] and rc.parse_gpu_list("0-1,3,3") == [0, 1, 3]
    for bad in ("", "a", "3-1", "-2"):
        with pytest.raises(SystemExit):
            rc.parse_gpu_list(bad)
    cmd, changed = rc.rewrite_nproc(["torchrun", "--nproc_per_node=8", "x.py"], 2)
    assert cmd == ["torchrun", "--nproc_per_node=2", "x.py"] and changed
    cmd, changed = rc.rewrite_nproc(["torchrun", "--nproc-per-node", "8", "x.py"], 1)
    assert cmd == ["torchrun", "--nproc-per-node", "1", "x.py"] and changed
    assert rc.rewrite_nproc(["python", "x.py"], 2) == (["python", "x.py"], False)


def test_serve_layout_must_match_serve_gpus():
    with pytest.raises(SystemExit, match="TP=1 x DP=6"):
        rc.resolve_role_cmd("serve", "moe_serve_qwen7b_tp8_mixed", None, [0, 1, 2, 3, 4, 5, 6],
                            ["--tp", "1", "--dp", "6"], None)
    label, cmd = rc.resolve_role_cmd("serve", "moe_serve_qwen7b_tp8_mixed", None, [0, 1, 2, 3],
                                     ["--tp", "4", "--dp", "1"], 120)
    assert label == "moe_serve_qwen7b_tp8_mixed" and cmd[-2:] == ["--duration", "120"] and cmd.count("--tp") == 2


def test_role_workload_source_is_exclusive():
    with pytest.raises(SystemExit):
        rc.resolve_role_cmd("train", None, None, [0], [], None)
    with pytest.raises(SystemExit):
        rc.resolve_role_cmd("train", "ddp_resnet18_cifar10", "python x.py", [0], [], None)


def test_registry_colocate_labels_and_launcher_refusal():
    from workloads.registry import WORKLOAD_REGISTRY as R

    for label in ("colocate_serve7_train1_qwen", "colocate_serve_tp4_train_ddp2"):
        cmd = R[label]["command"]
        assert R[label]["telemetry_owner"].endswith("run_colocate_serve_train.py")
        r = subprocess.run(cmd + ["--dry-run"], capture_output=True, text=True, timeout=120, cwd=ROOT)
        assert r.returncode == 0, r.stderr[-1500:]
    assert "GPUs [0, 1, 2, 3, 4, 5, 6]" in r.stderr or "GPUs [0, 1, 2, 3]" in r.stderr

    # run_workload_multi_gpu.py must refuse to wrap a label that owns its telemetry (stub NVML/pandas)
    stubs = {m: mock.MagicMock() for m in ("pynvml", "pandas", "pyarrow", "pyarrow.parquet")}
    with mock.patch.dict(sys.modules, stubs), mock.patch.object(
            sys, "argv", ["run_workload_multi_gpu.py", "--workload", "colocate_serve7_train1_qwen"]):
        sys.modules.pop("scripts.run_workload_multi_gpu", None)
        import scripts.run_workload_multi_gpu as launcher
        with pytest.raises(SystemExit) as e:
            launcher.main()
        sys.modules.pop("scripts.run_workload_multi_gpu", None)
    assert e.value.code == 2


# ══ W25: orchestration with fake collectors + real child processes ═══════════

class FakeCollector:
    made: list = []

    def __init__(self, **kw):
        self.kw = kw
        self._workload_role = kw.get("workload_role")
        self.metadata = types.SimpleNamespace(gpu_name="Fake H100")
        self.started = self.stopped = False
        FakeCollector.made.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def save(self, path):
        Path(path).write_text(json.dumps([{**{k: self.kw.get(k) for k in (
            "gpu_index", "workload_label", "workload_role", "role_label", "run_id")}}]))
        return path


class UntaggedCollector(FakeCollector):
    def __init__(self, **kw):                  # an "old" collector: silently ignores the role
        kw.pop("workload_role", None)
        super().__init__(**kw)
        self._workload_role = None


SERVE_PY = """
import json, os, sys, time
json.dump({k: v for k, v in os.environ.items() if k.startswith('SPAR_') or k == 'CUDA_VISIBLE_DEVICES'},
          open(sys.argv[1], 'w'))
time.sleep(0.5); print('Server healthy - starting traffic', flush=True)
time.sleep(float(sys.argv[2]))
sys.exit(int(sys.argv[3]) if len(sys.argv) > 3 else 0)
"""
TRAIN_PY = """
import json, os, sys, time
json.dump({'t': time.time(), **{k: v for k, v in os.environ.items() if k.startswith('SPAR_') or k == 'CUDA_VISIBLE_DEVICES'}},
          open(sys.argv[1], 'w'))
time.sleep(float(sys.argv[2]))
"""


def _ns(tmp_path, **over):
    import argparse
    (tmp_path / "serve.py").write_text(SERVE_PY)
    (tmp_path / "train.py").write_text(TRAIN_PY)
    base = dict(label="colo_test", serve_workload=None, train_workload=None, num_gpus=8,
                serve_gpus="0-5", train_gpus="6,7", train_count=2, duration=None, serve_args=[], train_args=[],
                serve_cmd=f"{sys.executable} {tmp_path}/serve.py {tmp_path}/serve_env.json 3",
                train_cmd=f"{sys.executable} {tmp_path}/train.py {tmp_path}/train_env.json 1.5",
                serve_ready_regex="Server healthy", serve_ready_timeout=30, train_delay_s=0.0, timeout=None,
                output_dir=str(tmp_path / "out"), interval=1.0, baseline_s=0.1, no_dcgm=True, no_fabric=True,
                no_telemetry=False, dry_run=False)
    base.update(over)
    return argparse.Namespace(**base)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("SPAR_TELEMETRY_OWNER", raising=False)
    FakeCollector.made.clear()


def test_orchestrator_tags_every_row_env_and_window(tmp_path):
    manifest, code = rc.run(_ns(tmp_path), collector_factory=FakeCollector)
    assert code == 0 and manifest["usable_as_w25_evidence"] is True

    # one collector per GPU, each carrying its role; one shared run_id
    by_gpu = {c.kw["gpu_index"]: c.kw for c in FakeCollector.made}
    assert sorted(by_gpu) == [0, 1, 2, 3, 4, 5, 6, 7]
    assert {g: by_gpu[g]["workload_role"] for g in by_gpu} == {**{g: "serve" for g in range(6)}, 6: "train", 7: "train"}
    assert len({c.kw["run_id"] for c in FakeCollector.made}) == 1
    assert all(c.started and c.stopped for c in FakeCollector.made)
    files = manifest["telemetry"]["files"]
    assert len(files) == 8 and {f["role"] for f in files} == {"serve", "train"}
    saved = json.loads(Path(files[0]["path"]).read_text())[0]
    assert saved["workload_role"] in ("serve", "train") and saved["run_id"] == manifest["run_id"]

    # children got disjoint GPU sets + role tags in env
    s_env, t_env = (json.loads((tmp_path / f"{r}_env.json").read_text()) for r in ("serve", "train"))
    assert s_env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5" and t_env["CUDA_VISIBLE_DEVICES"] == "6,7"
    assert (s_env["SPAR_WORKLOAD_ROLE"], t_env["SPAR_WORKLOAD_ROLE"]) == ("serve", "train")
    assert s_env["SPAR_RUN_ID"] == t_env["SPAR_RUN_ID"] == manifest["run_id"]
    assert s_env["SPAR_TELEMETRY_OWNER"] == "run_colocate_serve_train"

    # window truth: train starts after serve is ready, and the two overlap
    sv, tr = manifest["roles"]["serve"], manifest["roles"]["train"]
    assert sv["t_start"] < sv["t_ready"] <= tr["t_start"] < tr["t_end"] <= sv["t_end"] + 1.0
    assert manifest["overlap_window"][1] - manifest["overlap_window"][0] > 1.0
    assert (Path(manifest["telemetry"]["files"][0]["path"]).parent / f"colo_test_{manifest['run_id'][:12]}_roles.json").exists()


def test_orchestrator_refuses_to_mix_silently(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="overlap"):
        rc.run(_ns(tmp_path, serve_gpus="0-3", train_gpus="3-4"), collector_factory=FakeCollector)
    with pytest.raises(SystemExit, match="role tag"):
        rc.run(_ns(tmp_path), collector_factory=UntaggedCollector)
    monkeypatch.setenv("SPAR_TELEMETRY_OWNER", "run_workload_multi_gpu")
    with pytest.raises(SystemExit, match="another telemetry launcher"):
        rc.run(_ns(tmp_path), collector_factory=FakeCollector)
    monkeypatch.delenv("SPAR_TELEMETRY_OWNER")
    manifest, _ = rc.run(_ns(tmp_path, no_telemetry=True), collector_factory=FakeCollector)
    assert manifest["telemetry"] is None and manifest["usable_as_w25_evidence"] is False


def test_serve_dying_early_invalidates_the_run(tmp_path):
    manifest, code = rc.run(_ns(tmp_path, serve_cmd=f"{sys.executable} {tmp_path}/serve.py "
                                                    f"{tmp_path}/serve_env.json 0.5 3",
                                train_cmd=f"{sys.executable} {tmp_path}/train.py {tmp_path}/train_env.json 30"),
                            collector_factory=FakeCollector)
    assert code == 3 and manifest["usable_as_w25_evidence"] is False
    assert manifest["roles"]["train"]["t_end"] - manifest["roles"]["train"]["t_start"] < 20     # train was stopped


def test_role_tags_reach_real_workload_logs(tmp_path):
    """Env tags -> lora_ddp comm/steps JSONL + meta, and the vLLM-side traffic log (mock server)."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    pf = _free_port()
    serve = (f"{sys.executable} {ROOT}/workloads/llm_moe_serving.py --model qwen25_7b --tp 1 --dp 1 --port {pf} "
             f"--traffic decode --concurrency 1 --duration 3 --decode-max-tokens 2 --server-ready-timeout 60 "
             f"--traffic-log {tmp_path}/serve_traffic.jsonl")
    train = (f"{sys.executable} -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --master_addr=127.0.0.1 "
             f"--master_port={_free_port()} {ROOT}/workloads/lora_ddp.py --backend gloo --model random-tiny "
             f"--batch-size 2 --seq-len 16 --duration 3 --comm-log {tmp_path}/train_comm.jsonl")
    old = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = str(ROOT / "tests" / "mock_vllm") + os.pathsep + old
    if sys.platform == "darwin":
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0")
    try:
        manifest, code = rc.run(_ns(tmp_path, serve_gpus="0", train_gpus="1", serve_cmd=serve, train_cmd=train,
                                    serve_ready_regex="Server healthy"), collector_factory=FakeCollector)
    finally:
        os.environ["PYTHONPATH"] = old
    assert code == 0, manifest
    run_id = manifest["run_id"]
    tr = events(tmp_path / "serve_traffic.jsonl")
    assert tr and all(e["workload_role"] == "serve" and e["run_id"] == run_id for e in tr)
    steps = events(tmp_path / "train_comm.steps.jsonl")
    assert steps and all(e["workload_role"] == "train" and e["run_id"] == run_id for e in steps)
    meta = json.loads((tmp_path / "train_comm.meta.json").read_text())
    assert meta["workload_role"] == "train" and meta["role_label"] == "custom_train"
