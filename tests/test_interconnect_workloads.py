"""CPU/gloo checks for the SPAR interconnect workloads (W03-W05, W10, W15-W19, W23).

No GPU needed: each workload takes ``--backend gloo``, and the numerically checkable ones
(W03, W04, W18) have a ``--verify`` flag that fails the process if the parallel result
deviates from a single-process reference. These tests launch small multi-process runs with
``python -m torch.distributed.run`` and assert on what the workloads write to their comm logs.

Skipped when torch / gloo is missing (registry + W23 tests need no torch).
"""

import ast
import collections
import importlib.util
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch", reason="interconnect workloads need torch")
import torch.distributed as dist  # noqa: E402

needs_gloo = pytest.mark.skipif(not (dist.is_available() and dist.is_gloo_available()),
                                reason="torch.distributed gloo not available")


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def torchrun(nproc, script, *args, timeout=300):
    # Explicit loopback master address (not --standalone): on a laptop whose hostname resolves only
    # to ::1 (e.g. offline), --standalone rendezvous hangs. GLOO_SOCKET_IFNAME pins gloo to loopback.
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nnodes=1", f"--nproc_per_node={nproc}",
           "--master_addr=127.0.0.1", f"--master_port={_free_port()}",
           str(ROOT / "workloads" / script), "--backend", "gloo", *map(str, args)]
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    if sys.platform == "darwin":
        env.setdefault("GLOO_SOCKET_IFNAME", "lo0")
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout, env=env)
    assert r.returncode == 0, f"{script} failed:\n{r.stdout[-2000:]}\n{r.stderr[-3000:]}"
    return r.stdout


def events(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


# ── registry ────────────────────────────────────────────────────────────────

NEW_LABELS = [
    "disagg_kv_synth_1gb", "disagg_kv_synth_256mb_burst",
    "train_kv_disguise_ddp", "train_kv_disguise_p2p",
    "ddp_powersgd_resnet18", "ddp_powersgd_150m", "ddp_topk_resnet18", "ddp_topk_150m",
    "train_lora_only_ddp_qwen15", "train_lora_only_ddp_tinyllama",
    "train_ddp_rare_sync_accum8", "train_ddp_dilute_1of10",
    "train_hybrid_wire_H50", "train_hybrid_wire_H500",
    "train_ddp_nvl_only_ibquiet", "train_fsdp7b_nvl_only_ibquiet",
    "moe_ep_train_synthetic", "moe_ep_train_synthetic_dp",
    "train_pp2_1f1b_150m", "train_pp4_1f1b_1b",
    "train_tp2_dp4_1b", "train_tp2_dp2_150m",
    "diloco_1b_H500", "diloco_streaming_1b_H512",
]


def test_registry_labels_unique_and_resolvable():
    tree = ast.parse((ROOT / "workloads" / "registry.py").read_text())
    keys = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "WORKLOAD_REGISTRY":
            keys = [k.value for k in node.value.keys]
    assert keys and [k for k, c in collections.Counter(keys).items() if c > 1] == []

    from workloads.registry import WORKLOAD_REGISTRY

    for label in NEW_LABELS:
        entry = WORKLOAD_REGISTRY[label]
        assert {"command", "description", "default_timeout"} <= set(entry)
        for arg in entry["command"]:
            if isinstance(arg, str) and arg.endswith(".py"):
                assert os.path.exists(arg), f"{label}: {arg} missing"


# ── W23 (pure python: fake sysfs tree) ──────────────────────────────────────

def _load_ibquiet():
    spec = importlib.util.spec_from_file_location("ibq", ROOT / "scripts" / "run_nvl_only_ibquiet.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_ib(root, xmit, rcv):
    d = Path(root) / "mlx5_0" / "ports" / "1" / "counters"
    d.mkdir(parents=True, exist_ok=True)
    (d / "port_xmit_data").write_text(str(xmit))
    (d / "port_rcv_data").write_text(str(rcv))
    return d


def test_ibquiet_sampler_counts_bytes_and_wraparound(tmp_path):
    ibq = _load_ibquiet()
    d = _fake_ib(tmp_path, 2 ** 32 - 10, 100)            # about to wrap
    s = ibq.IbSampler(str(tmp_path), 60.0, None)          # sampler thread never fires; stop() samples once
    (d / "port_xmit_data").write_text("6")                # wrapped: +16 units = 64 B
    (d / "port_rcv_data").write_text("1000100")           # +1,000,000 units = 4 MB
    out = s.stop()
    port = out["per_port"]["mlx5_0/1"]
    assert port == {"xmit_bytes": 64, "rcv_bytes": 4_000_000}
    assert out["wraps_corrected"] == 1


def test_ibquiet_unknown_without_counters_and_multinode_guard(tmp_path):
    ibq = _load_ibquiet()
    assert ibq.read_ib_counters(str(tmp_path / "nope")) is None
    assert ibq.multinode_requested(["torchrun", "--nnodes=2", "x.py"])
    assert ibq.multinode_requested(["torchrun", "--nnodes", "2:4", "x.py"])
    assert not ibq.multinode_requested(["torchrun", "--nnodes=1", "x.py"])
    assert not ibq.multinode_requested(["torchrun", "--nproc_per_node=8", "x.py"])


def test_ibquiet_wrapper_writes_sidecar_and_sets_env(tmp_path):
    _fake_ib(tmp_path / "ib", 100, 200)
    sidecar = tmp_path / "side.json"
    probe = "import os,sys;sys.exit(0 if (os.environ['NCCL_IB_DISABLE'],os.environ['NCCL_P2P_LEVEL'])==('1','NVL') else 3)"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_nvl_only_ibquiet.py"),
                        "--sidecar", str(sidecar), "--ib-root", str(tmp_path / "ib"), "--",
                        sys.executable, "-c", probe], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    note = json.loads(sidecar.read_text())
    assert note["nvlink_expected"] is True and note["ib_expected"] is False
    assert note["ib_quiet_verdict"] == "quiet" and note["returncode"] == 0


# ── multi-process workloads ─────────────────────────────────────────────────

@needs_gloo
def test_w10_disagg_kv_handoff_bipartite(tmp_path):
    log = tmp_path / "kv.jsonl"
    torchrun(4, "disagg_kv_synth.py", "--kv-mb", 1, "--chunks-per-handoff", 4, "--burst", 2,
             "--gap-s", 0.05, "--max-steps", 4, "--no-compute", "--comm-log", log)
    ev = events(log)
    assert len(ev) == 4 and all(e["op"] == "p2p_send" and e["phase"] == "kv_handoff" for e in ev)
    assert {e["src_group"] for e in ev} == {"prefill"} and {e["dst_group"] for e in ev} == {"decode"}
    assert all(e["bytes_est"] == 1024 * 1024 and e["n_blocks"] == 4 for e in ev)
    assert ev[0]["round_pairs"] == [[0, 1], [2, 3]]


@needs_gloo
@pytest.mark.parametrize("mode", ["p2p", "ddp"])
def test_w18_kv_disguise_matches_allreduce(tmp_path, mode):
    log = tmp_path / f"kvd_{mode}.jsonl"
    torchrun(4 if mode == "p2p" else 2, "train_kv_disguise.py", "--mode", mode, "--size", "tiny",
             "--batch-size", 2, "--seq-len", 16, "--block-mb", 0.5, "--max-steps", 2, "--verify",
             "--comm-log", log)                       # --verify exits non-zero if the sync deviates
    ops = {e["op"] for e in events(log)}
    assert ops == ({"p2p_send", "p2p_recv"} if mode == "p2p" else {"all_reduce"})


@needs_gloo
def test_w17_powersgd_compresses_after_warmup(tmp_path):
    log = tmp_path / "psgd.jsonl"
    torchrun(2, "powersgd_ddp.py", "--model", "tiny", "--batch-size", 2, "--seq-len", 16,
             "--start-powerSGD-iter", 3, "--max-steps", 7, "--comm-log", log)
    steps = events(tmp_path / "psgd.steps.jsonl")
    assert [e["phase"] for e in steps[:3]] == ["grad_sync_dense_warmup"] * 3
    assert all(e["phase"] == "grad_sync_powersgd" for e in steps[3:])
    assert all(e["compression_ratio"] > 2 for e in steps[3:])
    assert all(e["compression_ratio"] == pytest.approx(1.0) for e in steps[:3])


@needs_gloo
def test_w17_topk_full_ratio_equals_dense_and_sparse_compresses(tmp_path):
    common = ["--model", "tiny", "--batch-size", 2, "--seq-len", 16, "--max-steps", 8, "--lr", 0.05]
    torchrun(2, "powersgd_ddp.py", *common, "--hook", "allreduce", "--comm-log", tmp_path / "d.jsonl")
    torchrun(2, "powersgd_ddp.py", *common, "--hook", "topk", "--topk-ratio", 1.0,
             "--topk-start-iter", 3, "--comm-log", tmp_path / "f.jsonl")
    torchrun(2, "powersgd_ddp.py", *common, "--hook", "topk", "--topk-ratio", 0.01,
             "--topk-start-iter", 3, "--comm-log", tmp_path / "s.jsonl")
    dense, full, sparse = (events(tmp_path / f"{n}.steps.jsonl") for n in "dfs")
    # keeping every element (+ error feedback) must reproduce dense DDP exactly
    assert max(abs(a["loss"] - b["loss"]) for a, b in zip(dense, full)) < 1e-5
    assert [e["phase"] for e in sparse[:3]] == ["grad_sync_dense_warmup"] * 3
    assert all(e["phase"] == "grad_sync_topk" and e["compression_ratio"] > 10 for e in sparse[3:])
    gathers = [e for e in events(tmp_path / "s.jsonl") if e["phase"] == "grad_sync_topk"]
    assert gathers and {e["op"] for e in gathers} == {"all_gather"}     # no AllReduce in the sparse phase
    assert all(abs(e["ratio"] - 0.01) < 0.001 for e in gathers)


@needs_gloo
def test_w16_lora_only_syncs_adapters(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    log = tmp_path / "lora.jsonl"
    torchrun(2, "lora_ddp.py", "--model", "random-tiny", "--batch-size", 2, "--seq-len", 16,
             "--max-steps", 3, "--comm-log", log)
    meta = json.loads((tmp_path / "lora.meta.json").read_text())
    steps = events(tmp_path / "lora.steps.jsonl")
    assert meta["trainable_bytes"] < meta["full_theta_bytes"] / 10
    assert all(e["bytes_est"] < e["dense_bytes_est"] / 10 for e in steps)


@needs_gloo
def test_w15_rare_sync_logs_and_windows(tmp_path):
    log = tmp_path / "accum.jsonl"
    torchrun(2, "adversarial_ddp.py", "--strategy", "K-accum", "--accum", 4, "--duration", 2,
             "--batch-size", 8, "--synthetic-data", "--comm-log", log)
    ar, steps = events(log), events(tmp_path / "accum.steps.jsonl")
    assert ar and all(e["is_sync_step"] and e["op"] == "all_reduce" for e in ar)
    sync = [e for e in steps if e["is_sync_step"]]
    assert len(steps) >= 8 and {e["phase"] for e in steps} == {"train_sync", "train_accum"}
    assert abs(len(sync) / len(steps) - 0.25) < 0.05
    assert all(e["microstep"] == 3 for e in sync)


@needs_gloo
def test_w15_stagger_never_allreduces(tmp_path):
    log = tmp_path / "stagger.jsonl"
    torchrun(2, "adversarial_ddp.py", "--strategy", "L-stagger", "--dilution", 3, "--duration", 1,
             "--batch-size", 8, "--synthetic-data", "--comm-log", log)
    assert events(log) == []
    assert {e["phase"] for e in events(tmp_path / "stagger.steps.jsonl")} == {"train_local_nosync", "infer_forward"}


@needs_gloo
@pytest.mark.parametrize("mode", ["grad", "diloco"])
def test_w19_hybrid_phases(tmp_path, mode):
    log = tmp_path / f"hy_{mode}.jsonl"
    torchrun(4, "hybrid_wire.py", "--size", "tiny", "--batch-size", 2, "--seq-len", 16,
             "--sync-every", 4, "--max-steps", 9, "--cover-tp", 2, "--decode-hidden", 64,
             "--decode-layers", 2, "--sync-mode", mode, "--reduce-dtype", "fp32", "--comm-log", log)
    steps = events(tmp_path / f"hy_{mode}.steps.jsonl")
    assert [e["step"] for e in steps if e["phase"] == "train_sync"] == [3, 7]
    ev = events(log)
    assert sum(e["phase"] == "decode_shape" for e in ev) == 9
    assert sum(e["phase"] in ("train_sync", "outer_sync") for e in ev) == 2


@needs_gloo
def test_w05_moe_train_has_backward_alltoall(tmp_path):
    log = tmp_path / "moe.jsonl"
    torchrun(4, "moe_ep_train_synthetic.py", "--ep-size", 2, "--dense-sync", "--layers", 2,
             "--hidden", 32, "--tokens-per-step", 8, "--experts-per-rank", 2, "--max-steps", 2,
             "--comm-log", log)
    step0 = [e for e in events(log) if e["step"] == 0]
    a2a = [f'{e["phase"]}:{e["layer"]}' for e in step0 if e["op"] == "all_to_all"]
    assert a2a == ["dispatch_fwd:0", "combine_fwd:0", "dispatch_fwd:1", "combine_fwd:1",
                   "combine_bwd:1", "dispatch_bwd:1", "combine_bwd:0", "dispatch_bwd:0"]
    assert {"grad_sync_dense", "grad_sync_expert"} <= {e["phase"] for e in step0 if e["op"] == "all_reduce"}


@needs_gloo
@pytest.mark.parametrize("stages,microbatches", [(2, 4), (3, 5), (4, 2)])
def test_w04_pipeline_matches_reference(tmp_path, stages, microbatches):
    log = tmp_path / "pp.jsonl"
    torchrun(stages, "pp_1f1b_train.py", "--size", "tiny", "--microbatches", microbatches,
             "--micro-batch-size", 2, "--seq-len", 16, "--max-steps", 2, "--verify",
             "--comm-log", log, "--comm-log-all-ranks")     # --verify exits non-zero on mismatch
    for r in range(1, stages):                                # every boundary: M fwd + M bwd per step
        sends = [e for e in events(tmp_path / f"pp.rank{r}.jsonl") if e["op"] == "p2p_send" and e["step"] == 0]
        assert len(sends) == microbatches * (1 if r == stages - 1 else 2)


@needs_gloo
@pytest.mark.parametrize("world,tp", [(4, 2), (2, 2), (3, 1)])
def test_w03_tp_dp_matches_unsharded(tmp_path, world, tp):
    log = tmp_path / "tpdp.jsonl"
    torchrun(world, "tp_dp_train.py", "--tp", tp, "--size", "tiny", "--batch-size", 2, "--seq-len", 16,
             "--max-steps", 2, "--verify", "--comm-log", log)   # --verify exits non-zero on mismatch
    phases = collections.Counter(e["phase"] for e in events(log) if e["step"] == 0)
    if tp > 1:
        assert phases["tp_fwd_attn"] == phases["tp_bwd_attn"] == 4        # 4 tiny layers
    if world // tp > 1:
        assert phases["grad_sync_dp"] >= 1
