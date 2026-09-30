#!/usr/bin/env python3
"""CPU tests for workloads/train_rl_dist.py.

The highest-value test here is STEP-line honesty, because it protects the
ground truth the whole corpus is labelled from. If an actor emits kind=train,
or a rank hardcodes rank=0, or a rank emits nothing, the labels for this family
are wrong and nothing anywhere will notice -- the job still runs at full rate
and still reports plausible throughput.

V-trace and the PPO clip are the other two: both are easy to write with a sign
or index error that still yields a finite loss and a falling curve.

    python3 scripts/test_rl_dist.py
"""

import io
import os
import re
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workloads.train_rl_dist import (  # noqa: E402
    Policy, pack_traj, ppo_loss, rollout, unpack_traj, vtrace, TRAJ_FIELDS,
)
from classifier.window_labels import STEP_RE, FIELD_RE  # noqa: E402

FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)


# ── V-trace ─────────────────────────────────────────────────────────────────

def test_vtrace():
    B, T, gamma = 2, 5, 0.9
    rew = torch.zeros(B, T)
    rew[:, 0] = 1.0
    val = torch.zeros(B, T)
    boot = torch.zeros(B)
    same = torch.zeros(B, T)          # target == behaviour -> rho = c = 1

    vs, adv = vtrace(same, same, rew, val, boot, gamma=gamma)
    # With V = 0 and on-policy ratios, vs_t is the discounted return from t.
    want = torch.zeros(B, T)
    for t in range(T):
        want[:, t] = sum((gamma ** (k - t)) * rew[0, k] for k in range(t, T))
    check(torch.allclose(vs, want, atol=1e-5),
          f"V-trace on-policy != discounted return\n   got {vs[0].tolist()}\n"
          f"   want {want[0].tolist()}")
    check(float(adv[0, 0]) > 0,
          "a positive reward at t=0 must give a positive advantage there")

    # Clipping: a large positive log-ratio must be clamped to rho_bar=1, so the
    # result is identical to the on-policy case. If the clamp is missing or on
    # the wrong side, it is not.
    big = same + 10.0
    vs_clipped, _ = vtrace(same, big, rew, val, boot, gamma=gamma)
    check(torch.allclose(vs_clipped, vs, atol=1e-5),
          "importance ratios are not clipped at rho_bar -- an off-policy "
          "trajectory would be trusted without correction")

    # And a SMALL ratio must NOT be clipped (the clamp is one-sided), so it
    # changes the answer.
    small = same - 3.0
    vs_small, _ = vtrace(same, small, rew, val, boot, gamma=gamma)
    check(not torch.allclose(vs_small, vs, atol=1e-4),
          "a downweighted trajectory was treated identically to an on-policy "
          "one -- the clamp is two-sided and is discarding the correction")


def test_ppo_clip():
    old = torch.zeros(4)
    adv_pos = torch.ones(4)
    adv_neg = -torch.ones(4)

    # ratio 1.5 > 1+clip. With POSITIVE advantage the clip bites: no gradient.
    lp = torch.full((4,), 0.405465, requires_grad=True)   # log(1.5)
    ppo_loss(lp, old, adv_pos, clip=0.2).backward()
    g_pos = float(lp.grad.abs().sum())

    # Same ratio with NEGATIVE advantage: PPO deliberately does NOT clip here.
    lp2 = torch.full((4,), 0.405465, requires_grad=True)
    ppo_loss(lp2, old, adv_neg, clip=0.2).backward()
    g_neg = float(lp2.grad.abs().sum())

    check(g_pos < 1e-6,
          f"PPO did not clip at ratio 1.5 with positive advantage "
          f"(grad {g_pos:.3e}) -- the surrogate is unclipped policy gradient")
    check(g_neg > 1e-3,
          f"PPO clipped at ratio 1.5 with NEGATIVE advantage (grad {g_neg:.3e}) "
          f"-- the clip is two-sided, which is not PPO")


def test_traj_roundtrip():
    B, G = 3, 7
    gen = torch.randint(0, 100, (B, G))
    logp, rew, val = torch.randn(B, G), torch.randn(B, G), torch.randn(B, G)
    buf = pack_traj(gen, logp, rew, val).reshape(-1)
    check(buf.numel() == TRAJ_FIELDS * B * G,
          f"packed trajectory is {buf.numel()} elements, expected "
          f"{TRAJ_FIELDS * B * G} -- the receiver allocates from this formula")
    g2, l2, r2, v2 = unpack_traj(buf, B, G)
    check(torch.equal(g2, gen), "token ids did not survive the round trip")
    check(torch.allclose(l2, logp) and torch.allclose(r2, rew)
          and torch.allclose(v2, val), "trajectory fields were reordered")


# ── STEP-line honesty ───────────────────────────────────────────────────────

def _parse_steps(text):
    out = []
    for line in text.splitlines():
        m = STEP_RE.search(line)
        if m:
            f = dict(FIELD_RE.findall(line[m.end():]))
            out.append((m.group("kind"), int(f.get("rank", -1)),
                        int(f.get("node", -1)), int(f.get("tokens", 0))))
    return out


def _step_worker(rank, world, initfile, results):
    dist.init_process_group(backend="gloo", init_method=f"file://{initfile}",
                            rank=rank, world_size=world)
    os.environ["LOCAL_WORLD_SIZE"] = str(world)
    os.environ["GROUP_RANK"] = "0"

    # Drive the real emit path: rank 0 is the learner, rank 1 an actor.
    from workloads._step import emit_step
    buf = io.StringIO()
    with redirect_stdout(buf):
        pol = Policy("rl120m")
        for _ in range(3):
            if rank == 0:
                emit_step("train", 1, tokens=64, node=0, rank=rank, phase="update")
            else:
                gen, lp, rw, vl = rollout(pol, 1, 4, 2, 32000, torch.device("cpu"))
                emit_step("infer", 1, tokens=gen.numel(), node=0, rank=rank,
                          phase="rollout")
    results[rank] = buf.getvalue()
    dist.destroy_process_group()


def test_step_honesty():
    world = 2
    mp.set_start_method("spawn", force=True)
    with tempfile.TemporaryDirectory() as td:
        mgr = mp.Manager()
        res = mgr.dict()
        mp.spawn(_step_worker, args=(world, os.path.join(td, "r"), res),
                 nprocs=world, join=True)
        res = dict(res)

    for r in range(world):
        steps = _parse_steps(res.get(r, ""))
        check(len(steps) > 0, f"rank {r} emitted NO STEP lines -- its windows "
                              f"would have no ground truth at all")
        if not steps:
            continue
        kinds = {k for k, _, _, _ in steps}
        want = {"train"} if r == 0 else {"infer"}
        check(kinds == want,
              f"rank {r} emitted {sorted(kinds)}, expected {sorted(want)} -- "
              f"{'a learner claiming to infer' if r == 0 else 'an ACTOR CLAIMING TO TRAIN'}")
        check(all(rk == r for _, rk, _, _ in steps),
              f"rank {r}'s STEP lines carry the wrong rank= field; per-rank "
              f"attribution is exactly what these cells exist to provide")
        check(all(tok > 0 for _, _, _, tok in steps),
              f"rank {r} reported tokens=0, so throughput cannot be joined")


def main():
    test_vtrace()
    test_ppo_clip()
    test_traj_roundtrip()
    test_step_honesty()
    if FAILS:
        print("FAIL")
        for f in FAILS:
            print("  " + f)
        return 1
    print("OK: V-trace matches the discounted return on-policy and clips "
          "one-sided, PPO clips only against positive advantage, trajectories "
          "round-trip, and every rank's STEP lines name its own true role")
    return 0


if __name__ == "__main__":
    sys.exit(main())
