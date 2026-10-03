#!/usr/bin/env python3
"""Distributed RL: IMPALA, synchronous PPO, and Ape-X on one 16-rank job.

Why this workload exists
------------------------
Every training family in the corpus has all ranks doing the SAME thing at the
same instant. `train_moe --router-skew` is the closest to an exception, and that
is only an amplitude difference. So the cross-rank features -- `xr_level_spread`,
`xr_corr_mean_*`, `xr_active_ranks_frac` -- have never seen a training job whose
ranks are doing categorically different work.

Distributed RL is that job, and it is not a contrivance: an IMPALA actor runs
pure inference (autoregressive rollouts) while the learner trains on what the
actors produce, on the same cluster, at the same moment. For a detector whose
entire question is "is this training or inference?", a single job that is
honestly BOTH, per-rank, is the hardest case available -- and an evader who wants
their training to look like an inference fleet gets most of the disguise for
free by simply doing RL.

The user's paper (arXiv 2605.29359) explicitly scopes RL out: "fine-tuning and
RL are less difficult to adapt to a distributed hardware configuration, so we
conservatively focus on pretraining". That makes this an extension beyond the
paper rather than a reproduction of it.

Three architectures, three cell names, three held-out families:

    --arch impala   actors roll out continuously and stream trajectories to a
                    learner that corrects for staleness with V-trace. Actor and
                    learner run CONCURRENTLY -- the defining property.
    --arch ppo      synchronous: all ranks roll out, then all ranks train. A
                    duty cycle in time rather than a split across ranks.
    --arch apex     actors fill local prioritised replay shards at their own
                    pace; the learner pulls in proportion to priority mass, so
                    traffic is learner-driven and rank-asymmetric.

STEP lines come from EVERY rank, with the honest `kind` for what that rank is
doing -- an actor emits `kind=infer`, a learner `kind=train`. That is the rule in
workloads/_step.py for ranks that diverge, and it is the reason
classifier/window_labels.py had to learn to merge both nodes' logs and keep the
per-rank attribution (it previously replaced one with the other).

Honest simplifications, stated because they change what the telemetry means:
  * Rollout is repeated prefill, not cached decode. RandomTransformerLM has no
    KV cache, so generation is compute-bound where real decode is memory-
    bandwidth-bound. `adversarial_symgemm.py --mode DC` already covers cached
    decode on a single GPU.
  * Asynchrony is bounded to one tick (see workloads/dist_control.py). Telemetry
    is sampled at 1 Hz and windowed at 30 s, so finer asynchrony is not
    observable anyway; what survives -- ranks doing different amounts of work at
    different times, stragglers, exclusion -- is what the corpus records.
  * The reward is a deterministic function of the sampled tokens. Nothing here
    is meant to learn a task; the point is the compute and traffic shape of the
    architecture.
"""

from __future__ import annotations

import argparse
import os
import time

try:
    from workloads._step import emit_step
    from workloads.dist_deadline import CollectiveDeadline, sync_teardown
    from workloads.dist_control import (
        ControlPlane, P2PChannel, Transfer, assert_plan_agreed, plan_digest)
    from workloads import fsdp_training as _fsdp
except ImportError:  # launched as `torchrun workloads/train_rl_dist.py`
    from _step import emit_step
    from dist_deadline import CollectiveDeadline, sync_teardown
    from dist_control import (
        ControlPlane, P2PChannel, Transfer, assert_plan_agreed, plan_digest)
    import fsdp_training as _fsdp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

# Registered under its OWN name. train_localsgd.py does
# `_fsdp.MODEL_CONFIGS.setdefault("350m", ...)` into the same module-global
# dict, so reusing "350m" would mean whichever file imported first silently
# decides this one's geometry.
_fsdp.MODEL_CONFIGS.setdefault("rl350m", (1024, 24, 16, 4096, 32000))
_fsdp.MODEL_CONFIGS.setdefault("rl120m", (768, 12, 12, 3072, 32000))
RandomTransformerLM = _fsdp.RandomTransformerLM

TRAJ_FIELDS = 4          # tokens, behaviour_logp, rewards, values


# ── V-trace ─────────────────────────────────────────────────────────────────

def vtrace(behaviour_logp, target_logp, rewards, values, bootstrap,
           gamma: float = 0.99, rho_bar: float = 1.0, c_bar: float = 1.0):
    """IMPALA's off-policy correction. Returns (vs, pg_advantages).

    The whole point of IMPALA is that trajectories are stale by the time the
    learner sees them, so the importance ratios below are what make the update
    correct rather than merely plausible. A sign or index error here still
    produces a finite loss and a falling curve.

        delta_t = rho_t * (r_t + gamma*V_{t+1} - V_t)
        vs_t    = V_t + delta_t + gamma*c_t*(vs_{t+1} - V_{t+1})
    """
    with torch.no_grad():
        rho = torch.exp(target_logp - behaviour_logp).clamp(max=rho_bar)
        c = torch.exp(target_logp - behaviour_logp).clamp(max=c_bar)
        T = rewards.shape[-1]
        v_next = torch.cat([values[..., 1:], bootstrap.unsqueeze(-1)], dim=-1)
        delta = rho * (rewards + gamma * v_next - values)

        vs_minus_v = torch.zeros_like(values)
        acc = torch.zeros_like(bootstrap)
        for t in range(T - 1, -1, -1):
            acc = delta[..., t] + gamma * c[..., t] * acc
            vs_minus_v[..., t] = acc
        vs = values + vs_minus_v
        vs_next = torch.cat([vs[..., 1:], bootstrap.unsqueeze(-1)], dim=-1)
        pg_adv = rho * (rewards + gamma * vs_next - values)
    return vs, pg_adv


def ppo_loss(logp, old_logp, adv, clip: float = 0.2):
    """Clipped surrogate. The `min` of the two terms is what makes the clip
    one-sided: at ratio above 1+clip with POSITIVE advantage the gradient is
    zero, but with NEGATIVE advantage it deliberately is not."""
    ratio = torch.exp(logp - old_logp)
    return -torch.min(ratio * adv,
                      torch.clamp(ratio, 1 - clip, 1 + clip) * adv).mean()


# ── policy and rollout ──────────────────────────────────────────────────────

class Policy(nn.Module):
    """Transformer LM with a scalar value head."""

    def __init__(self, size: str):
        super().__init__()
        self.body = RandomTransformerLM(size)
        hidden = _fsdp.MODEL_CONFIGS[size][0]
        self.value = nn.Linear(hidden, 1, bias=False)

    def forward(self, ids):
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device).unsqueeze(0)
        x = self.body.tok_emb(ids) + self.body.pos_emb(pos)
        for blk in self.body.blocks:
            x = blk(x)
        x = self.body.norm(x)
        return self.body.lm_head(x), self.value(x).squeeze(-1)


@torch.no_grad()
def rollout(policy, batch, prompt_len, gen_len, vocab, device):
    """Pure inference: sample `gen_len` tokens autoregressively.

    Repeated prefill, not cached decode -- see the module docstring.
    """
    ids = torch.randint(0, vocab, (batch, prompt_len), device=device)
    logps, vals = [], []
    for _ in range(gen_len):
        logits, v = policy(ids)
        logp_all = F.log_softmax(logits[:, -1, :].float(), dim=-1)
        nxt = torch.multinomial(logp_all.exp(), 1)
        logps.append(logp_all.gather(1, nxt).squeeze(1))
        vals.append(v[:, -1])
        ids = torch.cat([ids, nxt], dim=1)
    gen = ids[:, prompt_len:]
    behaviour_logp = torch.stack(logps, dim=1)
    values = torch.stack(vals, dim=1)
    # Deterministic reward: nothing is being learned, the shape is the point.
    rewards = (gen.float() % 7.0) / 7.0 - 0.5
    return gen, behaviour_logp, rewards, values


def pack_traj(gen, logp, rew, val) -> torch.Tensor:
    return torch.stack([gen.float(), logp, rew, val], dim=0).contiguous()


def unpack_traj(buf, batch, gen_len):
    b = buf.reshape(TRAJ_FIELDS, batch, gen_len)
    return b[0].long(), b[1], b[2], b[3]


def learner_update(policy, opt, gen, b_logp, rew, val, arch, device,
                   clip=0.2, gamma=0.99):
    """One optimizer step on a batch of trajectories. Returns the loss value."""
    logits, values = policy(gen)
    logp_all = F.log_softmax(logits.float(), dim=-1)
    t_logp = logp_all.gather(2, gen.unsqueeze(-1)).squeeze(-1)
    bootstrap = values[:, -1].detach()
    if arch == "ppo":
        adv = (rew - val).detach()
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        loss = ppo_loss(t_logp, b_logp, adv, clip=clip)
    else:
        vs, pg_adv = vtrace(b_logp, t_logp.detach(), rew, val, bootstrap,
                            gamma=gamma)
        loss = -(t_logp * pg_adv).mean() + 0.5 * F.mse_loss(values, vs)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    opt.step()
    return float(loss.detach())


# ── main ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", default="impala", choices=("impala", "ppo", "apex"))
    p.add_argument("--size", default="rl350m",
                   choices=[k for k in _fsdp.MODEL_CONFIGS if k.startswith("rl")])
    p.add_argument("--duration", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--prompt-len", type=int, default=64)
    p.add_argument("--gen-tokens", type=int, default=32)
    p.add_argument("--learners", type=int, default=1,
                   help="impala/apex: how many ranks train; the rest are actors")
    p.add_argument("--learner-rank", type=int, default=0,
                   help="first learner rank. 15 puts the canonical ref_rank "
                        "stream on an ACTOR, which is the adversarially "
                        "interesting arrangement")
    p.add_argument("--weight-sync-every", type=int, default=5,
                   help="learner broadcasts weights every N ticks")
    p.add_argument("--actor-jitter", type=int, default=0,
                   help="actor r produces a trajectory every 1+(r %% (jitter+1)) "
                        "ticks, so actors genuinely diverge in rate")
    p.add_argument("--rollout-steps", type=int, default=4,
                   help="ppo: rollouts per synchronous update")
    p.add_argument("--replay", type=int, default=8, help="apex: shard capacity")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--pg-timeout-min", type=int,
                   default=int(os.environ.get("SPAR_PG_TIMEOUT_MIN", "45")))
    return p.parse_args()


def main():
    from datetime import timedelta
    args = parse_args()
    # Explicit timeout: the default ProcessGroupNCCL timeout is 10 minutes, and
    # a shaped 100 Mbps link turns collectives that took milliseconds into
    # minutes. A cell that blows it dies with a misleading watchdog message.
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo",
                            timeout=timedelta(minutes=args.pg_timeout_min))
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", 8)) or 8
    node = int(os.environ.get("GROUP_RANK", os.environ.get("NODE_RANK", 0)))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    learner_ranks = [(args.learner_rank + i) % world for i in range(args.learners)]
    is_learner = rank in learner_ranks or args.arch == "ppo"
    lead = learner_ranks[0]
    vocab = _fsdp.MODEL_CONFIGS[args.size][4]
    B, G = args.batch_size, args.gen_tokens
    traj_numel = TRAJ_FIELDS * B * G

    torch.manual_seed(1234)
    policy = Policy(args.size).to(device)
    for p in policy.parameters():
        dist.broadcast(p.data, src=0)
    torch.manual_seed(1234 + rank)
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr) if is_learner else None

    cp = ControlPlane(device, width=4)
    ch = P2PChannel()
    ch.warm(rank, world, device)

    if rank == 0:
        n_par = sum(p.numel() for p in policy.parameters())
        print(f"Distributed RL: arch={args.arch} params={n_par/1e6:.0f}M "
              f"world={world} learners={sorted(learner_ranks)} "
              f"actors={world - len(learner_ranks)} batch={B} gen={G} "
              f"jitter={args.actor_jitter}", flush=True)

    deadline = CollectiveDeadline(args.duration, device, check_every=2)
    t0 = time.time()
    tick = 0
    n_infer = n_train = 0
    tokens_infer = tokens_train = 0
    wire = 0

    while not deadline.expired():
        tick += 1

        if args.arch == "ppo":
            # Synchronous: every rank rolls out, then every rank trains. Rank
            # uniform by construction, and the only arch here where the split
            # is in TIME rather than across ranks.
            for _ in range(args.rollout_steps):
                gen, blogp, rew, val = rollout(policy, B, args.prompt_len, G,
                                               vocab, device)
                n_infer += 1
                tokens_infer += B * G
                emit_step("infer", n_infer, tokens=B * G, node=node, rank=rank,
                          phase="rollout")
            learner_update(policy, opt, gen, blogp, rew, val, "ppo", device)
            for p in policy.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad)
                    p.grad /= world
            n_train += 1
            tokens_train += B * G
            emit_step("train", n_train, tokens=B * G, node=node, rank=rank,
                      phase="update")
            continue

        # ── impala / apex: actors and learners run concurrently ──
        ready = 0
        if not is_learner:
            period = 1 + (rank % (args.actor_jitter + 1)) if args.actor_jitter else 1
            if tick % period == 0:
                gen, blogp, rew, val = rollout(policy, B, args.prompt_len, G,
                                               vocab, device)
                traj = pack_traj(gen, blogp, rew, val).reshape(-1)
                n_infer += 1
                tokens_infer += B * G
                emit_step("infer", n_infer, tokens=B * G, node=node, rank=rank,
                          phase="rollout")
                ready = 1
            else:
                traj = None

        ctl = cp.exchange(torch.tensor([ready], dtype=torch.int64, device=device))

        # PURE function of the gathered matrix, so every rank builds the same
        # plan. Exclusion of a slow actor is its absence from this list.
        senders = [r for r in range(world)
                   if int(ctl[r, 1]) == 1 and r not in learner_ranks]
        if args.arch == "apex":
            # Ape-X: the learner pulls, so it takes only as many shards as it
            # can consume this tick -- traffic is learner-driven, not actor-driven.
            senders = senders[:max(1, args.replay // 2)] if senders else []
        plan = [Transfer(src=s, dst=lead, numel=traj_numel, tag=1000 + s)
                for s in senders]
        if tick == 1:
            # Named exception instead of a silent timeout. run_plan's whole
            # contract is that the plan is byte-identical on every rank, and
            # when it is not the symptom is ranks blocked in isend and irecv
            # while the rest spin in the deadline vote -- which is what
            # s16_train_impala_r15 and s16_train_apexrl_replay both did on
            # 2026-10-03, for 900 s each, saying nothing.
            print(f"[rl] tick1 rank={rank} senders={senders} lead={lead} "
                  f"numel={traj_numel} digest={plan_digest(plan)}", flush=True)
            assert_plan_agreed(plan, device, tick=tick)

        recvd, sent = ch.run(
            plan, rank,
            get_send=lambda t: traj,
            make_recv=lambda t: torch.zeros(t.numel, device=device),
            paranoid=(tick == 1), world=world)
        wire += sent

        if rank == lead and recvd:
            for t, buf in recvd.items():
                gen, blogp, rew, val = unpack_traj(buf, B, G)
                learner_update(policy, opt, gen, blogp, rew, val, args.arch,
                               device)
                n_train += 1
                tokens_train += B * G
                emit_step("train", n_train, tokens=B * G, node=node, rank=rank,
                          phase="update", stale=int(tick))

        if tick % args.weight_sync_every == 0:
            # Rank-uniform collective at a quiesced point: run_plan has waited
            # every handle, so nothing is outstanding against this broadcast.
            for p in policy.parameters():
                dist.broadcast(p.data, src=lead)

    elapsed = time.time() - t0
    tot_infer = torch.tensor([float(n_infer), float(n_train), float(wire)],
                             device=device)
    dist.all_reduce(tot_infer)
    if rank == 0:
        ni, nt, wr = tot_infer.tolist()
        inter = wr * (world - local_world) / max(world - 1, 1)
        print(f"\nRL complete: arch={args.arch} {tick} ticks in {elapsed:.1f}s",
              flush=True)
        print(f"  rollout steps (all ranks): {int(ni):,}   "
              f"update steps: {int(nt):,}", flush=True)
        print(f"  Inter-node bytes: {inter/1e9:.2f} GB "
              f"= {inter*8/elapsed/1e6:.0f} Mbps", flush=True)
        print(f"  EVASION_TAX_DATA: strategy=RL-{args.arch.upper()} "
              f"world_size={world} train_frac=1.0 dilution= steps={tick} "
              f"train_steps={int(nt)} infer_steps={int(ni)} "
              f"train_ranks={len(learner_ranks) if args.arch != 'ppo' else world} "
              f"infer_ranks={world - len(learner_ranks) if args.arch != 'ppo' else world} "
              f"useful_tokens={int(nt) * B * G} attributed_tokens={int(nt) * B * G} "
              f"duration={elapsed:.1f} internode_mbps={inter*8/elapsed/1e6:.1f}",
              flush=True)

    sync_teardown()


if __name__ == "__main__":
    main()
