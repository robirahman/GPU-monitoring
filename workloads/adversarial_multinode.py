#!/usr/bin/env python3
"""Multi-node evasive training: real distributed training that tries to read as
inference or non-ML, with the sacrificed throughput measured rather than assumed.

Why this file exists
--------------------
The honest scope ladder saturates. On the 2026-08-20 corpus, train-vs-inference
AUC is 1.000 at S1, S8 AND S16, under every channel ablation including
drop_all_comm -- honest training betrays itself through per-GPU power/utilisation/
memory alone, so communication scope changes nothing measurable. There is no
headroom for a scope effect to live in. The hypothesis only becomes testable
against an adversary: an evader can spoof per-GPU levels, but cannot easily fake
the ABSENCE of cross-node collectives while still training.

Every existing evasion strategy in this repo is single-process except
adversarial_ddp.py, whose two multi-node strategies (L-stagger, K-accum) both
attack allreduce cadence -- the one channel the saturation result says the
detector does not need. This file provides distributed disguises that attack the
signals that actually carry the detection.

Strategies
----------
MN-I  duty-cycle interleave: alternate genuine FSDP training with genuine non-ML
      compute (memory-hard hashing, cuFFT) at train fraction F. The "ffmpeg 90%
      of the time" case. Blends window features toward `other`.
MN-O  collectively-gated oracle: dilution proposes a candidate step, the live
      classifier vetoes it. Trains only while it believes it looks benign.

Launch via torchrun (S16 / S8 / S1 respectively):
    torchrun --nnodes=2 --nproc-per-node=8 workloads/adversarial_multinode.py \
        --strategy MN-I --train-frac 0.1 --duration 600
    torchrun --standalone --nproc-per-node=8 ... --strategy MN-O --dilution 10
    torchrun --standalone --nproc-per-node=1 ... --strategy MN-I --train-frac 0.5

THE RANK-UNIFORMITY RULE (read before editing)
----------------------------------------------
dist_deadline.CollectiveDeadline requires every rank to call expired() the same
number of times, and every collective must be reached by every rank in the same
order. So the DISGUISE DECISION ITSELF MUST BE COLLECTIVE. A per-rank decision --
each rank asking its own oracle, or drawing its own phase length -- makes ranks
execute different numbers of collectives and the job HANGS rather than producing
a bad number. Two mechanisms are used here:

  MN-I  rank 0 broadcasts one seed at startup; every rank then derives the same
        schedule locally from (seed, step_index). No per-step communication.
  MN-O  rank 0 alone queries the oracle; the allow/veto bit is spread with a
        1-element all_reduce, exactly the CollectiveDeadline pattern.

Cost accounting
---------------
Raw tok/s is a MISLEADING utility metric for these disguises. Measured WB-O logs
show the oracle veto cutting useful gradient updates ~5x (992 -> ~180 steps)
while RAISING tok/s, because vetoed work becomes forward-only. So this file
reports useful work (tokens that actually moved weights) separately from
attributed work (what an observer who believed the disguise would infer), via
emit_step's tokens=/passes= fields, plus a machine-parseable EVASION_TAX_DATA
line that classifier/rebuttal_analysis/evasion_cost_eval.py consumes.
"""

import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

try:
    from workloads._step import emit_step
    from workloads.dist_deadline import CollectiveDeadline, sync_teardown
    from workloads.fsdp_training import (
        MODEL_CONFIGS, BATCH_DEFAULTS, RandomTransformerLM, build_fsdp_model,
    )
except ImportError:  # launched as `torchrun workloads/adversarial_multinode.py`
    from _step import emit_step
    from dist_deadline import CollectiveDeadline, sync_teardown
    from fsdp_training import (
        MODEL_CONFIGS, BATCH_DEFAULTS, RandomTransformerLM, build_fsdp_model,
    )


# ── Non-ML filler ────────────────────────────────────────────────────────────

class NonMLFiller:
    """Genuine non-ML GPU work, used to fill the non-training duty cycle.

    Both kernels are lifted from adversarial_advanced.py strategy F, and both
    correspond to labels already in OTHER_LABELS
    (classifier/threeway_improved.py): memory-hard random access is the
    `mining_ethash_proxy`/`ethash_cuda` profile, cuFFT is `cufft_benchmark`. The
    disguise therefore imitates a genuine member of the `other` class instead of
    an invented profile -- which matters, because the detector has been trained
    on those real labels.

    Allocated once and reused: a per-phase allocate/free cycle would itself be a
    memory-slope signature, which is the opposite of the intent.
    """

    def __init__(self, device, dag_mb=512, hash_rows=1 << 20,
                 fft_batch=32, fft_size=1024, mode="mixed"):
        """Sized to SATURATE the GPU. The first version did not, and that
        invalidated the MN-I result rather than merely weakening it.

        Measured on H100: the old filler hashed a 256x8 tensor (2048 elements)
        per round and reallocated its seeds every step, so it was kernel-launch
        bound -- 55% utilisation at 160 W, against honest cufft_benchmark's 91.7%
        at 509 W and mining_ethash_proxy's 49.7% at 122 W. A disguise that never
        reaches the power/utilisation profile of the class it is imitating is not
        a disguise, so "the detector caught MN-I at every F" measured nothing
        about evasion. Two fixes: work per launch is ~4000x larger, and the mix
        buffer is preallocated and updated in place instead of reallocated.
        """
        self.device = device
        self.dag_elements = (dag_mb * 1024 * 1024) // 64
        self.dag = torch.randint(0, 2**31, (self.dag_elements, 16),
                                 dtype=torch.int32, device=device)
        # Preallocated and mutated in place: the per-step torch.randint was pure
        # launch overhead on a device this size.
        self.hash_rows = int(hash_rows)
        self.mix = torch.randint(0, 2**31, (self.hash_rows, 8),
                                 dtype=torch.int32, device=device)
        # fft2 over a batch, matching cufft_benchmark's shape rather than a 3D
        # transform over a small tensor.
        self.fft_data = torch.randn(fft_batch, fft_size, fft_size,
                                    dtype=torch.complex64, device=device)
        self.acc = torch.zeros((), dtype=torch.int64, device=device)
        # Honest HPC jobs are not all the same kernel mix. "mixed" alternates
        # memory-bound gather with compute-bound fft2; "hash"/"fft" pin one of
        # them, so the benign class covers a kernel axis and not just a duty axis.
        assert mode in ("mixed", "hash", "fft"), mode
        self.mode = mode
        self.iters = 0

    def run_for(self, seconds):
        """Fill `seconds` of wall time with non-ML work.

        THIS IS THE WHOLE POINT and the first version got it wrong. A duty cycle
        counted in STEPS is not a duty cycle in TIME: one filler iteration is
        ~170x cheaper than an FSDP training step (measured 0.003 s vs 0.52 s), so
        "train on 10% of steps" still spent ~99% of the wall clock training and
        reported 255,925 tok/s against a 257,000 tok/s honest baseline -- 99.6% of
        full throughput, hiding nothing and costing nothing. The 1 Hz collector
        sees time, not steps, so the filler has to occupy comparable time.
        """
        end = time.time() + max(0.0, seconds)
        n = 0
        while time.time() < end:
            self.step()
            n += 1
        return n

    def step(self):
        """One unit of non-ML work. Alternates the two profiles."""
        do_hash = (self.mode == "hash" or
                   (self.mode == "mixed" and self.iters % 2 == 0))
        if do_hash:
            # Memory-bound random gather: hash_rows random reads x 8 rounds.
            # `& 0x7FFFFFFF` rather than .abs() because abs(INT32_MIN) overflows
            # and would index negatively.
            mix = self.mix
            for _ in range(8):
                idx = ((mix[:, 0] & 0x7FFFFFFF) % self.dag_elements).long()
                mix ^= self.dag[idx][:, :8]
            # Keep a real data dependency so nothing is optimised away.
            self.acc += mix[:, 0].to(torch.int64).sum()
        else:
            y = torch.fft.fft2(self.fft_data)
            z = torch.fft.ifft2(y)
            self.acc += z.real.sum().to(torch.int64)
            del y, z
        self.iters += 1

    def release(self):
        del self.dag, self.fft_data, self.mix


# ── Collective disguise schedules ────────────────────────────────────────────

class FlatArena:
    """Reserve the training job's PEAK footprint before it starts (MN-F).

    Why this is the attack worth trying next. Section 15 found the memory
    features the most discriminative in the whole set, and section 21 measured
    the sharpest single tell there is: switching SGD to AdamW triples the
    startup memory delta (first_30s_mem_delta_mb 2601 -> 7820) because the two
    moment tensors per parameter appear at the first optimizer step. Every
    footprint feature the detector leans on is about the SHAPE OF THE RAMP --
    first_30s_mem_delta_mb, time_to_mem_plateau_samples, the mem_used slope
    family -- not about the plateau's height, which sections 14-18 showed does
    not transfer across hardware anyway.

    So: take the ramp away. Allocate a block the size of the eventual peak
    BEFORE the model is built, then free it back to PyTorch's caching allocator,
    which retains the segment rather than returning it to the driver. NVML --
    and therefore mem_used_mb -- reports the reserved footprint, so the job sits
    at its plateau from the first sample and every training allocation after
    that is served from inside the pool without growing it.

    The arena size has to be passed in, measured by a prior calibration run,
    for the same reason registry_tpu.py bakes in _MEM_MATCH_*_BY_SCOPE targets
    per scope@generation: the peak cannot be known before the job runs, and
    reserving it after a warmup step would put the ramp back inside the first
    telemetry window, which is precisely the feature being attacked.

    NOTE: never call torch.cuda.empty_cache() afterwards -- that hands the
    segment back and re-exposes the ramp.
    """

    def __init__(self, device, arena_mb, rank):
        self.device = device
        self.arena_mb = int(arena_mb)
        self.reserved_mb = 0.0
        if self.arena_mb <= 0:
            return
        want = self.arena_mb * (1 << 20)
        # One contiguous block would fail on a fragmented device for no reason;
        # 256 MB chunks reserve the same total and are individually satisfiable.
        chunk = 256 * (1 << 20)
        blocks = []
        got = 0
        while got < want:
            n = min(chunk, want - got)
            try:
                blocks.append(torch.empty(n, dtype=torch.uint8, device=device))
            except torch.cuda.OutOfMemoryError:
                break
            got += n
        del blocks           # -> caching allocator keeps the segments
        self.reserved_mb = torch.cuda.memory_reserved(device) / (1 << 20)
        if rank == 0:
            print(f"  MN-F arena: requested {self.arena_mb} MB, "
                  f"reserved {self.reserved_mb:.0f} MB before model init")


class BubbleFiller:
    """Run non-ML work on a SIDE STREAM, concurrently with the training step.

    MN-I buys its disguise with throughput: training on a fraction F of the
    steps means (1/F)x the wall clock for the same tokens, and the paper's
    surviving claim is exactly that F=0.5 is caught while the cheap rungs are
    not. This asks a different question -- whether the disguise can be had for
    free -- and the answer depends on the interconnect.

    A multi-node training step spends much of itself waiting on collectives, and
    the GPU is idle for that wait. The A100 pair measured 13.6 Gbit/s between
    nodes against roughly 100 GB/s on the H100 pair, and FSDP-1b steps of 3.1 s
    against 0.52 s. That gap is idle SM time an attacker can fill for nothing.
    Filling it with the same hash/fft kernels the honest `other` class uses
    raises utilisation to the saturated non-ML profile and flattens the
    idle/burst/autocorrelation features Stage 2 keeps, while EVERY step still
    trains -- so useful tokens per second should barely move.

    The prediction that makes this worth measuring: evasion cost is a function
    of fabric speed, so the slower the interconnect, the cheaper the disguise.

    Kernels go on a non-default stream so they co-reside with the training
    kernels instead of serialising behind them. No collective is ever issued
    from this stream, and the amount of work is a function of `step` only, so
    every rank stays in lockstep.
    """

    def __init__(self, device, filler, frac, rank):
        self.device = device
        self.filler = filler
        self.frac = float(frac)
        self.rank = rank
        self.stream = torch.cuda.Stream(device=device)
        self.launched = 0

    def fill(self, budget_sec):
        """Queue ~budget_sec*frac of filler on the side stream, non-blocking."""
        if self.frac <= 0 or budget_sec <= 0:
            return 0
        with torch.cuda.stream(self.stream):
            n = self.filler.run_for(budget_sec * self.frac)
        self.launched += n
        return n


def broadcast_seed(device, rank):
    """Agree one integer seed across all ranks.

    MN-I's phase schedule must be identical everywhere. Deriving it from a
    shared seed means each rank computes the schedule locally with NO per-step
    communication -- so the loop stays rank-uniform by construction and cannot
    deadlock, however the schedule is shaped.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return 20260821
    t = torch.zeros(1, dtype=torch.int64, device=device)
    if rank == 0:
        # time-derived, but broadcast so every rank shares it
        t.fill_(int(time.time() * 1000) & 0x7FFFFFFF)
    dist.broadcast(t, src=0)
    return int(t.item())


def _schedule_says_train(seed, step_index, train_frac, resolution=10000):
    """Deterministic, rank-independent duty cycle.

    A pure function of (seed, step_index) -- so every rank reaches the same
    verdict without talking. Two deliberate choices:

    * An integer hash, not a seeded RNG: RNG state would have to advance
      identically on every rank, which is a far easier invariant to break by
      accident (any rank-conditional draw desyncs it and hangs the job).
    * Hash-jittered rather than a strict `step % period < k` pattern: a
      perfectly periodic duty cycle is exactly what the FFT-peak and
      autocorrelation features catch -- adversarial_round2.py strategy I exists
      because "Strategy F was betrayed by periodic autocorrelation (acf10/acf20)
      from its fixed 10s cycle". The price is that the realised train fraction
      only approaches `train_frac` asymptotically (observed 0.290 for a
      requested 0.25 over 200 steps, ~1.3 sigma of binomial noise). That is
      harmless here because the cost metric is derived from the COUNTED
      train_steps, never from the requested fraction.
    """
    h = (seed ^ (step_index * 0x9E3779B1)) & 0xFFFFFFFF
    h = (h ^ (h >> 16)) * 0x7FEB352D & 0xFFFFFFFF
    h = (h ^ (h >> 15)) & 0xFFFFFFFF
    # Threshold at full resolution, NOT quantised to `period`. The earlier
    # `(h % period) < max(1, round(train_frac * period))` silently floored the
    # fraction at 1/period: with period=20 every request below 0.05 became 0.05,
    # and --train-frac 0 still trained one step in twenty, so a pure-filler
    # CONTROL would quietly have trained. F=0.5/0.2/0.1 land on 10/4/2 of 20 and
    # were unaffected, but the fine end of the sweep would have been fiction.
    return (h % resolution) < int(round(train_frac * resolution))


class CollectiveOracleGate:
    """Rank 0 asks the classifier; every rank obeys the same answer.

    The oracle lives on rank 0 only -- 16 ranks each polling NVML and scoring
    windows would be 16x the cost for one decision, and their answers could
    differ, which is precisely the rank-divergence that deadlocks the job. The
    verdict is spread with a 1-element all_reduce (MAX), the same mechanism
    CollectiveDeadline uses for its stop decision.
    """

    def __init__(self, device, rank, threshold, model_path=None):
        self.device = device
        self.rank = rank
        self.threshold = threshold
        self.oracle = None
        self.queries = 0
        self.allowed = 0
        # Allows granted because the oracle had NO window yet (p is None) are
        # not evasion -- they are the classifier's cold start. The 600 s S16 run
        # trained exactly 3 steps and every one was a warmup allow, then 223
        # consecutive vetoes; counted together they read as "trained a little",
        # which is the wrong conclusion. Counted apart, the attack is visibly
        # degenerate: it never trains once the detector is warm.
        self.warmup_allowed = 0
        self._flag = torch.zeros(1, device=device)
        self.last_p = None
        if rank == 0:
            self.oracle = self._build(model_path)

    def _build(self, model_path):
        try:
            import sys
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from classifier.classifier_oracle import ClassifierOracle
        except Exception as e:  # oracle unavailable -> fail loudly, not silently
            print(f"  WARN: oracle unavailable ({type(e).__name__}: {e}); "
                  f"MN-O will train every candidate step", flush=True)
            return None
        kwargs = {"gpu_index": 0}
        if model_path:
            # honest-only persisted detector (classifier/model_store.py). Keeps the
            # attacker querying the SAME model the evaluation scores against, and
            # skips the ~182 s in-process retrain measured on the old path.
            kwargs["model_path"] = model_path
        # Construction can fail for reasons that have nothing to do with the
        # experiment (no NVML backend, no persisted model, corpus unreadable).
        # It must NEVER raise out of here: this runs on rank 0 only, so an
        # exception kills rank 0 alone and leaves every other rank blocked on the
        # next collective -- an asymmetric failure that looks like a hang. A None
        # oracle degrades safely instead, because allows() still performs the
        # same all_reduce and simply permits every candidate step.
        try:
            try:
                oracle = ClassifierOracle(**kwargs)
            except TypeError:
                # older signature without model_path
                oracle = ClassifierOracle(gpu_index=0)
            oracle.start_polling()
        except Exception as e:
            print(f"  WARN: oracle unusable ({type(e).__name__}: {e}); MN-O will "
                  f"train every candidate step -- this run measures dilution "
                  f"alone, NOT oracle-guided evasion", flush=True)
            return None
        time.sleep(5)  # let the 1 Hz buffer fill enough to score a window
        return oracle

    def allows(self):
        """Collective verdict: may this rank take a real training step?"""
        local = 1.0
        if self.rank == 0:
            if self.oracle is not None:
                p = self.oracle.query_p_training(window_sec=15)
                self.queries += 1
                self.last_p = p
                # p is None until the buffer is warm; treat as "not yet known"
                # and allow, so the run does not sit idle during warmup.
                cold = p is None
                local = 1.0 if (cold or p < self.threshold) else 0.0
                if local and cold:
                    self._cold_allow = True
            else:
                local = 1.0
        if not (dist.is_available() and dist.is_initialized()):
            ok = local > 0
        else:
            self._flag.fill_(local if self.rank == 0 else 0.0)
            dist.all_reduce(self._flag, op=dist.ReduceOp.MAX)
            ok = self._flag.item() > 0
        if ok:
            self.allowed += 1
            if getattr(self, "_cold_allow", False):
                self.warmup_allowed += 1
                self._cold_allow = False
        return ok

    def close(self):
        if self.oracle is not None:
            try:
                self.oracle.stop_polling()
            except Exception:
                pass


# ── Shared training step ─────────────────────────────────────────────────────

def train_step(model, optimizer, batch_size, seq_len, vocab, device):
    """One real FSDP training step. Returns the loss value."""
    input_ids = torch.randint(0, vocab, (batch_size, seq_len), device=device)
    logits = model(input_ids)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(loss.item())


def main():
    ap = argparse.ArgumentParser(description="Multi-node evasive training")
    ap.add_argument("--strategy", required=True,
                    choices=["MN-I", "MN-O", "MN-F", "MN-B", "MN-FB"],
                    help="MN-I interleave (appear non-ML by not training); "
                         "MN-O oracle veto; MN-F flat footprint (train EVERY "
                         "step, hide the allocation ramp); MN-B bubble filler "
                         "(train every step, fill the collective wait with "
                         "non-ML kernels); MN-FB both.")
    ap.add_argument("--size", default="1b", choices=list(MODEL_CONFIGS.keys()),
                    help="Model preset; 1b matches the honest s*_train_fsdp1b rungs")
    ap.add_argument("--duration", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--seq-len", type=int, default=1024)
    # MN-I
    ap.add_argument("--train-frac", type=float, default=0.1,
                    help="MN-I: fraction of steps that really train")
    ap.add_argument("--schedule-resolution", type=int, default=10000,
                    help="MN-I: granularity of the train/fill threshold. The "
                         "default resolves --train-frac to 1e-4; it is NOT a "
                         "period (the schedule is hash-jittered, not cyclic).")
    ap.add_argument("--slot-sec", type=float, default=0.0,
                    help="MN-I: seconds of filler per non-training slot. 0 (the "
                         "default) auto-matches the measured training-step time, "
                         "which is what makes --train-frac a fraction of TIME.")
    # MN-O
    ap.add_argument("--dilution", type=int, default=10,
                    help="MN-O: every Nth step is a training CANDIDATE")
    ap.add_argument("--oracle-threshold", type=float, default=0.3,
                    help="MN-O: train only while P(training) < this")
    ap.add_argument("--arena-mb", type=int, default=0,
                    help="MN-F/MN-FB: reserve this many MB before model init. "
                         "Measure it from an honest run's peak "
                         "torch.cuda.memory_reserved; 0 disables (and makes "
                         "MN-F a no-op, which is a useful control).")
    ap.add_argument("--bubble-frac", type=float, default=1.0,
                    help="MN-B/MN-FB: fraction of the per-step time to queue as "
                         "side-stream filler. 1.0 targets the whole step.")
    ap.add_argument("--oracle-model", default=None,
                    help="MN-O: path to the persisted honest-only detector")
    args = ap.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    node_rank = int(os.environ.get("GROUP_RANK", 0))

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    batch_size = args.batch_size or BATCH_DEFAULTS[args.size]
    hidden, n_layers, n_heads, ffn, vocab = MODEL_CONFIGS[args.size]
    tokens_per_step = batch_size * world_size * args.seq_len

    if rank == 0:
        print(f"Adversarial multinode: {args.strategy}, {args.size} on {world_size} GPUs")
        print(f"  Per-GPU batch: {batch_size}, Seq len: {args.seq_len}")
        if args.strategy in ("MN-F", "MN-B", "MN-FB"):
            # Read args, NOT the `bubble` object: this print runs before the
            # object exists, and referencing it here killed all three cells
            # with UnboundLocalError 16 s into the run.
            _bf = args.bubble_frac if args.strategy in ("MN-B", "MN-FB") else 0
            print(f"  Trains EVERY step. arena_mb={args.arena_mb} "
                  f"bubble_frac={_bf}")
        elif args.strategy == "MN-I":
            print(f"  Train fraction: {args.train_frac:.0%} over a "
                  f"1/{args.schedule_resolution} threshold resolution")
        else:
            print(f"  Dilution: 1/{args.dilution}, oracle threshold: "
                  f"{args.oracle_threshold}")
        print(f"  Duration: {args.duration}s")

    # MN-F: reserve the plateau before ANY model allocation, so the first
    # telemetry window already sees the steady-state footprint.
    arena = (FlatArena(device, args.arena_mb, rank)
             if args.strategy in ("MN-F", "MN-FB") else None)

    with torch.device("meta"):
        model = RandomTransformerLM(args.size)
    fsdp_model = build_fsdp_model(model, device, activation_ckpt=True,
                                 cpu_offload=False)
    optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=1e-4)

    _needs_filler = args.strategy in ("MN-I", "MN-B", "MN-FB")
    filler = NonMLFiller(device) if _needs_filler else None
    bubble = (BubbleFiller(device, filler, args.bubble_frac, rank)
              if args.strategy in ("MN-B", "MN-FB") else None)
    gate = (CollectiveOracleGate(device, rank, args.oracle_threshold,
                                 args.oracle_model)
            if args.strategy == "MN-O" else None)
    seed = broadcast_seed(device, rank)

    if rank == 0:
        mem_gb = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"  Post-init GPU memory: {mem_gb:.1f} GB")
        print(f"  Schedule seed: {seed}")

    deadline = CollectiveDeadline(args.duration, device)
    wall_start = time.time()
    # EMA of how long a real training step takes, used as the length of a filler
    # slot so that "train fraction" means a fraction of TIME rather than of loop
    # iterations. Seeded with a rough guess and corrected after the first step.
    train_step_sec = args.slot_sec if args.slot_sec > 0 else 0.5
    step = 0            # every loop iteration
    train_steps = 0     # iterations that actually moved weights
    filler_steps = 0
    useful_tokens = 0
    attributed_tokens = 0
    last_print = wall_start
    loss_val = float("nan")

    while not deadline.expired():
        if args.strategy in ("MN-F", "MN-B", "MN-FB"):
            # The disguise is in the footprint and the idle time, NOT in
            # skipping work: every step is a real gradient step, so useful
            # tokens/s is the honest rate minus only contention.
            do_train = True
        elif args.strategy == "MN-I":
            do_train = _schedule_says_train(seed, step, args.train_frac,
                                            args.schedule_resolution)
        else:
            candidate = (step % args.dilution == 0) and step > 0
            # allows() is collective, so EVERY rank must call it whenever the
            # candidacy test passes -- candidacy depends only on `step`, which is
            # rank-independent, so this stays uniform.
            do_train = gate.allows() if candidate else False

        if do_train:
            if bubble is not None:
                # Queued BEFORE the step so the kernels are already resident
                # when the step's first collective blocks the compute stream.
                bubble.fill(train_step_sec)
            _t0 = time.time()
            loss_val = train_step(fsdp_model, optimizer, batch_size,
                                  args.seq_len, vocab, device)
            _dt = time.time() - _t0
            if args.slot_sec <= 0:
                # EMA so a slow first step (cudagraph capture, allocator warmup)
                # does not fix the slot length for the whole run.
                train_step_sec = 0.8 * train_step_sec + 0.2 * _dt
            train_steps += 1
            useful_tokens += tokens_per_step
            attributed_tokens += tokens_per_step
            if rank == 0:
                extra = {"loss": f"{loss_val:.4f}"}
                if gate is not None and gate.last_p is not None:
                    extra["p_train"] = f"{gate.last_p:.3f}"
                emit_step("train", step, tokens=tokens_per_step, passes=1,
                          node=node_rank, rank=rank, **extra)
        else:
            if filler is not None:
                # One fill slot costs about as much wall time as one training
                # step, so the step-level schedule delivers the requested TIME
                # fraction. (train_frac of steps train, each slot ~equal cost.)
                filler_steps += filler.run_for(train_step_sec)
            else:
                # MN-O's non-training work is a forward pass: it keeps the GPU
                # busy the way serving does, which is the whole point of the
                # disguise. Counted as attributed-but-not-useful.
                with torch.no_grad():
                    input_ids = torch.randint(0, vocab,
                                              (batch_size, args.seq_len),
                                              device=device)
                    fsdp_model(input_ids)
                attributed_tokens += tokens_per_step
            if rank == 0:
                emit_step("infer", step, tokens=0, passes=1,
                          node=node_rank, rank=rank)

        step += 1

        now = time.time()
        if rank == 0 and now - last_print >= 30:
            el = now - wall_start
            line = (f"  {el:.0f}s: steps={step} train={train_steps} "
                    f"useful={useful_tokens / el:.0f} tok/s")
            if filler is not None:
                line += f" filler={filler_steps}"
            if bubble is not None:
                line += f" bubble_iters={bubble.launched}"
            if arena is not None:
                line += f" reserved_mb={torch.cuda.memory_reserved(device)/(1<<20):.0f}"
            if gate is not None:
                line += f" oracle={gate.allowed}/{gate.queries}"
                line += f" warm_allow={gate.warmup_allowed}"
                if gate.last_p is not None:
                    line += f" p_train={gate.last_p:.3f}"
            print(line + f" loss={loss_val:.4f}", flush=True)
            last_print = now

    elapsed = time.time() - wall_start

    if rank == 0:
        print(f"\n{args.strategy} complete: {step} steps, {train_steps} training")
        print(f"  Useful tokens: {useful_tokens:,} ({useful_tokens / elapsed:.0f} tok/s)")
        print(f"  Measured train-step time: {train_step_sec * 1000:.0f} ms "
              f"(filler slot length)")
        print(f"  Realised TIME fraction training: "
              f"{train_steps * train_step_sec / elapsed:.1%}")
        print(f"  Attributed tokens: {attributed_tokens:,} "
              f"({attributed_tokens / elapsed:.0f} tok/s)")
        # Machine-parseable cost line, extending the idiom established at
        # adversarial_composite.py:182 with the fields the evasion ladder needs.
        # evasion_cost_eval.py parses this; keep the key=value shape stable.
        print(f"  EVASION_TAX_DATA: strategy={args.strategy} world_size={world_size} "
              f"train_frac={args.train_frac if args.strategy == 'MN-I' else ''} "
              f"dilution={args.dilution if args.strategy == 'MN-O' else ''} "
              f"steps={step} train_steps={train_steps} "
              f"useful_tokens={useful_tokens} attributed_tokens={attributed_tokens} "
              f"duration={elapsed:.1f} train_step_sec={train_step_sec:.4f} "
              f"oracle_queries={gate.queries if gate else 0} "
              f"oracle_allowed={gate.allowed if gate else 0} "
              f"oracle_warmup_allowed={gate.warmup_allowed if gate else 0}", flush=True)
        if train_steps == 0:
            # A disguise that never trains is not an evasion -- it is an
            # inference job. Say so loudly; evasion_cost_eval.py rejects it too.
            print("  WARN: zero training steps -- this run is not evidence of "
                  "evasion, it never trained.", flush=True)

    if gate is not None:
        gate.close()
    if filler is not None:
        filler.release()
    sync_teardown()


if __name__ == "__main__":
    main()
