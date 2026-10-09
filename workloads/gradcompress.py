#!/usr/bin/env python3
"""Delta compression codecs and compressed collectives for low-bandwidth training.

Why this exists
---------------
The 2026-09-30 leave-one-family-out put held-out training recall at 0.5687 and
showed that neither more data nor more NIC coverage moves it. What the corpus
lacks is training programs whose COMMUNICATION differs -- and the largest missing
piece is compression. There is none anywhere in this repo: no top-k, no error
feedback, no int8/fp8, no PowerSGD, no ddp_comm_hooks. Meanwhile the threat model
we are built against (arXiv 2605.29359) assumes a 100 Mbps consumer link and a
headline compression ratio of 150x, and Streaming DiLoCo (arXiv 2501.18512)
quantises its outer gradients to four bits.

Two rules this module exists to enforce
---------------------------------------
**1. Compress DELTAS, never parameters.** Top-k on a parameter tensor zeroes the
unselected 99% of the weights. Top-k on a delta merely postpones them, and error
feedback carries them to the next round. `local_sgd_sync` currently all-reduces
`p.data` directly, so any caller adding compression must first switch to delta
form: `d = p - prev; reduce(d); p = prev + mean(d)`.

**2. Report BYTES ON THE WIRE, not the logical tensor size.** This whole corpus
exists to characterise interconnect volume. A codec that reports what it would
have sent uncompressed is worse than no codec at all, because it puts a false
number into the analysis with no way to notice.

Rule 2 has a consequence that is easy to get wrong and expensive to get wrong.
**A quantised all-reduce is arithmetic nonsense**: NCCL will happily sum a buffer
of packed nibbles and hand back something with the right shape and the right byte
count, but the sum of quantised values is not the quantisation of the sum. So
every non-trivial codec here reduces by ALL-GATHER + local decode + local mean.
That is correct, and it is not free:

    ring all-reduce   2*S*(W-1)/W       ring all-gather   S*(W-1)

so the all-gather costs **W/2 times** the all-reduce for the same payload. That
factor, not the codec, decides whether compression is worth anything:

    W    penalty |  net wire ratio vs an uncompressed all-reduce
                 |  e3m0 (8x)   int8 (4x)   topk 1% (50x)
    2     1.0    |     8.00        4.00        50.00
    4     2.0    |     4.00        2.00        25.00
    8     4.0    |     2.00        1.00        12.50
    16    8.0    |     1.00        0.50         6.25

Read the bottom row. At a world of 16 -- which is what `train_localsgd.py` does
today, one replica per rank -- **E3M0 saves nothing at all**, and **int8 sends
twice as much as sending nothing would**, while both report a healthy payload
ratio. Only aggressive sparsification still pays.

The fix is not a cleverer collective, it is a smaller outer group. Streaming
DiLoCo runs M=2 replicas, and at W=2 the penalty is exactly 1.0 and the full
payload ratio is realised. That is the arithmetic reason to make a replica a
NODE rather than a rank: the outer exchange becomes a 2-rank collective between
node leaders, intra-node traffic stays on NVLink, and compression buys its
nominal factor instead of being cancelled.

`net_wire_ratio()` computes this, `CompressedAllReduce` warns when a
configuration would make traffic worse, and `scripts/test_gradcompress.py` pins
the whole table -- so the optimistic per-payload number cannot quietly stand in
for the wire number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.distributed as dist


# ── wire cost model ─────────────────────────────────────────────────────────
# One place, documented, so every byte figure in the corpus traces back here.

def ring_allreduce_bytes(payload_bytes: int, world: int) -> int:
    """Bytes THIS rank transmits in a ring all-reduce (reduce-scatter+all-gather)."""
    if world < 2:
        return 0
    return int(2 * payload_bytes * (world - 1) / world)


def ring_allgather_bytes(payload_bytes: int, world: int) -> int:
    """Bytes THIS rank transmits in a ring all-gather."""
    if world < 2:
        return 0
    return int(payload_bytes * (world - 1))


def net_wire_ratio(payload_ratio: float, world: int) -> float:
    """Wire saving actually achieved by a codec with this payload ratio.

    An uncompressed all-reduce of S bytes against a compressed all-gather of
    S/payload_ratio. Below 1.0 the "compression" increases traffic.
    """
    if world < 2:
        return 1.0
    # Float arithmetic directly: the byte helpers cast to int, which truncates
    # the sub-byte quantities this ratio is built from to zero.
    allreduce = 2.0 * (world - 1) / world
    allgather = (1.0 / payload_ratio) * (world - 1)
    return allreduce / allgather


# ── error-feedback state ────────────────────────────────────────────────────

@dataclass
class EFState:
    """Per-tensor residuals for error feedback.

    Keyed by POSITION in the caller's tensor list, not by `id()`: a tensor
    object can be reallocated between rounds while occupying the same logical
    slot, and keying on identity would silently drop its residual -- which looks
    exactly like a slightly worse learning rate.
    """
    residuals: list = field(default_factory=list)

    def get(self, i: int, like: torch.Tensor) -> torch.Tensor:
        while len(self.residuals) <= i:
            self.residuals.append(None)
        if self.residuals[i] is None:
            self.residuals[i] = torch.zeros_like(like)
        return self.residuals[i]

    def set(self, i: int, v: torch.Tensor) -> None:
        self.residuals[i] = v

    def reset(self) -> None:
        self.residuals = []


# ── codecs ──────────────────────────────────────────────────────────────────

class Codec:
    name = "abstract"
    uses_error_feedback = False

    def encode(self, t: torch.Tensor):
        raise NotImplementedError

    def decode(self, payload, meta) -> torch.Tensor:
        raise NotImplementedError

    def payload_bytes(self, payload, meta) -> int:
        raise NotImplementedError

    def needs_allgather(self) -> bool:
        return True


class NoOpCodec(Codec):
    """The uncompressed path. Reduces with all_reduce, which is what makes it
    the honest baseline: it pays the cheaper collective as well as the larger
    payload, so the ratio a compressed codec is measured against is the real
    one an uncompressed job would achieve."""
    name = "none"

    def encode(self, t):
        return t, {"numel": t.numel(), "shape": tuple(t.shape)}

    def decode(self, payload, meta):
        return payload

    def payload_bytes(self, payload, meta):
        return payload.numel() * payload.element_size()

    def needs_allgather(self):
        return False


def _pack_nibbles(codes: torch.Tensor):
    """uint8 codes in 0..15 -> two per byte. Returns (packed, n_original)."""
    n = codes.numel()
    flat = codes.reshape(-1)
    if n % 2:
        flat = torch.cat([flat, flat.new_zeros(1)])
    lo = flat[0::2]
    hi = flat[1::2]
    return (lo | (hi << 4)).to(torch.uint8), n


def _unpack_nibbles(packed: torch.Tensor, n: int) -> torch.Tensor:
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    out = torch.stack([lo, hi], dim=1).reshape(-1)
    return out[:n]


class E3M0Codec(Codec):
    """Streaming DiLoCo's 4-bit format: 1 sign bit + 3 exponent bits, no mantissa.

    Code 0 is zero. Codes 1..7 decode to ``sign * scale * 2**-(e-1)``, so the
    representable magnitudes are scale, scale/2, ... scale/64 -- a 64x dynamic
    range in four bits. Accumulation after decode is FP32, per the paper; the
    four bits are a transport format, never an accumulator.

    Round-to-nearest in the LOG domain is biased toward zero, because the
    midpoint between 2^-a and 2^-(a+1) in log space is not the midpoint in
    linear space. At four bits that bias is large enough to stall an outer
    optimizer while the run still reports plausible throughput, so
    ``stochastic=True`` offers an unbiased variant.
    """
    name = "e3m0"
    _MIN_EXP = 6          # smallest representable is scale * 2**-6

    def __init__(self, stochastic: bool = False):
        self.stochastic = stochastic

    def encode(self, t: torch.Tensor):
        flat = t.reshape(-1).float()
        scale = flat.abs().max()
        meta = {"numel": t.numel(), "shape": tuple(t.shape),
                "scale": float(scale.item())}
        if float(scale) == 0.0:
            packed, _ = _pack_nibbles(torch.zeros(flat.numel(), dtype=torch.uint8,
                                                  device=t.device))
            return packed, meta
        a = (flat.abs() / scale).clamp_(min=1e-30)
        lg = -torch.log2(a)                       # 0 at the top of the range
        if self.stochastic:
            # Choose between the two bracketing representable magnitudes
            # hi = 2**-fl and lo = 2**-(fl+1) with p(hi) = (a-lo)/(hi-lo).
            # Interpolating in LOG space instead would be unbiased in log
            # space -- which is not the domain the values are summed in, so
            # the accumulator would still drift.
            fl = torch.floor(lg)
            hi = torch.pow(2.0, -fl)
            lo = hi * 0.5
            p_hi = ((a - lo) / (hi - lo)).clamp_(0.0, 1.0)
            lg = torch.where(torch.rand_like(a) < p_hi, fl, fl + 1)
        else:
            lg = torch.round(lg)
        e = (lg + 1).clamp_(1, self._MIN_EXP + 1)
        # Anything below the smallest representable magnitude becomes the zero
        # code -- with error feedback the caller carries it forward, without it
        # the value is simply dropped, which is what a 4-bit format means.
        e = torch.where(lg > self._MIN_EXP + 0.5, torch.zeros_like(e), e)
        codes = e.to(torch.uint8) | (torch.signbit(flat).to(torch.uint8) << 3)
        packed, _ = _pack_nibbles(codes)
        return packed, meta

    def decode(self, payload, meta):
        n = meta["numel"]
        codes = _unpack_nibbles(payload, n).long()
        e = codes & 0x07
        neg = (codes & 0x08) != 0
        mag = torch.where(
            e > 0,
            meta["scale"] * torch.pow(2.0, -(e.float() - 1.0)),
            torch.zeros(1, dtype=torch.float32, device=payload.device))
        out = torch.where(neg, -mag, mag).to(torch.float32)
        return out.reshape(meta["shape"])

    def payload_bytes(self, payload, meta):
        # The scale rides along: one fp32 per tensor.
        return payload.numel() * payload.element_size() + 4


class Int8Codec(Codec):
    """Symmetric per-tensor int8. 4x payload reduction, far more accurate than
    E3M0 and the format SWARM uses for activations and gradients."""
    name = "int8"

    def encode(self, t: torch.Tensor):
        flat = t.reshape(-1).float()
        scale = flat.abs().max() / 127.0
        meta = {"numel": t.numel(), "shape": tuple(t.shape),
                "scale": float(scale.item()) if float(scale) > 0 else 0.0}
        if meta["scale"] == 0.0:
            return torch.zeros(flat.numel(), dtype=torch.int8, device=t.device), meta
        q = torch.round(flat / scale).clamp_(-127, 127).to(torch.int8)
        return q, meta

    def decode(self, payload, meta):
        return (payload.float() * meta["scale"]).reshape(meta["shape"])

    def payload_bytes(self, payload, meta):
        return payload.numel() * payload.element_size() + 4


class TopKCodec(Codec):
    """Top-k magnitude sparsification with error feedback.

    Sends ``k = ceil(k_frac * numel)`` (index, value) pairs: 4 bytes of int32
    index plus 4 bytes of fp32 value, so the payload is ``8*k`` regardless of
    the tensor's dtype. At k_frac=0.01 that is a 50x payload reduction against
    fp32 -- the operating point an evader on a 100 Mbps link would actually
    choose, since it turns a 216 s DiLoCo sync into about 3 s.

    Error feedback is what makes it converge rather than merely run: the
    unsent residual is added back before the next round's selection, so every
    coordinate is eventually transmitted instead of permanently dropped.
    """
    name = "topk"
    uses_error_feedback = True

    def __init__(self, k_frac: float = 0.01, error_feedback: bool = True):
        self.k_frac = float(k_frac)
        self.error_feedback = bool(error_feedback)

    def k_for(self, numel: int) -> int:
        return max(1, min(numel, math.ceil(self.k_frac * numel)))

    def encode(self, t: torch.Tensor):
        flat = t.reshape(-1).float()
        k = self.k_for(flat.numel())
        idx = torch.topk(flat.abs(), k, sorted=False).indices.to(torch.int32)
        val = flat[idx.long()]
        meta = {"numel": flat.numel(), "shape": tuple(t.shape), "k": k}
        return torch.cat([idx.view(torch.float32), val]), meta

    def decode(self, payload, meta):
        k = meta["k"]
        idx = payload[:k].view(torch.int32).long()
        val = payload[k:]
        out = torch.zeros(meta["numel"], dtype=torch.float32, device=payload.device)
        out.index_copy_(0, idx, val)
        return out.reshape(meta["shape"])

    def payload_bytes(self, payload, meta):
        return payload.numel() * payload.element_size()

    def residual_for(self, t: torch.Tensor, payload, meta) -> torch.Tensor:
        """What this codec failed to send, in the original shape."""
        return t.float() - self.decode(payload, meta).to(t.device)


def make_codec(spec: str) -> Codec:
    """'none' | 'e3m0' | 'e3m0:sr' | 'int8' | 'topk:0.01' | 'topk:0.01,noef'"""
    spec = (spec or "none").strip()
    head, _, tail = spec.partition(":")
    opts = [o for o in tail.split(",") if o]
    if head == "none":
        return NoOpCodec()
    if head == "e3m0":
        return E3M0Codec(stochastic="sr" in opts)
    if head == "int8":
        return Int8Codec()
    if head == "topk":
        frac = float(opts[0]) if opts and opts[0][0].isdigit() else 0.01
        return TopKCodec(k_frac=frac, error_feedback="noef" not in opts)
    raise ValueError(f"unknown compression spec {spec!r}")


# ── the compressed collective ───────────────────────────────────────────────

class CompressedAllReduce:
    """Mean-reduce a list of deltas across a group, compressed, in bounded memory.

    ``bucket_bytes`` is not an optimisation, it is a correctness requirement. A
    1.3B-parameter fp32 delta is 5.2 GB; packed to E3M0 it is 650 MB; all-gathered
    at world 16 that is 10.4 GB of scratch on a 40 GB card already holding a
    21 GB replica. Every 2-rank CPU test and every small-model smoke run passes
    without bucketing, and then the rented pair OOMs.
    """

    def __init__(self, codec: Codec, group=None, world_size: int | None = None,
                 bucket_bytes: int = 64 << 20):
        self.codec = codec
        self.group = group
        self.world = world_size or (dist.get_world_size(group)
                                    if dist.is_initialized() else 1)
        self.bucket_bytes = int(bucket_bytes)
        self.logical_bytes = 0
        self.n_collectives = 0
        self.max_scratch_bytes = 0

    def mean_(self, tensors: list, ef: EFState | None = None) -> int:
        """In place: each tensor becomes its mean over the group.

        Returns the number of bytes THIS rank transmitted, under the cost model
        at the top of this module. Returns 0 for a world of one -- a single-rank
        run transmits nothing, and reporting otherwise would put a fictional
        number into the corpus.
        """
        self.logical_bytes = sum(t.numel() * t.element_size() for t in tensors)
        if self.world < 2:
            return 0

        wire = 0
        for lo, hi in self._buckets(tensors):
            wire += self._reduce_bucket(tensors, lo, hi, ef)
        return wire

    def _buckets(self, tensors):
        """Contiguous [lo, hi) index ranges whose payloads fit the budget."""
        out, lo, acc = [], 0, 0
        for i, t in enumerate(tensors):
            # Upper bound on this tensor's payload; exact for every codec here.
            sz = t.numel() * t.element_size()
            if isinstance(self.codec, E3M0Codec):
                sz = (t.numel() + 1) // 2 + 4
            elif isinstance(self.codec, Int8Codec):
                sz = t.numel() + 4
            elif isinstance(self.codec, TopKCodec):
                sz = 8 * self.codec.k_for(t.numel())
            if acc and acc + sz > self.bucket_bytes:
                out.append((lo, i))
                lo, acc = i, 0
            acc += sz
        out.append((lo, len(tensors)))
        return out

    def _reduce_bucket(self, tensors, lo, hi, ef) -> int:
        codec = self.codec
        if not codec.needs_allgather():
            # Uncompressed: a real all-reduce, the cheaper collective.
            wire = 0
            for i in range(lo, hi):
                dist.all_reduce(tensors[i], group=self.group)
                tensors[i] /= self.world
                wire += ring_allreduce_bytes(
                    tensors[i].numel() * tensors[i].element_size(), self.world)
            self.n_collectives += (hi - lo)
            return wire

        payloads, metas = [], []
        for i in range(lo, hi):
            src = tensors[i].float()
            if codec.uses_error_feedback and ef is not None and codec.error_feedback:
                src = src + ef.get(i, src)
            p, m = codec.encode(src)
            if codec.uses_error_feedback and ef is not None and codec.error_feedback:
                # The residual is what THIS round failed to send, measured
                # against the tensor that was actually offered to the codec
                # (delta + carried residual) -- not against the raw delta, or
                # the carried part is counted twice and the feedback diverges.
                ef.set(i, src - codec.decode(p, m).to(src.device))
            payloads.append(p)
            metas.append(m)

        flat = torch.cat([p.reshape(-1).view(torch.uint8) for p in payloads])
        gathered = [torch.empty_like(flat) for _ in range(self.world)]
        dist.all_gather(gathered, flat, group=self.group)
        self.n_collectives += 1
        self.max_scratch_bytes = max(self.max_scratch_bytes,
                                     flat.numel() * self.world)

        # Scales are tiny but must travel; send them as one fp32 vector.
        scales = torch.tensor([m.get("scale", 0.0) for m in metas],
                              dtype=torch.float32, device=flat.device)
        gscales = [torch.empty_like(scales) for _ in range(self.world)]
        dist.all_gather(gscales, scales, group=self.group)

        offs, o = [], 0
        for p in payloads:
            n = p.reshape(-1).view(torch.uint8).numel()
            offs.append((o, o + n))
            o += n

        for j, i in enumerate(range(lo, hi)):
            acc = torch.zeros(metas[j]["numel"], dtype=torch.float32,
                              device=tensors[i].device)
            a, b = offs[j]
            for r in range(self.world):
                m = dict(metas[j])
                m["scale"] = float(gscales[r][j].item())
                raw = gathered[r][a:b]
                acc += codec.decode(self._as_payload(raw, payloads[j]),
                                    m).reshape(-1)
            tensors[i].copy_((acc / self.world).reshape(tensors[i].shape)
                             .to(tensors[i].dtype))

        wire = ring_allgather_bytes(flat.numel(), self.world)
        wire += ring_allgather_bytes(scales.numel() * 4, self.world)
        return wire

    @staticmethod
    def _as_payload(raw_u8: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        """Reinterpret gathered bytes back to the payload's own dtype."""
        if like.dtype == torch.uint8:
            return raw_u8
        return raw_u8.view(like.dtype)
