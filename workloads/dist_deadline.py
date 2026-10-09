"""Wall-clock run deadline that every rank agrees on.

Why this exists
---------------
The distributed workloads used to end on a purely local test::

    while time.time() - wall_start < args.duration:

Each rank owns its own clock, so ranks leave the loop on *different*
iterations. Within one node that is merely untidy. Across two nodes it is
fatal: the ranks still inside a collective see their peers disappear and the
job dies with

    ncclRemoteError: remote process exited or there was a network error

which aborts the surviving process. Observed 2026-08-20 on the 16-rank
S16 FSDP cell -- node1 exited 0 while node0 crashed mid-ReduceScatter, so a
cell that had trained correctly for its full duration was recorded as failed.

A barrier before teardown cannot fix this on its own: if one rank exits the
loop a step early it waits at the barrier while its peers try to run another
step, and that step's collectives can never complete. The stop decision itself
has to be collective, which is what `expired()` does -- rank 0 owns the clock
and its verdict is reduced to every rank, so all ranks break on the same
iteration.

Requires rank-uniform loops (every rank calls `expired()` the same number of
times). That already holds for DDP/FSDP, which require every rank to take part
in every step.
"""

import time

import torch
import torch.distributed as dist


class CollectiveDeadline:
    """Agreed stop signal for a timed distributed run.

    Args:
        duration: run length in seconds. Falsy means "never expire".
        device: CUDA device for the reduction tensor.
        check_every: vote once per N calls. The vote is a 1-element all_reduce,
            negligible next to the multi-GB gradient collectives, but there is
            no reason to pay it every step; the cost of a larger value is only
            that the run can overshoot by up to N-1 steps.
    """

    def __init__(self, duration, device, check_every=5):
        self.duration = duration
        self.device = device
        self.check_every = max(1, int(check_every))
        self.start = time.time()
        self._calls = 0
        self._stopped = False
        self._flag = torch.zeros(1, device=device)

    @property
    def elapsed(self):
        return time.time() - self.start

    def _distributed(self):
        return dist.is_available() and dist.is_initialized()

    def expired(self):
        """True once the deadline has passed, identically on every rank."""
        if not self.duration:
            return False
        if self._stopped:
            return True

        self._calls += 1
        # Single-process runs (S1 cells) have nobody to agree with.
        if not self._distributed():
            self._stopped = self.elapsed >= self.duration
            return self._stopped
        if self._calls % self.check_every:
            return False

        # Rank 0 owns the clock; MAX spreads its verdict to everyone.
        local = 1.0 if (dist.get_rank() == 0 and self.elapsed >= self.duration) else 0.0
        self._flag.fill_(local)
        dist.all_reduce(self._flag, op=dist.ReduceOp.MAX)
        self._stopped = bool(self._flag.item() > 0)
        return self._stopped


def sync_teardown():
    """Barrier, then tear down the process group.

    The barrier keeps a fast rank from destroying its NCCL communicator while a
    peer is still finishing its last step -- which surfaces as the same
    ncclRemoteError the deadline exists to prevent.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return
    try:
        dist.barrier()
    except Exception:  # teardown must never be the thing that fails a cell
        pass
    dist.destroy_process_group()
