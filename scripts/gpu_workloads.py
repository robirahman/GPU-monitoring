import time
import argparse
from datetime import datetime, timezone

import torch


def log(msg: str):
    print(f"{datetime.now(timezone.utc).isoformat()} | {msg}", flush=True)


def warmup():
    # quick warmup to avoid first-op overhead dominating
    x = torch.randn((2048, 2048), device="cuda", dtype=torch.float16)
    for _ in range(10):
        y = x @ x
    torch.cuda.synchronize()


def compute_bound(seconds: int, m: int):
    log(f"PHASE=compute_bound start seconds={seconds} m={m}")
    a = torch.randn((m, m), device="cuda", dtype=torch.float16)
    b = torch.randn((m, m), device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()

    t_end = time.time() + seconds
    it = 0
    while time.time() < t_end:
        c = a @ b
        # keep something live so the op isn't optimized away
        a = c
        it += 1
    torch.cuda.synchronize()
    log(f"PHASE=compute_bound end iters={it}")


def mem_stream(seconds: int, n: int):
    """
    Not perfect HBM bandwidth benchmark, but creates sustained large tensor ops.
    """
    log(f"PHASE=mem_stream start seconds={seconds} n={n}")
    x = torch.randn((n,), device="cuda", dtype=torch.float16)
    y = torch.randn((n,), device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()

    t_end = time.time() + seconds
    it = 0
    while time.time() < t_end:
        # elementwise ops over a large vector -> more memory traffic
        x = x + y
        x = x * 1.0001
        it += 1
    torch.cuda.synchronize()
    log(f"PHASE=mem_stream end iters={it}")


def idle(seconds: int):
    log(f"PHASE=idle start seconds={seconds}")
    time.sleep(seconds)
    log("PHASE=idle end")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idle", type=int, default=15)
    ap.add_argument("--compute", type=int, default=20)
    ap.add_argument("--mem", type=int, default=20)
    ap.add_argument("--m", type=int, default=8192, help="matmul size (square)")
    ap.add_argument("--n", type=int, default=256_000_000, help="vector length for mem phase (fp16)")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA not available"
    log(f"GPU={torch.cuda.get_device_name(0)}")
    warmup()

    idle(args.idle)
    compute_bound(args.compute, args.m)
    idle(args.idle)
    mem_stream(args.mem, args.n)
    idle(args.idle)

    log("DONE")


if __name__ == "__main__":
    main()
