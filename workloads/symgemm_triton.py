#!/usr/bin/env python3
"""One Triton GEMM kernel, used for all three matmuls of a training step.

This is the literal form of the attack described to reviewer M7wU: a custom
kernel that issues the backward-pass matmuls in the same shape, tile
configuration and launch geometry as a forward pass, so that at the kernel
level a training step is three forward GEMMs.

Two things make it a faithful disguise rather than a relabelling:

1. **One kernel, one tile config, one launch geometry.** FWD, dX and dW all
   go through ``_gemm_kernel`` with the same BLOCK_M/N/K and the same grid
   function. Transposition is expressed purely through strides, so no operand
   is ever materialised in a different layout and no distinct transpose kernel
   is launched.
2. **The optimizer lives in the epilogue.** With ``BETA=1`` the dW launch
   loads the weight tile, applies ``W -= lr * dW`` and stores it back, inside
   the same kernel. There is no gradient tensor, no optimizer kernel, and no
   optimizer state anywhere in the step.

What it does NOT change, and this is the point of running it: the compulsory
DRAM traffic. Fusing the update still requires reading and writing every
trained weight once per step. See classifier/rebuttal_analysis/symgemm_traffic.py
-- the gap between this and matched-throughput inference is +12.7% of traffic
at M=2048 and +4.4% at M=8192, exactly as for the cuBLAS version. Kernel
identity and launch count are not among the nine signals NVML exposes at 1 Hz.

Triton is slower than cuBLAS, so any run using this backend needs its own
kernel-matched control (mode L1 with --backend triton). Otherwise the
comparison measures "is this Triton?" rather than "is this training?".
"""

import torch
import triton
import triton.language as tl


# One config for every GEMM in the step. Deliberately NOT autotuned per shape:
# autotuning would give FWD, dX and dW different tile shapes and launch grids,
# which is precisely the signature the disguise is meant to remove.
BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 128, 128, 64, 8
NUM_WARPS, NUM_STAGES = 8, 3


@triton.jit
def _gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    alpha,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, BETA: tl.constexpr,
):
    """C = alpha * (A @ B) + BETA * C, grouped-ordered for L2 reuse.

    BETA is a constexpr so the accumulate-into-C path is compiled away
    entirely for the plain-store case; FWD/dX/dW-discard emit identical code.
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_rem, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_rem, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    acc = acc * alpha

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    if BETA:
        # The optimizer step, inside the GEMM epilogue: read the weight tile,
        # apply the update, write it back. No gradient is ever materialised.
        prev = tl.load(c_ptrs, mask=c_mask, other=0.0)
        acc = acc + prev.to(tl.float32)

    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)


def _grid(M, N):
    return (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)


def gemm(a, b, out=None, alpha=1.0, accumulate=False):
    """C = alpha * (A @ B) [+ C].

    ``a`` and ``b`` may be non-contiguous views (e.g. ``x.t()``); the
    transposition rides on the strides, so no layout conversion kernel is
    launched and every call has the same launch geometry.
    """
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"inner dims disagree: {a.shape} @ {b.shape}"
    if out is None:
        assert not accumulate, "accumulate=True needs an existing out tensor"
        out = torch.empty((M, N), device=a.device, dtype=a.dtype)

    _gemm_kernel[_grid(M, N)](
        a, b, out,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
        alpha,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M, BETA=1 if accumulate else 0,
        num_warps=NUM_WARPS, num_stages=NUM_STAGES,
    )
    return out


def self_test(device="cuda", dtype=torch.bfloat16):
    """Check the three GEMM roles against torch, including the fused update."""
    torch.manual_seed(0)
    M, K, N, lr = 2048, 512, 768, 1e-3
    tol = 3e-2  # bf16 accumulation in fp32; loose but catches real errors

    x = torch.randn(M, K, device=device, dtype=dtype)
    w = torch.randn(N, K, device=device, dtype=dtype) * 0.02
    g = torch.randn(M, N, device=device, dtype=dtype)

    def rel(a, b):
        return ((a.float() - b.float()).norm() / b.float().norm()).item()

    # FWD: Y = X W^T
    y = gemm(x, w.t())
    e_fwd = rel(y, torch.matmul(x, w.t()))

    # dX = dY W
    dx = gemm(g, w)
    e_dx = rel(dx, torch.matmul(g, w))

    # dW = dY^T X, plain store
    dw = gemm(g.t(), x)
    e_dw = rel(dw, torch.matmul(g.t(), x))

    # Fused update: W <- W - lr * (dY^T X), in the epilogue
    w_fused = w.clone()
    gemm(g.t(), x, out=w_fused, alpha=-lr, accumulate=True)
    e_upd = rel(w_fused, w - lr * torch.matmul(g.t(), x))

    print(f"triton self-test: fwd {e_fwd:.2e}  dX {e_dx:.2e}  "
          f"dW {e_dw:.2e}  fused-update {e_upd:.2e}")
    for name, e in (("fwd", e_fwd), ("dX", e_dx), ("dW", e_dw),
                    ("fused-update", e_upd)):
        assert e < tol, f"triton {name} rel err {e:.2e} exceeds {tol}"

    # Throughput vs cuBLAS: gate G3 aborts Stage B below 40%, where the
    # slowdown would dominate the telemetry and the attack stops being credible.
    import time
    a = torch.randn(4096, 4096, device=device, dtype=dtype)
    b = torch.randn(4096, 4096, device=device, dtype=dtype)
    for fn in (lambda: gemm(a, b), lambda: torch.matmul(a, b)):
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    times = []
    for fn in (lambda: gemm(a, b), lambda: torch.matmul(a, b)):
        t0 = time.time()
        for _ in range(20):
            fn()
        torch.cuda.synchronize()
        times.append(time.time() - t0)
    frac = times[1] / times[0]
    print(f"  triton throughput = {frac:.1%} of cuBLAS "
          f"({'OK' if frac >= 0.40 else 'BELOW G3 GATE (0.40) -- abort Stage B'})")
    print("triton self-test PASS")
    return frac


if __name__ == "__main__":
    self_test()
