# Cloud Provider Notes for GPU Profiling

## Requirements

This project needs **GPU performance counter access** for Nsight Compute (NCU) and DCGM profiling fields. This requires the host kernel parameter `RmProfilingAdminOnly=0`, which is only available on true bare-metal or properly configured PCIe passthrough hosts. Most cloud providers use virtualization that blocks this.

## Provider Comparison

| Provider | NCU/Profiling? | A100 $/hr | Notes |
|----------|---------------|-----------|-------|
| **Vast.ai** | No (not bare-metal despite claims) | ~$0.70 | Confirmed: does not work |
| **Lambda Labs** | **No** — explicitly unsupported due to virtualization | ~$1.29 | Staff confirmed "nsight compute is not currently supported" ([source](https://deeptalk.lambda.ai/t/nvidia-profiling-support/4763)) |
| **RunPod Secure Cloud** | **Likely yes** for DCGM; NCU host-dependent | ~$1.14-1.39 | Mentioned as supporting profiling; blocks mining software |
| **Nebius** | **Reportedly yes** | ~$1-2 | Mentioned as a profiling-capable alternative |
| **Massed Compute** | **Likely yes** — has Nsight documentation for A100/H100 | Varies | [Has FAQ on Nsight profiling](https://massedcompute.com/faq-answers/?question=How+do+I+use+NVIDIA's+Nsight+Systems+to+profile+memory+usage+on+A100+and+H100+GPUs?) |

## Decision

**RunPod Secure Cloud** is the best first option:
- Confirmed by other users as supporting profiling tools
- A100 at $1.14-1.39/hr — cheapest among confirmed-working providers
- DCGM is pre-installed
- **Caveat**: Blocks mining software — need to write a custom Ethash-like CUDA kernel instead of using T-Rex miner
- NCU may still be host-dependent — do a 1-hour test first

If RunPod doesn't work for NCU, try Nebius or Massed Compute next.

## Testing Strategy

1. Rent a RunPod Secure Cloud A100 for 1 hour (~$1.39)
2. Run the NCU capability test from week 3 instructions
3. If NCU works, proceed with full collection there
4. If not, collect Tier 1 + Tier 2 only on RunPod (DCGM should work), and find an NCU-capable host separately

At $1.14-1.39/hr, the full 8-hour collection session would cost ~$9-11, well within budget.
