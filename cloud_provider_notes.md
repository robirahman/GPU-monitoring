# Cloud Provider Notes for GPU Profiling

## Requirements

This project needs **data-center GPUs** (A100, H100, or newer) for DCGM profiling fields (Tier 2). For Nsight Compute (Tier 3), the host kernel parameter `RmProfilingAdminOnly=0` is required, which is only available on true bare-metal or properly configured hosts. Most cloud providers use virtualization that blocks this.

**Key distinction:**
- **Tier 2 (DCGM profiling fields)**: Works on data-center GPUs with GPU passthrough. Does NOT require bare metal. Should work on most cloud A100/H100 instances.
- **Tier 3 (NCU)**: Requires `RmProfilingAdminOnly=0` in the host kernel. Most cloud providers restrict this even on "bare-metal" instances. True bare-metal or specially configured hosts are needed.

## Provider Comparison

| Provider | Tier 2 (DCGM profiling) | Tier 3 (NCU) | GPU options | $/hr | Notes |
|----------|------------------------|--------------|-------------|------|-------|
| **RunPod Secure Cloud** | Likely yes (data-center GPUs with passthrough) | Probably no | A100, H100 | $1.14-1.39 (A100), ~$2-3 (H100) | Best bet for Tier 1+2; blocks mining software |
| **Massed Compute** | Yes | **Likely yes** -- has Nsight docs, offers bare-metal on request | A100, H100 | Varies (check website) | Most promising for NCU |
| **Nebius** | Likely yes | Reportedly yes | H100 (A100 may be retired) | ~$2/hr (H100) | Explorer Tier at $1.99/hr H100 |
| **Vast.ai** | Untested | **No** -- not bare-metal despite some claims | A100, H100 | ~$0.70+ | Confirmed: NCU does not work |
| **Lambda Labs** | Likely yes | **No** -- staff confirmed unsupported | A100, H100 | ~$1.29 (A100), ~$2.49 (H100) | Virtualization blocks NCU |
| **CoreWeave** | Likely yes | Unknown | A100, H100 | Enterprise pricing | Probably overkill for this project |

Sources:
- Lambda NCU unsupported: https://deeptalk.lambda.ai/t/nvidia-profiling-support/4763
- Massed Compute Nsight docs: https://massedcompute.com/faq-answers/?question=How+do+I+use+NVIDIA+Nsight+Systems+and+NVIDIA+Nsight+Compute+in+a+cloud+environment?
- Nebius pricing: https://nebius.com/prices

## Decision

**Split collection across two providers:**

### Tier 1 + 2 bulk collection: RunPod Secure Cloud
- Confirmed data-center GPUs with DCGM support
- A100 at $1.14-1.39/hr or H100 at ~$2-3/hr
- DCGM profiling fields (tensor_active, fp16/32/64 pipes, SM occupancy, DRAM bandwidth) should work
- **Caveat**: Blocks mining software -- need custom Ethash-like CUDA kernel instead of T-Rex miner
- Budget: ~$10-14 for 8-10 hours of collection

### Tier 3 NCU profiling: Massed Compute (or Nebius)
- Massed Compute offers bare-metal on request and has explicit Nsight documentation
- Do a 1-hour test to confirm NCU works before committing
- Only need ~1 run per workload type (much less time than Tier 1+2 collection)
- Nebius is the backup if Massed Compute doesn't work out

### Fallback
If no provider supports NCU:
- Collect Tier 1 + Tier 2 only (this is the core data for the project)
- The project's novel contribution is temporal patterns from Tier 1+2 -- NCU is supplementary
- Document the provider search in the report

## Testing Strategy

1. Rent a RunPod Secure Cloud A100 or H100 for 1 hour
2. Run DCGM profiling field test (see week 3 instructions Step 2)
3. If Tier 2 works, proceed with full collection on RunPod
4. Separately, rent a Massed Compute bare-metal instance for 1 hour
5. Run the NCU capability test
6. If NCU works, do Tier 3 collection there

## GPU Selection: A100 vs H100

Either A100 or H100 works for this project. Choose based on price:
- A100 40/80GB: Cheaper (~$1-1.40/hr), sufficient VRAM, full DCGM support
- H100 80GB: More expensive (~$2-3/hr), but may be the only option on some providers (e.g., Nebius)
- Both have tensor cores, DCGM profiling fields, and NCU support (if host allows)
- The telemetry pipeline works identically on both -- just different absolute values for metrics
