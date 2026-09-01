# The TPU rung on real hardware: what provisions, what the chip reports, and
# eight defects that only a genuinely multi-process slice could expose

Status as of 2026-09-01 20:30Z. A 2x v6e-1 multislice is live in us-east5-a and
collecting; the first fully valid cell landed at 19:54Z.

## 1. Capacity decided the experiment design, not us

Over a week of continuous parking across zones, GCP granted:

| shape | scope | granted |
|---|---|---|
| v6e-1 | tpuT1 | freely -- 13 times |
| 2x v6e-1 multislice | tpuTMS | 3 times (08-31, and twice on 09-01) |
| v6e-16 | tpuT16 | once (08-26) |
| v6e-8 | tpuT8 | **never** -- driver log reads `none provisioned` |
| 2x v6e-8 | tpuTMS | **never** |
| v5litepod-8 / -16 / 2x v5litepod-1 | | **never** |

So 2 of 7 target shapes have ever reached ACTIVE, and the planned
tpuT4 -> tpuT8 -> tpuT16 -> tpuTMS ladder is not obtainable. But N x v6e-1
multislice *is* obtainable, gives N chips, and puts **every** collective on DCN --
the channel the TPU rung exists to test and the one a single slice leaves dark.
The ladder is therefore now slice COUNT rather than slice size:
tpuTMS (2) -> tpuTMS4 (4) -> tpuTMS8 (8). The cells needed no changes: each reads
`jax.devices()` and `--multihost auto` forms the cross-slice mesh.

**Label hazard, recorded because it would corrupt the analysis silently.** The
registry describes `tpuTMS` as "2x v6e-8 = 16 chips". That has never been what
ran: every tpuTMS cell to date executed on 2x v6e-1 = **2 chips**. Chip count must
be read from telemetry (distinct `gpu_uuid`, `mem_total_mb`), never inferred from
the scope name.

## 2. What a v6e actually reports

Measured on `tpuTMS_nonml_duty30_rep2`, a full 600 s cell: **708 rows at ~1.2 Hz**,
`tpu_metrics_live = 1`, `mem_total_mb = 31996` (32 GB confirmed).

**Live (8):** `gpu_utilization_pct`, `mem_used_mb`, `mem_total_mb`,
`tpu_duty_cycle_pct`, `tensorcore_util_pct`, `hlo_exec_us_{p50,p90,p99}`,
`hlo_queue_size`, `dcn_collective_lat_us_{p50,p90,p99}`, `buffer_xfer_lat_us_p90`.

**Dead (11), as documented:** `power_draw_w`, `temperature_c`, `sm_clock_mhz`,
`mem_clock_mhz`, `pcie_tx_mbps`, `pcie_rx_mbps`, `encoder_util_pct`,
`decoder_util_pct`, `fan_speed_pct`, `mem_utilization_pct`, `hbm_bw_util_pct`.

Two silent redefinitions that make GPU-calibrated thresholds non-transferable:
`gpu_utilization_pct` on TPU **is** TensorCore duty cycle (the collector writes
the same `duty` value to it and to `tpu_duty_cycle_pct`, differing only in the
missing-value sentinel -- so the two are the same channel and only one is
featurised), and `mem_utilization_pct` means HBM *bandwidth* where NVML means
fraction of sample period spent accessing memory.

Only 2 of the 8 live channels were being featurised. Now 9 TPU-native features
enter through `EXTRA_SIGNALS`, with the two comm channels and tensorcore util also
in `XR_SIGNALS` (a straggler slice appears as cross-rank dispersion).

## 3. Calibration of the matched control, from measurement

The apparatus retraction depends on the F=0 disguise and the honest non-ML cell
having the same footprint. `_ALLOC_MATCHED` was **12.0 GB, an uncalibrated guess**.

| cell | mem_used_mb | note |
|---|---|---|
| `advtrain_satfiller_only` (F=0) | 5196 median, 5794 max | 758 samples |
| `nonml_duty30` at `--alloc-gb 2` | 3504 median | => non-alloc baseline ~1456 |

Matching 5196 MB needs 5196 - 1456 = 3740 MB = **3.65 GB**. The old value was
2.3x the real footprint *and* large enough to OOM a 32 GB chip, so every
`nonml_sat` cell died in ~55 s and controlled nothing. Re-derive per scope.

## 4. Eight defects, and why they were invisible until now

The first four are plumbing; the last four would have corrupted results.

1. **Per-slice worker identity.** Both slices reported `agent-worker-number=0`, so
   both resolved to worker 0 and slice 1 overwrote slice 0's parquet in GCS. Every
   cell failed `1/2 workers` while silently discarding half the data. The queue
   now exports `SPAR_TPU_SLICE_ID` per slice node.
2. **Zone-name collision.** Request names were derived with `cut -c3-7`, and
   `us-east5-a`/`us-east5-b` both truncate to `east5`. After claiming one, the
   release loop's identity guard matched the *other* zone's identically-named
   request and skipped it: a second 2-chip multislice provisioned and billed idle,
   reclaimable only by the campaign's final sweep up to MAX_HOURS later.
3. **Double HBM allocation.** The resident dummy did `jnp.zeros(...)` then
   `device_put` into the sharding -- allocating twice, and asking for one buffer
   that large. An 8 GB request failed with `Attempting to reserve 8.00G at the
   bottom of memory. There are 7.25G free`. Now allocated in 1 GB chunks built
   directly at their sharding, reporting `ALLOC_REALISED_GB` (a matched control
   that quietly allocated less than asked would break the pairing).
4. **Teardown hang.** `nonml_comm` completed its full 600 s on all three reps --
   1078 psum slots of real DCN traffic -- then sat idle until the 1000 s cell
   timeout killed it. `jax.distributed`'s coordinator connection can block at
   interpreter exit once a collective has been in flight. Now a bounded 20 s
   shutdown then `os._exit`.

5. **Replicated batches must be process-uniform.** The batch is placed with a
   replicated `PartitionSpec()`, which asserts the value is identical on every
   process -- but the data key was seeded `PRNGKey(base + pidx)`, deliberately
   different per host. Every multi-host training cell aborted at its first train
   step. Invisible on tpuT8 because a v6e-8 is one process. The same bug sat in
   `tpu_fsdp_training.py` and `tpu_sharded_inference.py`, so it would have killed
   every tpuT16 cell too.
6. **Wall-clock loop bounds deadlock a multislice.** This is why only one cell of
   the run passed. `while time.time() - t0 < duration` is unsafe once the body
   runs collectives: the hosts drift, and eventually the faster one's deadline
   fires and it leaves while the slower one is still blocked in a collective with
   no partner. `tpuTMS_advtrain_satint_f20` trained correctly -- real train steps
   interleaved with filler, loss 10.5717 -> 10.5709 -- then died 16 minutes after
   its last step with

       error_reporter.cc:141 Aborting the coordinator after collecting errors
       from all workers as megascale_error_reporter_abort_on_hang is set to true

   The evidence is clean: `advtrain_satfiller_only` was the **only** cell that
   returned OK, and it is exactly the cell with no collectives, because F=0 never
   trains. comm mode was doubly exposed -- its inner loop was time-bounded too, so
   one host could issue 100 all_reduces in a slot where the other issued 98.

   Fixed by `T.agreed_step_budget()`: extrapolate the step rate from a reference
   taken *after* the first collective step (excluding XLA compile), then take host
   0's total for everyone. One collective, once, before the steady state --
   deliberately not a per-step agreement, because these cells' DCN traffic **is**
   the measurement and a per-step barrier would manufacture the signal we are
   trying to detect. At F=0 no collective is needed at all.
7. **TPU filenames were parsed into labels that defeat workload-held-out CV.** The
   loader only knew `_NVIDIA_`, so
   `tpuTMS_nonml_duty30_rep2_w0c0_Google_TPU_v6e_<run>.parquet` fell through to a
   generic `rsplit('_', 3)` and produced `workload_label =
   'tpuTMS_nonml_duty30_rep2_w0c0_Google'`. That label does not match
   `workload_group`'s `_rep\d+$` anchor, so **every rep and every chip becomes its
   own workload** and held-out-by-workload CV silently degenerates into
   run-grouped CV -- exactly the leak that criterion was adopted to remove, and it
   would have reported inflated accuracy with nothing visibly wrong. Fixed by
   splitting on `_Google_TPU_` and stripping the `_w<W>c<C>$` rank token before
   the rep token. Verified inert on the GPU side: 0 of 362 labels change group.
8. **Six of eight live TPU channels reached no model.** See section 2.

## 5. Verified healthy

- **No infinities** in the TPU feature matrix -- the Phase 1 `util_per_sm_pct`
  guard holds on real TPU data. All 2038 non-finite cells are NaN, every one from
  skew/kurtosis of a constant series (the dead channels, plus `mem_used` in 32
  flat windows); `fillna(0)` handles them.
- **The window builder works end-to-end on TPU telemetry**: 1466 raw rows -> 141
  windows -> 171 features, correct three-way classes, correct workload groups.
- **Stage-2's WB-O defence covers the new TPU features**: the generalised suffix
  rule drops mean/p50/p95 as levels and keeps cv/skew/acf/burstiness/slope as
  shape, and reproduces the original hand-written 73-name LEVEL list with **0
  disagreements** -- so the TPU-native channels cannot leak into Stage 2 as
  pseudo-shape.

## 6. Open

- No `tpuTMS` cell with training has yet returned OK; the fix for defect 6
  reached the workers at 20:24Z, so the first clean rep is rep2.
- `_ALLOC_HIMEM = 48.0` exceeds a 32 GB chip. With chunked allocation it now
  degrades to "as much as fits" and logs the realised figure rather than dying,
  but the cell no longer means what its name says at this scope.
- The scale ladder above 2 slices (tpuTMS4, tpuTMS8) is registered and parked but
  has never provisioned.
