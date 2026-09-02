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

## 2b. The `_p50/_p90/_p99` TPU columns are not percentiles

Established from the collector source, not inferred. libtpu's distribution
payload exposes only `count/mean/min/max`, and
`collect_tpu_telemetry.py` deliberately passes `[min, mean, max]` into
`_percentiles()` rather than inventing quantiles. `np.percentile` over three
points therefore yields:

| column | what it actually is |
|---|---|
| `*_p50` | **the interval mean** (median of 3 sorted points) |
| `*_p90` | an interpolation 80% of the way from mean to max -- no statistical meaning |
| `*_p99` | **~the interval max** (98% of the way to max) |

Worked example with min=100, mean=5000, max=2e6: reported p50=5,000,
p90=1,601,000, p99=1,960,100. So `hlo_exec_us_p90` is essentially a
max-dominated quantity, which is why it reaches 13.6 **seconds** in places -- that
is a compile-time HLO execution, not a steady-state one.

**Are they cumulative?** Two of the metric names say so outright:
`megascale.collective_end_to_end_latencies.microsecond.cumulative.distribution`
and `megascale.dcn_transfer_latencies.microsecond.cumulative.distribution`. But
the observed series **decrease** within a run -- `dcn_collective_lat_us_p90` on
`nonml_duty30` falls from 16,100,725 to 83,884 -- and a cumulative maximum cannot
decrease. So reads evidently reset the distribution, i.e. each sample is the
interval since the previous read, and the enormous first value is the
`jax.distributed.initialize()` collective landing in the first interval.

Consequences for the analysis, none of them optional:

- Prefer `_p50` (interval mean) and `_p99` (interval max) and treat `_p90` as
  redundant with `_p99`; the three are not independent statistics.
- `dcn_collective_lat_us_*` is **nonzero on cells that run no cross-slice
  collectives at all** (`nonml_duty30`, an `hpc`-mode cell, medians ~86,000).
  That is initialisation traffic, not workload traffic. Any claim that the DCN
  channel substitutes for the missing power channel has to survive this: the
  channel is live but its early samples are dominated by process startup.
- The `hlo_exec_us_*` series rises by 2 orders of magnitude across a run in some
  cells (500 -> 35,000 for `_p50`), so a window's absolute level partly encodes
  how far into the run it sits. A level feature on it risks encoding window
  position rather than workload -- the same shape of hazard as the label-parsing
  bug in section 4.7, and it must be checked against window index before any of
  these features is trusted.

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

## 3b. F=0 is not the F->0 limit: the disguise carries a 3.2x memory tell

Measured from `mem_used_mb` over whole runs on the 2x v6e-1 slice:

| cell | footprint | when |
|---|---|---|
| F=0 (`satfiller_only`) | ramps to **5196 MB**, stays | from startup |
| F=0.2 (both slices) | 5.2 GB until t=124s, then **16495 MB** | jump at first train step |
| F=0.5 | same jump to **16495 MB** | t=126s |

The jump is the first train step compiling: activations, gradients and the
optimizer moments are materialised and stay resident. It is **independent of F** --
0.2 and 0.5 both land on exactly 16495 MB -- because it is a one-off workspace
allocation, not anything per-step.

This matters more than it first looks. A disguised-training run at *any* F>0
holds 16.5 GB while the filler-only cell holds 5.2 GB, and on TPU -- where power,
clocks and temperature do not exist -- `mem_used_mb` is one of the few live
**level** channels. Two consequences:

1. **The F=0 cell is not the limit of the MN-I ladder.** It is a structurally
   different workload: it never allocates the training workspace at all. The
   footprint is discontinuous at F -> 0, so F=0 cannot be read as "MN-I with the
   training turned down to zero".
2. **One matched control is not enough.** `_ALLOC_MATCHED = 3.65 GB` matches F=0
   (total ~5196 MB) and is the right partner for the apparatus test. It is the
   WRONG partner for the headline F=0.5 comparison: memory alone would separate
   disguise from control by 3.2x, and any detection would be reading footprint
   rather than training dynamics -- precisely the confound a matched control
   exists to remove. So a second control, `nonml_sat_16g` at 14.69 GB of dummy
   (+~1456 MB baseline = ~16.5 GB), is now registered for every multislice rung.

**Naming hazard, hit while adding it:** the cell may not be called
`nonml_sat_train`. `threeway_label()` routes on the substring `train`, so
`tpuTMS_nonml_sat_train` labels an HONEST non-ML control as `ml_training` --
verified, it returns exactly that. The keyword grammar silently wins over intent,
so any new honest cell must be checked through `threeway_label` before it runs.

## 3c. What the failed cells still contain

`FAIL` is a verdict on the cell's exit status, not on its telemetry:

| cell | rows / span | usable? |
|---|---|---|
| F=0.2 rep1 w0 | 1284 / 1283s | **partly** -- real interleave at util 100% from t=124s to ~700s, then idle padding while it hung |
| F=0.2 rep1 w1000 | 683 / 682s | **mostly** -- busy throughout |
| F=0.5 rep1 w0 | 475 / 474s | **no** -- util ~0 the whole time: the crash hit at the first train step, so the run is compile then hang |
| `nonml_comm` reps 1-3 | full 600s | **yes** -- all 1078 slots ran; only the exit hung |
| `nonml_duty30` reps 1-3 | full 600s | **yes** |
| `satfiller_only` reps 1-3 | full 600s | **yes** |

The idle tail is a hang artifact and must be trimmed, not averaged in: it drags
`gpu_utilization_pct` median from 100 to 0 on the w0 stream. Any salvage pass has
to cut at the last STEP line in the worker log, not at the end of the parquet.

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


## 7. First workload-held-out result on TPU -- and why it is not yet believable

Corpus: 44 streams / 28,521 rows / 28,487 cell-seconds from the 2x v6e-1
multislice, ingested with hang tails trimmed. 2,743 windows (30 s, 10 s stride),
8 workload groups, classes balanced 1371 ml_training / 1372 other.
Leave-one-WORKLOAD-out, RandomForest, p = P(ml_training).

| feature set | window TPR@0.8 | window FPR@0.8 |
|---|---|---|
| all 171 | 0.734 | 0.164 |
| **memory features removed (114)** | **0.700** | **0.003** |
| memory features only (57) | 0.754 | 0.499 |

Per cell, with memory features removed:

| cell | class | p_med | frac > 0.8 |
|---|---|---|---|
| `advtrain_satint_f05` | training | 0.996 | 0.901 |
| `advtrain_satint_f20` | training | 0.997 | 0.967 |
| `advtrain_satint_f50` | training | 0.996 | 0.975 |
| `advtrain_satfiller_only` (F=0) | training | 0.349 | 0.095 |
| `nonml_comm` | other | 0.594 | 0.000 |
| `nonml_duty30` | other | 0.359 | 0.000 |
| `nonml_duty50_hash` | other | 0.413 | 0.005 |
| `nonml_sat` | other | 0.322 | 0.008 |

Two things worth keeping:

**The memory channel is a liability on TPU, not an asset.** With memory features
in, the false-positive rate at tau=0.8 is 16.4%, driven by `nonml_duty50_hash` at
58%; memory-only puts that cell at 94%. Drop them and its false-alarm rate falls
to 0.5%. This is the same cell that false-alarmed hardest on GPU, and the reason
is visible in section 3b: every F>0 cell sits at exactly 16495 MB because of a
one-off arena allocation, so a tree learns a footprint threshold that has nothing
to do with training and then misfires on any honest cell that happens to allocate.

**F=0 evades here too** -- 0.095 of windows above tau -- which is consistent with
the GPU finding, and is the expected result rather than a failure: the cell runs
no training steps at all.

### Why these numbers must not be quoted

1. **Family identity is not held out.** All four adversarial cells are the SAME
   SCRIPT with the same model and filler; the four honest cells come from two
   scripts. Holding out `f05` leaves `f20`, `f50` and `satfiller_only` in the
   training fold, so the classifier can score a held-out cell by recognising its
   family rather than by detecting training. The GPU corpus has ~170 workloads to
   dilute this; here there are 8. That is the most likely explanation for F=0.05
   appearing 90% detectable when the GPU result found nothing below F=0.5
   detectable at any usable threshold -- a discrepancy too convenient to accept.
2. **No ml_inference class**, so this is ML-vs-other, not the two-stage p_train
   the paper's operating point is defined on. Stage 2 cannot be trained at all.
3. **`nonml_sat` is not footprint-matched** (2880 and 6464 MB against a 5196
   target), so the apparatus contrast is not yet valid -- see section 3.
4. **8 groups** means every fold trains on 7 cells; the variance is large and no
   confidence interval here would be honest.

The fix is more workload FAMILIES, not more reps: inference, honest undisguised
training, `other_fft`, `idle`, and the same ladder at single-slice and v5e scopes.
That is what the relaunched campaign collects.


## 8. v5e pre-flight (2026-09-02), run on an idle grant before the campaign claimed it

`v5litepod-8` provisioned for the first time ever, in TWO zones at once. While the
campaign was still in its parking pass and unable to claim, the idle slice was
used to pre-flight the pinned stack -- worth doing because a wrong libtpu pairing
would have failed setup and cost the grant.

    jax 0.6.2  jaxlib 0.6.2          libtpu 0.0.17 -- WORKS on v2-alpha-tpuv5-lite
    devices 8  local 8               single slice, 8 chips, ONE host
    kind "TPU v5 lite"
    bytes_limit  16126 MB/chip       HALF the v6e's 31996
    memory_stats keys: bytes_in_use, bytes_limit, bytes_reservable_limit,
                       bytes_reserved, largest_alloc_size,
                       largest_free_block_bytes, num_allocs,
                       peak_bytes_in_use, peak_bytes_reserved

Consequences:

* The pin needs no change for v5e. That risk is retired.
* **v5e has 16 GB/chip**, so `_MEM_MATCH_TRAIN = 16495` exceeds the whole chip and
  is unsatisfiable there. The hard cap at 80% of `bytes_limit` stops the loop and
  logs `match hit the cap`, so it degrades visibly rather than OOMing -- but the
  cell is then NOT a matched control.
* `_MEM_MATCH_F0 = 5196` is likewise a v6e-multislice number. The 1b model sharded
  over 8 v5e chips holds a different per-chip footprint than over 2 v6e chips, so
  neither target transfers.
* Therefore: **treat nonml_sat and nonml_sat_16g as calibrated ONLY for tpuTMS on
  v6e.** For every other scope, read that scope's own
  `MEM_BYTES_IN_USE_STEADY` from its first F>0 cell and re-derive. The disguise
  cells print it precisely so this needs no extra rental.
* `bytes_reserved` exists on v5e as well, so the arena-tracking match key is
  available on both generations.
