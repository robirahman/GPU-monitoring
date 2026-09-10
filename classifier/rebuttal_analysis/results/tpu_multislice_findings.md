# TPU rung: findings

> **SUMMARY (2026-09-02 18:45Z).** Sections below are append-only and in
> discovery order, including three claims of mine that were later withdrawn. Read
> this box first.

## Corpus in hand

| corpus | shape | groups | classes | notes |
|---|---|---|---|---|
| `tpuTMS@v6e` | 2x v6e-1 multislice | 8 | train + other | no inference; predates the class references |
| `tpuT8@v5e` | v5litepod-8 single slice | 11 | all three | 3 full reps, 264 streams |
| `tpuTMS@v5e` | 2x v5litepod-1 multislice | 11 | all three | + the MN-T attack ladder |
| `tpuT1@v6e` | v6e-1 single slice | 12 | all three | first with a footprint-matched honest control |

Both topologies and both generations are covered. All 11 libtpu metrics are logged
in data collected after 07:40Z; earlier data has 7.

## Established

* **DCN is multislice-only.** `dcn_collective_lat_us_*` and
  `buffer_xfer_lat_us_p90` are NaN on any single slice and carry values on
  multislice, on both generations. The premise the rung rests on, confirmed from
  data (section 9).
* **The memory channel carries real signal on TPU.** Removing all memory features
  costs 0.3-0.4 in false-alarm rate on every corpus tried -- notable for a platform
  with no power channel at all (section 18).
* **The GPU headline claim is not subject to the footprint-coverage artifact.**
  17 of 56 honest GPU groups sit above the median training footprint; the honest
  class genuinely contests the training range (section 17).
* **The within-TPU-generation detector keys on the SCRIPT, not on training.**
  Honest training scores 0.254 while a cell that never trains scores 0.968. Four
  of eight groups are one script and leave-one-workload-out cannot separate them
  (section 10). GPU has 335 groups and does not share this.
* **F=0 is not the F->0 limit.** Every F>0 cell carries a training workspace the
  F=0 cell never allocates -- 16495 vs 5196 MB/chip on v6e, 7013 vs 1089 on v5e
  (section 3b).
* **Stage 2 cannot order the classes.** It puts honest training (0.505) below both
  inference cells (0.733, 0.637) (section 12).

## Withdrawn

* "Memory features are the liability; drop them" (section 7) -- the filter caught
  52 of 62 memory features (section 14).
* "The footprint confound survives the ablation" (section 13b) -- it leaked
  through the 10 features the filter missed (section 14).
* "Drop memory levels, keep memory shape" (section 15) and its footprint-coverage
  mechanism (section 16) -- three corpora give three different winners, and the
  mechanism's own prediction fails on `tpuT1` (section 18).

* "Plain SGD training evades the detector completely" (section 20) -- it evades on
  v5e (0.000 above tau) and is CAUGHT on v6e (0.494), same script, same holdout,
  same code. That was a property of the v5e corpus, not of the optimizer
  (section 21).

**No feature-subset rule, and no evasion claim, is defensible from a single
corpus.** Four claims in this file were proposed and withdrawn -- sections 7, 13b,
15/16 and 20 -- and every one looked convincing on the corpus it was fitted to and
failed on data it had not seen. 8-15 groups with 1-3 reps is too small, with
section 10's family leakage only partly addressed.

## The surviving adversarial results

* **MN-T's shaping does nothing, on either generation.** Raising TensorCore
  utilisation toward the inference profile, cutting cross-slice collectives
  32-fold and adding a device-to-host trickle do not move the type decision. On
  v6e the STRONGEST shaping is the most detected (`a32` 0.799 vs `off` 0.494), so
  if anything it is counterproductive.
* **The optimizer has a real but partial effect where the detector works.** On
  v6e, same script and topology: SGD 0.494 vs AdamW 0.875 above tau, with
  `mem_used` 12750 vs 17900 MB and the startup delta 2601 vs 7820 MB -- the
  moment tensors are visible at allocation time, as predicted, but they do not
  buy evasion.
* **An unseen training script is far harder to detect on v5e than on v6e.** That
  is the open question worth pursuing, and a more interesting one than the
  optimizer: how detectable a novel workload is depends on the generation the
  detector was fitted to. The v5litepod-8 re-collect now running gives a v5e
  SINGLE-SLICE corpus with the same cell set as the v6e multislice one, which
  separates generation from topology for the first time.

## Open

* **MN-T is untested, not refuted.** The attack moved `p(train|ML)` by nothing,
  but on a subcorpus where run FPR is 0.375 and honest cells alert as often as
  training ones. There was no working detector to fool (sections 13, 13b).
* A multislice corpus with the full honest class set (5 non-ML cells) is the
  missing piece for MN-T, for Stage 2, and for any feature claim.
* `~/win_full.parquet` predates the legacy-TPU quarantine and should be rebuilt.

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


## 9. First v5e cell (tpuT8, v5litepod-8, single slice, 8 chips) -- 2026-09-02

    tpuT8_advtrain_satint_f50_rep1  OK (695s, 1/1 workers)
    1177 steps in 599.8s, realised_frac=0.5072, useful_tokens=4,890,624
    chips=8 hosts=1     8 per-chip streams: w0c0 .. w0c7
    gpu_name='Google TPU v5e'   mem_total_mb=16126   mem_used_mb=7013

Confirmations:

* **8 distinct per-chip streams from one host.** On the old `(ts, run_id)` dedup
  key this would have collapsed to a single chip -- the Phase 1 fix (adding
  `gpu_uuid`) was written for precisely this case and this is its first real test.
* **The DCN channels are DARK on a single slice**: `dcn_collective_lat_us_p90` and
  `buffer_xfer_lat_us_p90` are both NaN, while they carry values on the 2-slice
  v6e runs. This is the first direct confirmation of the premise the whole rung
  rests on -- megascale.* accumulates only ACROSS slices -- rather than an
  inference from metric names.
* The steady-state budget correction works here too: initial estimate 1847 steps,
  re-agreed to 1177 at 0.43 s/step, landing 599.8 s against a 600 s target.

Cautions:

* **`hlo_queue_size` is topology-dependent**: median 1 on v5e single-slice against
  ~519 on the v6e multislice adversarial cells. Within the v6e corpus, where every
  cell shares one topology, its 519-vs-0 separation is still a workload signal --
  but its SCALE does not transfer, so scopes must not be pooled on this feature
  without normalising.
* **Footprints do not transfer between scopes.** v5e F>0 sits at 7013 MB/chip
  against v6e's 16495, because the same 1b model shards eight ways instead of two,
  and the chip is half the size (16126 vs 31996 MB). `_MEM_MATCH_TRAIN = 16495`
  therefore exceeds an entire v5e chip. Targets are now a per-scope map
  (`_MEM_MATCH_{F0,TRAIN}_BY_SCOPE`); a scope with no measured entry falls back to
  the open-loop `--alloc-gb` knob and is explicitly NOT a matched control.
* Consequently `tpuT8_nonml_sat` in rep 1 is unmatched (open-loop 3.65 GB global
  = ~467 MB/chip against a target near 7013). Its F=0 partner had not run yet when
  rep 1 was queued. Excluded from the apparatus contrast; reps 2-3 can be matched
  once that scope's satfiller_only footprint is read.


## 10. The within-generation numbers are keyed on the SCRIPT, not on training

Corpus at this point: 148 streams, two generations -- 2,743 windows from 2x v6e-1
multislice (`tpuTMS`, 8 workload groups, no inference class) and 6,488 from
v5litepod-8 single-slice (`tpuT8`, 11 groups, all three classes). Harness:
`classifier/rebuttal_analysis/tpu_rung_eval.py`.

### The decisive number is an honest training run

Within v5e, held out by workload, full two-stage `p_train`, memory features
dropped, tau=0.8:

| cell | class | p_med | frac > tau |
|---|---|---|---|
| `advtrain_satint_f05` | disguised training | 1.000 | **1.000** |
| `advtrain_satint_f20` | disguised training | 1.000 | **1.000** |
| `advtrain_satint_f50` | disguised training | 0.984 | **0.976** |
| `advtrain_satfiller_only` (F=0) | disguised, trains NEVER | 0.912 | **0.968** |
| `train_fsdp1b` | **honest, trains ALWAYS** | 0.504 | **0.254** |
| `inferrep_1b` | inference | 0.703 | 0.119 |
| `infershard_1b` | inference | 0.390 | 0.356 |
| `nonml_comm` | other | 0.141 | 0.000 |
| `nonml_duty30` | other | 0.227 | 0.000 |
| `nonml_duty50_hash` | other | 0.721 | 0.048 |
| `nonml_sat` | other | 0.528 | 0.000 |

Read the last two training rows together. `train_fsdp1b` performs a gradient step
on EVERY step of its 600 s; `advtrain_satfiller_only` performs NONE, by
construction. If the model were detecting training, the honest run would score
highest and F=0 lowest. The order is exactly inverted: **0.254 for real training,
0.968 for a cell that never trains**. What the four `advtrain_*` cells share is
not training -- it is the MN-I script, its model, and its filler.

So the 0.98 "TPR" is a script detector. The GPU corpus has ~170 workloads to
dilute this; a single generation here has 11 groups of which 4 are one script, and
leave-one-workload-out cannot separate the two. This supersedes section 7: those
numbers are not a weaker version of the claim, they are a different quantity.

### Memory features flip which cell evades, which is the same tell

With memory features IN, F=0 evades (0.079 above tau) and `duty50_hash`
false-alarms at 0.710 (FPR 0.172). With them OUT, F=0 is "caught" at 0.968 and
`duty50_hash` falls to 0.048 (FPR 0.012). Nothing about the workloads changed --
only whether the model can see that F=0 holds 1089 MB/chip while F>0 holds 7013.
Given memory, it separates F=0 from its siblings and scores it low; denied memory,
it scores F=0 as just another member of the family. Both readings are artifacts.

### Cross-generation transfer, which is the test that cannot be faked

Train on one generation, test on the other. Different chip (31996 vs 16126 MB),
different sharding of the same model (2 chips vs 8), different interconnect for
the same collective (DCN vs ICI), and `dcn_collective_lat_us_*` live on one side
and NaN on the other.

| direction | features | disguised TPR@0.8 | honest FPR@0.8 |
|---|---|---|---|
| v6e -> v5e | all | 0.508 | **0.000** |
| v6e -> v5e | no memory | 0.011 | 0.000 |
| v5e -> v6e | all (Stage 1 only) | 0.559 | 0.451 |
| v5e -> v6e | no memory (Stage 1) | -- | see script |

Two things follow. **v6e -> v5e at 0.508 TPR with a 0.000 false-alarm rate is the
only operating point here that is not obviously an artifact** -- it was fitted
without ever seeing a v5e chip, so family recognition cannot explain the honest
cells all scoring below tau. And it depends on the memory features: dropping them
collapses TPR to 0.011, the reverse of the within-v6e ablation in section 7. The
arena footprint is apparently the one channel that means the same thing on both
generations, even though its absolute value differs.

The reverse direction is poor (FPR 0.451), which is expected: a v5e-trained model
has never seen DCN traffic, and every v6e cell has it.

### What this changes

* **Do not quote a within-generation TPR** until the corpus has several
  independent training scripts per generation. The fix is more workload FAMILIES,
  and `train_fsdp1b`, `inferrep_1b` and `infershard_1b` now exist at every scope
  precisely so the ml_training class is not just MN-I.
* **`train_fsdp1b` is the control that exposes this**, and it only entered the
  corpus on 2026-09-02. Every earlier TPU analysis lacked the one cell that could
  have detected the problem.
* Stage 2 works at least partly: inference scores 0.119-0.356, below every
  disguised cell.
* Two-stage `p_train` is now computable on TPU data (v5e has all three classes),
  so the paper's actual operating point is reproducible on this hardware -- on
  v6e it still is not, since that corpus has no inference cells.


## 11. Overnight plan of 2026-09-02, and two gaps closed before it started

Robi's brief: cover real inference, training and non-ML on BOTH topologies with all
available telemetry, then attack the classifier's TYPE decision on a multi-slice
training run. Budget up to $3000, commit data frequently.

### Telemetry was incomplete: 7 of 11 channels

`ListSupportedMetrics`, enumerated on a live v5litepod-8 (the metrics server lives
inside the workload process, so this is only possible while a cell holds the chip),
serves **eleven** names. Seven were mapped. The four discarded on every run so far:

| metric | now |
|---|---|
| `megascale.host_to_device_transfer_latencies...` | mapped + featurised |
| `megascale.device_to_host_transfer_latencies...` | mapped + featurised |
| `slice.error.detected.gauge` | logged as a data-quality flag, not featurised |
| `tpu.runtime.uptime.seconds.gauge` | logged, deliberately not featurised |

The transfer-latency pair matters beyond completeness: **training streams batches
host->device while inference streams generated tokens device->host**, so these are
the channels most likely to separate the two classes -- which is Stage 2's job, and
Stage 2 is exactly what this corpus could not train until inference cells existed.
`tpu_uptime_s` is excluded from features because it rises monotonically and would
encode window position, the hazard already flagged for `hlo_exec_us_*`.

**Note for analysis:** these four channels exist only in data collected AFTER
2026-09-02 07:40Z. The 44 v6e multislice streams and the 264 v5e streams predate
them, so a model using h2d/d2h cannot be compared against those corpora directly.

### MN-T: attacking the TYPE decision

`workloads/tpu_adversarial_typeconf.py`. MN-I attacks Stage 1 (make training look
non-ML); MN-T attacks Stage 2 (keep it plainly ML, make train read as INFER).
Harder by construction: Stage 2 sees only shape features after the WB-O level drop,
so no single level can be moved to win.

Each knob targets a channel measured to separate the classes on v5e:

| channel | training | inference | knob |
|---|---|---|---|
| `tensorcore_util_pct` | 17 | 88 | `--tc-fill` dense matmul bursts |
| `mem_used_mb` | 7013 | 2668 | `--optimizer sgd` (no Adam moments) |
| cross-slice collectives | every step | none | `--accum K` (local SGD) |
| d2h transfer | idle | streaming | `--d2h-trickle` |

Ladder: `_off` (SGD alone -- the control that says how much of any effect is just
the optimizer), `_a8`, `_a32`. It remains REAL training: genuine forward/backward
and parameter update on the same 1b model, useful tokens reported so the evasion
tax is measurable.

**Stated limitation.** Under `--accum K` the parameters are replicated per slice
and averaged only every K steps, which is what cuts DCN events to 1/K. But the
batch is process-uniform (device_put asserts a replicated value is identical
across processes), so every host trains on the same data and the averaging is
numerically redundant. The telemetry -- compute, footprint, collective cadence,
transfer pattern -- matches true local SGD, and the telemetry is what the detector
sees, so the evasion question is answered faithfully. Throughput-per-token from
this cell is not meaningful and must not be quoted.

**What to look for:** the attack succeeds if `p(infer | ml)` rises with attack
strength while ground truth stays training, i.e. `_off` -> `_a8` -> `_a32` shows
falling `p_train` on cells that all train every step. Because they DO train every
step, this is a cleaner test than MN-I: there is no "it trained less" confound, so
a drop in p_train is unambiguously the detector being fooled about TYPE.

### Durability

GCS is the durability layer -- each cell's parquets upload as that cell finishes,
so a lost slice costs at most the cell in flight (verified: 256 objects present
mid-run). `scripts/tpu_collection/corpus_autocommit.sh` additionally ingests every
campaign session every 30 minutes and commits the trimmed corpus, so there is a
versioned copy beside the code. git-lfs 3.5.1 was installed to make that possible:
the repo is LFS-configured but the binary was missing, so every push all session
had failed with a misleading "failed to push some refs".


## 12. Stage 2 ranks honest training BELOW actual inference

With all three v5e reps ingested (264 streams), held out by workload, memory
features dropped, reporting Stage 2's own output p(train | ML) -- the type
decision, separated from Stage 1's ML gate:

| cell | ground truth | p(train\|ML) median | frac > 0.8 |
|---|---|---|---|
| `advtrain_satint_f05` | training | 1.000 | 1.000 |
| `advtrain_satint_f20` | training | 1.000 | 1.000 |
| `advtrain_satint_f50` | training | 0.995 | 1.000 |
| `advtrain_satfiller_only` | training (trains never) | 0.985 | 1.000 |
| `inferrep_1b` | **inference** | **0.733** | 0.158 |
| `infershard_1b` | **inference** | **0.637** | 0.303 |
| `train_fsdp1b` | **training (every step)** | **0.505** | 0.260 |
| `nonml_duty50_hash` | other | 0.798 | 0.478 |

Stage 2 puts honest training (0.505) BELOW both inference cells (0.733, 0.637).
The discriminator is not ordering these classes at all: it has learned that the
MN-I script family is training, and everything outside that family lands in a band
between 0.5 and 0.75 regardless of what it actually is.

This reframes the MN-T experiment. MN-T is designed to push p(train\|ML) down on a
run that trains every step -- but the type decision is **already unreliable before
any attack**, since it cannot rank a real training run above a real inference run.
So a low p(train\|ML) on an MN-T cell will not by itself demonstrate a successful
attack: the same score appears on honest training. The MN-T ladder must therefore
be read as a CONTRAST -- `_off` vs `_a8` vs `_a32`, all training every step, all
from one script -- and only a monotone fall across that ladder is evidence the
shaping did something. Against the absolute p(train\|ML) scale, everything
non-MN-I already looks like this.

It also predicts what the multislice re-collect will show: once `train_fsdp1b`,
`inferrep_1b` and `infershard_1b` exist on multislice too, Stage 2 there should
become trainable and can be checked for the same pathology. If it reproduces, the
honest conclusion is that this corpus cannot yet support a train-vs-infer claim on
TPU at all -- independent of the adversarial question.


## 13. MN-T first result: no type confusion, but the test was inconclusive

Data: `v5ms` = 2x v5litepod-1 multislice, v5e, 3 reps of each MN-T rung plus
honest training and both inference cells on the SAME topology. Held out by
workload, memory features dropped, tau=0.8, scope restricted to tpuTMS so the
comparison is within one topology.

    MN-T ladder (every cell trains on EVERY step)
    cell                     p_ml    p(train|ML)   p_train   frac>tau
    typeconf_off            0.988         0.998     0.998      0.997
    typeconf_a8             1.000         0.985     0.985      0.997
    typeconf_a32            0.968         0.998     0.997      0.781

**The attack did not work.** p(train|ML) is flat at 0.985-0.998 across the whole
ladder: raising TensorCore utilisation toward the inference level, stripping the
optimizer moments, cutting cross-slice collectives to 1/32, and adding a
device->host trickle moved the type decision by nothing. Only the RATE above tau
falls at _a32 (0.781 vs 0.997), and its median p_train is unchanged.

**But this is not evidence the attack fails.** On this subcorpus the detector is
degenerate: the honest false-alarm rate is **0.736**, with `nonml_sat_hi` at 0.981
and the real inference cell `infershard_1b` at 0.983. A model that calls almost
everything training cannot be tricked into calling something inference -- there is
nothing left to fool. The MN-T ladder was measured against a broken instrument.

**Why the instrument is broken here, and it is our own doing.** The done-marker bug
(section 14) meant `v5ms` skipped `nonml_duty30`, `nonml_duty50_hash` and
`nonml_comm` -- their markers already existed from camp2's v6e run -- so the
"other" class on this topology is just `nonml_sat` (1 rep) and `nonml_sat_hi`,
against seven training groups. With 11 groups and that imbalance, leave-one-
workload-out cannot produce a usable operating point.

One sub-result does look real and is worth keeping: **`nonml_sat_hi` -- honest
non-ML held at the TRAINING footprint -- is called training at 0.981 even with
memory features dropped.** That is the cell added specifically to test whether
detection rides on footprint, and on this topology the detector cannot separate a
saturating non-ML job at a training-sized footprint from training itself. Treat it
as a hypothesis until the class balance is fixed, but it is the sharpest thing in
this table.

### What would make the MN-T test conclusive

A multislice claim that runs all 15 cells, giving 4 honest non-ML groups
(`sat`, `sat_hi`, `duty30`, `duty50_hash`, `comm`) against the training groups.
The marker namespaces have now been archived, and `tpuTMS4`/`tpuTMS8` (4x and 8x
v6e-1) have never had markers at all, so the next multislice claim will collect
the full set without any further intervention.


## 13b. Run level makes the degeneracy unmistakable -- and footprint leaks past the memory features

Same subcorpus (2x v5litepod-1 multislice, tau=0.8, memory dropped), now under the
3-of-5 run rule the claim is actually defined on:

| cell | class | runs | alert rate |
|---|---|---|---|
| `advtrain_typeconf_off` / `_a8` / `_a32` | training | 6 each | 1.000 |
| `train_fsdp1b` | training | 6 | 1.000 |
| `advtrain_satint_f05` / `f20` / `f50` | training | 2 each | 1.000 |
| `infershard_1b` | **inference** | 6 | **1.000** |
| `inferrep_1b` | inference | 6 | 0.167 |
| `nonml_sat_hi` | **other** | 6 | **1.000** |
| `nonml_sat` | other | 2 | 0.000 |

**run TPR 1.000, run FPR 0.750.** The detector alerts on everything except
`nonml_sat`. There is no operating point here, which settles how to read section
13: MN-T's null result says nothing about MN-T.

**The sharp clue.** `nonml_sat` and `nonml_sat_hi` are the SAME workload -- same
kernels, same duty, same script -- differing only in footprint: one matched to the
F=0 cell, one to the training cell. One never alerts; the other always does. And
this is the run with `mem_*` features DROPPED. So the footprint is reaching the
decision through channels that are not the memory features, and **dropping `mem_*`
does not remove the footprint confound.**

Plausible routes, all testable and none yet tested: a larger resident working set
changes HLO execution timing (`hlo_exec_us_*` is live and featurised), changes the
allocator's behaviour visible in `hlo_queue_size`, or changes the filler's cache
behaviour and hence its utilisation shape. This matters well beyond TPU: the
section 7 ablation concluded "memory features are the liability" and dropped them
as the remedy. That remedy is now shown to be incomplete -- the confound survives
the ablation, so the earlier clean-looking FPR of 0.003 on the v6e corpus may have
been masking the same effect rather than eliminating it.

Next step for this, independent of the adversarial question: pair `nonml_sat` and
`nonml_sat_hi` at equal reps on a topology with the full honest class set, and
compare per-feature importances between them. If a non-memory channel tracks the
footprint, it will show up as the top discriminator between two cells that differ
in nothing else.


## 14. CORRECTION: my memory ablation was incomplete, and the real result is better

**Retraction first.** Sections 7, 13 and 13b all report numbers "with memory
features removed". The filter was `c.startswith("mem_")`, which catches 52 of the
**62** memory-derived features and misses ten:

    acf1_mem_used, acf2_mem_used, acf5_mem_used, acf10_mem_used, acf20_mem_used,
    first_30s_mem_delta_mb, first_30s_mem_delta_frac, first_30s_mem_ratio,
    time_to_mem_plateau_samples, power_per_mem_util

So every "memory removed" figure was computed with ten memory features still
present, including all four run-level ones. **Section 13b's conclusion -- that the
footprint confound survives the ablation and leaks through non-memory channels --
is withdrawn.** It leaked through memory features I failed to exclude. Tested
directly on the controlled pair (`nonml_sat` vs `nonml_sat_hi`, identical
workloads differing only in resident footprint, held out by run):

| features | accuracy telling the two apart |
|---|---|
| all 171 | 1.000 |
| no `mem_*` (old filter) | 0.951 -- carried by `first_30s_mem_*` |
| no memory at all (109) | **0.713**, against a 0.75 majority baseline |

At or below chance once memory is genuinely gone. The confound is entirely in the
memory features. (That test also initially leaked its own label: `get_feature_cols`
picked up the `hi` column I had added to the frame, and it ranked 4th by
importance. Fixed by keeping the label out of the frame.)

### The real finding, on section 7's exact corpus

Same 2,743 windows and 8 groups as section 7, so only the filter varies:

| filter | features | TPR@0.8 | FPR@0.8 |
|---|---|---|---|
| ALL | 171 | 0.734 | 0.164 |
| no `mem_*` (section 7's filter) | 119 | 0.710 | **0.009** |
| **NO memory at all** | 109 | 0.670 | **0.415** |
| memory only | 62 | 0.737 | 0.162 |

The ten features I failed to drop are the most useful ones in the set. Keeping
only the memory **startup trajectory** -- how fast the footprint ramps and when it
plateaus -- while discarding the 52 per-window memory **levels** is what produced
the clean 0.009. Drop the trajectory too and false alarms rise to 0.415, far worse
than keeping everything.

That has a clean physical reading, and it lines up with the level-vs-shape split
Stage 2 already encodes: the *absolute* footprint is confoundable, because an
honest job can allocate just as much (which is exactly what `nonml_sat_hi`
demonstrates), while the *trajectory* of allocating a model plus optimizer states
is a signature of the apparatus and much harder to imitate incidentally.

So section 7's recommendation -- "drop the memory features" -- happened to land on
a good feature subset for the wrong reason. The defensible version is: **drop
memory LEVELS, keep memory SHAPE.**

Caveats unchanged: 8 groups, and the family-identity leakage of section 10 still
applies, so these are feature-utility observations inside one corpus, not
deployment numbers.

### Corrected multislice numbers (2x v5litepod-1, memory genuinely dropped)

Run level, 3-of-5: **run TPR 0.533, run FPR 0.375** (previously reported as
1.000/0.750 with the incomplete filter). The detector is weak rather than
saturated, so the MN-T verdict of section 13 stands but for a cleaner reason: there
is still no usable operating point to attack. The ladder's rate above tau does now
decline monotonically -- 0.850, 0.732, 0.703 for `_off`, `_a8`, `_a32` -- but
p(train|ML) stays flat at 0.92-0.96, so what little moves is Stage 1's ML gate,
not the type decision MN-T targets.


## 15. The rule, validated: drop memory LEVELS, keep memory SHAPE

Section 14 found the right feature subset by accident. Stated as a rule and
tested, using the level/shape classification the codebase ALREADY has --
`twostage.stage2_feature_mask`, the WB-O rule Stage 2 applies to defend against
level-spoofing -- restricted to the 62 memory-derived features (24 levels,
38 shapes). Section 7's corpus, 2,743 windows, 8 groups, leave-one-workload-out:

| feature set | n | TPR@0.8 | FPR@0.8 |
|---|---|---|---|
| ALL | 171 | 0.734 | 0.164 |
| **drop memory LEVELS, keep memory SHAPE** | 147 | 0.724 | **0.000** |
| drop memory SHAPE, keep memory LEVELS | 133 | 0.723 | 0.571 |
| drop ALL memory | 109 | 0.670 | 0.415 |

Both halves matter, in opposite directions:

* **Memory levels are the false-alarm source.** Keeping them and dropping the
  shapes gives FPR 0.571. Dropping them costs 0.010 of TPR and takes FPR to zero.
* **Memory shape is load-bearing.** Dropping it as well costs 0.054 of TPR and
  raises FPR to 0.415 -- worse than keeping everything.

The mechanism is demonstrated directly rather than inferred, by the controlled
pair from section 14: `nonml_sat` and `nonml_sat_hi` are the same workload
differing only in resident footprint, and with memory present they are perfectly
separable while with memory absent they are at chance. An honest job can allocate
exactly as much as a training job -- so the LEVEL is confoundable by construction,
and `nonml_sat_hi` exists to prove it. What an honest job does not reproduce is
the *trajectory*: the ramp and plateau of allocating parameters plus optimizer
states, which `first_30s_mem_delta_*`, `first_30s_mem_ratio`,
`time_to_mem_plateau_samples` and `acf*_mem_used` measure.

That the same level/shape rule Stage 2 uses against a white-box adversary also
turns out to be the right rule for the memory channel in Stage 1 is the most
satisfying thing in this file: one principle, two places, and in the second it was
found by measurement rather than assumed.

**Standing caveats.** 8 workload groups, and the family-identity leakage of
section 10 is unaddressed -- four of those groups are one script -- so this is a
feature-utility result within one corpus, not a deployment number. The next
multislice run, with the full honest class set, is what tests whether it holds
when the class balance is real.


## 16. Section 15's rule does not generalise -- and the reason is the real finding

Tested the same level/shape ablation on the v5e single-slice corpus (16,408
windows, 11 groups, all three classes -- larger and better balanced than section
15's). It **reverses**:

| feature set | tpuTMS@v6e (sect 15) | tpuT8@v5e |
|---|---|---|
| ALL | TPR 0.734 / FPR 0.164 | TPR 0.672 / FPR 0.016 |
| drop memory LEVELS, keep SHAPE | 0.724 / **0.000** | 0.665 / **0.244** |
| drop memory SHAPE, keep LEVELS | 0.723 / 0.571 | 0.665 / **0.008** |
| drop ALL memory | 0.670 / 0.415 | 0.469 / 0.308 |

On one corpus dropping levels takes FPR to zero; on the other it multiplies FPR
by 15. **Section 15's rule is withdrawn as a rule.** I generalised from a single
corpus and the second one disagrees.

### Why, and this part does generalise

Measured footprints, sorted, with the class of each cell:

    tpuTMS@v6e                              tpuT8@v5e
    honest duty30            2352           honest comm               96
    honest comm              4224           honest duty30            380
    honest sat               4672           INFER  infershard_1b     428
    TRAIN  satfiller_only    5196   <--      TRAIN  satfiller_only  1089
    honest duty50_hash       5424   <--      honest sat             1152
    TRAIN  typeconf_off     13177           honest duty50_hash      1152
    TRAIN  satint_f05/20/50 16495           INFER  inferrep_1b      2668
                                            TRAIN  train_fsdp1b     4978
                                            TRAIN  satint_f05/20/50 7013

On v6e multislice an honest cell (`duty50_hash`, 5424 MB) sits ABOVE a training
cell (`satfiller_only`, 5196 MB): the ranges interleave, so no level threshold
works and levels generate false alarms. On v5e single-slice every honest cell caps
at 1152 MB while training reaches 7017 MB: a threshold near 2000 MB separates the
classes almost perfectly, and levels look like an excellent feature.

**So the measured utility of memory-level features is an artifact of how far the
honest class's footprint range extends.** If the honest workloads in your corpus
never allocate as much as your training workloads, levels appear highly
discriminative with a near-zero false-alarm rate -- and that is a property of the
corpus, not of the world. Any honest job that happens to allocate a training-sized
footprint breaks it, and an adversary can trivially arrange to.

Two consequences:

1. **An FPR quoted for level features is meaningless without the honest class's
   footprint range alongside it.** The v5e 0.008 and the v6e 0.571 are the same
   feature set on the same detector; only the honest coverage differs.
2. **This is precisely what `nonml_sat_hi` was added for.** The v5e single-slice
   corpus has no honest cell at the training footprint -- that cell did not exist
   when v5t8 ran -- which is exactly why levels look good there. The camp4
   multislice runs do have it, and are the corpus that can settle this.

The honest statement for the paper is therefore not "use shape, not levels" but:
**level features on the memory channel are only as trustworthy as the footprint
coverage of the honest class used to validate them**, and a corpus without a
footprint-matched honest control will overstate them.


## 17. The GPU headline claim does NOT suffer the section 16 artifact

Section 16 says a level-feature FPR is only as trustworthy as the honest class's
footprint coverage. The obvious next question is whether the paper's GPU result
-- 100% of F=0.5 runs caught at a 0.6% false-alarm rate -- rests on the same
artifact. It does not.

From the cached GPU window corpus (`~/win_full.parquet`, 335 workload groups):

    training footprint range      0 .. 121,997 MB   (208 groups)
    honest   footprint range    449 ..  20,634 MB   ( 56 groups)
    ranges interleave                          yes
    honest groups above the LOWEST  training group   56 of 56
    honest groups above the MEDIAN  training group   17 of 56

    highest-footprint honest workloads
      s8_nonml_himem        20,634 MB
      s8_nonml_lp_d30m16    20,634 MB
      s16_nonml_himem       19,132 MB
      s1_nonml_himem        18,254 MB
      scientific_nbody      15,909 MB

Seventeen of the 56 honest groups sit above the median training footprint, and
every honest group sits above the lowest. So the GPU false-alarm rate is measured
against honest workloads that genuinely contest the training footprint range,
which is the condition section 16 identifies as necessary for a level-feature FPR
to mean anything. The v5e single-slice corpus fails that condition (every honest
cell below every training cell); the GPU corpus passes it comfortably.

This also puts a number on why the `nonml_himem` and low-power/partial-utilisation
cells were worth adding: they are what carries the honest class from ~15 GB
(`scientific_nbody`, an incidental high-memory workload) up to 20.6 GB, i.e. into
the range where the disguised training cells live. Without them the GPU honest
class would top out near the incidental workloads and the reported FPR would be
open to exactly the objection section 16 raises.

Two smaller notes from the same query:

* **335 workload groups on GPU against 8-11 per TPU corpus.** The family-identity
  leakage of section 10 -- four of eight groups being one script -- is a
  consequence of TPU corpus size, not a defect the GPU result shares.
* `tpu_jax_training` appears at 0 MB, because this cached window file was built
  on 2026-08-25, before the legacy-TPU contamination was quarantined. The corpus
  on disk is quarantined; the cache is stale and should be rebuilt before it is
  used for anything but this kind of range check.


## 18. Three corpora, three answers: the memory ablation is not stable, and that IS the result

Ran the identical level/shape ablation on a third corpus, `tpuT1` on v6e single
slice -- which is the first TPU corpus to contain a footprint-matched honest
control (`nonml_sat_hi`) and whose footprints genuinely interleave (honest
`duty50_hash` 9663 MB and `sat_hi` 9695 MB sit above the F=0 training cell at
9197 MB, just below honest training at 10018 MB).

FPR@0.8 on the honest class, same features, same detector, three corpora:

| feature set | tpuTMS@v6e (8 grp) | tpuT8@v5e (11 grp) | tpuT1@v6e (12 grp) |
|---|---|---|---|
| ALL (171) | 0.164 | 0.016 | **0.001** |
| drop memory LEVELS | **0.000** | 0.244 | 0.175 |
| drop memory SHAPE | 0.571 | **0.008** | 0.200 |
| drop ALL memory | 0.415 | 0.308 | 0.374 |

Every column has a different winner. **Section 16's prediction is falsified**: it
argued that levels look good only where the honest class fails to reach the
training footprint range, and predicted `tpuT1` would therefore behave like
`tpuTMS@v6e`. `tpuT1` has the matched control, its ranges interleave, and
all-features still reaches FPR 0.001 while dropping levels raises it to 0.175.
Interleaving is necessary but not sufficient; with enough classes present the
shape features disambiguate what the levels alone cannot.

### What is actually robust

Only two things survive all three corpora:

1. **Removing all memory features always hurts, substantially** -- FPR 0.415,
   0.308, 0.374 and the worst TPR in two of three. The memory channel carries real
   signal on TPU, which is worth stating given that TPU has no power channel.
2. **Which memory SUBSET helps is not determined by these corpora.** The ablation
   outcome swings with corpus composition -- 8 vs 11 vs 12 groups, presence or
   absence of an inference class, presence or absence of honest training -- by
   more than it swings with the feature set.

### Withdrawn, and why I am not replacing it with a third rule

Section 15 proposed "drop memory levels, keep memory shape" from one corpus.
Section 16 withdrew it and offered a mechanism. Section 18 falsifies the
mechanism's prediction. The pattern in my own reasoning is the informative part:
each rule was fitted to the corpus in front of me and failed on the next one, and
these corpora are 8-12 workload groups with 1-3 reps -- far too small to support a
feature-selection claim, especially with the family-identity leakage of section 10
unaddressed.

So no rule. The defensible statements are (1) and (2) above, plus: **any
feature-subset recommendation for the TPU rung needs a corpus with several
independent workload families per class, which none of these three has.** The GPU
corpus has 335 groups and is the only one here that could support such a claim --
and section 17 shows its honest class does contest the training footprint range,
so its numbers are not subject to this instability.


## 19. Standing cost of one-at-a-time claiming: v5litepod-16 declined four times

`v5litepod-16` has now been granted four times (2026-09-02 at 13:50, 20:56, 21:18
and once earlier) and used zero times. Three of those were released correctly --
the campaign holds one scope at a time, and the reaper drops anything outside the
claimed group -- and one was the reaper's stale-claim bug, since fixed.

This is a design consequence, not a defect, but it has a real cost. `v5litepod-16`
is a **16-chip single slice spanning two hosts**, and the corpus has no multi-host
single-slice data at all: every single-slice corpus so far is one host (v6e-1,
v5litepod-8). It is the only shape on offer that isolates cross-HOST ICI traffic
without DCN, which sits exactly between the topologies already collected.

The decision is Robi's, because it trades money for coverage:

* **Keep one-at-a-time.** Simple, and the reaper's one-claimed-group model is what
  makes unattended running safe. Cost: shapes granted while another scope runs are
  always declined, and v5litepod-16 appears to be offered precisely when something
  else is claimed.
* **Allow two concurrent claims.** Roughly $19/hr for a v5litepod-16 on top of the
  current claim. The obstacle is not the money -- ~$300 of the authorised $3000 is
  spent -- but that the reaper would need to track a SET of claimed groups rather
  than one, and a bug there bills unsupervised. I would not make that change
  unattended.

Recorded rather than acted on. If the answer is "take it", the change is small
(claim list instead of claim scalar in both campaign.sh and camp_reaper.sh) and
worth doing while someone is watching.


## 20. MN-T result on a balanced corpus: the shaping does nothing; the OPTIMIZER evades

camp5 collected the first balanced multislice corpus: 15 groups on 2x
v5litepod-1, five honest non-ML controls, honest training, both inference cells,
the MN-I ladder and the MN-T ladder. 4,932 windows.

### The detector is no longer degenerate

Leave-one-workload-out, full two-stage, tau=0.8, h2d/d2h excluded (they exist
only in camp5 rows, so their presence would identify the campaign and hence
partly the cell):

    window TPR@0.8 = 0.940      window FPR@0.8 (honest) = 0.139
    train_fsdp1b (honest training)   p=0.843  rate=0.902   (was 0.254)
    nonml_comm / duty50_hash / sat   rate 0.000
    nonml_duty30 0.426, nonml_sat_hi 0.201   <- the remaining false alarms

Honest training is now detected at 0.90 where the imbalanced corpus put it at
0.254, and three of five honest controls sit at zero. Section 13's verdict ("the
detector was too degenerate to attack") no longer applies.

### The MN-T ladder needed a family holdout, and it changes everything

Under leave-one-WORKLOAD-out the ladder reads p(train|ML) = 0.991 / 0.994 /
0.993 -- apparently untouched. But the three rungs are ONE SCRIPT, so holding out
`a32` leaves `off` and `a8` in the training fold and the model can score it by
family membership. Holding out **all three rungs together**, so the detector has
never seen the typeconf script:

| rung | p_train | p(train\|ML) | frac > tau |
|---|---|---|---|
| `typeconf_off` | 0.264 | 0.369 | **0.000** |
| `typeconf_a8` | 0.262 | 0.360 | **0.000** |
| `typeconf_a32` | 0.234 | 0.378 | **0.000** |

For comparison, MN-I under the same family holdout is still partly caught:

| cell | p_train | frac > tau |
|---|---|---|
| `advtrain_satint_f50` | 0.926 | 0.785 |
| `advtrain_satint_f20` | 0.805 | 0.549 |
| `advtrain_satint_f05` | 0.765 | 0.230 |
| `advtrain_satfiller_only` | 0.699 | 0.033 |

Two conclusions.

**The 0.99 ladder was family recognition, not robustness.** A 4x drop when the
siblings leave the training fold. My first MN-T reading was measuring family
membership, which is the same trap as section 10 and I walked into it again --
this time the fix was available (hold out the family, not the workload).

**The shaping knobs do nothing.** 0.264 -> 0.262 -> 0.234 across off -> a8 ->
a32 is flat. Raising TensorCore utilisation toward the inference level, cutting
cross-slice collectives 32-fold, and adding a device-to-host trickle moved
nothing. What evades is the BASELINE cell.

### What actually evades: the optimizer

`typeconf_off` is training with no adversarial shaping at all -- plain SGD, no
filler, no interleave, real gradient steps every step -- and it scores 0.264 with
**zero** windows above tau, while `train_fsdp1b` (AdamW, same model, same
topology) scores 0.843. The detector was trained on AdamW training only: MN-I and
`train_fsdp1b` both use it.

The likely mechanism is the optimizer-state signature. AdamW carries two moment
tensors per parameter, which dominates both the footprint and its allocation
trajectory -- the very features section 15 found most discriminative. Plain SGD
carries none. So switching optimizer is a one-line change, needs no adversarial
intent, and appears to cost the detector more than any of the deliberate shaping
in MN-T.

**This is not yet isolated.** `typeconf_off` also differs from `train_fsdp1b` in
script structure (replicated parameters with periodic averaging rather than
sharded FSDP). The clean test is one cell: the same typeconf script with
`--optimizer adamw`, which changes nothing but the optimizer. Registered as
`{scope}_advtrain_typeconf_adamw` for the next run.

Caveats: 15 groups, 1-2 reps per cell, and FPR 0.139 is not a usable operating
point. Treat the magnitudes as indicative and the ORDERING -- SGD training evades
where AdamW training does not -- as the finding worth testing properly.


## 20b. Mechanism evidence: the SGD/AdamW split is carried by the memory startup trajectory

Discriminating `typeconf_off` (SGD) from `train_fsdp1b` (AdamW) on existing data,
948 windows:

    typeconf_off (SGD)    mem_used_mb 12345-12430   first_30s_mem_delta_mb 2693-2902
    train_fsdp1b (AdamW)  mem_used_mb 14196         first_30s_mem_delta_mb 3975

    memory features carry 86.8% of total importance
    top three: first_30s_mem_delta_frac, first_30s_mem_ratio, first_30s_mem_delta_mb

The **startup delta differs by 48%** (3975 vs 2693 MB) while the total footprint
differs by only 15% (14196 vs 12345). That asymmetry is what an optimizer-state
signature should look like: AdamW's two moment tensors per parameter are allocated
during startup, so they show up far more strongly in the allocation TRAJECTORY
than in the steady-state level -- and the trajectory features are precisely the
ones section 15 found most discriminative and the ones section 20 predicts the
SGD cell lacks.

Two caveats, both mine:

* **The 1.000 accuracy is inflated.** I grouped by `run_id`, which is a per-CHIP
  stream, so holding out some streams leaves same-cell streams in the training
  fold. The importances are informative about what differs; the accuracy is not a
  generalisation estimate.
* **The confound is unresolved.** `typeconf` REPLICATES parameters (a full copy
  per chip) while `train_fsdp1b` SHARDS them, so structure contributes to the
  footprint difference as well as the optimizer. A 1b model is ~5.2 GB in fp32:
  replicated that is 5.2 GB/chip, sharded over two chips 2.6 GB/chip, and AdamW
  moments would add ~5.2 GB total. The observed 1.85 GB/chip gap is the same order
  as either effect, which is exactly why it cannot be attributed from these two
  cells.

`{scope}_advtrain_typeconf_adamw` -- identical to `typeconf_off` but AdamW -- is
registered and queued adjacent to its comparator. One cell settles it.


## 21. CORRECTION to section 20: the evasion was generation-specific, not the optimizer

The isolating cell ran (9/9 cells, 3 reps, 2x v6e-1) and it does not confirm
section 20. `typeconf_adamw` is identical to `typeconf_off` except
`--optimizer adamw`.

**The optimizer's effect on the telemetry is real and large.** Same script, same
model, same topology:

| cell | optimizer | mem_used_mb | first_30s_mem_delta_mb |
|---|---|---|---|
| `typeconf_off` | SGD | 12750 | 2601 |
| `typeconf_adamw` | AdamW | 17900 | **7820** |
| `train_fsdp1b` | AdamW + sharded FSDP | 14134 | 3977 |

Switching SGD to AdamW **triples the startup memory delta** (2601 -> 7820 MB) and
adds 5.1 GB of footprint, exactly as the moment-tensor hypothesis predicts.

**But it does not produce evasion.** Holding out all three cells and training on
the other 13 groups of the v6e multislice corpus (7,708 windows):

| cell | optimizer | p_ml | p(train\|ML) | p_train | frac > tau |
|---|---|---|---|---|---|
| `typeconf_off` | SGD | 0.977 | 0.978 | 0.977 | 0.765 |
| `typeconf_adamw` | AdamW | 0.629 | 0.983 | 0.972 | 0.950 |
| `train_fsdp1b` | AdamW | 0.959 | 0.913 | 0.904 | 0.933 |

SGD training is caught at 0.765, not the 0.000 section 20 reported.

### What section 20 actually measured: a generation difference

Running section 20's exact holdout (MN-T family out) on both generations, same
features and code:

| corpus | `typeconf_off` | `_a8` | `_a32` | `_adamw` |
|---|---|---|---|---|
| **v5e** (12 grp, 5,510 win) | 0.259 / **0.000** | 0.264 / **0.000** | 0.235 / **0.000** | -- |
| **v6e** (12 grp, 7,692 win) | 0.799 / 0.494 | 0.798 / 0.495 | 0.853 / 0.799 | 0.875 / 0.875 |

(p_train median / frac above tau.)

The same script, the same holdout, the same feature set: **complete evasion on
v5e, detection on v6e.** So section 20's finding was a property of the v5e
multislice corpus, not of the optimizer. Training-set size differs (5,510 vs
7,692 windows) and is a partial confound, but a 40% data difference does not
plausibly explain 0.000 versus 0.5-0.875.

### What survives

* **The MN-T shaping still does nothing**, on either generation. v5e: 0.259 /
  0.264 / 0.235 across off/a8/a32. v6e: 0.494 / 0.495 / 0.799 above tau -- if
  anything the strongest shaping is the MOST detected. Raising TensorCore
  utilisation, cutting cross-slice collectives 32-fold and adding a d2h trickle
  do not move the type decision.
* **The optimizer has a real but partial effect**, visible where the detector
  works at all: on v6e, SGD 0.494 against AdamW 0.875 above tau, in the direction
  the footprint evidence predicts.
* **An unseen training script is much harder to detect on v5e than on v6e.** That
  is now the open question, and it is a more interesting one than the optimizer:
  it says detectability of a novel workload depends on the hardware generation the
  detector was fitted to.

### The methodological lesson, stated plainly

Section 20 was built on one corpus, one generation, and a holdout I had to fix
once already. It produced a striking claim -- "an evader needs only a different
optimizer" -- that survived exactly as long as it took to run the isolating cell
and the cross-generation check. That is four claims in this file now proposed and
withdrawn (sections 7, 13b, 15/16, 20). Every one failed on data it had not been
fitted to, and every one looked convincing beforehand. Nothing in this file should
be quoted from a single corpus.


## 22. Precise telemetry accounting: 11 mapped, but only 5 live everywhere

Section 11 said the four channels added on 2026-09-02 completed the telemetry, and
singled out the host<->device transfer latencies as "the channels most likely to
separate training from inference". Measured on both topologies, that needs
qualifying: **they are megascale metrics and exist only on multislice.**

    h2d/d2h_transfer_lat_us_*   multislice: live 666/683   single slice: live 0/661

Same family as `dcn_collective_lat_us_*` and `buffer_xfer_lat_us_p90`, and dark on
a single slice for the same reason -- megascale.* accumulates only ACROSS slices.

The honest inventory of the 11 mapped metrics:

| channel | availability |
|---|---|
| `duty` -> `gpu_utilization_pct`, `tpu_duty_cycle_pct` | **always live** |
| `hbm_used` -> `mem_used_mb` | **always live** |
| `hbm_total` -> `mem_total_mb` | **always live** |
| `hlo_exec` -> `hlo_exec_us_p50/p90/p99` | **always live** |
| `hlo_queue` -> `hlo_queue_size` | **always live** |
| `dcn_lat` -> `dcn_collective_lat_us_p50/p90/p99` | multislice only |
| `buf_xfer` -> `buffer_xfer_lat_us_p90` | multislice only |
| `h2d_lat` -> `h2d_transfer_lat_us_p50/p90/p99` | multislice only |
| `d2h_lat` -> `d2h_transfer_lat_us_p50/p90/p99` | multislice only |
| `slice_error` -> `slice_error_detected` | **never carries a value** (0/683 even on multislice) |
| `uptime` -> `tpu_uptime_s` | live, deliberately excluded from features |

So a single-slice corpus has **five** usable accelerator channels, not eleven, and
`tensorcore_util_pct` / `mem_utilization_pct` on top where served. A multislice
corpus has nine. That is the number to quote for "all available TPU telemetry",
and it is a strong statement in its own right: on a platform with no power,
temperature, clock, PCIe or fan channel, a single slice offers five.

**Consequence for the v5litepod-8 re-collect now running.** Its stated
justification -- upgrading 264 streams from 7 channels to 11 -- is wrong: on a
single slice the four added channels contribute nothing (two are megascale, one
never carries a value, one is excluded by design). The claim is still worth its
~6 h, but for a different reason: it adds `nonml_sat_hi`, the MN-T ladder and
`typeconf_adamw` on single-slice v5e, which is exactly what section 21's open
question needs -- a v5e corpus with the same cell set as the v6e multislice one,
so generation can be separated from topology.

## 22. The interconnect channel: multi-node training is ~1,000x louder than multi-node inference

Robi's hypothesis for the multi-node adversarial program was that a training job
big enough to *require* more than one node is fundamentally detectable through
its use of the inter-node fabric, and that the only available disguise is to
resemble multi-node **inference**. The existing corpus already answers the first
half, because the 2026-08-20 H100 session collected `nic_*` host-NIC sidecars
alongside the per-GPU telemetry.

Host NIC counters (tx+rx summed, one interface `enp0s12`, same 2x H100 pair and
same session for every row, `s16_` only so every cell genuinely spans two nodes):

| cell | median Mbps | p95 | idle % | acf1 |
|---|---|---|---|---|
| `s16_train_fsdp30b` | **34,650** | 36,336 | 2.8 | 0.91 |
| `s16_train_fsdp1b` | 16,491 | 17,330 | 2.8 | 0.95 |
| `s16_nonml_comm` | 1,650 | 1,684 | 53.0 | 0.97 |
| `s16_infershard_qwen72b_tp8pp2` | **30.9** | 36.2 | 10.3 | 0.87 |
| `s16_infershard_qwen1_5b_tp4pp4` | 8.5 | 10.7 | 9.2 | 0.95 |
| `s16_inferrep_qwen1_5b` | 0.1 | 0.1 | 100.0 | 0.36 |

**30B FSDP training against 72B TP8/PP2 inference is a 1,123x separation** on a
single scalar. Nothing in the 174-feature per-GPU set comes close; this channel
alone separates the two classes with no model at all.

The mechanism is the parallelism strategy, not the model size -- note that the
*larger* model (72B) is the quiet one:

* FSDP shards parameters and optimizer state across all 16 ranks, so every step
  all-gathers parameters and reduce-scatters gradients ACROSS the node boundary.
  Traffic scales with model size and runs continuously (2.8% idle, acf1 0.91).
* TP8/PP2 inference puts the tensor-parallel group INSIDE a node, where it rides
  NVLink and never touches the NIC, and sends only pipeline activations between
  stages -- small tensors, paced by token generation.
* Replicated inference shares nothing at all: 0.1 Mbps, idle in every sample.

### What this means for the disguise

The hypothesis is right that a multi-node FSDP job cannot hide on this channel:
to pass as sharded inference it must cut inter-node bytes by three orders of
magnitude, which no shaping of the existing job achieves.

But the table also says the reference class matters more than the disguise does,
and cuts against inference as the target:

* **Against an inference reference the disguise is architectural.** Train the
  too-big model with tensor/FSDP parallelism WITHIN each node and pipeline
  parallelism ACROSS nodes. Then the inter-node channel carries activations and
  activation gradients -- the same kind of traffic, at roughly 2x the volume,
  as the PP inference it is imitating. This is not an exotic attack; it is a
  legitimate parallelism choice. Its cost is the pipeline bubble, and QUANTIFYING
  THAT COST IS THE EXPERIMENT: evasion cost becomes the throughput penalty of PP
  against FSDP, which is a very different quantity from MN-I's 1/F.
* **Against a non-ML reference the disguise is nearly free.** `nonml_comm` sits
  only 21x below 30B training and 53x above sharded inference. An attacker whose
  cover story is a bandwidth-heavy HPC job barely has to change anything on this
  channel -- so "is this ML training" is much weaker here than "is this training
  rather than inference".

### Caveats before this goes in the paper

* 30B on 8x H100-80GB (640 GB/node) does NOT strictly require two nodes; AdamW
  on 30B is roughly 360 GB. The cell is multi-node because it was RUN that way.
  The genuinely-forced case needs either a larger model or the A100-40GB pair
  (320 GB/node), where 30B AdamW does not fit on one node. That pair exists and
  is stopped, not deleted, so it can be restarted onto the same disks.
* `s16_nonml_comm` was collected under the over-enqueue loop fixed in 50109cb,
  so its absolute level is not a clean measurement of a 256 MB all-reduce cell.
* These NIC counters are host-wide, so they include the telemetry sync uploads
  (~220 Mbps while syncing). That is immaterial at 34 Gbps and visible only in
  the p95 of the quiet inference rows.

## 23. Multi-node baseline, held out by workload: the corpus is underpowered

`multinode_baseline.py` fits the honest-only two-stage detector per PLATFORM
(pooling H100, A100 and TPU would let the model score a window by recognising
its hardware -- sections 14-18), holds out by workload, and scores at tau=0.80
with the 3-of-5 run rule.

| platform | distinct honest TRAINING workloads | win TPR | win FPR | run TPR | run FPR |
|---|---|---|---|---|---|
| H100 s16 (2 nodes x 8) | 6 | 0.324 | 0.050 | 0.453 | 0.116 |
| H100 s8 (1 node x 8) | 3 | 0.158 | 0.000 | 0.150 | 0.000 |
| A100-40GB s16 (2 nodes) | 2 | 0.000 | 0.016 | 0.000 | 0.000 |
| TPU v6e multislice | 1 | -- | 0.031 | -- | 0.031 |
| TPU v5e multislice | 1 | -- | 0.016 | -- | 0.194 |
| TPU v5e 16chip/4host | 1 | -- | 0.002 | -- | 0.000 |

The TPR column tracks the training-workload count and nothing else, and the
mechanism is not subtle: leave-one-workload-out removes an entire training
workload from the fitting fold, so

* with **1** training workload (every TPU topology -- only `train_fsdp1b`), the
  fold has NO training windows at all, Stage 2 is single-class, the fold is
  skipped, and TPR is undefined. It is not that the detector fails on TPU; the
  measurement cannot be taken.
* with **2** (A100: `train_fsdp1b`, `train_ddp_resnet50`), one held-out
  workload leaves exactly one behind and the detector generalises to the
  held-out one not at all: TPR 0.000.
* with **6** (H100 s16) it reaches 0.453 at the run level -- still far below
  the single-GPU headline, at a 0.116 run-level false-alarm rate.

So the honest multi-node baseline is **0.45 run-TPR at 0.12 run-FPR on H100**,
and unmeasurable elsewhere. This is a statement about the CORPUS, not about
detectability: three of six multi-node topologies cannot support a
held-out-by-workload claim at all.

What it costs to fix is small and specific -- more DISTINCT honest multi-node
training workloads per topology, not more repetitions of the ones we have. The
H100 run in progress adds `train_fsdp70b` (a 7th), and the TPU topologies need
at least a second training cell each before any TPR from them means anything.

Note also that `s16_train_fsdp30b`, which supplied the loud row in section 22,
is one of the six H100 training workloads -- and it did not need two nodes
(32.5B params, ~390 GB against 640 GB per node). `s16_train_fsdp70b` (78.4B,
~940 GB) is the first cell in this corpus that genuinely cannot run on one node.

### 22a. Verifying that the obligatory-inference cell really is obligatory

`s16_infershard_qwen72b_kv32k` was designed so that KV cache, not weights,
forces the second node. vLLM's own startup accounting on the 2x H100 pair
confirms it, and the numbers are worth keeping because the design margin is
only ~1.5x:

    Available KV cache memory: 62.59 GiB   (per GPU, TP8/PP2)
    GPU KV cache size: 3,214,400 tokens    (16 GPUs)
    Maximum concurrency for 32,768 tokens per request: 98.10x
    -> 1001 GiB / 3.21M tokens = 0.327 MB/token   (predicted 0.328)

Capacity is not usage, so the prompts had to be checked too. vLLM reported
`est. speed input: 870,765 tok/s` over ~2.3 s per batch of 64, i.e. ~2.0M input
tokens per iteration = **~31,300 tokens per prompt**: `--prompt-tokens 32000`
did what it claims, and peak concurrent KV is ~2.0M tokens.

Could one node hold it? Under PP2 each node carries half the weights (~9-10
GiB/GPU, per vLLM's "consumed memory (weights + non-torch)"). On a single node
the full 145 GB sits on 8 GPUs = 18.1 GiB/GPU, so KV per GPU falls to about
62.59 - 9.06 = 53.5 GiB, or 428 GiB across the node = **~1.34M tokens**.

    needed 2.0M tokens  vs  ~1.34M available on one node  ->  OBLIGATORY

So this cell cannot be served by one node, by a factor of ~1.5. Note how
narrow that is: dropping --batch-size to 40 would have made the cell fit on one
machine and quietly stopped being an obligatory-inference sample.

## 24. Scoring the obligatory cells: MN-P does not evade, but obligatory INFERENCE false-alarms at 100%

The four multi-node-obligatory cells were scored against the PERSISTED honest
detector (`classifier/models/honest_detector.json`, 40,175 honest windows, 174
features, `advtrain`/`adversarial`/`whitebox`/`symgemm` families excluded from
fitting). All four cells were collected on 2026-09-08, AFTER that detector was
fitted on 2026-09-07, so every row below is out-of-sample.

| cell | n_win | p_ml | p(train\|ML) | p_train | frac>0.80 | run 3-of-5 |
|---|---|---|---|---|---|---|
| `train_fsdp70b` (honest training) | 4152 | 0.958 | 0.866 | 0.853 | 0.742 | **92/96 = 0.958** |
| `infershard_qwen72b_kv32k` (honest inference) | 2288 | 0.990 | 0.867 | 0.864 | 0.639 | **48/48 = 1.000** |
| `nonml_himem_obl` (honest non-ML) | 2096 | 0.739 | 0.011 | 0.008 | 0.000 | 0/48 = 0.000 |
| `advtrain_pipe1b_rank0` (MN-P) | 1968 | 0.980 | 0.821 | 0.811 | 0.579 | 48/48 = 1.000 |
| `advtrain_pipe70b_mb1` (MN-P, obligatory) | 3648 | 0.965 | 0.736 | 0.695 | 0.378 | 42/48 = 0.875 |

Two results, and the second one matters more than the first.

### MN-P defeats the interconnect channel and not the per-GPU one

Section 22 measured MN-P cutting inter-node traffic 44x (373 Mbps against
`train_fsdp1b`'s 16,491). It does not follow that the job is hidden: on per-GPU
telemetry the obligatory-scale disguise still scores p_train 0.695 and is caught
by the run rule 42 times in 48. `pipe1b_rank0` is caught 48 of 48.

So MN-P is **not** a successful evasion as it stands. Choosing pipeline
parallelism moves the job from very loud to nearly quiet on the NIC while
leaving the GPUs doing recognisably training-shaped work. An attacker would have
to compose it with a per-GPU disguise (MN-F/MN-B, or MN-I's duty interleave) --
and MN-I's cost curve then applies on top of the pipeline bubble.

### The detector cannot tell obligatory-scale inference from training

`infershard_qwen72b_kv32k` is HONEST inference -- Qwen2.5-72B, TP8/PP2, serving
64 concurrent ~31k-token requests -- and the detector alerts on **48 of 48 runs**
with median p_train 0.864, indistinguishable from honest training's 0.853.

This is a false-alarm rate of 1.000 on the exact workload class the deployed
claim has to separate, against the 0.6% FPR the single-GPU headline reports. The
mechanism is straightforward once stated: at 31k tokens per request the cell is
**prefill-dominated**, and prefill is a large dense forward pass -- compute-bound,
high utilisation, big activation footprint -- which is what training's forward
pass looks like. The short-prompt inference cells the detector was fitted on are
decode-dominated and memory-bound, a completely different profile.

Consequences:

* Every earlier train-vs-infer number in this project used decode-dominated
  inference negatives. They do not characterise long-context serving, which is
  now the dominant production inference regime.
* This is a stronger objection to the detector than any adversarial result so
  far: it needs no attacker. A lab running ordinary long-context inference trips
  the monitor continuously.
* The fix is a corpus fix, not a threshold fix -- prefill-heavy inference has to
  be in the honest negatives before any FPR claim generalises.

### 24a. CORRECTION to section 24's mechanism: it is a LEVEL overlap, not prefill

Section 24's measurement stands -- obligatory long-context inference alerts on
48/48 runs at p_train 0.864 against the persisted detector. Its stated
*mechanism* does not. Section 24 argued that at ~31k tokens per request the
cell is prefill-dominated, and that prefill is "a large dense compute-bound
forward pass, which is what training looks like", while the fitted inference
negatives are decode-dominated and memory-bound.

Median per-GPU telemetry (H100 node0/gpu0) says otherwise:

| cell | util | mem_util | power W | mem_used MB |
|---|---|---|---|---|
| `train_fsdp1b` (in fit) | 100 | 25 | **436** | 8,574 |
| `train_fsdp7b` (in fit) | 100 | 15 | 317 | 14,004 |
| `train_fsdp30b` (in fit) | 100 | 4 | 188 | 30,516 |
| `train_fsdp70b` (obligatory) | 100 | 3 | 190 | 49,086 |
| `infershard_qwen72b_kv32k` (obligatory infer) | 99 | **41** | **421** | 78,552 |
| `infershard_qwen72b_tp8pp2` (decode infer) | 100 | 57 | 381 | 74,038 |
| `inferrep_qwen1_5b` (in fit) | 100 | 47 | 309 | 74,616 |

On DRAM activity the prefill-heavy cell (41%) sits nearer decode inference
(57%) than training (3-25%), which is the opposite of the claimed mechanism.

What the table does show is a **level overlap**. In the fitted training class,
power and mem_util fall monotonically as the model grows -- 436 W / 25% at 1b,
317 / 15% at 7b, 188 / 4% at 30b -- because at batch 1 a larger model spends
proportionally more of each step in inter-node collectives with the GPU idle.
`kv32k` at 421 W / 41% lands almost exactly on `train_fsdp1b` (436 W / 25%),
the high-power end of the fitted training distribution, while the fitted
inference cells sit lower (309 W). The detector calls it training because its
LEVELS look like the training it was shown, not because prefill resembles a
backward pass.

### The scale effect is the more alarming half

`train_fsdp70b` -- the only genuinely obligatory training cell -- draws **190 W
at 3% DRAM activity**, statistically alongside `train_fsdp30b` and far below
`train_fsdp1b`'s 436 W. Obligatory-scale training is communication-bound, so
per-GPU it looks *quieter* than small-model training.

That inverts the deployment story: the larger and more consequential the
training run, the less it resembles the training the detector was fitted on,
while a long-context serving job moves INTO that region. Both errors push the
same way, and neither is an attack.

### 24b. The prefill ladder refutes BOTH proposed mechanisms

The ladder varies prompt length at fixed model (72B) and topology (TP8/PP2),
with `kv32k_out2k` as the control that keeps the 30-32k context but returns the
wall clock to decode by asking for 2048 output tokens instead of 128. Median
per-GPU telemetry, H100 node0, non-idle streams only:

| cell | prompt/output | util | mem_util | power W |
|---|---|---|---|---|
| `kv1k` | 1k / 128 | 100 | 46 | 368 |
| `kv8k` | 8k / 128 | 98 | 32 | 359 |
| `kv32k` | 32k / 128 (prefill-dominated) | 99 | 41 | 421 |
| `kv32k_out2k` | 30k / 2048 (decode-dominated) | 100 | 41 | **488** |
| `train_fsdp7b_seq2k` | seq 2048 | 100 | 16 | 328 |
| `train_fsdp1b` (in fit) | | 100 | 25 | 436 |
| `train_fsdp70b` (obligatory) | | 100 | 3 | 190 |

**Section 24's prefill hypothesis fails.** If the false alarm came from prefill
resembling a dense training forward pass, the decode-dominated control should
look markedly less training-like. It does not; it is the most extreme cell here.

**Section 24a's level-overlap hypothesis also fails, and it was mine.** I
predicted that returning the wall clock to decode would LOWER power and move
the cell off `train_fsdp1b`'s 436 W perch. `kv32k_out2k` draws 488 W -- the
highest of any cell, training or inference. Power does not track prefill
fraction at all: 368 -> 359 -> 421 -> 488 is not monotonic in it.

What the table does establish is stronger than either mechanism and worse for
the detector: **on power the inference cells (359-488 W) now bracket AND exceed
every training cell (190-436 W).** Power is not a discriminator in this regime,
in either direction. Nor is DRAM activity: inference spans 32-46% and training
3-25%, which separates them here but in the direction that makes the largest,
most consequential training runs look LEAST like training (section 24a).

The surviving candidate is CONCURRENCY rather than prefill: every high-power
inference cell serves 64 sequences at once, and `kv32k_out2k` sustains 64
simultaneous decodes of a 30k-token context, which is a continuous stream of
large batched GEMMs. That is a hypothesis, not a result. It is testable by
varying only `--batch-size` at fixed context and output length, which is a
cheaper cell than anything in this ladder.

Recorded as a caution: two mechanisms were proposed for this false alarm and
both were refuted by the first control that could distinguish them. The
MEASUREMENT (obligatory inference alerting 48/48) has held up across every
check; the story about why has not.

### 24c. The concurrency ladder: a real but insufficient effect, and the signal we are not using

Varying ONLY `--batch-size` at 32k context and 128 output tokens (3 reps x 8
GPUs per rung, node0, no idle streams):

| batch | util | mem_util | power W |
|---|---|---|---|
| 8 | 100 | **60** | 378 |
| 16 | 100 | 54 | 400 |
| 32 | 100 | 50 | 416 |
| 64 (`kv32k`) | 99 | **41** | 421 |

**Concurrency is partially vindicated and still insufficient.** Power does rise
monotonically with batch, which is what 24b predicted. But the whole 8x range
spans 43 W, and every rung -- including batch 8 -- sits inside the training band
(`train_fsdp70b` 190 W to `train_fsdp1b` 436 W). Concurrency cannot explain why
`kv32k` is mistaken for training, because de-concentrating it 8-fold does not
move it out of the region.

**What the ladder does show is the channel we should be using.** DRAM activity
moves further and in the opposite direction, 60% down to 41%, and across every
inference cell measured so far it stays in 32-60%:

    inference   kv32k_b8 60, b16 54, b32 50, b64 41, kv1k 46, kv8k 32,
                out2k 41, tp8pp2 57, inferrep 47      -> 32-60%
    training    fsdp1b 25, fsdp7b 15, fsdp7b_seq2k 16,
                fsdp30b 4, fsdp70b 3                  -> 3-25%

That is a clean separation with no overlap, on a single feature, holding across
model size, context length, output length and concurrency. Power, by contrast,
overlaps badly in both directions (inference 359-488, training 190-436).

So the discriminating signal EXISTS in the telemetry and the fitted detector is
not using it decisively -- `kv32k` alerts on 48/48 runs at p_train 0.864 anyway.
This is no longer a question about which mechanism makes inference look like
training; it is a question about what the detector weights. mem_util is one of
the 174 features and Stage 2 keeps it (it is shape, not level), yet the model
still calls this cell training.

The next step needs no collection: score `kv32k` against the refitted detector
(42,259 honest windows) and read the feature attributions, then ablate. If
mem_util is present and ignored, the fix is corpus composition -- every training
cell in the fit is DRAM-quiet because it is either small-model or
communication-bound, so the model has no reason to learn that boundary.

Three mechanisms have now been proposed for this false alarm (prefill, level
overlap, concurrency) and all three have failed to explain it, while the
measurement has survived every check. That pattern is itself the finding: the
error is in what was fitted, not in what the hardware does.
