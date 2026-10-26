# Camera-ready corrections, 2026-10-22

What changed in the pipeline, what it does to the numbers, and what the
recorded data cannot answer. Companion to
[the corpus manifest](../camera_ready_manifest_20261022/README.md).

## 1. The abstract's 98.2% is the pre-fix number

`8ee55e53` (2026-07-24) recorded the effect of closing the within-run temporal
leak on the same corpus:

| configuration | binary CV |
| :-- | --: |
| `run_level_mode="orig"` (leaky, what the abstract quotes) | 98.2% |
| `run_level_mode="causal"` (deployed since `850c34b8`) | **98.0%** |
| `run_level_mode="none"` (run-level features off) | 97.7% |

The causal figure is the one the deployed classifier produces and the one the
camera-ready should quote.

## 2. Two leaks survived the 2026-08 causal sweep

`850c34b8` flipped `sliding_windows` to `run_level_mode="causal"` and fixed
~25 call sites at once, but only swept `run_level_features`.

**`add_evasion_resistant_features`** never sliced `run_raw`. Its own comment
said so: *"approximation: use whole-run features as window-level proxies"*.
`cumulative_energy_wh` was total run energy -- knowable only once the run is
over -- stamped on every window of it, along with `power_mem_correlation`,
`util_bimodal_score`, `power_efficiency_ratio` and `sustained_compute_frac`.
Four of the five move under truncation; all five now see the run only up to
each window's own end.

These are the Round 3 features (Table 5 row 3), and `eval_catchup` strips the
already-fixed plateau features only to re-admit these -- so the leak survived
on exactly the path that claims to control for it. **Round 3 and the catch-up
comparison have to be regenerated.**

**`attach_raw_features`** broadcasts one short per-run capture across every
window of its run. The snippet carries no usable epoch, so the windows that
precede it cannot be masked; it now refuses window-level use and takes an
explicit `allow_run_constant=True` for run-level analysis. No live caller on
main, so no published number moves.

`tests/test_causality.py` states the property -- delete a window's future,
rebuild, require its vector to be unchanged -- and carries a positive control
asserting `run_level_mode="orig"` still fails, so the suite cannot go blind.
Scope features, BMC features and both windowers pass, including multi-node
window admission: the cadence gate does not in fact change membership under
truncation, contrary to the initial audit.

## 3. Window labels now reach every evaluation path

`e54fc687` labelled windows by observed work "in both caches and the LOFO", and
only there; ~45 other call sites stamped the job objective onto every window.
`work_events` now defaults to `"auto"` on both windowers, which is how
`850c34b8` propagated the causal fix, with the same opt-out (`work_events=None`)
for reproducing archived numbers.

Measured over the rebuttal-era STEP logs, windows that contain no training step
at all:

| cell | training-free windows | | cell | training-free windows |
| :-- | --: | --- | :-- | --: |
| `L_diluted_10` | 70.6% | | `wb_lora_7b_N500_long` | 77.9% |
| `L_diluted_20` | 41.2% | | `advtrain_oracle` | 97.4% |
| `ppo_rlhf_r32` | 32.8% | | `advtrain_interleave_f02` | 33.3% |
| `I_stochastic` | 7.4% | | `advtrain_interleave_f05` | 5.1% |

Moderate dilutions (N <= 100 at an unstarved batch rate) put at least one step
in every 30 s window, so their job label was right all along. The correction
bites where the paper reports its weakest numbers.

## 4. The camera-ready corpus has no recorded per-window truth

STEP logging was added on 2026-07-24, after every tier A run had been
collected. Of tier A's 1,371 jobs, 375 are mixed-work -- 9,185 windows, the
whole adversarial table -- and **none has a STEP log**. Nor is there anything
else to mine: of 1,739 launcher logs, 1,217 carry `STEP` lines, 522 carry no
work evidence at all, and **none** carries the per-30 s progress record without
also carrying STEP lines.

`classifier/schedule_reconstruction.py` recovers what the schedule allows.
Strategy L trains on batch indices 0, N, 2N, ... with batch 0 at the start of
the run, and a batch rate measured from a reference log places those steps. The
rate is carried as an interval and a window is labelled only where its
endpoints agree.

The reference has to match on **both** the cell and the device. The rate is a
property of the run, not of the schedule: `L_diluted_10` and its 8-worker
sibling differ by three orders of magnitude, and the camera-ready
`L_diluted_10` cell ran on six GPU models of which only the A100 has a logged
sibling. Borrowing across either axis relocates every reconstructed step.

| cell | device | windows | reconstructed |
| :-- | :-- | --: | --: |
| `adversarial_L_diluted_10` | A100-SXM4-80GB | 20 | 1 training, 19 inference |
| `adversarial_L_diluted_20` | A100-SXM4-80GB | 20 | 1 training, 19 inference |

**40 of 9,185 windows.** The other 9,145 stay unresolved: their cell has no
logged sibling on their device, or is not deterministic in the batch index at
all (`F_interleave` is phase-interleaved, `I_stochastic` is stochastic). The
module refuses rather than guessing.

> An earlier revision of this document reported 398 recovered windows. That
> figure applied the A100 batch rate to the same cell on the RTX 5080, B200,
> H200, MI300X and RX 9070 XT. The device match is now required and the correct
> figure is 40.

## 5. The strategy-L cells are dataloader-starved, and it is severe

`adversarial_L_diluted_10` on an A100-SXM4-80GB, running the exact camera-ready
command (`--strategy L --duration 300 --dilution 10 --batch-size 128`), reports
in its own summary line:

```
352s: 512 samples, 1 img/s, train=1, infer=3
Strategy L complete. 512 samples (1 img/s), train_batches=1, infer_batches=3, train_frac=25.0%
```

Four batches in 352 seconds. Its 8-worker sibling did **1,939 img/s** -- about
2,000x faster -- on the same device.

`workloads/adversarial_round3.py:219` makes this explicit: `--num-workers`
defaults to 0 "to preserve the behaviour every existing corpus run had", and
runs with workers get a distinct label so they are not pooled with the
`num_workers=0` runs. So the starvation is known and deliberate as a corpus
property. What it means for the paper is less comfortable:

- The cell is not training diluted to 10%. It is a near-idle GPU that takes one
  gradient step, and the program itself reports `train_frac=25.0%`.
- A detector that misses it is missing one gradient step in six minutes of idle,
  which is a much weaker claim than "dilution defeats the classifier".
- Round 3's 52-60% and Table 7's `L_diluted` row are measured on these runs.

Whatever is decided about labels, the camera-ready should not describe these as
10%-diluted training without saying what they actually did.

## 6. What re-runs would buy

`scripts/plan_step_log_reruns.py` enumerates the (cell, device) pairs that each
need one logged reference run. Three reps per pair gives a rate interval rather
than a point estimate, which is what the reconstruction consumes.

| device | pairs | windows in those pairs | pairs reconstruction can use | windows it can label | GPU-hours |
| :-- | --: | --: | --: | --: | --: |
| GeForce RTX 5080 | 46 | 1,886 | 16 | 526 | 11.1 |
| AMD RX 9070 XT | 28 | 1,963 | 6 | 342 | 7.3 |
| B200 | 20 | 1,212 | 6 | 456 | 3.8 |
| H200 | 19 | 1,325 | 6 | 456 | 5.1 |
| A100-SXM4-80GB | 17 | 1,273 | 4 | 416 | 3.9 |
| AMD Instinct MI300X VF | 14 | 843 | 6 | 360 | 3.8 |
| H100 NVL | 14 | 328 | 0 | 0 | 2.0 |
| GeForce RTX 5090 | 14 | 269 | 7 | 134 | 3.7 |
| RTX 4050 Laptop | 9 | 0 | 0 | 0 | 0.1 |
| **total** | **181** | **9,099** | **51** | **2,690** | **40.7** |

> An earlier revision of this section reported the fourth column as "windows
> unlocked" and totalled it at 9,099. That is the window count of every pair
> that lacks a reference, not the count a reference can label.
> `schedule_reconstruction` only handles cells deterministic in the batch
> index, and 130 of the 181 pairs are not: `F_interleave` is phase-interleaved,
> `I_stochastic` is stochastic, and `composite_*`, `H_mimicry`, `E_memory_minimal`,
> `G_clock_throttled`, `J_pid` and `K_online_learning` declare no schedule at
> all. The reconstructable figure is **2,690 windows**, and it reached that
> only after `adversarial_L_ddp_<N>` was added to `DETERMINISTIC_DILUTION`;
> before that it was 1,562.

Forty GPU-hours therefore recovers per-window truth for under a third of the
camera-ready adversarial table, not essentially all of it. The rest needs
either a per-cell schedule model or -- better, and the same price -- a re-run
that emits STEP lines, which is now what every adversarial strategy does. Two of the nine devices are local machines
(RTX 5080 desktop, RTX 4050 laptop: 55 pairs, 11.1 GPU-hours) and cost nothing
but time; the AMD parts need the non-GCP provider.

Three caveats:

1. A re-run measures the rate of *a* run of that cell on that device, not of
   the archived run. For a dataloader-bound cell the rate is set by the host
   CPU, so a re-run on a different host measures a different bottleneck. Run
   the reference on the same machine where one still exists, and record the
   host in the log.
2. It only helps cells deterministic in the batch index. `F_interleave`,
   `I_stochastic`, `composite_*` and `M_composite_memmin_*` need their own
   schedule model, or a re-run that simply emits STEP lines directly -- which
   is strictly better and costs the same. As of `05bb03e4` strategies E, G, H
   and J emit them too, so a re-run of those cells needs no schedule model at
   all; H's mimicry slots are real cuFFT/mining/n-body compute and emit
   `kind=nonml`, which the work-label path reads as `other`.
3. The cheapest version of all of this is to re-run with STEP logging enabled
   and skip the reconstruction entirely. The reconstruction exists because the
   archived runs cannot be re-run; anything you actually re-run should just log.

## 6b. The A100 re-runs: what they found, 2026-10-23

All 17 A100-SXM4-80GB pairs were re-run with STEP logging, three reps each:
51 runs, 177 parquets. Per-cell numbers in `rerun_references_20261023b.csv`
(single-GPU) and `rerun_references_20261023c.csv` (data-parallel). H200 and
B200 were not reachable -- this project has no quota metric for either, and
H100 NVL and RTX 5090 are not GCP SKUs.

**Four of the eleven single-GPU cells never ran.** Their archived runs stop
about 30 s into a 300 s workload with the GPU never once registering
utilisation:

| cell | archived span | archived mean util | samples with any util | windows in corpus |
| :-- | --: | --: | --: | --: |
| `F_interleave_30` | 27 s | 0.0% | 0% | 0 |
| `H_mimicry_cufft` | 32 s | 0.0% | 0% | 1 |
| `H_mimicry_mining` | 31 s | 0.0% | 0% | 1 |
| `J_pid_75` | 30 s | 0.0% | 0% | 1 |

These are aborted runs, not slow ones. The re-runs of the same cells on the
same GPU model hold 97% of samples busy for the full 308 s. Together these
four cells contribute three windows to the camera-ready A100 table, so little
is corrupted -- but the table reports them as evasion results, and they are
not results at all.

**Three more ran to length while doing almost nothing.**

| cell | archived util / busy | re-run util / busy |
| :-- | --: | --: |
| `K_online_learning` | 0.9% / 2% | 64.0% / 97% |
| `L_diluted_2` | 0.9% / 2% | 89.5% / 97% |
| `L_diluted_5` | 0.2% / 1% | 53.7% / 97% |

This is §5's starvation, measured against a healthy sibling rather than
inferred: the archived `L_diluted_2` GPU was idle in 98% of its samples. The
remaining four cells (`E_memory_minimal`, `F_interleave_50`, `I_stochastic_30`,
`G_clock_throttled`) are comparable or close, and all six data-parallel cells'
archived runs are healthy (85-93% mean utilisation).

**The reconstruction yield is zero.** Of the four A100 cells deterministic in
the batch index -- `L_diluted_2`, `L_diluted_5`, `L_ddp_10`, `L_ddp_5` -- none
has a transferable rate: the first two because their archived runs are starved,
the DDP pair because the archived runs held 85-91% utilisation where the
re-runs held 44-51%. A rate measured on a run that does not resemble the
archived one is not that run's rate, so none was transferred. Section 6's
estimate of 416 reconstructable A100 windows is an upper bound on what a
reference *could* unlock, not what these references did.

What the campaign does deliver is 51 runs carrying their own per-window truth,
which is what caveat 3 recommended over reconstruction in the first place.

One incidental confirmation: `L_ddp_stagger_5` measures 49.3 batches/s against
`L_ddp_5`'s 6.4, a ratio of 7.7 on an 8-rank job. That is every rank emitting,
exactly as `_step.py` warns, and why `edf784c6` left stagger out of
`DETERMINISTIC_DILUTION`.

## 6c. Actual adversarial coverage per device, 2026-10-26

`scripts/audit_run_health.py` classifies every tier-A adversarial job by what
its telemetry shows it did. 46 of 294 are unusable, and they are not spread
evenly -- the coverage a device appears to have and the coverage it has differ
most exactly where the paper's weakest numbers come from.

| device | cells usable / claimed | jobs usable / total | windows usable / total |
| :-- | --: | --: | --: |
| `GeForce RTX 5080` | 29 / 29 | 96 / 96 | 1555 / 1555 |\n| `AMD RX 9070 XT` | 19 / 19 | 76 / 76 | 1444 / 1444 |\n| `H200` | 14 / 19 | 17 / 22 | 1225 / 1325 |\n| `A100-SXM4-80GB` | 10 / 19 | 10 / 19 | 1210 / 1313 |\n| `B200` | 8 / 20 | 8 / 21 | 1112 / 1212 |\n| `AMD Instinct MI300X VF` | 6 / 6 | 18 / 18 | 360 / 360 |\n| `H100 NVL` | 9 / 14 | 21 / 31 | 258 / 328 |\n| `GeForce RTX 5090` | 2 / 2 | 2 / 2 | 39 / 39 |\n| `GeForce RTX 4050 Laptop GPU` | 0 / 9 | 0 / 9 | 0 / 0 |

`GeForce RTX 4050 Laptop GPU` has **no usable adversarial runs at all**: all
nine jobs stop within 6-7 s with the GPU never activating. Any statement of
the form "N evasion families across 9 GPU models" is wrong for this device,
and incomplete for B200 (8 of 20 cells), A100-SXM4-80GB (10 of 19), H100 NVL
(9 of 14) and H200 (14 of 19).

The three verdicts, and what each costs:

- **aborted** (19 jobs, 3 windows) -- stopped short with the GPU idle. Costs
  coverage, not correctness: a 6-32 s run yields 0-1 windows, so these cells
  are absent from the evaluation rather than wrong in it.
- **truncated** (12 jobs, 70 windows) -- stopped short with the GPU working.
  The H100 NVL `composite_*` jobs ran 128 s of 300 at ~90% busy and their
  seven windows apiece are sound, so these are **not** excluded; they are
  listed for re-collection at full length.
- **starved** (15 jobs, 300 windows) -- ran to length with the GPU idle in
  99% of samples. These are excluded, because their windows are labelled
  adversarial training while the device did essentially nothing.

Exclusions are recorded in `data/quality_exclusions.json` by job identity, so
`apply_quality_exclusions` drops them on every path that already honours it,
and the raw files are preserved. The full list, including the truncated jobs
that are kept, is `recollection_list.csv`.

Prevention is in `scripts/run_health.py`, called by both runners: a run that
aborts, truncates or leaves the GPU idle now says so when it finishes and
writes a `.health.json` beside its telemetry, and `--require-gpu-activity`
turns that into a non-zero exit for campaigns that should never produce one.
Replayed over the corpus it agrees with the post-hoc audit on 293 of 294 jobs
and would have caught all 46 on the day they were collected.

## 7. Other corrections in this branch

- `run_level_detection.per_run_sequences` called a run training by its **first**
  window's label, so a diluted job that opened on inference became a benign run
  and left the missed-run denominator. Now any window.
- `rebuttal_analysis/multinode_baseline` re-derived `cls` -- its fitting target
  and scoring truth -- from `workload_label`, putting the job objective back on
  every window after the windower had labelled them.
- `eval_mi300x_experiments` fitted per-device normalisation over the whole
  frame, so the MI300X statistics came from the evaluation rows themselves and
  from windows later in each run. It now holds out 30% of the target device's
  runs for unsupervised calibration and scores the rest. **Appendix transfer
  numbers move.**

## Reproducing

```bash
cd /srv/gpu-monitoring/cloudshell-workspace
export SPAR_LFS_OBJECTS=/srv/gpu-monitoring/lfs-objects
PYTHONPATH=. python3 scripts/camera_ready_manifest.py
PYTHONPATH=. python3 scripts/window_label_impact.py
PYTHONPATH=. python3 -m unittest tests.test_causality tests.test_schedule_reconstruction
```

`SPAR_LFS_OBJECTS` matters: the working tree holds LFS pointers, and 11,842 of
12,198 accelerator parquets are unreadable without a populated object store.
That store was assembled on `spar-analysis-0` from the October follow-up run's
archive; 36 payloads (two October A100 pilot jobs, tier C) are still missing
and need `git lfs pull` with GitHub credentials.
