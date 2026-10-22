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
whole adversarial table -- and **none has a STEP log**.

`classifier/schedule_reconstruction.py` recovers what the schedule allows:
strategy L trains on batch indices 0, N, 2N, ... with batch 0 at the start of
the run, and a batch rate measured from reference logs of the *same cell*
places those steps. The rate is carried as an interval and a window is labelled
only where its endpoints agree.

| cell | windows | reconstructed as inference | share |
| :-- | --: | --: | --: |
| `adversarial_L_diluted_10` | 253 | 240 | 94.9% |
| `adversarial_L_diluted_20` | 145 | 138 | 95.2% |

Every one of those 378 windows is currently scored as a missed detection. They
are a substantial part of what Round 3's 52-60% measures.

**The remaining 8,787 windows (95.7%) stay unresolved.** `adversarial_composite_*`,
`adversarial_N_grad_accum_*`, `adversarial_M_composite_memmin_*`,
`whitebox_diluted_*`, `whitebox_lora_*`, `dilution_sweep_N*` and
`adversarial_L_diluted_5` have neither a STEP log nor a reference run whose
rate would place their steps. Their per-window truth is not recoverable from
what was recorded, and the module refuses rather than guessing.

Three ways forward, in order of what they cost:

1. Report the adversarial tables as **run-level** detection, which is what they
   have always measured, and say so in the caption.
2. Extend the reconstruction. `adversarial_composite_<pct>` and
   `adversarial_N_grad_accum_<k>` are also deterministic; they need a reference
   run with STEP logging to fix their rate.
3. Re-collect those families with logging, as the rebuttal already did for
   `L_diluted`, `F_interleave` and `I_stochastic`.

## 5. A workload-validity problem these runs surfaced

The reference runs for `adversarial_L_diluted_10` are dataloader-starved.
`L_diluted_10_rep1` completed **4 batches in 300 s** (~70 s each) while its
8-worker sibling `L_diluted_10_mw8_rep1` completed **4,647**. At four batches a
1-in-10 schedule takes exactly one gradient step, so the run is a nearly idle
GPU with one step in it rather than training diluted to 10%.

That is independent of labelling and it affects what the cell demonstrates.
Whatever is decided about the labels, the camera-ready should not describe
these runs as 10%-diluted training without saying what they actually did.

## 6. Other corrections in this branch

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
