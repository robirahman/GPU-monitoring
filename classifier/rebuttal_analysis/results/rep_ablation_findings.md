# How many reps per cell? Measured, 2026-09-23

Most cells in this project have been collected at `REPS=3`. That is a third of
all pair-time, so it is worth knowing what it buys.

## Method

Leave-one-family-out on the honest multi-node corpus (12 families, 95,410
windows), capping the TRAINING half of every fold at N reps per workload while
always scoring the held-out family in full — so the comparison isolates what
extra reps give the model, rather than also shrinking the test set. Reps are
ordered by first window, which is collection order.

    PYTHONPATH=. python3 classifier/rebuttal_analysis/mn_binary_lofo.py --max-reps N

## Result: reps buy nothing measurable

| reps | windows available to fit | TRAIN mean | train_fsdp | train_ddp | not-train mean |
|---|---|---|---|---|---|
| 1 | 30,960 | 0.0244 | 0.0018 | 0.0469 | 0.9063 |
| 2 | 54,500 | **0.0432** | 0.0329 | 0.0535 | 0.9092 |
| 3 (all) | 95,410 | 0.0234 | 0.0051 | 0.0416 | 0.8988 |

Tripling the training data moves held-out accuracy by about one point, and not
monotonically: 2 reps scores HIGHEST on the training side and 3 lowest. The
spread across rep counts (0.023–0.043) is larger than any trend in it.

## Why this is the expected answer, not a surprise

The detector is held out by workload FAMILY, so the effective sample size is the
number of independent programs, not the number of windows. A rep is the same
code, same config, same hardware and same duration as its siblings — it adds
rows that are near-duplicates of something the model has already seen. It cannot
add what the holdout is actually testing for.

This is the same lesson `heldout_calibration.md` recorded from the other
direction: going from 14 to 32 honest non-ML workloads did not move held-out FPR,
because all 32 were parameter variants of one script.

## Decision

**`REPS=2`.** The second rep is insurance against a transient failure —
preemption, a vLLM load failure, a rendezvous timeout, all of which this project
has hit — not statistical power. Against `REPS=3` that returns a third of the
pair-time, which is better spent on another program.

Keep `REPS=3` only where run-to-run variance is itself the measurement.

## Caveat

At 12 families the estimate is noise-dominated: every number in the table above
sits in a regime where held-out training recall is 0.002–0.05, i.e. the detector
does not work across programs at all. This ablation says reps do not fix that. It
does not say reps would be worthless in a corpus with enough families for the
estimate to be stable — re-run it after the 2026-09-23 collection round, when the
training side goes from 2 families to 6.
