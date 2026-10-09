# 2026-10-05 LOFO: the distributed-training round, measured

## What was measured

Leave-one-family-out over the multi-node corpus after ingesting sessions
20261001 and 20261003 (and recovering 20260904, which predated the idle
screen), plus the drop-one-round ablation that separates "the rounds changed
the corpus" from "the rounds changed the answer".

    lofo_full_20261005.json       14,241 windows, 353 jobs, 113 workloads, 48 families
    lofo_ablation_20261005.json   13,011 windows, 326 jobs,  88 workloads, 39 families

The ablation drops the 1,230 windows belonging to the new rounds' cells.

## The six new families are detectable

Held-out recall, families that did not exist in the corpus before this round:

    train_swarm            1.000   (78 win, 1 fold)
    train_apexrl           1.000   (40 win, 1 fold)
    train_decoupdiloco     1.000   (88 win, 2 folds)
    train_ppodist          1.000   (40 win, 2 folds)
    train_streamdiloco     1.000  (167 win, 2 folds)
    train_impala           0.995  (125 win, 3 folds)

and from the preceding round:

    train_moe              1.000  (119 win, 3 folds)
    train_tp               1.000   (80 win, 1 fold)
    train_localsgd         0.908  (493 win, 4 folds)

CAVEAT, and it is not small: these are 40-167 windows over 1-3 folds. A
single-fold 1.000 is one draw, not a measurement. train_localsgd, with the
most windows and folds, is also the lowest of them at 0.908 -- which is the
direction you would expect if the others are optimistic.

## What the new data did to the families that were already there

Matched comparison, the 8 training families tested in BOTH runs:

    family                  ablation    full     delta
    train_fsdp                 0.661   0.950   +0.289
    advtrain_flatarena         0.862   1.000   +0.138
    train_ddp                  0.858   0.928   +0.070
    stalled_train              0.599   0.664   +0.064
    advtrain_pipe              0.807   0.845   +0.038
    train_accumsync            0.000   0.000    0.000
    advtrain_satfiller         0.000   0.000    0.000
    advtrain_satint            0.943   0.764   -0.179

    mean over the same 8:      0.5915  0.6439  +0.0524

train_fsdp and train_ddp were the two worst held-out families on 2026-09-30
(recall 0.5687 overall, train_ddp 0.0255). Adding nine distributed-training
families to the TRAINING pool moves train_fsdp by +0.289. That is the result
this programme was built to test: the gap was training-program diversity, and
supplying it helps on families that were already present.

advtrain_satint goes the other way, -0.179. One regression out of eight, on an
adversarial family, which honest_mask excludes from the headline anyway.

## The honest caveat on the headline

Seed-level test accuracy rises 0.6946 -> 0.7802, but macro-F1 falls slightly,
0.6552 -> 0.6419. Accuracy is up because the added families are easy and
numerous; the per-class balance is marginally worse. Quoting only the accuracy
would overstate this.

## Still unmeasured

Whether a detector trained on unshaped telemetry transfers to the SHAPED
regime. The _bw100 cells are inside train_localsgd / train_streamdiloco /
train_decoupdiloco / train_impala by design, so LOFO reports one number per
family and cannot split them. That needs per-window predictions split by cell,
which this protocol does not emit.

## A process note worth keeping

The first run of this analysis completed cleanly, published results, and was
wrong. build_mn_window_cache.py skips its raw load when raw_mn.parquet exists,
/var/tmp survives reboots, and the file was dated 2026-09-25 -- so the cache,
and every number in it, was September's corpus: 34 families, none of the six
new ones. It was caught only because the full and ablation outputs were
byte-identical in size, which led to the ablation having dropped 0 windows.

The runner now discards a stale raw cache by default, dies if the ablation
drops nothing, and dies if the six new families are absent from the output.
