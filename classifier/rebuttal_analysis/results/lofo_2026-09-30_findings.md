# 2026-09-30 LOFO: the 20260929 round was neutral

## What was measured

The 20260929a round collected 13 clean cells with NIC sidecars (18 queued, 5
failed). This is the leave-one-family-out re-measurement after ingesting it,
plus two ablations against the SAME window cache to separate "the round changed
the corpus" from "the round changed the answer".

    mn_binary_lofo_20260929.csv   full corpus, 31 families, 10,119 honest windows
    lofo_noALL.csv                identical minus the 27 runs of 20260929
    lofo_noLP.csv                 identical minus the 14 lp_*/lowpower runs

## Result

                          TRAIN mean      not-train mean
    without the round       0.5795            0.9136
    with the round          0.5687            0.9419
    without lp_* only       0.5685            0.9318

-0.011 on held-out training recall, +0.028 on the not-train side. Neutral.

The low-power cells are not responsible for anything: dropping all six moves
TRAIN mean by 0.0002.

## Two claims this retires

**"train_ddp regressed from 0.42."** It did not. `lofo_mn_ALL.csv` (0.4194) was
built on a different, smaller corpus -- 28 families against this cache's 31 --
so it is not a control. On THIS cache train_ddp is 0.0223 without the round and
0.0255 with it. It was already broken.

**"NIC coverage is worth +0.198 TRAIN mean."** That number came from restricting
both train and test to the NIC-covered subset. This round bought sidecar
coverage on twelve not-train cells and added 670 not-train windows and ZERO
training windows to the honest set -- the advtrain_* cells and the stall control
are excluded from it by design. Coverage went to the side that was not the
bottleneck.

## Coverage is no longer the constraint

    family              NIC coverage    held-out recall
    train_ddp              80.6%            0.0255
    train_lora             98.2%            0.2060
    train_fsdp             50.5%            0.2754
    train_pipe             98.3%            0.9056
    train_contrastive     100.0%            1.0000
    train_diffusion       100.0%            1.0000

DDP has sidecars on four windows in five and is detected in one window in forty.
Collecting more NIC data cannot fix DDP or LoRA.

## What the corpus actually needs

train_fsdp has 8 workloads and 1,445 windows and still scores 0.275 -- the
negative volume correlation (r = -0.30 windows, -0.25 jobs) showing up again.
The gap is training-program DIVERSITY: parallelism machinery the held-out fold
has never seen (ZeRO stages, tensor/sequence parallelism, gradient-accumulation
regimes), not more variants of what is already sampled.
