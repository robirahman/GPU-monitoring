# Wiring in NIC + cross-rank features (2026-09-25)

Job-level windows from `build_mn_window_cache.py`: 63,070 windows, 107 `xr_` and
38 `xragg_` features, 18 of them NIC-derived. Shape-masked (199 of 315), which is
what production uses at Stage 2. Canonical `family()` from
`multinode_family_holdout` -- 6 training families, not the 16 a local regex gave.

## train_pipe: 0.056 -> 0.972

The single clearest result. Pipeline parallelism was the worst-detected training
family in the corpus and is now near-perfect, which is mechanistically what you
would expect: GPipe moves activations rank->rank in a ring, so the signature is
cross-rank LAG STRUCTURE, not level. `xr_lag_median_*` and `xr_dip_coalign_*`
are exactly that, and no per-stream featuriser can see it -- the information
lives strictly between ranks.

`train_ddp` improves too, 0.108 -> 0.419, and to 0.972 once restricted to
windows where the NIC was actually measured.

## The coverage artifact was real and worth +0.198

Same pipeline, same features; the only difference is dropping windows where no
NIC sidecar existed (`meta_nic_present`).

| held-out TRAIN family | all s16 | NIC-covered | delta |
|---|---|---|---|
| train_ddp | 0.4194 | 0.9718 | **+0.5524** |
| train_lora | 0.3240 | 0.6967 | **+0.3727** |
| train_fsdp | 0.1847 | 0.4135 | **+0.2288** |
| train_pipe | 0.9722 | 1.0000 | +0.0278 |
| train_contrastive | 0.9937 | 1.0000 | +0.0063 |
| train_diffusion | 1.0000 | 1.0000 | 0.0000 |
| **mean** | 0.6490 | **0.8470** | **+0.1980** |

25 of 76 s16 cells predate the sidecar. Their NIC features read 0.0, which is
indistinguishable from "measured, and quiet" -- and it falls hardest on
`train_fsdp` (65% of windows) and `train_pipe` (67%), whose own sidecars measure
a median 16,077 Mbps. Feeding that to a model teaches it when a cell was
collected. The +0.198 is the price that artifact was charging.

## It is NOT a free win

| held-out not-train | all s16 | NIC-covered | delta |
|---|---|---|---|
| **infershard_qwen** | 0.9951 | **0.3904** | **-0.6047** |
| nonml_fftshard | 0.9936 | 0.9776 | -0.0160 |
| inferrec_dlrm | 0.9968 | 0.9873 | -0.0095 |
| not-train mean | 0.9965 | 0.9523 | -0.0422 |

Sharded 72B inference -- the largest inference family, 1,401 covered windows --
goes to a 61% false-alarm rate, its median training probability rising 0.255 ->
0.536. On the NIC-covered subset it starts looking like training. So the
interconnect signal buys training recall and spends it on the single family this
detector most needs to keep straight.

That is consistent with the raw throughput: FSDP is ~440x sharded inference, but
the new HPC and recommender cells (fftshard 6,676 Mbps, dlrm 5,907) are LOUDER
than four of the six training families, and LoRA is silent at 0.003 Mbps. Volume
does not separate the classes; only structure might.

## Caveats

* Job-level windows (63,070) are not per-stream windows (205,867), so the
  absolute numbers are not comparable to `lofo_2026-09-23_findings.md`. The
  ALL-vs-NIC-covered delta IS clean: same pipeline, one filter.
* The NIC-covered arm has 20 families, not 28. Eight were dropped for lack of
  coverage, so the task differs as well as the data.
* `train_fsdp` (0.185 all / 0.414 covered) is worse here than the 0.646 the
  per-stream shape model got. Granularity confound, not a regression -- but it
  means job-level windowing is not strictly better and both should be kept.
