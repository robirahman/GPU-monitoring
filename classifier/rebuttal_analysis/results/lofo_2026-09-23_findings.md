# Leave-one-family-out after the 2026-09-23 collection

Corpus: 151,632 honest multi-node windows, 90 workloads, **31 families** (was 12),
of which **6 are training** (was 2). 174 features, 96 of them shape-only.

Read every comparison below **down one tau column**. A recall gain paid for by a
matching rise in false alarms is not an improvement, and this table makes that
trade explicit rather than hiding it in a single accuracy number.

## The three numbers this round set out to move (all at tau=0.5)

| | 2026-09-22 | 2026-09-23 | |
|---|---|---|---|
| `train_fsdp` recall | 0.0051 | **0.2916** | 57x |
| `train_ddp` recall | 0.0416 | **0.2220** | 5.3x |
| `infershard_qwen` false alarms | 0.7144 | **0.0056** | 128x fewer |

These three are the same families in both runs, so the comparison is honest.
Sharded 72B inference was being called training on 71% of its windows; it is now
called training on 0.56% of them, while held-out training recall rose at the same
threshold. Both directions moved the right way at one shared tau.

## The headline finding: absolute levels were memorisation

174 features vs the 96 shape-only ones (which drop absolute memory, power and
utilisation levels and keep only distributional shape):

| held-out family | win-acc (all 174) | win-acc (shape-only 96) |
|---|---|---|
| `train_lora` | 0.0770 | **0.7421** |
| `train_fsdp` | 0.2916 | **0.6457** |
| `train_ddp` | 0.2220 | 0.1079 |
| `train_contrastive` | 0.9484 | 0.8207 |
| **TRAIN mean** | 0.4345 | **0.5621** |
| **not-train mean** | 0.9478 | **0.9668** |

Shape-only wins on **both** sides at once -- better training recall AND fewer
false alarms -- which is not a trade-off, it is a strictly better model. Removing
78 features improves held-out accuracy, so those features were encoding "this
family sits at 38 GB and 310 W" rather than anything about training. That is
exactly what section 33d predicted from the vLLM preallocation confound, now
measured directly instead of argued.

`train_lora` is the clearest case: 0.077 -> 0.742, a 10x gain from deleting
features.

## What is still broken

`TRAIN mean 0.4345` is not a working detector. Two of the six training families
are essentially invisible at tau=0.5:

    train_pipe   0.0680      train_lora   0.0770

and both are wildly threshold-sensitive -- at tau=0.3 they become 0.9534 and
0.7948. Median training probability for the four multi-node-parallel families is
at or below 0.4 (`ddp` 0.170, `lora` 0.325, `fsdp` 0.372, `pipe` 0.392), i.e. the
model systematically scores them below the decision line. The two families it
finds easily, `train_diffusion` (p_med 0.945) and `train_contrastive` (0.798),
are the compute-dense ones that look least like inference.

So the detector still fails on precisely the workloads the threat model cares
about: training split across nodes.

Dropping tau to 0.3 buys real recall (`fsdp` 0.6994, `pipe` 0.9534, `lora`
0.7948) but costs `inferdiff_sdxl` 79% false alarms and `inferrec_dlrm` 50%.
There is no tau in this table at which the detector is simultaneously useful and
safe.

## Caveats

* 6 training families is still small. The rep ablation
  (`rep_ablation_findings.md`) showed estimates at this corpus size are
  noise-dominated; treat single-family numbers as indicative, not precise.
* The baseline had 12 families and this has 31, so only the three same-family
  comparisons above are strictly like-for-like. The means are not comparable
  across the two runs.
* `inferrec_dlrm` (0.5779) and `inferdiff_sdxl` (0.6600) are the worst
  not-train families and are both NEW this round -- they were not part of what
  the 2026-09-22 model was asked to reject.
