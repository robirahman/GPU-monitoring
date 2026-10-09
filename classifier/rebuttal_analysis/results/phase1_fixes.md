# Phase 1: four TPU-blocking fixes, and two larger problems they exposed

## The fixes

1. **`util_per_sm_pct` was `+inf` on any collector that writes `-1` clocks.**
   `threeway_improved.py`: `util.mean() / (sm_clk.mean() + 1.0)` with
   `sm_clock_mhz = -1` gives a denominator of exactly `0.0`. Every hot path uses
   `.fillna(0)`, which does not touch infinities, so `model_store --fit`,
   `twostage --train`, `evasion_cost_eval` and `scope_ladder_eval` would all have
   raised `Input X contains infinity` on the first TPU row -- and the cross-vendor
   evaluators, which call `np.nan_to_num` without `posinf=`, would have silently
   turned it into `3.4e38`. Fixed by clamping the clock mean at `>= 0`, so a `-1`
   sentinel behaves exactly like an absent column. **Verified: 0 columns contain inf.**
   (346,855 non-finite cells remain and are all NaN, pre-existing, from skew/kurtosis
   on constant channels; `fillna(0)` handles those.)
2. **The dedup key would have collapsed a 16-chip TPU slice to one chip** -- and was
   already destroying GPU data. See below.
3. **Stage-2 level mask generalised** from an explicit 73-name set to a suffix rule,
   so TPU-native levels (`hlo_exec_us_p90_mean`, `hbm_bw_util_pct_p95`) cannot enter
   Stage 2 disguised as shape features. Guarded against over-reach: `_per_`/`_minus_`
   names are dimensionless ratios and stay -- this keeps `mem_util_per_gpu_util_p95`,
   one of the Round-6 DRAM features the disguised-training evidence rests on.
   Net effect on the current vocabulary: 73 -> 77 dropped, the four additions being
   genuine levels (`dcgm_tensor_active_{mean,max}`, `encoder/decoder_util_pct_mean`).
4. **`_vendor_of` gained a `tpu` branch** in all three evaluators; previously
   "Google TPU v6e" returned `unknown` and rows were dropped with no count.
5. **Unreadable-file skips now report at WARNING** with a count, instead of vanishing
   at DEBUG.

## Problem A: the dedup key was silently discarding 8% of the GPU corpus

    raw loaded                      2,048,285 rows
    old key (ts, run_id)            1,573,224      <- every result on this branch
    new key (ts, run_id, gpu_uuid)  1,698,463      <- +125,239 recovered (+8%)

I had reasoned NVIDIA runs were immune because each GPU gets its own collector process
with an independent `datetime.now()`. **That was wrong.** Enough per-GPU streams share
an identical float epoch within a run that the old key was collapsing them. This was
never a TPU-only latent bug.

Window count is **unchanged at 54,028**, because `sliding_windows` groups by `run_id`,
so recovered rows add samples to existing windows rather than creating new ones. The
operating point is likewise unchanged (see attribution below), so no published number
moves -- but the windows are now computed over 8% more underlying samples.

## Problem B: 3,605 parquet files under `data/` are unreadable

They fail with "Parquet magic bytes not found in footer" -- they are **LFS pointer
text, not data**, because git-lfs is not installed in the Cloud Shell this work runs
in. They have been skipped at DEBUG level with no visible line.

Every result on this branch was therefore computed on whatever subset of the corpus
had real bytes locally. Internal comparisons remain valid (all arms saw the same
subset) but the corpus is materially smaller than the file count implies. Installing
git-lfs and fetching would change the denominator substantially and is worth doing
before any cross-hardware claim.

## Attribution: which fix moved the operating point?

Both arms below use the same post-fix cache, so the only difference is the mask.

| | tau at 0.6% FPR | F=0.5 there | F=0.2 at 4.2% FPR |
|---|---|---|---|
| OLD mask (73 dropped) | 0.80 | 100% | 0% |
| NEW mask (77 dropped) | **0.75** | 100% | **50%** |

**The dedup fix is a no-op on the metric**: with the old mask it reproduces the
pre-fix baseline exactly (tau=0.80 -> 0.6% FPR, 100% F=0.5). The mask change shifts
the probability *scale* -- dropping four features lowers p_train slightly, moving the
0.6%-FPR threshold from 0.80 to 0.75 -- while giving identical F=0.5 performance and
**adding F=0.2 sensitivity the old mask lacked entirely** (50% of runs at the same
4.2% FPR where the old mask gets 0%). Strictly a Pareto improvement.

**Restated headline: tau=0.75, run-level k=3-of-5, held out by workload -> 0.6%
false-alarm rate at 100% F=0.5 run-TPR.** The claim is unchanged; only the threshold
constant moves.
