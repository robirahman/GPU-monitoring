# Phase 0 gate: is the F=0.5 result expressible without power telemetry?

TPU exposes duty cycle, HBM bandwidth utilisation and HBM capacity. It does NOT
expose power, temperature, clocks, PCIe, encoder/decoder or DCGM --
`scripts/collect_tpu_telemetry.py:74-78` hard-wires nine columns to `-1`. This
re-runs the winning operating point on only the features that survive that loss, on
the clean corpus (54,028 windows), held out by workload, both arms in one process so
the CV folds are byte-identical.

Feature subset derived programmatically and asserted: **174 -> 72, 102 dropped**.
Retained deliberately: `gpu_util_fft_peak_power`, `mem_used_fft_peak_power`,
`mem_util_fft_peak_power` -- these contain "power" but are FFT MAGNITUDES, not
power-draw derived. A naive `grep power` overcounts the dead set by 4 and would
wrongly discard 3 survivors. Full lists in `/home/robi/phase0_features.json`.

## Result (run-level, k=3 of m=5, held out by workload)

| tau | FPR_run A(174) | FPR_run B(72) | F=0.2 A | F=0.2 B | F=0.5 A | F=0.5 B |
|---|---|---|---|---|---|---|
| 0.50 | 4.2% | 10.9% | 0% | **50%** | 100% | 100% |
| 0.60 | 3.0% | 7.3% | 0% | **50%** | 100% | 100% |
| 0.70 | 1.8% | 4.8% | 0% | 0% | 100% | 100% |
| **0.80** | **0.6%** | **3.0%** | 0% | 0% | **100%** | **100%** |
| 0.85 | 0.0% | 3.0% | 0% | 0% | 0% | 0% |
| 0.90 | 0.0% | 1.8% | 0% | 0% | 0% | 0% |

## Verdict: the pre-registered gate FAILS as written; the mechanism survives

The gate was "F=0.5 run-TPR at <=1% held-out FPR". On 72 features **no threshold
achieves <=1% FPR with non-zero TPR** -- at tau=0.90 the FPR is still 1.8% and the
TPR is already 0. So the criterion I set in advance is not met, and I am not
relaxing it retroactively.

What *does* survive: **F=0.5 is still detected at 100% of runs at tau=0.80, at 3.0%
FPR instead of 0.6%.** The qualitative mechanism -- F=0.5 detectable, F<=0.05 never
-- reproduces. The cost is a **5x worse false-alarm rate**, which is itself the
finding: power is carrying most of the discrimination at the high-precision end.

Corroborating detail: the worst honest false positive degrades sharply,
`s8_nonml_lp_d50hash` 0.574 -> 0.837. That is exactly the workload power was
distinguishing -- a memory-bound duty-cycled kernel looks like training once you can
only see duty cycle and HBM.

Unexpected upside: the 72-feature arm gains **F=0.2 sensitivity that the full set
lacks entirely** (50% of runs at tau=0.50-0.60, against 0% for all 174). This is the
same trade seen in the earlier shape-only arm: dropping level features costs
high-precision and buys mid-fraction sensitivity.

## Why this is a FLOOR, not an estimate

Arm B removes 102 channels and adds nothing. Real TPU telemetry adds ~11 native
channels the GPU has no equivalent for (`collect_tpu_telemetry.py:81-88`):
`hlo_exec_us_p50/p90/p99`, `hlo_queue_size`, `dcn_collective_lat_us_p50/p90/p99`,
`buffer_xfer_lat_us_p90`, `tpu_duty_cycle_pct`, `tensorcore_util_pct`,
`hbm_bw_util_pct`. The DCN collective-latency channel in particular has no NVIDIA
counterpart and speaks directly to the cross-host-collectives hypothesis the scope
ladder was built for. So a TPU detector has 72 + ~11 features, and Arm B is the
pessimistic bound on what it can do.

Caveat that cannot be resolved without collecting: TPU's `mem_utilization_pct` is
HBM *bandwidth* utilisation, while NVML's is the fraction of the sample period spent
accessing memory. Arm B uses the NVML semantics. Those 39 memory/util marginal-stat
features therefore survive numerically while changing meaning, so Arm B is a
statement about *how much information is lost*, not a simulation of TPU data.
