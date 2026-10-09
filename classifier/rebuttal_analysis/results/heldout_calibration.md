# Held-out calibration: the FPR problem is not a data-quantity problem

Method: whole honest non-ML workloads are held out of the fit (grouped 5-fold) and
scored only as test data. The deployment question is whether an HPC job the
detector has NEVER SEEN trips it; calibrating on in-sample honest workloads is
what produced tau=0.1.

Population: substantive busy honest non-ML. Excludes `s*_nonml_(comm|hpc)` (broken
measurements), `idle*` (degenerate -- gated on utilisation in deployment, and no
disguise imitates an idle GPU), and workloads under 30 windows (noise).
TPR from a detector fit on all valid honest data; disguises are never in that fit.

## Adding 18 honest non-ML workloads changed almost nothing

| tau | FPR_win 14 wl | FPR_win 32 wl | FPR_run 32 wl |
|---|---|---|---|
| 0.10 | 55.5% | 57.4% | 56.4% |
| 0.30 | 26.5% | 29.3% | 32.1% |
| 0.50 | 3.1% | 4.1% | 5.5% |
| 0.60 | 2.3% | 3.0% | 3.0% |
| 0.80 | 1.0% | 1.2% | 2.4% |

The honest non-ML class went from 14 substantive busy workloads to 32 -- covering
duty 0.15-0.7, allocation 2-16 GB, and three kernel mixes -- and the held-out FPR
did not improve. **More of the same kind of honest workload does not teach a
general ML-vs-non-ML boundary.** This is a model/feature problem, not a corpus-size
problem.

## Only one usable operating point, and it catches only F=0.5

| tau | FPR_win | FPR_run | F=0 | F=0.05 | F=0.2 | F=0.5 |
|---|---|---|---|---|---|---|
| 0.10 | 57.4% | 56.4% | 0.0% | 90.0% | 100% | 100% |
| 0.30 | 29.3% | 32.1% | 0.0% | 8.8% | 93.8% | 100% |
| **0.50** | **4.1%** | **5.5%** | 0.0% | 0.0% | 1.2% | **100%** |
| 0.60 | 3.0% | 3.0% | 0.0% | 0.0% | 0.0% | 97.5% |
| 0.80 | 1.2% | 2.4% | 0.0% | 0.0% | 0.0% | 5.0% |

At tau 0.50-0.60: FPR 3-5%, F=0.5 caught at 97.5-100%, F=0.2 at <=1.2%, F=0.05 at
0%. F=0.2 has no viable threshold -- 93.8% TPR costs 29.3% FPR.

**Evasion cost is ~2x, not the ~20x claimed earlier.** The adversary needs only to
drop below roughly half duty cycle. The earlier ~20x figure came from the frozen
detector with honest workloads inside its own training set.

## The worst false positives are memory-bound duty-cycled jobs

Held-out mean p_train, selected:

    s8_nonml_lp_d50hash   0.736   (honest, hash-only kernel, duty 0.5)
    s16_nonml_lp_d50hash  0.671   (honest)
    mining_ethash_proxy   0.429   (honest, REAL, 1372 windows)
    blender_bmw           0.367   (honest, REAL)
    s16_nonml_lowpower    0.354   (honest)
    s16_nonml_lp_d50fft   0.212   (honest, fft-only, SAME duty as d50hash)
    cufft_benchmark       0.051   (honest, REAL)
    nbody_sim             0.022   (honest, REAL)

Two things follow. First, kernel choice at identical duty moves p_train 3.5x
(d50hash 0.671 vs d50fft 0.212), so memory-bound periodic work is what reads as
training -- and that is a large class of real HPC jobs. Second, the synthetic
low-power cells sit alongside REAL mining_ethash_proxy (0.429) and blender_bmw
(0.367), so the false positive is a property of the region, not of the synthetic
design.

## Implication

Before more hardware, the detector needs a general boundary: workload-held-out CV
as the model-selection criterion (it is currently fit without it), and features
that do not encode workload identity. The defensible claim as it stands is narrow:
half-duty disguised training is detectable at ~3-5% FPR, and nothing below F=0.5 is.

---

## AMENDMENT (corpus quarantine): conclusions unchanged

43 windows of legacy TPU telemetry with all nine raw signals zero-filled were
removed from the corpus (`gcs_c_quarantine/legacy_tpu_schema/README.md`). Re-measured
on 54,028 clean windows, the run-level operating point is **identical**: tau=0.80
gives 0.6% held-out FPR at 100% F=0.5 run-TPR, tau=0.85 collapses both to zero, and
F=0.2/F=0.05 stay at 0% throughout. The ~2x evasion cost stands.
