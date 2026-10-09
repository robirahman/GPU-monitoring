# TPU v5/v6 experiment plan — scope-ladder on Trillium + v5p + v5e

Load-and-run brief for the TPU collection campaign. Read this, then `git pull` on
the cloud side and go. Design rationale lives in `.claude/plans/spicy-soaring-hamming.md`
and the GPU-side result docs in `classifier/rebuttal_analysis/results/`.

---

## 0. What we are actually testing (read first — it changed)

The **honest** train-vs-inference ladder **saturates**: AUC 1.000 at every scope on
GPU, even after dropping all comm features. So the scope hypothesis has no headroom
on honest workloads. The question that survives is **adversarial**:

> Against a competent disguised trainer (**MN-I**: a resident model that trains only
> fraction **F** of steps and fills the rest with genuine non-ML work), do TPU's
> native communication channels — **`hlo_exec_us_*` (always-on) and
> `dcn_collective_lat_us_*` (cross-slice)** — recover the high-precision detection
> that *power* provided on GPU (which TPU does not expose)?

Phase-0 (simulation on the GPU corpus, `results/phase0_tpu_feature_parity.md`):
dropping power, F=0.5 disguise is still caught at 100% of runs but at **5× worse FPR
(3.0% vs 0.6%)**, and F≤0.2 at **0%**. The disguise hides per-chip duty cycle but
**cannot hide the periodic cross-host collectives of its real training steps** — and
that is exactly what HLO-exec / DCN-latency measure, and what NVIDIA has no
equivalent for. **A yes is the headline: a content-agnostic signal NVIDIA can't see
defeats the strongest evasion.** This is why the multislice/DCN rentals are core.

## 1. Scope → slice → host/interconnect (⚠ corrected — governs everything)

The repo comments said "v6e-16 = 4 hosts × 4 chips." **Wrong: v6e is 8 chips/host**
(official docs: v6e-8 is a *full single host*). The cells are scope-agnostic
(`jax.devices()` + `--multihost auto`) so they run correctly regardless; only the
interpretation and the worker count change.

| scope | v6e slice | chips | hosts / workers | comm regime | DCN (`megascale.*`) |
|---|---|---|---|---|---|
| tpuT1 | v6e-1 | 1 | 1 | none | dark |
| tpuT4 | v6e-4 | 4 | 1 | intra-host ICI | dark |
| **tpuT8** | v6e-8 | 8 | **1 (full host)** | intra-host ICI | dark |
| **tpuT16** | v6e-16 | 16 | **2 hosts / 2 workers** | **cross-host ICI** (HLO-exec sees it) | still dark (1 slice) |
| **tpuTMS** | **2× v6e-8** | 16 | **2 slices / 2 workers** | **cross-slice DCN** | **FIRES** ✅ |

Two boundaries: single→multi-**host** at **T8→T16** (HLO-exec engages);
single→multi-**slice** at **T16→TMS** (DCN engages). v5p (4 chips/host) and v5e
(8 chips/host) map analogously (§4).

**⚠ MUST VERIFY ON FIRST PROVISION:** confirm the worker count of each slice
(`gcloud compute tpus tpu-vm describe <name> --format='value(networkEndpoints)'` or
count SSH workers). `tpu_queue.sh` now derives `NUM_WORKERS` from the live slice, but
if you override it, a wrong value makes the `.done` accounting mark every cell FAILED
(it expects ≥NUM_WORKERS per-worker markers). This was an untested assumption in the
old code — the earlier v6e-16 Spot slices were preempted before a cell ever ran.

**Live TPU feature set** (libtpu 0.0.17): duty→`gpu_utilization_pct`, HBM
usage→`mem_used_mb`, `hlo_exec_us_p50/p90/p99`, `hlo_queue_size`,
`tensorcore_util_pct`, (+`dcn_collective_lat_us_*` **only at tpuTMS**). **Dark:**
`hbm_bw_util_pct`/`mem_utilization_pct` (`tpu.runtime.hbm.utilization.percent` =
NOT_FOUND), and all power/temp/clock/PCIe (=-1). Do not build analysis on HBM-bw.

## 2. Instances to rent (on-demand — Spot was preempted 3×; you can't afford that)

Rent per generation, priority **v6e → v5e → v5p**. On-demand quota is ample (v6e
512-chip limit; no request needed). Multislice via `NODE_COUNT=2`.

| Gen | single-host | multi-host (ICI) | **multislice (DCN)** | $/chip-hr | notes |
|---|---|---|---|---|---|
| **v6e** (primary; validated) | v6e-8 | v6e-16 (2 hosts) | **2× v6e-8** | ~$2.70 | + v6e-1 for T1 baseline & cheap iteration (~60–110 s to provision) |
| **v5e** (cheapest) | v5e-8 (`ct5lp-hightpu-8t`) | v5e-16 | **2× v5e-8** | ~$1.20 | 16 GB HBM (1b fits, 7b tight) |
| **v5p** (training silicon) | v5p-8 (4 chips) | v5p-16 (8 chips) | **2× v5p-8** | ~$4.20 | 95 GB HBM; **verify quota per zone — v5p was QUOTA/permission-blocked before** |

- **Zones:** use `orchestration/mz_common.sh`'s table + `probe_driver.sh` claim-first
  (us-east5-a/b, europe-west4-a, us-central1-a). Don't hard-retry one zone
  (us-east1-d churned v6e Spot). v5p usually us-east5-a.
- **Unattended is mandatory:** launch `orchestration/mz_watchdog.sh`
  (`ACTIVE_MAX_SECS=3600`) from `spar-analysis-0`, NOT Cloud Shell — it guarantees
  teardown on DONE / deadline / active-timeout. `--valid-until-duration` bounds the
  *request*, not the node's life.
- **Cost (decisive subset, ~2–3 h wall/gen):** v6e ≈ $180–260, v5e ≈ $80–120,
  v5p ≈ $250–350; total ≈ **$500–750**. Cloud-Shell time ≈ launch + monitor only.

## 3. Workloads to run (decisive first)

At each of **single-host / multi-host-ICI / multislice-DCN** per gen:

**Tier 1 — the adversarial contrast (run first, every gen):**
`*_advtrain_satint_f50` (F=0.5 — the "100% @ 0.6% FPR" GPU claim), `_f20`, `_f05`
(0% on GPU), `*_advtrain_satfiller_only` (F=0 — evades on GPU); honest partners
`*_nonml_sat` (footprint-matched apparatus test), `*_nonml_comm` (**cross-chip
collectives — directly probes DCN**), `*_nonml_duty30` (the half-idle HPC job that
false-alarmed *harder* than F=0.05 on GPU), `*_nonml_duty50_hash` (worst GPU FP).

**Tier 2 — honest reference (confirms saturation + arm/cross-gen anchor):**
`*_train_fsdp1b`, `*_inferrep_1b`, `*_infershard_1b`, `*_other_fft`, `*_idle`.

**Reps ≥3** per (scope × cell) to clear the eval's `<8 jobs → LOW-N` gate.
`DUR≥600 s` (XLA compile is 5–10 min — else you measure compile time). Run order:
`ADVERSARIAL=1` (Tier 1, decisive-cell-first) then `STRETCH=1` (Tier 2). All Tier-1
cells are registered at tpuT1/4/8/16 and (new) **tpuTMS**.

## 4. Per-generation launch (Cloud-Shell-minimal)

For each gen set `ACCEL`/`ZONE`/`RUNTIME` and run the same flow:

```bash
# 0. metric re-probe GATE (new gens can drift libtpu metric names)
ACCEL=v6e-1 bash scripts/tpu_collection/provision_tpu.sh create   # 1 chip, ~90s
#   ... setup_tpu_all.sh, then:
bash scripts/tpu_collection/tpu_queue.sh --probe-only              # fails loudly if names drift

# 1. cheap validation incl. DCN: 2 single-chip slices as a multislice
NODE_COUNT=2 ACCEL=v6e-1 SCOPE=tpuTMS bash scripts/tpu_collection/provision_tpu.sh create
SMOKE=1 SCOPE=tpuTMS NODE_COUNT=2 bash scripts/tpu_collection/tpu_queue.sh
#   ASSERT in the smoke parquet: tpu_metrics_live=1 AND dcn_collective_lat_us_p50 != -1
#   (if DCN is still -1, the mesh isn't crossing slices — fix before the big rentals)

# 2. decisive run, unattended (launch watchdog from spar-analysis-0 first):
#    single-host, multi-host, multislice — each is one provision + one queue:
SCOPE=tpuT8  ACCEL=v6e-8               ADVERSARIAL=1 REPS=3 bash scripts/tpu_collection/tpu_queue.sh
SCOPE=tpuT16 ACCEL=v6e-16             ADVERSARIAL=1 REPS=3 bash scripts/tpu_collection/tpu_queue.sh
SCOPE=tpuTMS ACCEL=v6e-8 NODE_COUNT=2 ADVERSARIAL=1 REPS=3 bash scripts/tpu_collection/tpu_queue.sh
# then STRETCH=1 ... for the Tier-2 honest reference on each.
```

Then v5e (`ACCEL=v5litepod-8`/`v5litepod-16`, `RUNTIME=v2-alpha-tpuv5litepod`), then
v5p (`ACCEL=v5p-8`/`v5p-16`, `RUNTIME=v2-alpha-tpuv5p`) — same flow, re-probe first.

## 5. The one integration to validate on hardware — multislice DCN (tpuTMS)

`tpuTMS` is the only rung that populates the DCN channel, and it is the **one path
not yet run end-to-end**. The pieces (all committed):
- `provision_tpu.sh NODE_COUNT=2` creates the 2-slice group (`--node-count`/`--node-prefix`).
- `registry_tpu.py` now registers the tpuTMS Tier-1 cells (§3).
- `tpu_queue.sh` SCOPE=tpuTMS fans the cell to **all workers of both slice nodes** so
  `jax.distributed.initialize()` forms one mesh across the slices, and the TPU
  runtime's megascale env makes the collectives cross DCN.

**What to verify on the 2× v6e-1 smoke (step 4.1), in order:**
1. **Mesh spans both slices** — the workload logs `jax.device_count()==2` (all
   slices), not 1. If 1, the megascale coordinator env wasn't set: for a
   `--node-count` queued resource it should be automatic; else export
   `MEGASCALE_COORDINATOR_ADDRESS` on the workers and confirm `--multihost auto`
   took the multislice branch.
2. **DCN channel is live** — `dcn_collective_lat_us_p50 != -1` and
   `tpu_metrics_live=1` in the parquet. If still -1, collectives aren't crossing DCN
   (mesh built per-slice) — fix before renting the big multislice.
3. **`.done` markers don't collide across slices** — `TPU_WORKER_ID` is per-slice, so
   slice-0-worker-0 and slice-1-worker-0 could both write `<cell>.w0.done` and the
   queue's `got >= NUM_WORKERS` check would undercount and mark a healthy cell FAILED.
   If the smoke shows `got < NUM_WORKERS` on a cell that clearly ran, namespace the
   marker by slice in `run_workload_tpu.py` (add a slice/process-index component to
   `--worker-id`, e.g. `jax.process_index()` which is globally unique across slices).

Do NOT rent the big multislice until the cheap 2× v6e-1 smoke passes all three.

## 6. Analysis (no cloud; reuse the harness)

TPU parquets carry GPU-schema column names, so `classifier/multinode_features.py` +
`classifier/rebuttal_analysis/scope_ladder_eval.py` ingest them (`get_feature_cols`
auto-picks `hlo_exec_us_*`/`dcn_collective_lat_us_*`). Headline numbers:
- **Disguised-train (advtrain F=0.5/0.2/0.05) run-level TPR at ≤1% held-out FPR**, per
  scope × channel-ablation (`drop_dcn`, `drop_hlo`, `drop_all_comm`, `duty+hbm only`).
  Does adding HLO-exec (T16) then DCN (TMS) recover power's high-precision point, and
  does F=0.2/0.05 — invisible on GPU — become visible once the comm channel is present?
- **Monotonicity:** job-cluster bootstrap ΔAUC(single→multi-host) and (multi-host→
  multislice) on the *adversarial* discrimination.
- **Cross-gen** (v6e/v5p/v5e) and **cross-accelerator** (GPU-trained detector on TPU).
- Carry the GPU cautions: group CV **by workload not run** (88.9% workload-ID leakage);
  worst confusable = memory-bound half-duty HPC job; report a **detection cost**
  (evasion ≈ ½-duty ≈ 2× slowdown), not an absolute floor.

## 7. Follow-ups by initial finding

- **① DCN/HLO recovers high-precision detection** (advtrain F=0.5 caught at ≤1% FPR at
  TMS where power-free single-host can't; ideally F=0.2 becomes visible): the win →
  quantify DCN's marginal contribution, push F=0.05, replicate v5p/v5e, scale the
  multislice, write the paper's TPU section.
- **② Partial** (multi-host > single-host but FPR above GPU-with-power): characterize
  the gap + confusable workloads, test HLO+DCN+duty+tensorcore combined, report
  "harder without power, but a real multi-host advantage + a quantified cost."
- **③ No advantage / DCN doesn't separate:** first suspect instrumentation (the Aug-31
  "HLO channels dark" class): verify `tpu_metrics_live=1`, DCN≠-1, cadence not capped
  at 5 s, and that the disguise actually induces cross-slice collectives. If genuinely
  null → the governance finding: content-agnostic TPU telemetry can't recover power's
  discrimination; TPUs are structurally harder to monitor this way.
- **④ v5p/v5e metric surprise** (probe shows different names / DCN differs): update the
  collector substring map + re-pin `requirements_tpu.txt` for that gen; a portability
  finding.
- **⑤ Capacity/preemption:** `probe_driver.sh` claim-first across zones; fall back to a
  smaller multislice (2× v6e-1 still lights DCN) or another gen; watchdog stops billing.

## 8. Teardown & security (every session)

- Watchdog auto-teardown is primary; still `provision_tpu.sh delete` + `verify-zero`
  (nodes + queued-resources across all zones = 0; **multislice deletes all N nodes**).
  TPU VMs bill from CREATE, not SSH.
- **Revoke the leaked GitHub PAT** (`HANDOFF_CLOUDSHELL.md`: pasted into a chat + shell,
  cached `/etc/spar/git_pat`, maybe in worker git remotes). New PAT via
  `read -rsp … > ~/.spar_pat` (never a visible arg); `git -C /opt/spar/repo remote -v`
  to check each host.
