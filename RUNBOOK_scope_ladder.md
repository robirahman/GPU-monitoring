# Scope-ladder collection runbook (multi-node GPU + multi-host TPU)

Tests the hypothesis **train-vs-inference separability rises with communication
scope (S1 < S8 < S16)** by collecting matched training / replicated-inference /
sharded-inference telemetry at 1, 8, and 16 GPUs (and the TPU v6e-4/8/16
analogue). Design + rationale: `.claude/plans/spicy-soaring-hamming.md`
(local) — this file is the operational checklist.

Everything is pre-written so a GCP session is pure execution. Land, clone this
branch, bootstrap, run. **Cheapest decisive result: the S16 train-vs-sharded
contrast is the first two cells (~$45, ~35 min).**

---

## 0. One-time, from your workstation

```bash
git fetch && git checkout multinode-tpu-experiments
source scripts/gcp/env.sh          # PROJECT/ZONE/NODE names/BUCKET; override by exporting first
bash scripts/gcp/provision_gpu_pair.sh    # creates 2x a3-highgpu-8g Spot + bucket + firewall
```
`env.sh` defaults: project `focus-surfer-494820-g0`, zone `us-central1-a`, nodes
`spar-gpu-0/1`. Override any value by exporting it before sourcing.

---

## 1. GPU track (2x a3-highgpu-8g = 16x H100)

**Bootstrap each node (over SSH — a startup-script races cloud-init):**
```bash
gcloud compute scp scripts/gcp/bootstrap_node.sh spar-gpu-0:~ --zone=$ZONE
gcloud compute ssh spar-gpu-0 --zone=$ZONE --command="NODE_INDEX=0 bash ~/bootstrap_node.sh"
# same for spar-gpu-1 with NODE_INDEX=1  (installs driver+fabric-manager+DCGM,
# two venvs, chrony<50ms gate; asserts NVLink counters live via dcgmi)
gcloud compute ssh spar-gpu-0 --zone=$ZONE --command="bash /opt/spar/repo/scripts/gcp/verify_pair.sh"
```

**Smoke, then collect. On BOTH nodes** (node0 first), from `$REPO_DIR`:
```bash
export MASTER_ADDR=$(gcloud compute instances describe spar-gpu-0 --zone=$ZONE \
    --format='value(networkInterfaces[0].networkIP)')
export NODE_RANK=0            # 0 on spar-gpu-0, 1 on spar-gpu-1
export RUN_STAMP=20260820a    # any string, identical on both nodes
export PY=$VENV_MAIN/bin/python PY_VLLM=$VENV_VLLM/bin/python

# shakedown (60s cells -> logs/smoke_data/, never data/):
SMOKE=1 bash scripts/rebuttal_collection/multinode_queue.sh
# real run (priority-ordered: S16 decisive cells first, then S8, then S1):
nohup bash scripts/rebuttal_collection/multinode_queue.sh &
```
The c10d rendezvous at `$MASTER_ADDR:29500` is the per-cell barrier; both nodes
derive the same per-cell `run_id` (sha1 of `RUN_STAMP_cell_rep`) with no
coordination channel. Telemetry streams to `data/` and a host-NIC sidecar
records the inter-server channel. `sync_data.sh --loop 120 &` mirrors to GCS.

**On preemption of one node:** the survivor records `.failed`, rendezvous
times out in 120s, and downshifts to the single-node S8/S1 phases (fully
usable). Re-`start` the node under the same name and re-run the queue — `.done`
markers skip completed cells.

**Teardown (always):** `bash scripts/gcp/teardown.sh` — best-effort final sync,
delete instances/TPUs/queued-resources, verify-zero. Idle pair burns ~$78/h.

---

## 2. TPU track (v6e-16 = 4 hosts, the multi-node analogue)

```bash
bash scripts/tpu_collection/provision_tpu.sh create      # Spot v6e-16, us-east1-d
gcloud compute tpus tpu-vm scp scripts/tpu_collection/setup_tpu_host.sh spar-tpu-0:~ --zone=$TPU_ZONE --worker=all
gcloud compute tpus tpu-vm ssh spar-tpu-0 --zone=$TPU_ZONE --worker=all --command="bash ~/setup_tpu_host.sh"
SMOKE=1 bash scripts/tpu_collection/tpu_queue.sh         # then without SMOKE for the real run
bash scripts/tpu_collection/provision_tpu.sh delete
```
TPU exposes no power/clock/PCIe (those map to `-1`), but does expose the
inter-server signal NVIDIA lacks — `hlo_exec_timing` (always-on) and DCN
collective latency. Stage-1 (ML-vs-other) will be weaker; Stage-2 (shape-only)
transfers.

---

## 3. Analysis (from your workstation, after pulling data)

```bash
bash scripts/gcp/pull_results.sh && bash scripts/gcp/ingest_session.sh   # GCS -> data/gcp/<SESSION>/
python -m classifier.check_job_grouping | tee classifier/results/job_grouping_check.txt   # honest baseline
python -m classifier.rebuttal_analysis.scope_ladder_eval                 # the hypothesis test
# outputs: classifier/results/scope_ladder_{summary,monotonicity}.csv + report/figures/
# validate the harness with no cloud data:  ... scope_ladder_eval --synthetic
```
Headline = `scope_ladder_monotonicity.csv`: PASS iff ΔAUC(S8−S1) and
ΔAUC(S16−S8) are both > 0 at p<0.05, reported per feature arm
(`pergpu` = paper-faithful, `crossrank` = joins streams across servers) and per
inference negative. Every table prints `n_jobs` beside `n_windows`; the harness
withholds a verdict on any cell with < 8 independent jobs.

---

## 4. Verify against your gcloud version before spending

The provisioning scripts encode facts from the 2026-08-18 session; a few flags
drift across gcloud releases (details in each script header):
1. **DWS Flex Start** flags (`--provisioning-model=FLEX_START --max-run-duration`) — left commented; verify with `gcloud compute instances create --help | grep -A2 provisioning-model`.
2. **DLVM image family** `common-cu123-ubuntu-2204-py310` — confirm via `gcloud compute images list --project deeplearning-platform-release --filter="family~cu12"`.
3. **Kernel-pinned nvidia module** package name (`linux-modules-nvidia-<branch>-gcp`) — the most fragile bit; if the driver phase reboots without recovering, set `NVIDIA_BRANCH=<n>` from `apt-cache search linux-modules-nvidia`.
4. **TPU service account** `tpu-service-account@$PROJECT...` must pre-exist with the roles in `docs/cloud_tpu.md`.
5. **`libtpu.sdk.tpumonitoring`** metric names/shapes vary by version — the collector logs one `repr(data())` per metric at startup so the name→column map can be corrected on-TPU at first run.

---

## Cost / time (Spot)

| Deliverable | Wall | Cost |
|---|---|---|
| Decisive S16 train-vs-sharded (A1+A2, 1 rep) | ~35 min | ~$45 |
| Lean 1-rep GPU pass (all A+B+C) | ~2.4 h | ~$190 |
| Full GPU program (12 jobs/cell + slack) | 2 sittings | ~$0.6–2.0k |
| TPU v6e ladder | ~3–5 h | ~$150–300 |

Well inside the >$5–10k envelope.
