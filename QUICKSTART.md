# QUICKSTART — scope-ladder collection (READ FIRST)

> 💸 **The 2× H100 pair burns ~$78/h; v6e-16 ~$15/h.** Kill-loss is ≤5 min of data
> (per-cell GCS sync + 60 s checkpoints). **When in doubt, tear down (§3).** Set a timer.
> Full detail & rationale: `RUNBOOK_scope_ladder.md`. Design: `.claude/plans/spicy-soaring-hamming.md`.

## 0. Workstation, once
```bash
git checkout multinode-tpu-experiments && source scripts/gcp/env.sh
# defaults: project focus-surfer-494820-g0, zone us-central1-a, nodes spar-gpu-0/1, bucket gs://<proj>-telemetry
# pin the session if it may cross UTC midnight:  export SESSION=20260820_scope
```
⚠️ First time only: verify 5 gcloud-flag items vs your CLI version — **RUNBOOK §4**
(Flex Start syntax, DLVM image family, kernel nvidia-module pkg, TPU service account, libtpu metrics).

## 1. GPU track (16× H100) — decisive result is the FIRST 2 cells (~35 min, ~$45)
```bash
# provision + bucket + firewall
bash scripts/gcp/provision_gpu_pair.sh
# bootstrap BOTH nodes over SSH (driver+fabric-manager+DCGM+venvs+chrony; ~20 min each)
for i in 0 1; do n=spar-gpu-$i
  gcloud compute scp scripts/gcp/bootstrap_node.sh $n:~ --zone=$ZONE
  gcloud compute ssh $n --zone=$ZONE --command="NODE_INDEX=$i bash ~/bootstrap_node.sh"; done
# gate: must PASS (16 GPUs, 2-node NCCL, NVLink counters, clock <50ms)
gcloud compute ssh spar-gpu-0 --zone=$ZONE --command="bash $REPO_DIR/scripts/gcp/verify_pair.sh"
```
Then **on BOTH nodes**, from `$REPO_DIR` (node0 first):
```bash
export MASTER_ADDR=$(gcloud compute instances describe spar-gpu-0 --zone=$ZONE \
    --format='value(networkInterfaces[0].networkIP)')
export NODE_RANK=0                 # 0 on spar-gpu-0, 1 on spar-gpu-1  ← CHANGE PER NODE
export RUN_STAMP=20260820a         # any string, IDENTICAL on both nodes
export PY=$VENV_MAIN/bin/python PY_VLLM=$VENV_VLLM/bin/python
SMOKE=1 bash scripts/rebuttal_collection/multinode_queue.sh          # shakedown (~10 min)
nohup bash scripts/rebuttal_collection/multinode_queue.sh &          # real run (priority-ordered)
nohup bash scripts/gcp/sync_data.sh --loop 120 &                     # continuous GCS egress
```
Rep-major & priority-first: safe to **stop after any rep**. Preempted node → survivor auto-downshifts
to single-node S8/S1; `gcloud compute instances start <name>` (same name) + re-run queue (`.done` skips finished cells).

## 2. TPU track (v6e-16 = 4 hosts, the multi-node analogue)
```bash
bash scripts/tpu_collection/provision_tpu.sh create                 # Spot, us-east1-d
gcloud compute tpus tpu-vm scp scripts/tpu_collection/setup_tpu_host.sh $TPU_NAME:~ --zone=$TPU_ZONE --worker=all
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$TPU_ZONE --worker=all --command="bash ~/setup_tpu_host.sh"
SMOKE=1 bash scripts/tpu_collection/tpu_queue.sh                     # then again WITHOUT SMOKE
```

## 3. ⛔ BEFORE YOU LEAVE — teardown checklist (the money step)
- [ ] **Results are in GCS?** queue syncs per-cell; force if unsure: on each node `bash scripts/gcp/sync_data.sh --final`
- [ ] **`bash scripts/gcp/teardown.sh`** — deletes instances + TPUs + queued-resources, then prints verify-zero
- [ ] **Read the verify-zero output**: instances / **DISKS (STOP-mode disks keep billing!)** / TPUs / queued-resources / reservations must all be **0**
- [ ] **`bash scripts/tpu_collection/provision_tpu.sh delete`** if you didn't use teardown for the TPU
- [ ] **Pull results home**: `bash scripts/gcp/pull_results.sh && bash scripts/gcp/ingest_session.sh` → `data/gcp/$SESSION/`
- [ ] Double-check the GCP console **Billing → Reports** next day; confirm spend stopped.

## 4. Analyze (workstation, after ingest)
```bash
python -m classifier.check_job_grouping | tee classifier/results/job_grouping_check.txt   # honest baseline
python -m classifier.rebuttal_analysis.scope_ladder_eval                                   # the S1<S8<S16 test
# → classifier/results/scope_ladder_{summary,monotonicity}.csv   (PASS = both ΔAUC>0 at p<0.05)
```
