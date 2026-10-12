# Seeded collection pilot

Implementation is ready for a live smoke, but **no new VM or collection job was
started in this execution**. `gcloud auth list` confirms an active account;
`gcloud compute instances list` fails here because the restricted agent session
cannot resolve `compute.googleapis.com`. No further login is currently indicated.
The operator was asked to save a live instance listing to the ignored local file
`gcloud-instances.local.json` from regular Cloud Shell. VM identity, availability
and current workloads have therefore not yet been established.

## Changes prepared

- All six implementations behind the 25 planned configurations accept `--seed`
  or `SPAR_WORKLOAD_SEED`. The default remains 1234. Existing rank, replica, stage
  and tensor-shard offsets are preserved. Python/NumPy seeds and the base seed
  are recorded; asynchronous scheduling is not claimed to be deterministic.
- The regenerated 125-launch plan has a separate base workload seed per launch.
  Its schedule seed does not stand in for a model/data seed.
- Local SGD, TP, MoE and distributed RL emit per-rank final progress, including
  completed optimizer updates or actor rollout steps. Tokens are local counts;
  TP rank counts must not be summed as unique data. Task learning is unmeasured.
- Distributed and telemetry-only GPU collectors write rolling snapshots every
  60 seconds. The bounded pilot's NIC recorder stays alive through model startup
  and teardown instead of expiring at `duration + 120`.
- Queue opt-ins refuse to reap other GPU processes, abort on failed cloud
  barriers and check GPU availability before changing shaping. Filtered bandwidth
  cells no longer install shaping at all.
- `launch_followup_pilot.py` deploys a source-only working-tree bundle with hashes
  to a fresh directory on both nodes. It neither archives stale local HEAD nor
  copies the corpus or Git credentials. The old repo/venvs remain available.
- The driver requires two already-running idle H100 hosts, eight GPUs each. It
  starts one cell under detached systemd services with a runtime cap and shutdown
  after completion/failure. It does not provision or implicitly restart machines.
- The node runner uploads partial outputs every 120 seconds. Final artifacts are
  downloaded again and SHA256 checked. Receipts require eight GPU filenames, a
  NIC file, a success marker, correct seeds and progress from every local rank.
  Receipts are local completion evidence, **not corpus admission**: inspect actual
  parquet contents, duration, clock alignment, all-worker completion and peer
  results before admitting the data. Failed prefixes remain separate.

## Cloud Shell execution

First establish the pair's current state; do not substitute remembered names:

```bash
gcloud compute instances list --project=focus-surfer-494820-g0 --format=json \
  > /home/robi/GPU-monitoring/gcloud-instances.local.json
```

After identifying a dedicated idle pair, set `ZONE`, `NODE0`, and `NODE1` to its
actual values. A normal invocation prints the concrete plan. `--launch` performs
it. Both hosts will shut down afterwards, including on a failed launch. Each
service allows `duration + 1800` seconds, followed by at most 300 seconds of stop
grace and a 120-second shutdown delay. The 125-launch plan is not started.

```bash
cd /home/robi/GPU-monitoring
python3 scripts/gcp/launch_followup_pilot.py \
  --zone "$ZONE" --node0 "$NODE0" --node1 "$NODE1" \
  --run-stamp followup_tp8_smoke_01 --cell s16_train_tp8 --smoke --launch
```

Fresh stamps are mandatory. If a deployment directory already exists, the driver
fails rather than overwriting it. The pilot choices are TP8, local SGD H=1,
balanced MoE and IMPALA with learner rank 0. These are synthetic mechanism probes;
production training and benign matched controls remain open checklist items.

Inspect the detached service on either host while running:

```bash
gcloud compute ssh "$NODE0" --project=focus-surfer-494820-g0 --zone="$ZONE" \
  --command='sudo journalctl -u spar-followup-followup_tp8_smoke_01 --no-pager -n 60'
```

Receipts are stored at:
`gs://focus-surfer-494820-g0-telemetry/runs/followup_20261012/control/<stamp>/<node>.complete.json`.
Telemetry, partial snapshots, per-rank work logs and source manifests are under
`runs/followup_20261012/<node>/`. A missing receipt must be investigated; absence
is never interpreted as successful completion.

Only after successful smoke receipts from both hosts and confirmation they have
stopped, explicitly restart the intended pair. Then collect the same cell:

```bash
gcloud compute instances start "$NODE0" "$NODE1" \
  --project=focus-surfer-494820-g0 --zone="$ZONE"
python3 scripts/gcp/launch_followup_pilot.py \
  --zone "$ZONE" --node0 "$NODE0" --node1 "$NODE1" \
  --run-stamp followup_tp8_collect_01 --cell s16_train_tp8 \
  --smoke-stamp followup_tp8_smoke_01 --seed 8123401 --duration 1800 --launch
```

The collection gate compares the smoke's exact source hashes against the current
working tree and checks both receipts. Any code change requires another smoke.
Confirm the instances are stopped after every bounded batch; an OS shutdown
request cannot itself prove the cloud instance's final state. These commands have
been dry-run/CPU-tested here, not validated on the live pair.

## Remaining execution

The full evaluation still needs the 420 MB of missing selected LFS objects
identified in `input_summary.json`, restored to external storage. The new pilot
launcher does not fetch or hydrate the historical corpus. GPU smoke, collection,
GCS hash receipts, real evaluation results and useful-learning validation remain
pending, and must not be marked complete from unit-test success.
