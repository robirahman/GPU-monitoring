# Seeded collection pilot

Implementation is ready for a live smoke, but **no new VM or collection job was
started in this execution**. `gcloud auth list` confirms an active account;
`gcloud compute instances list` fails here because the restricted agent session
cannot resolve `compute.googleapis.com`. No further login is currently indicated.
The operator supplied the live listing in `gcloud-instances.local.json`. It
contains `spar-a100-0` and `spar-a100-1` in `us-central1-c`, both stopped,
`a2-highgpu-8g` with eight A100 accelerators each. No H100 instances are listed.
The pilot now explicitly supports a matching A100 pair. This is an A100 hardware
stratum and does not satisfy the H100 repetition counts in the original plan.
The running `spar-analysis-0` remains available for the separate evaluation.

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
- The driver requires two idle hosts with eight matching A100 or H100 GPUs each. It
  starts one cell under detached systemd services with a runtime cap and shutdown
  after completion/failure. It does not provision machines. `--start-stopped`
  explicitly starts a stopped SPAR pair and stops that pair on launch failure.
- The node runner uploads partial outputs every 120 seconds. Final artifacts are
  downloaded again and SHA256 checked. Receipts require eight GPU filenames, a
  NIC file, a success marker, correct seeds and progress from every local rank.
  Receipts are local completion evidence, **not corpus admission**: inspect actual
  parquet contents, duration, clock alignment, all-worker completion and peer
  results before admitting the data. Failed prefixes remain separate.

## One-command A100 campaign

Run this in regular Cloud Shell (the agent session still cannot resolve the
Cloud API hostname). No additional login is indicated:

```bash
cd /home/robi/GPU-monitoring
nohup python3 -u scripts/gcp/run_a100_followup_pilot.py --launch \
  > /tmp/followup-a100-pilot.log 2>&1 < /dev/null &
```

This uses the two named A100 nodes and does the following:

1. Require both instances to be stopped before taking ownership; start the pair.
2. Wait for SSH and check actual GPU model, memory and idle state. The Python
   process probe matches executable/script arguments rather than its own source.
3. Stage a fresh source bundle and run a 60-second TP8 smoke with seed 20261012.
4. Wait for both durable success receipts. Download every final artifact, verify
   its SHA256, read the parquet files, and check sample counts, timestamp order,
   recording spans and per-rank completed work. Require 8 GPU and 1 NIC file per
   node and positive optimizer progress on all 16 TP workers.
5. Confirm both hosts stopped. Restart them and collect TP8 for 1800 seconds with
   seed 8123401, gated on the exact smoke source hashes and receipts.
6. Repeat the download checks, confirm both hosts stopped, and upload the campaign
   summary. Any failed gate aborts and requests shutdown of the owned pair.

The script prints a concrete plan without `--launch`. Campaign identifiers are
UTC-stamped. Remote artifacts go under
`gs://focus-surfer-494820-g0-telemetry/runs/followup_a100_<timestamp>/`, and checked
local downloads and `campaign.json` go under `/tmp/followup_a100_<timestamp>/`.
The cloud copy is durable; `/tmp` is only a local inspection cache.

The 30-minute collection is a nominal eight accelerator hours (16 × 0.5 h), plus
smoke/startup/teardown. Each node service has a `duration + 1800` second cap,
300-second stop grace and 120-second shutdown delay. SSH startup waits are bounded
and failed startup requests are followed by stop requests. Logs explicitly report
any failed cleanup. No success claim is made without checking cloud stop state.
The full 125-launch campaign is not launched.

Follow progress with:

```bash
tail -n 60 /tmp/followup-a100-pilot.log
```

This workflow has passed CPU failure-path tests and a CLI dry run; live execution
is still pending. A successful run establishes recorded synthetic TP work and
basic telemetry integrity. Further clock/phase-label review and scientific corpus
admission are still required, and task learning has not been measured.

For another dedicated pair, `launch_followup_pilot.py` exposes explicit node,
zone, GPU-model and startup arguments. It supports TP8, local SGD H=1, balanced
MoE and IMPALA with learner rank 0. Production workloads and benign matched
controls remain open checklist items.

## Remaining execution

The full evaluation still needs the 420 MB of missing selected LFS objects
identified in `input_summary.json`, restored to external storage. The new pilot
launcher does not fetch or hydrate the historical corpus. GPU smoke, collection,
GCS hash receipts, real evaluation results and useful-learning validation remain
pending, and must not be marked complete from unit-test success.
