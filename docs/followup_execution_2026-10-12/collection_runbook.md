# Seeded collection pilot

The [first live startup attempt](first_cloud_attempt.json), at 23:11 UTC on
October 12, failed for both A100 nodes with `ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS`
(`stockout`). **No smoke or collection workload ran in that first attempt.**
Cloud API and SSH access were restored at 23:24 UTC. Both nodes then started,
and TP8 completed on all 16 workers. The gate rejected that smoke because the
NIC recorder did not flush final parquet on SIGTERM. This is now fixed and
covered by a real subprocess regression. The [fresh smoke passed](verified_a100_smoke.json)
with 16 GPU streams, two NIC streams and 26 downloaded, hash-verified artifacts.
The 30-minute collection started at 23:48 UTC on October 12, but Google Cloud
preempted `spar-a100-0` at 23:51:50 UTC. The distributed workload failed and
both hosts stopped. [Failure evidence](failed_a100_collection.json) preserves
the cloud audit events and partial-upload records; this run is incomplete.
A fresh smoke-gated retry on the same pair started at
01:14 UTC on October 13. Its smoke gate passed and collection launched at 01:27
UTC, but Google Cloud preempted the same node again at 01:53:22 UTC, about 26
minutes after launch. [Retry evidence](failed_a100_retry.json) records the partial
data and absent completion receipts. Both GPU hosts are now stopped; no new retry
was launched during the 02:22 UTC status check.
Both interrupted runs now have [validated observed intervals in the corpus](preempted_recovery.md).
The success-receipt gate establishes completed-run status and controls pilot
progression; it is not a blanket exclusion of earlier valid telemetry.
[Live execution](live_execution.json) records the campaign and separate analysis run.
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

Run this in an authenticated shell. Cloud API access now works directly from
the agent session; do not start a duplicate controller while a campaign is active:

```bash
cd /home/robi/GPU-monitoring
nohup python3 -u scripts/gcp/run_a100_followup_pilot.py --launch --inspect-analysis \
  >> logs/followup_a100_collection.log 2>&1 < /dev/null &
```

This uses the two named A100 nodes and does the following:

1. Optionally inspect `spar-analysis-0` without changing its repo or starting/
   stopping jobs; save readiness to `logs/followup_analysis_probe.local.json`.
   Freeze the campaign source in external storage so ongoing code edits cannot
   change the smoke/collection comparison.
2. Require both instances to be stopped before taking ownership; start the pair.
   Retry only the exact capacity error, at most six attempts per stage with
   ten-minute pauses. Stop any partial allocation and confirm both hosts are
   `TERMINATED` before each pause. Authentication, quota, and unconfirmed-cleanup
   failures stop immediately.
3. Wait for SSH and check actual GPU model, memory and idle state. The Python
   process probe matches executable/script arguments rather than its own source.
4. Stage a fresh source bundle and run a 60-second TP8 smoke with seed 20261012.
5. Wait for both durable success receipts. Download every final artifact, verify
   its SHA256, read the parquet files, and check sample counts, timestamp order,
   recording spans and per-rank completed work. Require 8 GPU and 1 NIC file per
   node and positive optimizer progress on all 16 TP workers.
6. Confirm both hosts stopped. Restart them and collect TP8 for 1800 seconds with
   seed 8123401, gated on the exact smoke source hashes and receipts.
7. Repeat the download checks, confirm both hosts stopped, and upload the campaign
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
tail -n 60 logs/followup_a100_collection.log
```

The updated workflow passes 24 focused CPU tests, including final NIC parquet
saving after a real SIGTERM, and a CLI dry run. See the live execution record for
the current smoke and collection status. Google identifies this error as transient resource
availability, separate from quota, and recommends trying later as the first
option. [Google Cloud guidance](https://docs.cloud.google.com/compute/docs/troubleshooting/troubleshooting-resource-availability). A successful run establishes recorded synthetic TP work and
basic telemetry integrity. Further clock/phase-label review and scientific corpus
admission are still required, and task learning has not been measured.

For another dedicated pair, `launch_followup_pilot.py` exposes explicit node,
zone, GPU-model and startup arguments. It supports TP8, local SGD H=1, balanced
MoE and IMPALA with learner rank 0. Production workloads and benign matched
controls remain open checklist items.

## Remaining execution

All historical evaluation inputs are now restored and hash-verified in
`/var/tmp/spar_followup/20261012T232425Z/repo` on `spar-analysis-0`. Most missing
payloads came from the VM's existing checkout; the remaining 633 objects
(17,121,923 bytes) were fetched using the existing Cloud Shell GitHub login.
No credentials were copied to the VM. The first cache build reached its 24 GiB memory cap. Removing redundant dataframe copies let the retry pass ingestion, but raw
cache serialization then exposed legacy `N/A` sensor strings and mixed throttle
status types. Numeric missing values and categorical status are now normalized,
with a parquet round-trip regression. The current attempt runs in `schema_retry/`
with two cache workers, a four-vCPU quota and the same memory cap. The shared analysis
VM remains running after this job. At 02:22 UTC on October 13, five of six feature
chunks had completed (61,138 windows); the evaluation service was still active.
Model evaluation had not started. Real results and useful-learning validation
remain pending and must not be inferred from unit-test success.
