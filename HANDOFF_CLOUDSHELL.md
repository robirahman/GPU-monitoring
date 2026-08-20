# HANDOFF — scope-ladder session (PAUSED 05:06Z Aug 20)

Supersedes the 00:45Z handoff. That version predates the smoke triage; three of
its instructions are now WRONG and are called out below.

## PAUSED — RESUME HERE (05:06Z Aug 20)

### TOMORROW'S AGREED PLAN (Robi, ~14:00Z Aug 20)
In order:
1. `prune_unmarked_cells.py` over whatever is already pulled/synced, so no
   failed-cell telemetry reaches the classifier. Dry run first, then --apply.
2. Start both instances, diff `.done` markers between nodes for symmetry, then
   run the 3-cell validation smoke below. Do NOT start a long run until it passes.
3. Finish the remaining H100 cell-reps (33 of 36 outstanding; ~6 h for all three
   reps, ~2 h for the rest of rep 1).
4. Bring up a NON-PREEMPTIBLE TPU: `SPOT=0 ZONE=us-east1-d bash
   scripts/tpu_collection/provision_tpu.sh create`. No quota request needed
   (v6e on-demand limit 512 chips vs 16 required). Check on-demand pricing first.

### NEW PAT — WHERE IT GOES
Write it to `~/.spar_pat` in Cloud Shell, mode 0600. That is the single location
every script reads (`setup_tpu_all.sh` uses PAT_FILE=${PAT_FILE:-$HOME/.spar_pat}),
and Cloud Shell's $HOME persists across sessions and VM recycles.

Write it in a REAL terminal tab (the `+` in the Cloud Shell toolbar). Claude
Code's `!` prefix has no TTY, so `read` there just times out -- and typing the
token as a visible argument is what leaked the last one:

    read -rsp "PAT: " P && printf '%s' "$P" > ~/.spar_pat && chmod 600 ~/.spar_pat && unset P

`read -rsp` never echoes, so the token stays out of the terminal, out of
~/.bash_history, and out of any assistant transcript.

Scope it as a FINE-GRAINED token limited to robirahman/GPU-monitoring:
  * Contents: Read       -- enough for TPU worker clones and `git pull` on nodes
  * Contents: Read+Write -- needed if the branch is to be pushed from Cloud Shell
Set a short expiry (this is a one-or-two-day job).

After revoking the old token, the copy cached at `/etc/spar/git_pat` on both
retained boot disks is dead. Nothing on resume needs it (the repo is already
cloned), but to `git pull` the pushed fixes onto the nodes, refresh it:
    gcloud compute ssh spar-gpu-$i --zone=us-central1-a \
      --command="sudo install -m 600 /dev/stdin /etc/spar/git_pat < ~/.spar_pat"
Then `sudo git -C /opt/spar/repo pull` picks up commits 7e391f8 + ceeeaf1, which
is cleaner than the scp'd copies currently on the disks.


Both GPU instances are STOPPED (status TERMINATED, no compute billing). The two
500 GB pd-balanced BOOT DISKS ARE DELIBERATELY RETAINED (~$0.14/h total, ~$3.30/
day) because they hold everything expensive to rebuild: the dkms-built nvidia
610.57.04 driver, venv-main + venv-vllm, ~148 GB of preloaded Qwen weights, the
CIFAR-10 staging, and the .done markers. Re-provisioning from scratch instead
would cost ~$50 of H100 time. Do NOT delete these disks without deciding to
abandon that state.

To resume (nothing else to set up):
    gcloud config set project focus-surfer-494820-g0
    gcloud compute instances start spar-gpu-0 spar-gpu-1 --zone=us-central1-a
    # wait for ssh, then on BOTH nodes (node0 first):
    gcloud compute ssh spar-gpu-$i --zone=us-central1-a \
      --command="sudo RUN_STAMP=20260820b bash /opt/spar/repo/scripts/gcp/launch_queue.sh"
    # and the GCS sync loop on both:
    #   cd /opt/spar/repo && nohup bash scripts/gcp/sync_data.sh --loop 120 &
CAVEAT: these are Spot instances, so `start` succeeds only if Spot capacity is
available in us-central1-a at the time.

FIRST THING TO DO ON RESUME: validate the three fixes that have never been
observed working end-to-end -- vLLM sampler (VLLM_USE_FLASHINFER_SAMPLER=0), the
S1 rung at --batch-size 1 --seq-len 512, and the adversarial_ddp collective
deadline. Cheapest validation is a targeted smoke:
    sudo RUN_STAMP=20260820val SMOKE=1 \
      ONLY_CELLS='^(s1_train_fsdp7b|s16_inferrep_qwen7b|s16_train_ddpstagger_k)' \
      bash /opt/spar/repo/scripts/gcp/launch_queue.sh
Do NOT start a multi-hour run before those three pass.

ALSO: keep the nodes SYMMETRIC. Any marker present on one node but not the other
desyncs every S16 cell (the c10d rendezvous is the only cross-node barrier, and a
lone node fails after RENDEZVOUS_TIMEOUT=120 s). Before relaunching, diff
`logs/.done/$RUN_STAMP/` between the nodes and delete any S16 marker that is not
on both. Two such markers were removed at 05:03Z (node1's telemetry-only
"successes" for s16_infershard_vllm_tp8pp2, where node0's vLLM never served).

## LIVE RESOURCES (all stopped/deleted -- see above)
- spar-gpu-0/1: a3-highgpu-8g Spot, us-central1-a, RUNNING since 23:24Z (~$78/h pair).
- NO TPU. The v6e-16 Spot was preempted THREE times in one night and collected
  ZERO cells: ~9 min (00:40-00:49), <=80 min (01:31-02:53), <=20 min (03:05-03:26,
  killed mid-G1-gate right after setup finished). Queued resource and node are
  both deleted -- `queued-resources list` and `tpu-vm list` return 0 items, so
  nothing is billing. v6e-16 Spot capacity in us-east1-d was being reclaimed
  aggressively; a 4th attempt needs a different zone, a non-Spot reservation, or
  a smaller slice, not another retry of the same shape. The gate bugs (11, 12)
  are fixed, so the track is READY to run if capacity ever holds -- setup is
  fully scripted (scripts/tpu_collection/setup_tpu_all.sh, ~5 min for 4 workers).
- Bucket gs://focus-surfer-494820-g0-telemetry

## ⚠️ CORRECTIONS TO THE PREVIOUS HANDOFF
1. **SESSION is `20260820`, NOT `20260820_scope`.** /etc/spar/node_env is
   authoritative and env.sh sources it LAST, so it overrides any pinned export.
   Data is at `gs://.../runs/20260820/{spar-gpu-0,spar-gpu-1,control}/`.
   `pull_results.sh` / `ingest_session.sh` with the pinned value find NOTHING.
   Use: `export SESSION=20260820` before pulling.
2. Do not run `gcloud config set project` fresh each shell -- it is now set.
3. The GPU SMOKE from 00:41Z was NOT a shakedown pass: it failed almost every
   cell. Do not treat those `.done` markers or `logs/smoke_data/` as evidence.

## THE REAL RUN
Launched 01:24Z, RUN_STAMP=20260820b, both nodes, detached ON the node:
```
gcloud compute ssh spar-gpu-$i --zone=us-central1-a \
  --command="sudo RUN_STAMP=20260820b bash /opt/spar/repo/scripts/gcp/launch_queue.sh"
```
`scripts/gcp/launch_queue.sh` (new) reads this node's rank from /etc/spar/node_env,
so the SAME command line is correct on both nodes. Add `SMOKE=1` and/or
`ONLY_CELLS='<regex>'` for a targeted run. GCS sync loops also running on both.

Progress: `s16_train_fsdp7b` OK on BOTH nodes (622s/619s) = the headline S16
train datapoint is banked. Expect ~2 h/rep, ~5.5-6 h for REPS=3 (~$450).
Rep-major: safe to stop after any rep.

## NINE BUGS FOUND BY THE SMOKE (all fixed unless noted)
Three of these produced *silent* wrong results -- worse than a crash.
1. `torchrun` is a venv console-script and is NOT on root's secure_path, and the
   launcher execs argv[0] with no shell -> EVERY train cell died in ~5 s.
   Fixed: queue resolves `$TORCHRUN` from `$PY`'s bindir + preflight gate.
2. Same for `ray`, but its failure was swallowed by `|| true` -> vLLM started a
   LOCAL 8-GPU Ray and blocked forever on a 16-GPU placement group.
   Fixed: `$RAY` resolved, failures fatal, `RAY_ADDRESS` pinned, pre-emptive
   `ray stop`, and a wait_ray_gpus(16) barrier before vLLM launches.
3. `.done` markers were keyed on cell+node only -> (a) smoke markers made the
   real run SKIP cells, (b) **REPS=3 collapsed to 1 rep** because rep 2/3 saw
   rep 1's markers. Fixed: markers/logs keyed on `$rid` (=sha1 of stamp_cell_rep)
   under `logs/.done/$RUN_STAMP/`, so resume still works and reps stay distinct.
4. `--parallel` mode never called `sys.exit(rc)` -> a cell whose 8 replicas ALL
   died reported OK and wrote a .done marker. Fixed.
5. Registry commands hardcode `sys.executable` = venv-main, which has no vllm ->
   every inferrep replica `ModuleNotFoundError`. Fixed: `--workload-python`.
6. Replicated cells recorded `workload_label=llm_infer_qwen25_7b_fp16`, but
   `scope_ladder_eval.py` derives the rung from the `s1_/s8_/s16_` PREFIX ->
   the whole inferrep arm scored as scope "na" and was dropped from the ladder.
   Fixed: `--label` records the cell name.
7. Train loops stopped on a PER-RANK wall clock with no barrier before teardown
   -> ranks left the loop on different iterations, node1 exited 0 while node0
   aborted mid-ReduceScatter (`ncclRemoteError`). A barrier alone cannot fix
   this (a rank that exits early strands peers inside a collective), so the STOP
   DECISION itself is now collective: `workloads/dist_deadline.py`
   (`CollectiveDeadline` + `sync_teardown`), wired into fsdp_training.py,
   pytorch_training_ddp.py and all 4 loops of adversarial_ddp.py.
8. flashinfer (optional vllm dep) annotates `array.array[int]` at module scope;
   subscriptable only in py3.13, venv is 3.10 -> TypeError, which vllm's
   `except ImportError` does NOT catch -> every TP>1 worker died in init_device().
   Patching the annotation made it WORSE: flashinfer then JIT-compiles, needs
   nvcc (driver-only image), and that failure HANGS the engine for the full
   ~26 min timeout. Fixed by UNINSTALLING flashinfer-python from venv-vllm, which
   makes the import a clean ImportError -> fi_ar_available=False, falls back to
   custom allreduce/pynccl. Do NOT install the CUDA toolkit to "fix" this: it
   would JIT inside the measurement window. Codified in bootstrap_node.sh.
9. 72B/7B weights were not on disk; a ~145 GB download would land INSIDE the
   timed cell. Fixed: `scripts/gcp/preload_vllm_models.sh`, run on both nodes
   (~148 GB each, done). Re-run it after any re-provision.

## WHAT IS ACTUALLY BANKED (verified via .done markers on BOTH nodes)
Target matrix is 36 cell-reps: rep1 13 cells, rep2 13, rep3 10 (reps 1-2 add
fsdp30b + ddpstagger + ddpkaccum; rep3 drops them). Confirmed valid:

    s16_train_fsdp30b     rep1            (both nodes, 619 s)
    s8_train_ddp_resnet50 rep1 + rep2     (both nodes -- the CIFAR fix working)

That is 3 of 36 cell-reps. Everything else is still to collect. Note the
s16/s8 fsdp7b cells DID collect successfully earlier, but were deliberately
invalidated by the matched --batch-size 1 --seq-len 512 decision and their
markers cleared, so they will be re-collected.

Rough cost to finish: ~33 cell-reps at ~11 min for train / ~7 min inferrep /
~17 min infershard is ~6 h ~= $480. Rep 1 alone (11 remaining cells) is ~2 h
~= $160 and is enough for a first ladder evaluation, though with no bootstrap CI
across reps.

## TPU: QUOTA IS NOT THE PROBLEM
Checked 2026-08-20 via `gcloud alpha services quota list --service=tpu.googleapis.com`:
    tpu-v6e              (on-demand)   global limit = 512 chips
    tpu-v6e-preemptible                global limit = 1536 chips
A v6e-16 needs 16 chips, so NO quota request is required for on-demand. The three
preemptions were Spot capacity reclamation, not a quota ceiling.

To switch to on-demand, provision_tpu.sh already takes a flag -- it is one env var:
    SPOT=0 ZONE=us-east1-d bash scripts/tpu_collection/provision_tpu.sh create
(SPOT=1 is the default and appends --spot; SPOT=0 omits it.) On-demand cannot be
preempted, but creation still depends on capacity being available in the zone at
request time. On-demand v6e costs materially more than Spot -- CHECK THE PRICING
PAGE before committing; do not trust a remembered figure.

## BUGS 10-14 (found after 01:45Z)
10. CIFAR-10 was not on the nodes, and the workloads pass
    `download=(rank == 0)` -- rank 0 lives on node0, and with no shared
    filesystem node1's ranks ALWAYS fail. Both DDP cells and both adversarial
    cells died in ~15 s. Staged on both nodes (341 MB); codified in
    preload_vllm_models.sh. The upstream mirror runs at ~100 kB/s and served
    node0 a corrupt file, so the reliable path is: download once, then copy
    node-to-node via the bucket (1.4 GiB/s).
11. tpu_queue.sh's G1 gate ran the CPU-only self-test on the TPU, where jnp
    matmuls default to bfloat16. The finite-difference check (lp-lm ~ 4e-6
    against a loss of ~4.6) became pure noise: rel_err=6.1e-2 vs a 5e-2
    tolerance, aborting the whole TPU track even though the gradient was
    correct. Fixed by forcing JAX_PLATFORMS=cpu +
    XLA_FLAGS=--xla_force_host_platform_device_count=8 (what the self_test
    docstring already specified). On CPU: rel_err=9.4e-03, PASS.
12. The same gate piped ssh_all into `grep -q`, which exits on first match and
    SIGPIPEs the ssh upstream -- and under `set -o pipefail` that made the
    pipeline non-zero, so the gate aborted even when the self-test HAD printed
    PASS. Fixed by capturing the output and deciding on the captured text.
13. NODE DRIFT breaks cells that otherwise work. The queue relies on the c10d
    rendezvous as its only cross-node barrier, which holds while cells succeed --
    but a cell that fails in 15 s on one node and runs 600 s on the other injects
    ~10 min of skew, and subsequent S16 cells then fail at ~95-127 s (the 120 s
    RENDEZVOUS_TIMEOUT). s16_train_fsdp30b passed rep 1 at 619 s and failed rep 2
    at 126 s for exactly this reason. Any asymmetric change between the nodes
    (staging data on one, restarting one) must be followed by restarting BOTH
    queues together to re-sync them.
14. DATA HYGIENE: run_workload_multi_gpu.py saves telemetry whether or not the
    workload succeeded, and ingest_session.sh globs parquets with NO reference to
    the .done markers. On this session ~74% of ladder parquets came from failed
    cells -- including 48 files of near-idle GPUs labelled `s16_train_ddp*`, i.e.
    idle telemetry teaching a train-vs-inference classifier that idle looks like
    TRAINING. New `scripts/gcp/prune_unmarked_cells.py` keeps a parquet only if a
    .done marker vouches for its exact cell+node+rep, quarantines (never deletes)
    the rest, and leaves the ~1720 historical repo parquets alone. RUN IT BEFORE
    ingest_session.sh:
      python scripts/gcp/prune_unmarked_cells.py --data <tree> --markers <tree>   # dry run
      ... --apply
    Note: data/ IS CHECKED INTO THE REPO with ~1720 historical A100/B200/H200
    parquets, so raw file counts are NOT a measure of tonight's progress.

## S1 RUNG DECISION (2026-08-20, Robi)
s1_train_fsdp7b OOMs: FSDP FULL_SHARD on a single rank shards nothing, so 6.7B
params + optimizer (42.4 GB post-init) + activations peak at 83 GB on an 80 GB
H100. Activation checkpointing is already on by default. Decision: run ALL THREE
fsdp7b rungs (S1/S8/S16) at `--batch-size 1 --seq-len 512` so only SCOPE varies
across the ladder -- giving S1 alone a smaller batch would confound batch size
with scope in the headline dAUC(S8-S1) test. The 30B cell already used these
settings. Cost: the previously-banked fsdp7b cells must be re-collected, and
because the rid is deterministic in (stamp, cell, rep) the new parquets overwrite
the old ones cleanly.

## OPEN
- `s16_infershard_vllm_tp8pp2`: still failing as of 01:45Z. Ray forms 16 GPUs and
  all 16 workers spawn, then node1's PP-stage workers died on the flashinfer/nvcc
  error above and the engine hung. flashinfer is now gone, so rep 2 retries it on
  the fixed path; `s8_infershard_vllm_tp8` (later in rep 1) tests it sooner.
  If it hangs again, suspect the 26-min TIMEOUT_BIG masking a fast failure.
- adversarial_ddp.py fix is deployed but has NOT yet been exercised (cells 6-7).

## NEW/CHANGED FILES (working tree only -- NOT committed, NOT pushed)
scripts/gcp/launch_queue.sh, scripts/gcp/preload_vllm_models.sh,
scripts/tpu_collection/setup_tpu_all.sh, workloads/dist_deadline.py (all new);
scripts/rebuttal_collection/multinode_queue.sh, scripts/run_workload_multi_gpu.py,
scripts/gcp/bootstrap_node.sh, scripts/tpu_collection/setup_tpu_host.sh,
workloads/{fsdp_training,pytorch_training_ddp,adversarial_ddp}.py (modified).
All are scp'd to the nodes. **A fresh re-provision clones from GitHub and would
get the UNFIXED code** -- push before relying on preemption recovery from a new
disk. (A STOP/start on the preserved boot disk keeps the scp'd fixes.)

## TPU (step still needs a human)
Workers already have the PATCHED setup_tpu_host.sh (it now scrubs the PAT from
the saved git remote -- the old one left the token in .git/config on all 4).
Write the PAT WITHOUT putting it in a chat/history, in a real terminal:
```
read -rsp "PAT: " P && printf '%s' "$P" > ~/.spar_pat && chmod 600 ~/.spar_pat && unset P
bash scripts/tpu_collection/setup_tpu_all.sh     # scp + run on all 4 workers
```
Then: `tmux new -s tpu; ZONE=$TPU_ZONE SMOKE=1 bash scripts/tpu_collection/tpu_queue.sh`
then again without SMOKE.

## TEARDOWN (the money step)
On both nodes `bash scripts/gcp/sync_data.sh --final`; then
`bash scripts/gcp/teardown.sh` and READ its verify-zero output (instances /
DISKS / TPUs / queued-resources / reservations must all be 0 -- STOP-mode disks
keep billing); `ZONE=us-east1-d bash scripts/tpu_collection/provision_tpu.sh delete`;
then `export SESSION=20260820` and
`bash scripts/gcp/pull_results.sh && bash scripts/gcp/ingest_session.sh`.
Check Billing -> Reports tomorrow.

## SECURITY
REVOKE the GitHub PAT after teardown. It was pasted into a Claude chat and typed
as a stray shell command earlier, is cached at /etc/spar/git_pat on both GPU
nodes, and may exist in TPU worker git remotes if setup ran with the OLD script.
Verify with: `git -C /opt/spar/repo remote -v` on each host.
