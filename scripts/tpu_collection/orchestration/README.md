# Unattended TPU orchestration

These three scripts drive a TPU session with no operator present. They lived only
in `/tmp` for several sessions and were lost twice to Cloud Shell clearing it,
which is why they are in the repo now.

    mz_common.sh      the resource table (zone:queued-resource:node:accel) plus
                      qr_state / kill_one / kill_all. EDIT THE TABLE per session.
    mz_watchdog.sh    teardown guarantee. Fires on whichever comes first: the
                      DONE marker in GCS, a global deadline, or ACTIVE_MAX_SECS
                      since a node was first seen ACTIVE. Then deletes every
                      node AND queued resource across all zones, verifies zero,
                      and stops its own VM.
    probe_driver.sh   claims the first slice to provision, releases the others,
                      runs the work, and ALWAYS writes the DONE marker on exit.

## Why the watchdog is not optional

`--valid-until-duration` on a queued resource bounds how long the REQUEST stays
valid, not how long a provisioned node lives. A slice that comes up runs until
something deletes it. `provision_tpu.sh`'s docstring claims otherwise; do not
rely on it.

ACTIVE_MAX_SECS is the load-bearing trigger: a pending request is free, a
provisioned v6e-16 is not. If the driver dies after a slice comes up, this is
what stops the billing.

## Run them from a VM, not Cloud Shell

Both are launched with nohup on `spar-analysis-0` so they survive the operator
logging off and Cloud Shell recycling. They need only gcloud plus the resource
table.

## Where results go

* code + result write-ups -> this repo
* TPU telemetry, driver/watchdog logs, probe JSON
  -> gs://focus-surfer-494820-g0-telemetry/tpu/<session>/
     {control/DONE.txt, control/driver.log, probe/*.json, runs/<rid>/w*/}
  GCS is the right home because both VMs are stopped by the time anyone reads it.

## Known rough edges

* `gcloud compute scp` fails with rc 255 fairly often just after a VM boots;
  piping through `ssh --command="cat > file"` is reliable. Retry loops included.
* Capturing a remote file with `ssh --command="cat f" > local` can yield an empty
  local file (it did for one probe JSON). Verify size before trusting it.
* The driver's per-step status wrapper inherits the exit status of `| tail`
  pipelines under `pipefail`, and multi-worker `gcloud ssh` returns non-zero
  routinely -- so step-level rc has over-reported failure. Treat rc as a hint and
  read the log.
