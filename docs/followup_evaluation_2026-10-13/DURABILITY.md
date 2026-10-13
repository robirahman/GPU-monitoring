# Durable storage and unattended execution

The evaluation runs outside Cloud Shell on `spar-analysis-0` in `us-central1-a`,
project `focus-surfer-494820-g0`. Its `/var/tmp` directory is on the VM's ext4
persistent boot disk (`/dev/sda1`), not a local SSD or Cloud Shell temporary disk.
The VM is shared and is not shut down by this workflow.

The complete frozen source/input archive, feature cache, completed result bundles,
checksums, and publication source checkpoint are also backed up in Cloud Storage:

`gs://focus-surfer-494820-g0-telemetry/followup_analysis/20261012T232425Z/reviewed_20261013/`

The manifest records exact object generations and SHA-256 hashes. Inputs are never
restored to the small Google Cloud home volume. A Cloud Shell disconnect does not
stop the Compute Engine systemd jobs or remove the Cloud Storage objects.

The primary evaluation service has a 12-hour limit. The mechanism handoff service
uploads the main result bundle before stopping duplicate sensitivity stages. The
review service checks saved predictions against the frozen cache, writes the report
and figure, and saves `publication_ready.tar.gz`, `RESULTS.md`, `review.json`, and
`review_pipeline_status.json` to the prefix above. A failed audit is recorded as a
failure, not published as a passing result. Service scripts are retained under
`artifacts/`; they do not contain account credentials.

The independent 32-vCPU sensitivity worker completed both evaluations, uploaded
eight hash-verified artifacts, and stopped. Its VM and boot disk were deleted after
verification. Its source, logs, execution receipt, results and cleanup record remain
in Cloud Storage and this review bundle. No accelerator VM was launched for these
CPU evaluations.

If resuming from another machine, read `review_pipeline_status.json` first. Download
its completed publication bundle and verify the recorded SHA-256 before unpacking.
The publication checkpoint and `PUBLICATION_RESUME.json` preserve the intended base
commit and branch if the original Cloud Shell session disappears before publication.
GitHub credentials were not copied to either worker. An authenticated client performs
the reviewed branch push; a successful automated review alone does not imply that
GitHub publication happened.

The two Spot-preempted A100 captures remain admitted as observed intervals in the
corpus, with zero completed-repetition credit. Their original cloud objects and
recovery manifest are retained. This evaluation's frozen snapshot predates their
admission, so incorporating them requires a separately versioned evaluation.
