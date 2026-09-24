# data/gcp/20260821adv has moved to GCS

2,776 files (307 MB) from the 2026-08-21 adversarial session:

    gs://focus-surfer-494820-g0-telemetry/archive/sessions/20260821adv/

Moved 2026-09-24 to make room for the 20260923 ingest. Verified file-for-file
before deletion (2776 local == 2776 remote).

Worth noting why this was urgent independently of disk space: this session was
NOT in the bucket at all. `gs://.../runs/20260821adv/` did not exist and nothing
matching "20260821" existed under any prefix, so 2,635 parquets were single-copy
on a 4.8 GB disk that has hit 100% twice. Assuming it was mirrored -- as an
earlier note in this repo did -- would have destroyed it.

Restore:

    gsutil -m cp -r gs://focus-surfer-494820-g0-telemetry/archive/sessions/20260821adv data/gcp/
