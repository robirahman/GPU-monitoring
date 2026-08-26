# Quarantined: legacy TPU telemetry with a pre-rewrite schema

Nine files (three distinct payloads, duplicated across `data/`, `data/gcp/20260820/`
and `data/gcp/20260821adv/`; the two gcp copies are byte-identical to each other and
differ from the `data/` root copies).

They carry the schema the TPU collector emitted BEFORE the rewrite:

    timestamp_utc, timestamp_epoch, cpu_utilization_pct, sys_mem_used_gb,
    tpu_utilization_pct, tpu_mem_used_gb, [tpu_mem_total_gb,] workload_label,
    run_id, tpu_name, tpu_type

None of `tpu_utilization_pct` / `tpu_mem_used_gb` / `tpu_mem_total_gb` is in
`_normalize_columns`' rename map (classifier/threeway_improved.py:306-331), so all
nine `RAW_SIGNALS` were silently zero-filled by lines 324-326. The runs then entered
the corpus as ALL-ZERO feature vectors carrying real class labels:

    tpu_jax_training   rid 92e0b56e4e19  21 windows -> ml_training
    tpu_jax_inference  rid 5eccea234215  21 windows -> ml_inference
    tpu_idle           rid d7be74c0936b   1 window  -> other
                                          == 43 of 54,071 windows (0.08%)

Note `tpu_idle.parquet` carries `workload_label = "idle"`, not "tpu_idle", so its row
merged into the GPU `idle` workload group rather than announcing itself. It is only 1
of that group's 458 windows, so it does NOT explain `idle`'s high held-out p_train.

Quarantined rather than deleted: the payloads are the only TPU telemetry that has
ever been collected in this project, and the raw numbers may be worth something as a
sanity reference for the v6e-16 collection. They must not re-enter `data/` — the
loader rglobs `data/**/*.parquet` and would zero-fill them again. A collector
conforming to the current contract (scripts/collect_tpu_telemetry.py) is the
replacement.

## The payloads have been deleted from the working tree

Nothing was lost. Verified before deleting:

**The three real payloads are recoverable from git.** They were LFS-tracked, and the
objects are present in the local cache, byte sizes matching exactly:

    .git/lfs/objects/fb/30/fb30b2c6...  tpu_idle.parquet            8105 B
    .git/lfs/objects/../..              tpu_jax_training.parquet   17280 B
    .git/lfs/objects/../..              tpu_jax_inference.parquet  17374 B

To restore one (requires git-lfs, which is NOT installed in the Cloud Shell used for
this work), the pointers live at `data/<name>.parquet` in commit 77b43ee (i.e.
`cd935f7~1`), and `git show 77b43ee:data/tpu_idle.parquet` prints the oid directly.

**The six copies under `data/gcp/20260820/` and `data/gcp/20260821adv/` held no data
at all.** They were 129-130 byte LFS *pointer text* files, not parquet -- pyarrow
rejects them with "Parquet magic bytes not found in footer". That is why their md5
differed from the `data/` root copies, and why the contamination measured 43 windows
from 3 runs rather than 9: `load_all_data`'s `except Exception: log.debug(...)`
silently skipped the unreadable six.
