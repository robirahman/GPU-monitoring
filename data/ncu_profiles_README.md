# data/ncu_profiles has moved to GCS

Six raw Nsight Compute profiles (235 MB, captured 2026-04-11) now live at

    gs://focus-surfer-494820-g0-telemetry/archive/ncu_profiles/ncu_profiles/

    cufft_hpc  mining_proxy  nbody_hpc  rendering_proxy
    resnet18_training  resnet50_inference

Moved on 2026-09-23 because /home (a 4.8 GB disk) reached 100% and neither git
nor gcloud could write. Nothing in the analysis path reads them --
`classifier/rebuttal_analysis/multinode_baseline.py:173` excludes `ncu_profiles`
from its file scan by name -- and they are regenerable with
`scripts/run_ncu_profiling.py`.

Restore:

    gsutil -m cp -r gs://focus-surfer-494820-g0-telemetry/archive/ncu_profiles/ncu_profiles data/
