# TPU Deployment Findings (May 12, 2026)

This note documents the results of attempting to provision TPU resources across various Google Cloud regions and accelerator types for the SPAR GPU monitoring project.

## Successful Provisioning
- **Zone:** `europe-west4-a`
- **Accelerator Type:** `v6e-1`
- **Runtime Version:** `v2-alpha-tpuv6e`
- **Status:** Success. VM `tpu-vm-1` created and repository loaded on branch `feat/tpu-adaptation`.

## Availability & Quota Obstacles

### 1. `us-central2-b`
- **Accelerator Type:** `v4-8`
- **Obstacle:** `LOCATION_POLICY_VIOLATED` / Permission Denied.
- **Note:** This region/zone appeared to have restricted access or was not enabled for the project's current location policy.

### 2. `us-east1-d`
- **Accelerator Type:** `v3-8`
  - **Obstacle:** `INVALID_ARGUMENT`. The accelerator `v3-8` was not found in this zone despite being listed.
- **Accelerator Type:** `v5p-8`
  - **Obstacle:** `QUOTA_EXCEEDED`. Limit is 0 for `TPUV5PPerProjectPerZoneForTPUAPI`.

### 3. `us-central1-c`
- **Accelerator Type:** `v2-8`
  - **Obstacle:** `Insufficient capacity`. Resource was listed but unavailable at the time of request.

### 4. `europe-west4-a`
- **Accelerator Type:** `v3-8`, `v2-8`
  - **Obstacle:** `Insufficient capacity`. High demand for older generations in this region.

## Configuration Notes
- **Service Account:** `tpu-service-account@focus-surfer-494820-g0.iam.gserviceaccount.com` was successfully created with `tpu.admin`, `storage.admin`, `logging.logWriter`, and `monitoring.metricWriter` roles.
- **Deployment Method:** `gcloud compute tpus tpu-vm create` followed by `gcloud compute tpus tpu-vm scp` for code synchronization.
