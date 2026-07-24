# GCP TPU Quota Status & Telemetry Comparison Report

This report documents our findings from the TPU provisioning probes, analyzes the telemetry availability on Google Cloud TPUs compared to the previous NVIDIA GPU phase, and guides you on how to request quota increases and policy updates in GCP.

---

## 📈 TPU Quota and Policy Status Report

Based on the 39-zone provisioning scan, we categorized the GCP availability barriers into three actionable groups:

### 1. Quotas Needing Increase
These regions have the hardware available, but your GCP project's quota limit is set to 0 or is insufficient for the requested cores:
* **TPU v5p (`v5p-8`):** Failed with `Quota Exceeded` in `europe-west1-b`, `us-east1-d`, and `us-south1-a`.
  * *Required Metric:* `tpu.googleapis.com/tpu-v5p` (Cores in use by TPU API)
* **TPU v6e (`v6e-1`):** Failed with `Quota Exceeded` in `us-east4-a`.
  * *Required Metric:* `tpu.googleapis.com/tpu-v6e` (Cores in use by TPU API)
* **TPU v4 (`v4-8`):** Lacks any quota metrics in this project.
  * *Required Metric:* `tpu.googleapis.com/tpu-v4s` (Cores in use by TPU API)
* **TPU v7 (Ironwood) / v8 (8t / 8i):** Missing quota allocation entirely in standard billing projects.
  * *Required Action:* Whitelist and configure quota allocations for metrics like `tpu.googleapis.com/tpu-v7` or equivalent.

### 2. Policies Needing Update
These regions returned `Permission Denied` or `LOCATION_POLICY_VIOLATED`, which indicates that your Google Cloud Organization Policy restricts resource deployment in these regions:
* **TPU v5p (`v5p-8`):** Failed with `Permission Denied` in `us-central1-a` and `us-east5-a`.
* **TPU v6e (`v6e-1`):** Failed with `Permission Denied` in `us-east1-d`.
* **TPU v4 (`v4-8`):** Previously blocked in `us-central2-b` due to location policy constraints.
  * *Policy to Modify:* `constraints/gcp.resourceLocations` (Resource Location Restriction)

### 3. Insufficient Capacity (No Quota Changes Needed)
For older generations (`v2-8`, `v3-8`), provisioning failed in all zones due to `Insufficient Capacity`. This is a hardware availability issue on GCP's end; older ASICs are being retired, and requesting quota increases will not resolve this.

---

## 📋 How to File GCP Requests

### Step A: Requesting Quota Increases
1. Go to the **IAM & Admin > Quotas** page in your GCP Console: [GCP Quotas Console](https://console.cloud.google.com/iam-admin/quotas?project=focus-surfer-494820-g0).
2. Filter the service by `Cloud TPU API`.
3. Select the following metrics and click **Edit Quotas**:
   * **Metric:** `tpu.googleapis.com/tpu-v6e` | **Dimension:** `us-east4-a` | **Request:** Change limit from `0` to `8` (minimum size to test pods/instances).
   * **Metric:** `tpu.googleapis.com/tpu-v5p` | **Dimensions:** `us-east1`, `us-south1`, `europe-west1` | **Request:** Change limit from `0` to `8`.
   * **Metric:** `tpu.googleapis.com/tpu-v4s` | **Dimension:** `us-central2` | **Request:** Change limit from `0` to `8`.
4. Enter a business justification:
   > "Conducting vendor-invariant workload classification research for AI compute governance. Exposing telemetry across diverse TPU generations (v4, v5p, v6e) to validate training signatures."

### Step B: Updating Organization Location Policies
If your project is under an organization policy that restricts resource location:
1. Go to **IAM & Admin > Organization Policies** in the GCP Console.
2. Search for the constraint **Resource Location Restriction** (`constraints/gcp.resourceLocations`).
3. Edit the policy to add the desired regions to the allowed list:
   * `us-central1`, `us-east1`, `us-east5`, `us-south1`, `europe-west1`
4. Ensure the TPU Service Agent (`service-1066673230465@gcp-sa-tpu.iam.gserviceaccount.com`) has the `roles/tpu.serviceAgent` role bound to it.

---

## 🔍 Telemetry Comparison: NVIDIA vs. Google TPU

We audited the previous NVIDIA GPU phase's telemetry metrics against the native capabilities of Google Cloud TPUs (v5e/v6e) polled via the `tpu-info` library:

| NVIDIA NVML Signal (1 Hz) | TPU v5e/v6e Availability (via `tpu-info`) | Resolution / Fallback Strategy |
| :--- | :--- | :--- |
| **GPU utilization (%)** | **Yes** (`duty_cycle_pct`) | Directly maps to core training active duty cycle. |
| **Memory utilization (%)** | **Yes** (Derived) | Calculate using `memory_usage / total_memory * 100`. |
| **Memory used (MB)** | **Yes** (`memory_usage`) | Directly maps to device HBM bytes used. |
| **Power draw (W)** | ❌ **No** | Excluded at the host level. We must map `power_draw_w` to `-1` (sentinel) in the dataset. |
| **Temperature (°C)** | **Yes** (`chip_temp` / `device_temp`) | Available on newer TPU nodes. |
| **SM clock (MHz)** | ❌ **No** | Not exposed via local host-level counter. |
| **Memory clock (MHz)** | ❌ **No** | Not exposed via local host-level counter. |
| **PCIe TX Bandwidth** | ❌ **No** | Not exposed at the host level (HBM integration is unified). |
| **PCIe RX Bandwidth** | ❌ **No** | Not exposed at the host level (HBM integration is unified). |

### Impact on Workload Classifier
Because key physical signals like **Power Draw (W)** and **Core Clocks** are missing on TPUs, Stage-1 classifiers (which rely on level metrics) will experience degraded accuracy. We must rely on Stage-2 temporal features (autocorrelation, periodicity, and memory usage slope over time) to distinguish training from inference.

### Exposing Excluded Telemetry
To obtain the missing physical counters, we should suggest that GCP:
1. Expose average power draw (Watts) at the chip level via the local Metadata Server.
2. Provide host-to-device interconnect bandwidth counters (analogous to PCIe TX/RX) via the JAX `jax.devices()` telemetry hooks.

---

## 💡 Discussion: TPU Model Taxonomy

Besides the generations tested (`v2`, `v3`, `v5e`, `v5p`, `v6e`), here is the status of other TPU architectures:

* **TPU v4 (v4-8 / v4-16 / etc.):** A highly relevant model released in 2021 utilizing a 3D torus topology. It is fully supported by Google Cloud but was blocked in your project due to lack of quota and location policies.
* **TPU v6p:** There is currently no `v6p` model. The `v6e` generation (Trillium) is Google's current flagship Trillium architecture.
* **TPU v7 (Ironwood):** General availability was announced on March 31, 2026. It is optimized for large-scale training and inference, supporting native FP8 precision and 192 GB HBM3E per chip.
* **TPU v8 (8t / 8i):** Announced at Google Cloud Next in April 2026. This generation splits the architecture into specialized hardware: TPU 8t is optimized for massive frontier-model pre-training, while TPU 8i is designed for low-latency, real-time inference (agentic AI).
* **TPU v1:** Google's initial custom ASIC (2016) which was internal-only and inference-only.
