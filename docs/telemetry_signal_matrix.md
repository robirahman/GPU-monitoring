# Telemetry Signal Availability Across Accelerators

This document tracks which of the paper's 9 always-available telemetry
signals each accelerator architecture exposes. It is the deliverable for
**Phase 2** of the cross-architecture extension plan
(`~/.claude/plans/what-else-should-we-soft-kahn.md`) — the table is filled
in incrementally as we probe each platform.

The 9 signals come from §4.1.3 of the paper:

1. GPU utilization (%)
2. Memory utilization (%)
3. Memory used (MB)
4. Power draw (W)
5. Temperature (°C)
6. SM clock (MHz)  *(streaming-multiprocessor / GFX / tensor-core core clock)*
7. Memory clock (MHz)
8. PCIe TX bandwidth (MB/s)
9. PCIe RX bandwidth (MB/s)

## Availability matrix

Filled-in cells: ✓ = signal exists with the same semantics as NVML;
~ = exists but with materially different semantics (e.g. bundled / averaged /
named differently); ✗ = not exposed at 1 Hz polling rate.

| Signal | NVIDIA NVML | AMD ROCm amdsmi | Google TPU (libtpu / `tpu_info`) | AWS Trainium `neuron-monitor` |
|---|---|---|---|---|
| GPU utilization (%)    | ✓ `nvmlDeviceGetUtilizationRates().gpu` | ✓ `gfx_activity` | ✓ `duty_cycle_pct` → `gpu_utilization_pct` (0.0 when dark) | TODO |
| Memory utilization (%) | ✓ `nvmlDeviceGetUtilizationRates().memory` | ✓ `umc_activity` | ~ HBM **bandwidth** util → `mem_utilization_pct` (different semantics: BW, not occupancy) | TODO |
| Memory used (MB)       | ✓ `nvmlDeviceGetMemoryInfo().used` | ✓ `amdsmi_get_gpu_memory_usage` | ✓ HBM `hbm_capacity_usage` → `mem_used_mb` | TODO |
| Power draw (W)         | ✓ `nvmlDeviceGetPowerUsage` | ✓ `current_socket_power` (or `socket_power`/`average_socket_power` fallback) | ✗ not exposed at host level → **`-1` constant** | TODO |
| Temperature (°C)       | ✓ `nvmlDeviceGetTemperature(GPU)` | ✓ `temperature_edge` (also `_hotspot`, `_vram`) | ✗ not exposed to the collector → **`-1` constant** | TODO |
| SM clock (MHz)         | ✓ `nvmlDeviceGetClockInfo(SM)` | ✓ `amdsmi_get_clock_info(GFX).clk` | ✗ → **`-1` constant** | TODO |
| Memory clock (MHz)     | ✓ `nvmlDeviceGetClockInfo(MEM)` | ✓ `amdsmi_get_clock_info(MEM).clk` | ✗ → **`-1` constant** | TODO |
| PCIe TX (MB/s)         | ✓ `nvmlDeviceGetPcieThroughput(TX)` | ✗ not exposed on consumer cards (`-1` sentinel) | ✗ unified HBM, no host PCIe counter → **`-1` constant** | TODO |
| PCIe RX (MB/s)         | ✓ `nvmlDeviceGetPcieThroughput(RX)` | ✗ not exposed on consumer cards (`-1` sentinel) | ✗ → **`-1` constant** | TODO |

Additional signals available on some vendors but not in the 9-signal core set:

| Extra signal | NVIDIA | AMD | TPU | Trainium |
|---|---|---|---|---|
| Encoder utilization (%)   | ✓ NVENC engine % | ~ `mm_activity` (bundled enc+dec) | ✗ → `-1` | TODO |
| Decoder utilization (%)   | ✓ NVDEC engine % | ~ same as encoder | ✗ → `-1` | TODO |
| Hotspot / junction temp (°C) | ✗ | ✓ `temperature_hotspot` | ✗ | TODO |
| VRAM / HBM temp (°C)      | ✗ (some H100/H200/B200 expose it) | ✓ `temperature_vram` | ✗ | TODO |
| Fan speed (RPM or %)      | ✓ (consumer cards) | ✓ (consumer cards, `-1` on datacenter) | ✗ → `-1` | TODO |
| Power limit (W)           | ✓ | ✓ `power_limit` | ✗ | TODO |
| Throttling reasons / status | ✓ NVML throttle reason bitmask | ✓ `throttle_status` (boolean) | ✗ | TODO |
| TensorCore active (%)     | ✓ via DCGM (`SM_OCCUPANCY`, `TENSOR_ACTIVE`) | ✓ via RDC | ✓ `tensorcore_util` → `tensorcore_util_pct` (libtpu monitoring) | TODO |

### TPU-native communication signals (no NVIDIA equivalent)

The TPU exposes an inter-server communication channel NVML/DCGM does not, via
`libtpu.sdk.tpumonitoring` (`list_supported_metrics()` / `get_metric(n).data()`).
These become extra numeric columns (percentile-scalarised) that
`get_feature_cols` auto-picks; they are `-1` when the metric is unsupported/dark:

| TPU-native signal | Column(s) | Notes |
|---|---|---|
| HLO execution timing | `hlo_exec_us_p50/p90/p99` | **Always-on** comm channel — collective time is inside HLO exec, so this fires even inside a single slice where DCN metrics stay dark. |
| HLO queue depth | `hlo_queue_size` | Scheduler backpressure. |
| Collective / DCN latency | `dcn_collective_lat_us_p50/p90/p99` | Cross-**host** collective latency; guaranteed to fire on the `tpuTMS` (2× v6e-8 multislice) rung, may be dark within one slice (collectives ride ICI). |
| Buffer transfer latency | `buffer_xfer_lat_us_p90` | Host↔device / inter-chip buffer moves. |
| HBM bandwidth util | `hbm_bw_util_pct` | Native duplicate of `mem_utilization_pct` but `-1` (not 0.0) when dark. |
| Duty cycle | `tpu_duty_cycle_pct` | Native duplicate of `gpu_utilization_pct` but `-1` (not 0.0) when dark. |
| Metric liveness | `tpu_metrics_live` (0/1) | 1 when any device metric read this sample — lets the analysis filter dark rows. |

Power/temp/clock/PCIe are hard-wired to `-1` (fleet-internal; matches the
AMD-port sentinel precedent), so TPU Stage-1 (level features) is expected to be
weak — the TPU rung leans on Stage-2 shape features plus the comm channel above.

## Phase 2 — probe procedure

### Google TPU v6e (Trillium) on GCP

```bash
# After provisioning a TPU v6e-1 in europe-west4-a (cheapest single-chip):
pip install tpu-info
python -c "
from tpu_info import metrics
m = metrics.get_chip_usage()
print(m)  # Should print a dict-like structure
"
```

Then run `scripts/collect_tpu_telemetry.py` from `feat/tpu-adaptation`
branch for 30 seconds and inspect the parquet schema. Fill in the TODO
column above.

### AWS Trainium2 on AWS

```bash
# After provisioning a trn1.2xlarge (cheapest Trainium with 2 chips):
neuron-monitor --json | python -m json.tool | head -100
# Or programmatically:
python -c "
import subprocess, json
out = subprocess.run(['neuron-monitor', '-c', '/etc/neuron-monitor.conf'],
                    capture_output=True, text=True, timeout=10)
print(json.loads(out.stdout)['neuron_runtime_data'][0])
"
```

Fill in the Trainium column above.

## Architecture decision (TBD — depends on probe results)

Two candidate designs:

**(a) Base-class abstraction (`scripts/collect_telemetry_base.py`).** Common
interface (`start/stop/save/sample`) with vendor subclasses. Cleaner long-term;
requires refactoring the existing two collectors (`collect_telemetry.py` for
NVIDIA, `collect_amdsmi_telemetry.py` for AMD). Worth doing only if TPU and
Trainium expose ≥6 of the 9 core signals and have similar polling-thread
ergonomics.

**(b) Standalone per-vendor collectors + downstream normalizer.** Keep
existing collectors untouched; add `collect_tpu_telemetry.py` and
`collect_trainium_telemetry.py` as siblings. Each writes a parquet with
whatever signals it can. The cross-architecture classifier handles
schema differences in `classifier/threeway_improved.py`'s `_normalize_columns`
function (which already gracefully handles missing PCIe on AMD). Lower
upfront cost, more duplication.

**Likely decision (pre-probe):** (b), because:
- AMD already taught us that not every signal exists on every vendor
  (PCIe missing on consumer AMD).
- TPU and Trainium have fundamentally different compute models (dataflow
  vs warp-scheduled), so even shared signals may have different semantics.
- The classifier already tolerates `-1`/`NaN` sentinels via the
  normalizer.

The probe results will confirm or contradict this. Fill in the matrix,
then write a 1-paragraph decision here.

## Once decided

- If (a): refactor `collect_telemetry.py` + `collect_amdsmi_telemetry.py`
  into `scripts/collect_telemetry_base.py` + vendor subclasses.
- If (b): write `scripts/collect_tpu_telemetry.py` (extend from
  `feat/tpu-adaptation`'s scaffold) and `scripts/collect_trainium_telemetry.py`
  from scratch. Both write the same 26-column parquet schema as the AMD
  collector, with un-applicable signals set to `-1`.
