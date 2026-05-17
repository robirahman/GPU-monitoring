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

| Signal | NVIDIA NVML | AMD ROCm amdsmi | Google TPU `tpu_info` | AWS Trainium `neuron-monitor` |
|---|---|---|---|---|
| GPU utilization (%)    | ✓ `nvmlDeviceGetUtilizationRates().gpu` | ✓ `gfx_activity` | TODO | TODO |
| Memory utilization (%) | ✓ `nvmlDeviceGetUtilizationRates().memory` | ✓ `umc_activity` | TODO | TODO |
| Memory used (MB)       | ✓ `nvmlDeviceGetMemoryInfo().used` | ✓ `amdsmi_get_gpu_memory_usage` | TODO | TODO |
| Power draw (W)         | ✓ `nvmlDeviceGetPowerUsage` | ✓ `current_socket_power` (or `socket_power`/`average_socket_power` fallback) | TODO | TODO |
| Temperature (°C)       | ✓ `nvmlDeviceGetTemperature(GPU)` | ✓ `temperature_edge` (also `_hotspot`, `_vram`) | TODO | TODO |
| SM clock (MHz)         | ✓ `nvmlDeviceGetClockInfo(SM)` | ✓ `amdsmi_get_clock_info(GFX).clk` | TODO | TODO |
| Memory clock (MHz)     | ✓ `nvmlDeviceGetClockInfo(MEM)` | ✓ `amdsmi_get_clock_info(MEM).clk` | TODO | TODO |
| PCIe TX (MB/s)         | ✓ `nvmlDeviceGetPcieThroughput(TX)` | ✗ not exposed on consumer cards (`-1` sentinel) | TODO | TODO |
| PCIe RX (MB/s)         | ✓ `nvmlDeviceGetPcieThroughput(RX)` | ✗ not exposed on consumer cards (`-1` sentinel) | TODO | TODO |

Additional signals available on some vendors but not in the 9-signal core set:

| Extra signal | NVIDIA | AMD | TPU | Trainium |
|---|---|---|---|---|
| Encoder utilization (%)   | ✓ NVENC engine % | ~ `mm_activity` (bundled enc+dec) | TODO | TODO |
| Decoder utilization (%)   | ✓ NVDEC engine % | ~ same as encoder | TODO | TODO |
| Hotspot / junction temp (°C) | ✗ | ✓ `temperature_hotspot` | TODO | TODO |
| VRAM / HBM temp (°C)      | ✗ (some H100/H200/B200 expose it) | ✓ `temperature_vram` | TODO | TODO |
| Fan speed (RPM or %)      | ✓ (consumer cards) | ✓ (consumer cards, `-1` on datacenter) | TODO | TODO |
| Power limit (W)           | ✓ | ✓ `power_limit` | TODO | TODO |
| Throttling reasons / status | ✓ NVML throttle reason bitmask | ✓ `throttle_status` (boolean) | TODO | TODO |
| TensorCore active (%)     | ✓ via DCGM (`SM_OCCUPANCY`, `TENSOR_ACTIVE`) | ✓ via RDC | TODO | TODO |

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
