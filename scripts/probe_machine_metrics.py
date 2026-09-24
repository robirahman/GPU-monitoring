#!/usr/bin/env python3
"""Exhaustive discovery of metrics this machine can expose for SPAR interconnect work.

Run this FIRST on every new host (RunPod / VCF / laptop). Writes:

  * ``<out>.json``  — machine-readable inventory (present / absent / sample values)
  * ``<out>.md``    — human summary for teammates

It does **not** require a workload. It probes NVML, DCGM, NVLink, IB sysfs (all
counter files), Ethernet, optional NCCL build info, and common CLI tools.

Usage::

    python scripts/probe_machine_metrics.py --out runs/hour1_metrics_probe
    python scripts/probe_machine_metrics.py --out runs/probe --sample-sec 3
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _run(cmd: list[str], timeout: float = 30) -> dict[str, Any]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {
            "cmd": cmd,
            "ok": r.returncode == 0,
            "returncode": r.returncode,
            "stdout": (r.stdout or "")[-8000:],
            "stderr": (r.stderr or "")[-2000:],
        }
    except FileNotFoundError:
        return {"cmd": cmd, "ok": False, "error": "not_found"}
    except subprocess.TimeoutExpired:
        return {"cmd": cmd, "ok": False, "error": "timeout"}


def probe_host() -> dict[str, Any]:
    uname = _run(["uname", "-a"])
    return {
        "hostname": socket.gethostname(),
        "fqdn": socket.getfqdn(),
        "utc_now": datetime.now(timezone.utc).isoformat(),
        "uname": uname.get("stdout", "").strip(),
        "python": sys.version,
        "cwd": os.getcwd(),
        "env_subset": {
            k: os.environ.get(k)
            for k in (
                "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES",
                "NNODES", "NODE_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
                "MASTER_ADDR", "MASTER_PORT", "NCCL_DEBUG",
                "SLURM_JOB_ID", "RUNPOD_POD_ID", "RUNPOD_DC_ID",
            )
            if os.environ.get(k)
        },
    }


def probe_nvml() -> dict[str, Any]:
    out: dict[str, Any] = {"available": False, "gpus": []}
    try:
        import pynvml
    except ImportError as e:
        out["error"] = f"pynvml_missing: {e}"
        return out
    try:
        pynvml.nvmlInit()
    except Exception as e:
        out["error"] = f"nvmlInit_failed: {e}"
        return out
    out["available"] = True
    try:
        out["driver_version"] = pynvml.nvmlSystemGetDriverVersion()
        try:
            cv = pynvml.nvmlSystemGetCudaDriverVersion_v2()
            out["cuda_driver_version"] = f"{cv // 1000}.{(cv % 1000) // 10}"
        except Exception:
            out["cuda_driver_version"] = None
        n = pynvml.nvmlDeviceGetCount()
        out["gpu_count"] = n
        # Field IDs used for NVLink throughput (SPAR interconnect_collect)
        try:
            # NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX/RX = 70-something depending on version;
            # we probe by trying GpuNvlinkSampler below and also list link count.
            out["nvml_field_values_api"] = hasattr(pynvml, "nvmlDeviceGetFieldValues")
        except Exception:
            out["nvml_field_values_api"] = False

        for i in range(n):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            g: dict[str, Any] = {"index": i}
            g["name"] = pynvml.nvmlDeviceGetName(h)
            if isinstance(g["name"], bytes):
                g["name"] = g["name"].decode()
            g["uuid"] = pynvml.nvmlDeviceGetUUID(h)
            if isinstance(g["uuid"], bytes):
                g["uuid"] = g["uuid"].decode()
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            g["mem_total_mb"] = mem.total // (1024 * 1024)
            g["mem_used_mb"] = mem.used // (1024 * 1024)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            g["sample_gpu_util_pct"] = util.gpu
            g["sample_mem_util_pct"] = util.memory
            g["sample_power_w"] = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
            g["sample_temp_c"] = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            g["sm_clock_mhz"] = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
            g["mem_clock_mhz"] = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM)
            for label, fn in (
                ("pcie_gen", lambda: pynvml.nvmlDeviceGetCurrPcieLinkGeneration(h)),
                ("pcie_width", lambda: pynvml.nvmlDeviceGetCurrPcieLinkWidth(h)),
                ("encoder_util", lambda: pynvml.nvmlDeviceGetEncoderUtilization(h)[0]),
                ("decoder_util", lambda: pynvml.nvmlDeviceGetDecoderUtilization(h)[0]),
                ("fan_pct", lambda: pynvml.nvmlDeviceGetFanSpeed(h)),
            ):
                try:
                    g[label] = fn()
                except Exception as e:
                    g[label] = None
                    g[f"{label}_error"] = str(e)[:120]
            # NVLink link count
            n_links = 0
            for link in range(12):
                try:
                    # NVML_NVLINK_STATE / capability probe
                    st = pynvml.nvmlDeviceGetNvLinkState(h, link)
                    if st:
                        n_links += 1
                except Exception:
                    break
            g["nvlink_links_up"] = n_links
            out["gpus"].append(g)
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
    return out


def probe_spar_nvlink(sample_sec: float) -> dict[str, Any]:
    """Use the same sampler SPAR telemetry uses."""
    try:
        from scripts.interconnect_collect import GpuNvlinkSampler
    except ImportError:
        try:
            from interconnect_collect import GpuNvlinkSampler  # type: ignore
        except ImportError as e:
            return {"available": False, "error": str(e)}
    results = []
    try:
        import pynvml
        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        pynvml.nvmlShutdown()
    except Exception as e:
        return {"available": False, "error": f"gpu_count: {e}"}
    for i in range(n):
        s = GpuNvlinkSampler(i)
        s.sample(0.01)  # prime
        time.sleep(max(0.2, sample_sec))
        row = s.sample(max(0.2, sample_sec))
        results.append({"gpu_index": i, **row})
    return {"available": any(r.get("nvlink_available") for r in results), "per_gpu": results}


def probe_dcgm() -> dict[str, Any]:
    out: dict[str, Any] = {"available": False}
    try:
        sys.path.insert(0, "/usr/local/dcgm/bindings/python3")
        import dcgm_structs  # type: ignore
        import pydcgm  # type: ignore
    except Exception as e:
        out["error"] = f"import_failed: {e}"
        out["lib_candidates"] = [
            p for p in (
                "/usr/lib/x86_64-linux-gnu/libdcgm.so",
                "/usr/lib/x86_64-linux-gnu/libdcgm.so.3",
            )
            if os.path.exists(p)
        ]
        return out
    out["available"] = True
    out["note"] = "DCGM import OK — SPAR TelemetryCollector will try profiling fields 1001-1008 + NVLink 1011/1012"
    out["spar_prof_fields"] = {
        1001: "dcgm_gr_engine_active",
        1002: "dcgm_sm_active",
        1003: "dcgm_sm_occupancy",
        1004: "dcgm_tensor_active",
        1005: "dcgm_dram_active",
        1006: "dcgm_fp64_pipe_active",
        1007: "dcgm_fp32_pipe_active",
        1008: "dcgm_fp16_pipe_active",
        1011: "dcgm_nvlink_tx_bytes",
        1012: "dcgm_nvlink_rx_bytes",
    }
    out["spar_basic_fallback"] = {
        203: "dcgm_gpu_util",
        155: "dcgm_power_usage",
        150: "dcgm_gpu_temp",
        100: "dcgm_sm_clock",
        101: "dcgm_mem_clock",
    }
    # Live one-GPU sample via SPAR collector if possible
    try:
        from scripts.collect_telemetry import DcgmCollector
        dc = DcgmCollector(0)
        sample = dc.sample()
        out["live_sample_keys"] = sorted(sample.keys())
        out["live_sample"] = {k: sample[k] for k in list(sample)[:20]}
        dc.cleanup()
        out["live_ok"] = True
    except Exception as e:
        out["live_ok"] = False
        out["live_error"] = str(e)[:300]
    return out


def probe_ib(root: str = "/sys/class/infiniband") -> dict[str, Any]:
    out: dict[str, Any] = {"available": False, "devices": {}, "all_counter_files": {}}
    if not os.path.isdir(root):
        out["error"] = f"{root} missing (common on eth-only RunPod)"
        return out
    devices = sorted(os.listdir(root))
    if not devices:
        out["error"] = "infiniband class empty"
        return out
    out["available"] = True
    for dev in devices:
        ports_root = os.path.join(root, dev, "ports")
        if not os.path.isdir(ports_root):
            continue
        out["devices"][dev] = {}
        for port in sorted(os.listdir(ports_root)):
            cdir = os.path.join(ports_root, port, "counters")
            cdir_ext = os.path.join(ports_root, port, "counters_ext")
            files = sorted(os.listdir(cdir)) if os.path.isdir(cdir) else []
            files_ext = sorted(os.listdir(cdir_ext)) if os.path.isdir(cdir_ext) else []
            samples = {}
            for fn in files:
                try:
                    samples[fn] = open(os.path.join(cdir, fn)).read().strip()
                except OSError:
                    samples[fn] = None
            out["devices"][dev][port] = {
                "counters": files,
                "counters_ext": files_ext,
                "sample_values": samples,
            }
            out["all_counter_files"][f"{dev}/{port}"] = {
                "counters": files,
                "counters_ext": files_ext,
            }
    # What SPAR actually reads today
    out["spar_reads_today"] = ["port_xmit_data", "port_rcv_data"]
    out["spar_note"] = (
        "SPAR IbDeltaSampler only consumes port_xmit_data + port_rcv_data "
        "(×4 octets, wrap-corrected). All other counter files above are available "
        "on this host but NOT yet fused into TelemetryCollector rows."
    )
    return out


def probe_eth(root: str = "/sys/class/net") -> dict[str, Any]:
    skip = ("lo", "docker", "veth", "br-", "virbr", "tun", "tap", "cni", "flannel", "cali")
    out: dict[str, Any] = {"available": False, "ifaces": {}}
    if not os.path.isdir(root):
        out["error"] = f"{root} missing"
        return out
    for iface in sorted(os.listdir(root)):
        if any(iface.startswith(p) for p in skip):
            continue
        sdir = os.path.join(root, iface, "statistics")
        if not os.path.isdir(sdir):
            continue
        files = sorted(os.listdir(sdir))
        samples = {}
        for fn in files:
            try:
                samples[fn] = open(os.path.join(sdir, fn)).read().strip()
            except OSError:
                samples[fn] = None
        out["ifaces"][iface] = {"statistics_files": files, "sample_values": samples}
    out["available"] = bool(out["ifaces"])
    out["spar_reads_today"] = ["rx_bytes", "tx_bytes"]
    out["spar_skips_prefixes"] = list(skip)
    return out


def probe_spar_fabric(sample_sec: float) -> dict[str, Any]:
    try:
        from scripts.interconnect_collect import SharedFabricCache, infer_topology
    except ImportError:
        try:
            from interconnect_collect import SharedFabricCache, infer_topology  # type: ignore
        except ImportError as e:
            return {"error": str(e)}
    topo = infer_topology()
    cache = SharedFabricCache(interval_sec=0.5)
    time.sleep(max(0.5, sample_sec))
    snap = cache.snapshot()
    summary = cache.summary()
    return {"topology": topo, "snapshot": snap, "summary_after_one_interval": summary}


def probe_torch_nccl() -> dict[str, Any]:
    out: dict[str, Any] = {"torch": None, "cuda": None, "nccl": None}
    try:
        import torch
        out["torch"] = torch.__version__
        out["cuda_compiled"] = torch.version.cuda
        out["cuda_available"] = torch.cuda.is_available()
        out["device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if torch.cuda.is_available():
            out["device0_name"] = torch.cuda.get_device_name(0)
        out["nccl_available"] = torch.distributed.is_nccl_available()
        try:
            out["nccl_version"] = ".".join(str(x) for x in torch.cuda.nccl.version())
        except Exception as e:
            out["nccl_version_error"] = str(e)[:120]
    except ImportError as e:
        out["error"] = str(e)
    return out


def probe_clis() -> dict[str, Any]:
    cmds = [
        ["nvidia-smi", "-L"],
        ["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,driver_version,pcie.link.gen.current,pcie.link.width.current", "--format=csv"],
        ["nvidia-smi", "nvlink", "-s"],
        ["ibstat"],
        ["ofed_info", "-s"],
        ["dcgmi", "discovery", "-l"],
        ["vllm", "--version"],
    ]
    return {c[0]: _run(c) for c in cmds}


def render_md(inv: dict[str, Any]) -> str:
    host = inv["host"]
    nvml = inv["nvml"]
    ib = inv["ib"]
    eth = inv["eth"]
    dcgm = inv["dcgm"]
    lines = [
        f"# Machine metrics probe — {host.get('hostname')}",
        "",
        f"- UTC: `{host.get('utc_now')}`",
        f"- uname: `{host.get('uname', '')[:120]}`",
        f"- env: `{json.dumps(host.get('env_subset') or {})}`",
        "",
        "## Verdict (what SPAR can use here)",
        "",
        f"| Signal | Present? |",
        f"|---|---|",
        f"| NVML Tier-1 | **{nvml.get('available')}** ({nvml.get('gpu_count', 0)} GPUs) |",
        f"| DCGM Tier-2 | **{dcgm.get('available')}** live_ok={dcgm.get('live_ok')} |",
        f"| NVLink (SPAR sampler) | **{inv['spar_nvlink'].get('available')}** |",
        f"| InfiniBand sysfs | **{ib.get('available')}** |",
        f"| Ethernet stats | **{eth.get('available')}** ({len(eth.get('ifaces') or {})} ifaces) |",
        f"| torch + NCCL | torch={inv['torch_nccl'].get('torch')} nccl={inv['torch_nccl'].get('nccl_version')} |",
        "",
        "## GPUs (NVML sample)",
        "",
    ]
    for g in nvml.get("gpus") or []:
        lines.append(
            f"- GPU{g['index']}: **{g.get('name')}** {g.get('mem_total_mb')} MiB total, "
            f"util={g.get('sample_gpu_util_pct')}% power={g.get('sample_power_w')}W "
            f"temp={g.get('sample_temp_c')}C nvlink_links_up={g.get('nvlink_links_up')}"
        )
    lines += ["", "## InfiniBand counter files (exhaustive on this host)", ""]
    if not ib.get("available"):
        lines.append(f"- ABSENT: `{ib.get('error')}`")
        lines.append("- Use **Ethernet + NVLink** for fabric signal on this machine.")
    else:
        lines.append(f"- SPAR currently reads only: `{ib.get('spar_reads_today')}`")
        for key, dirs in (ib.get("all_counter_files") or {}).items():
            lines.append(f"- `{key}` counters ({len(dirs.get('counters') or [])}): "
                         + ", ".join(dirs.get("counters") or [])[:500])
            if dirs.get("counters_ext"):
                lines.append(f"  - counters_ext: {', '.join(dirs['counters_ext'])}")
    lines += ["", "## Ethernet statistics files", ""]
    if not eth.get("available"):
        lines.append(f"- ABSENT: `{eth.get('error')}`")
    else:
        lines.append(f"- SPAR currently reads only: `{eth.get('spar_reads_today')}`")
        for iface, meta in (eth.get("ifaces") or {}).items():
            lines.append(f"- `{iface}`: {', '.join(meta.get('statistics_files') or [])}")
    lines += ["", "## DCGM", ""]
    if dcgm.get("available"):
        lines.append(f"- live_sample_keys: `{dcgm.get('live_sample_keys')}`")
    else:
        lines.append(f"- not usable: `{dcgm.get('error')}`")
    lines += [
        "",
        "## SPAR fused snapshot (one interval)",
        "",
        "```json",
        json.dumps(inv.get("spar_fabric", {}), indent=2, default=str)[:4000],
        "```",
        "",
        "## How to record time series for real",
        "",
        "Use `scripts/run_with_telemetry.py` (opens NVML+DCGM+fabric on all GPUs, runs your command).",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="runs/metrics_probe", help="output prefix (writes .json and .md)")
    ap.add_argument("--sample-sec", type=float, default=2.0, help="seconds between fabric/NVLink delta samples")
    ap.add_argument("--ib-root", default="/sys/class/infiniband")
    ap.add_argument("--net-root", default="/sys/class/net")
    args = ap.parse_args()

    print("Probing host / NVML / DCGM / IB / eth / SPAR fabric …", flush=True)
    inv: dict[str, Any] = {
        "host": probe_host(),
        "nvml": probe_nvml(),
        "spar_nvlink": probe_spar_nvlink(args.sample_sec),
        "dcgm": probe_dcgm(),
        "ib": probe_ib(args.ib_root),
        "eth": probe_eth(args.net_root),
        "spar_fabric": probe_spar_fabric(args.sample_sec),
        "torch_nccl": probe_torch_nccl(),
        "clis": probe_clis(),
        "spar_telemetry_row_fields_expected": {
            "tier1_nvml": [
                "timestamp_utc", "timestamp_epoch",
                "gpu_utilization_pct", "mem_utilization_pct",
                "mem_used_mb", "mem_total_mb", "power_draw_w", "temperature_c",
                "sm_clock_mhz", "mem_clock_mhz",
                "pcie_tx_mbps", "pcie_rx_mbps",
                "encoder_util_pct", "decoder_util_pct", "fan_speed_pct",
                "gpu_index", "workload_label", "run_id",
            ],
            "fabric": [
                "nvlink_available", "nvlink_n_links",
                "nvlink_tx_bytes", "nvlink_rx_bytes", "nvlink_tx_mbps", "nvlink_rx_mbps",
                "ib_available", "ib_n_ports", "ib_tx_bytes", "ib_rx_bytes",
                "ib_tx_mbps", "ib_rx_mbps", "ib_ports_json",
                "eth_available", "eth_n_ifaces", "eth_tx_bytes", "eth_rx_bytes",
                "eth_tx_mbps", "eth_rx_mbps",
                "fabric_dt_sec", "fabric_timestamp_epoch",
                "hostname", "nnodes", "node_rank", "nproc_per_node", "world_size", "job_id",
            ],
            "dcgm_if_present": [
                "dcgm_gr_engine_active", "dcgm_sm_active", "dcgm_sm_occupancy",
                "dcgm_tensor_active", "dcgm_dram_active",
                "dcgm_fp64_pipe_active", "dcgm_fp32_pipe_active", "dcgm_fp16_pipe_active",
                "dcgm_nvlink_tx_bytes", "dcgm_nvlink_rx_bytes",
            ],
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json_path = out.with_suffix(".json") if out.suffix != ".json" else out
    md_path = Path(str(json_path).removesuffix(".json") + ".md")
    # normalize: --out runs/foo → runs/foo.json + runs/foo.md
    stem = Path(args.out)
    json_path = stem if stem.suffix == ".json" else Path(str(stem) + ".json")
    md_path = Path(str(json_path).removesuffix(".json") + ".md")

    with open(json_path, "w") as f:
        json.dump(inv, f, indent=2, default=str)
    md_path.write_text(render_md(inv))
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    # Console one-liner verdict
    print(
        "VERDICT:",
        f"gpus={inv['nvml'].get('gpu_count')}",
        f"dcgm={inv['dcgm'].get('available')}",
        f"nvlink={inv['spar_nvlink'].get('available')}",
        f"ib={inv['ib'].get('available')}",
        f"eth={inv['eth'].get('available')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
