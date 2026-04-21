#!/usr/bin/env python3
"""B200 Phase B orchestrator: 15 inference workloads × 3 reps with JIT model
downloads and careful deletions so peak disk stays below 230 GB.

Ordering rationale (see chat transcript for full disk trace):
  - Run all 3 reps of a workload consecutively so we can delete the model
    immediately after.
  - Run models that share weights (qwen3b ×3 variants, tinyllama ×2 variants)
    in adjacent slots.
  - Delete sdxl (72 GB) and qwen32b (62 GB) early to free headroom for the
    three big JIT downloads (AWQ 45 / mixtral 90 / qwen72b_fp16 144).
  - Chain JIT downloads so next model's download overlaps with current
    workload's 15-min runtime.

Waits for run_b200_ml.sh (Phase A / training) to exit before starting.
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

REPO = Path("/workspace/SPAR-GPU-monitoring")
DATA_DIR = REPO / "data"
LOG = REPO / "logs" / f"b200_phase_b_{time.strftime('%Y%m%d_%H%M%S')}.log"
HF_HOME = Path(os.environ.setdefault("HF_HOME", "/workspace/.hf_home"))
HF_HUB = HF_HOME / "hub"
os.environ["PYTHONPATH"] = (
    "/usr/share/datacenter-gpu-manager-4/bindings/python3:"
    + os.environ.get("PYTHONPATH", "")
)


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def disk_free_gb(path: str = "/workspace") -> float:
    s = shutil.disk_usage(path)
    return s.free / 1e9


def model_dir(repo_id: str) -> Path:
    safe = "models--" + repo_id.replace("/", "--")
    return HF_HUB / safe


def has_model(repo_id: str) -> bool:
    d = model_dir(repo_id)
    if not d.exists():
        return False
    # heuristic: has at least one snapshot with .safetensors or .bin weights
    snaps = d / "snapshots"
    if not snaps.exists():
        return False
    for snap in snaps.iterdir():
        for p in snap.rglob("*"):
            if p.suffix in (".safetensors", ".bin", ".gguf", ".pt", ".pth"):
                return True
    return False


def delete_model(repo_id: str) -> None:
    d = model_dir(repo_id)
    if d.exists():
        size_gb = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e9
        log(f"DELETE {repo_id} ({size_gb:.1f} GB)")
        shutil.rmtree(d)
    # also clean up any blob/download symlinks that may remain
    for pat in ("blobs", "refs"):
        p = d / pat
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)


class Downloader:
    """Background snapshot_download with cancellation support."""

    def __init__(self, repo_id: str):
        self.repo_id = repo_id
        self.thread: Optional[threading.Thread] = None
        self.done = threading.Event()
        self.error: Optional[Exception] = None

    def _run(self):
        try:
            from huggingface_hub import snapshot_download
            token = os.environ.get("HF_TOKEN")
            kwargs = {"token": token} if token else {}
            log(f"DL start {self.repo_id} (free={disk_free_gb():.1f} GB)")
            path = snapshot_download(repo_id=self.repo_id, **kwargs)
            log(f"DL done  {self.repo_id} → {path} (free={disk_free_gb():.1f} GB)")
        except Exception as e:
            self.error = e
            log(f"DL FAIL  {self.repo_id}: {type(e).__name__}: {e}")
        finally:
            self.done.set()

    def start(self) -> None:
        if has_model(self.repo_id):
            log(f"DL skip  {self.repo_id} (already cached)")
            self.done.set()
            return
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def wait(self) -> bool:
        self.done.wait()
        return self.error is None


def run_workload(label: str) -> int:
    """Run scripts/run_workload.py for the given registry label. Streams output."""
    cmd = [
        sys.executable,
        str(REPO / "scripts" / "run_workload.py"),
        "--workload", label,
        "--output-dir", "data/",
    ]
    log(f"RUN start {label}")
    t0 = time.time()
    with LOG.open("a") as f:
        proc = subprocess.run(cmd, cwd=str(REPO), stdout=f, stderr=subprocess.STDOUT)
    log(f"RUN end   {label} exit={proc.returncode} ({time.time()-t0:.0f}s)")
    return proc.returncode


@dataclass
class Slot:
    """One scheduled group of workloads sharing a model.

    All reps of all workloads in `workloads` run back-to-back; the slot holds
    the referenced model on disk. `prefetch_next` downloads are started at slot
    entry so they overlap with the runs. `delete_after` models are removed
    after the last rep completes.
    """
    name: str  # human-readable label for logs
    workloads: list[str] = field(default_factory=list)  # registry labels
    reps: int = 3
    prefetch: list[str] = field(default_factory=list)  # repo_ids to DL on entry
    delete_after: list[str] = field(default_factory=list)  # repo_ids to rm on exit
    wait_for_dl: list[str] = field(default_factory=list)  # block until DL complete before any run


# Model repo IDs (HF) for each inference workload's weights
MODEL = {
    "phi3": "microsoft/Phi-3-mini-4k-instruct",
    "qwen3b": "Qwen/Qwen2.5-3B-Instruct",
    "mistral_q4": "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ",
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "vit_l": "google/vit-large-patch16-224",
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen32b": "Qwen/Qwen2.5-32B-Instruct",
    "qwen72b_awq": "Qwen/Qwen2.5-72B-Instruct-AWQ",
    "qwen72b_fp16": "Qwen/Qwen2.5-72B-Instruct",
    "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
}


# JIT-ordered schedule. Reorder rationale:
#   1-7: small models, delete after → frees ~25 GB total
#   8: sdxl early → frees 72 GB (!)
#   9: vit_l
#   10-12: qwen 7B, 14B, 32B — chain DL of AWQ and mixtral during 14B/32B runs
#   13-15: JIT big — AWQ → mixtral → 72B_fp16 (72B downloaded only after mixtral
#          deleted; unavoidable 5 min critical-path wait for 144 GB)
SCHEDULE: list[Slot] = [
    Slot("01 phi3_mini", ["llm_infer_phi3_mini"], delete_after=[MODEL["phi3"]]),
    Slot("02 qwen3b ×3",
         ["llm_infer_qwen25_3b", "llm_infer_qwen25_3b_bs4", "llm_infer_qwen25_3b_bs16"],
         delete_after=[MODEL["qwen3b"]]),
    Slot("03 mistral_q4", ["llm_infer_mistral_7b_q4"], delete_after=[MODEL["mistral_q4"]]),
    Slot("04 tinyllama ×2",
         ["llm_infer_tinyllama_bs8", "llm_infer_tinyllama_seq4096"],
         delete_after=[MODEL["tinyllama"]]),
    Slot("05 sdxl (-72 GB)", ["llm_infer_sdxl"], delete_after=[MODEL["sdxl"]]),
    Slot("06 vit_l", ["llm_infer_vit_l_bs64"], delete_after=[MODEL["vit_l"]]),
    Slot("07 qwen7b", ["llm_infer_qwen25_7b_fp16"], delete_after=[MODEL["qwen7b"]]),
    # Start DLing AWQ during qwen14b run
    Slot("08 qwen14b + DL(AWQ)",
         ["llm_infer_qwen25_14b_fp16"],
         prefetch=[MODEL["qwen72b_awq"]],
         delete_after=[MODEL["qwen14b"]]),
    # Start DLing mixtral during qwen32b run (need auth for gated repo)
    Slot("09 qwen32b + DL(mixtral)",
         ["llm_infer_qwen25_32b_fp16"],
         prefetch=[MODEL["mixtral"]],
         delete_after=[MODEL["qwen32b"]]),
    Slot("10 qwen72b_fp8 (AWQ)",
         ["llm_infer_qwen25_72b_fp8"],
         wait_for_dl=[MODEL["qwen72b_awq"]],
         delete_after=[MODEL["qwen72b_awq"]]),
    Slot("11 mixtral",
         ["llm_infer_mixtral_8x7b"],
         wait_for_dl=[MODEL["mixtral"]],
         delete_after=[MODEL["mixtral"]]),
    # qwen72b_fp16 (144 GB) only fits after mixtral deleted. 5 min critical
    # path DL is unavoidable here — no earlier slot with enough headroom.
    Slot("12 qwen72b_fp16",
         ["llm_infer_qwen25_72b_fp16"],
         prefetch=[MODEL["qwen72b_fp16"]],
         wait_for_dl=[MODEL["qwen72b_fp16"]],
         delete_after=[MODEL["qwen72b_fp16"]]),
]


def wait_for_phase_a() -> None:
    """Block until run_b200_ml.sh exits."""
    log("Waiting for run_b200_ml.sh (Phase A) to finish...")
    while True:
        out = subprocess.run(
            ["pgrep", "-f", "run_b200_ml.sh"], capture_output=True, text=True
        )
        if not out.stdout.strip():
            log("Phase A finished. Starting Phase B.")
            return
        time.sleep(30)


def main() -> int:
    log(f"=== Phase B orchestrator starting ===")
    log(f"HF_HOME={HF_HOME}  HF_TOKEN={'set' if os.environ.get('HF_TOKEN') else 'NOT SET'}")
    log(f"Initial free: {disk_free_gb():.1f} GB")

    wait_for_phase_a()
    log(f"After Phase A: {disk_free_gb():.1f} GB free")

    downloaders: dict[str, Downloader] = {}

    for i, slot in enumerate(SCHEDULE, 1):
        log(f"=== Slot {slot.name} ===")

        # Kick off prefetch downloads now (run in background during slot workloads)
        for repo_id in slot.prefetch:
            if repo_id not in downloaders:
                dl = Downloader(repo_id)
                dl.start()
                downloaders[repo_id] = dl

        # If slot needs a model that's being downloaded, block here
        for repo_id in slot.wait_for_dl:
            if repo_id in downloaders:
                log(f"Waiting for {repo_id} download to complete...")
                if not downloaders[repo_id].wait():
                    log(f"ERROR: DL of {repo_id} failed; skipping slot {slot.name}")
                    # Don't delete in this case
                    for d in slot.delete_after:
                        if d == repo_id:
                            continue
                    continue
            elif not has_model(repo_id):
                # Not downloading, not cached — start DL now and wait
                dl = Downloader(repo_id)
                dl.start()
                downloaders[repo_id] = dl
                if not dl.wait():
                    log(f"ERROR: DL of {repo_id} failed; skipping slot")
                    continue

        # Run workloads (3 reps each)
        for wl in slot.workloads:
            for rep in range(1, slot.reps + 1):
                log(f"  {slot.name} :: {wl} rep {rep}/{slot.reps}")
                rc = run_workload(wl)
                if rc != 0:
                    log(f"  non-zero exit ({rc}); continuing")

        # Delete models
        for repo_id in slot.delete_after:
            delete_model(repo_id)
            log(f"After delete: {disk_free_gb():.1f} GB free")

    log("=== Phase B orchestrator done ===")
    return 0


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    sys.exit(main())
