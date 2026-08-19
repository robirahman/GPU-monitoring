"""TPU workload registry -- label -> {command, description, default_timeout}.

Same shape as workloads/registry.py, but every label follows the cross-cutting
label grammar from the plan (section 2.2):

    {scope}_{class}_{detail}[_rep{N}]
      scope in {tpuT1, tpuT4, tpuT8, tpuT16, tpuTMS}
      class in {train, infershard, inferrep, other, idle}

The SCOPE encodes the slice size (tpuT4 = v6e-4 / 1 host, tpuT8 = v6e-8 / 2
hosts, tpuT16 = v6e-16 / 4 hosts). The workload COMMAND is the same across
scopes -- the cell reads jax.devices(), which reflects whatever slice it was
provisioned on, and `--multihost auto` brings up the cross-host cluster only
when the slice spans >1 host. So one command definition serves the whole ladder;
the scope in the label (and the slice the queue targets) is what varies.

Routing check (threeway_improved.threeway_label, zero taxonomy edits):
    *_train_*      -> ml_training   ("train" keyword)
    *_infershard_* -> ml_inference  ("infer" keyword; no train/fsdp/ddp/bert/gpt)
    *_inferrep_*   -> ml_inference  ("infer" keyword)
    *_other_*      -> other
    *_idle         -> other
The per-chip suffix _w{W}c{C} (added by the collector filename) is stripped by
threeway_label before matching, so streams route on the base label.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WORKLOADS_DIR = os.path.join(_REPO_ROOT, "workloads")
_PY = sys.executable

# Ladder of communication scopes. tpuT1 (degenerate, no collectives) and tpuTMS
# (2x v6e-8 multislice, guarantees DCN metrics) are available for the stretch
# rungs; the core ladder the plan prioritises is T4 -> T8 -> T16.
_SCOPES = ["tpuT4", "tpuT8", "tpuT16"]

# (default) per-cell durations and timeouts (seconds). XLA's first-step compile
# can take minutes, so timeouts carry generous headroom over --duration.
_DUR = 600
_TO_TRAIN = 1500      # 7b compile + 600s run
_TO_INFER = 1500
_TO_OTHER = 1000
_TO_IDLE = 400


def _train_cmd(size, dur=_DUR, batch=None, seq=1024):
    cmd = [_PY, os.path.join(_WORKLOADS_DIR, "tpu_fsdp_training.py"),
           "--size", size, "--duration", str(dur),
           "--optimizer", "adamw", "--seq-len", str(seq),
           "--multihost", "auto", "--remat"]
    if batch is not None:
        cmd += ["--batch-size", str(batch)]
    return cmd


def _infer_cmd(size, mode, dur=_DUR):
    return [_PY, os.path.join(_WORKLOADS_DIR, "tpu_sharded_inference.py"),
            "--size", size, "--mode", mode, "--duration", str(dur),
            "--batch-size", "8", "--prompt-len", "128", "--gen-tokens", "64"]


def _other_cmd(dur=_DUR):
    return [_PY, os.path.join(_WORKLOADS_DIR, "tpu_jax_other.py"),
            "--mode", "fft", "--sharded", "--duration", str(dur)]


def _idle_cmd(dur=300):
    return [_PY, os.path.join(_WORKLOADS_DIR, "idle.py"), "--duration", str(dur)]


WORKLOAD_REGISTRY_TPU: dict[str, dict] = {}


def _add(label, command, description, timeout):
    WORKLOAD_REGISTRY_TPU[label] = {
        "command": command,
        "description": description,
        "default_timeout": timeout,
    }


# ── Core ladder x {train_fsdp7b, infershard_7b, inferrep_7b, other_fft, idle} ─
for _sc in _SCOPES:
    _add(f"{_sc}_train_fsdp7b", _train_cmd("7b"),
         f"FSDP-style 7B JAX transformer training on {_sc} (gradient collectives)",
         _TO_TRAIN)
    _add(f"{_sc}_infershard_7b", _infer_cmd("7b", "tp"),
         f"7B sharded (tensor-parallel) serving on {_sc} (cross-host activation psums)",
         _TO_INFER)
    _add(f"{_sc}_inferrep_7b", _infer_cmd("7b", "replicated"),
         f"7B replicated serving on {_sc} (per-host replicas, no cross-host sync)",
         _TO_INFER)
    _add(f"{_sc}_other_fft", _other_cmd(),
         f"Sharded 2-D FFT on {_sc} (comm-heavy non-ML guard)",
         _TO_OTHER)
    _add(f"{_sc}_idle", _idle_cmd(),
         f"Idle baseline on {_sc}",
         _TO_IDLE)

# ── 1b size across the ladder (small-model comparison point) ─────────────────
for _sc in _SCOPES:
    _add(f"{_sc}_train_fsdp1b", _train_cmd("1b"),
         f"FSDP-style 1B JAX transformer training on {_sc}",
         _TO_TRAIN)
    _add(f"{_sc}_infershard_1b", _infer_cmd("1b", "tp"),
         f"1B sharded serving on {_sc} (cross-host activation psums)",
         _TO_INFER)
    _add(f"{_sc}_inferrep_1b", _infer_cmd("1b", "replicated"),
         f"1B replicated serving on {_sc} (no cross-host sync)",
         _TO_INFER)

# ── tpuT16 30B stretch (largest that fits the 4-host slice comfortably) ──────
_add("tpuT16_train_fsdp30b",
     _train_cmd("30b", batch=1, seq=512),
     "FSDP-style 30B JAX transformer training on tpuT16, bs=1 seq=512 (stretch)",
     1800)
_add("tpuT16_infershard_30b",
     _infer_cmd("30b", "tp"),
     "30B sharded serving on tpuT16 (cross-host activation psums, stretch)",
     1800)


def get_tpu_workload(label: str) -> dict:
    """Look up a TPU workload by label. Raises ValueError if not found."""
    if label not in WORKLOAD_REGISTRY_TPU:
        available = ", ".join(sorted(WORKLOAD_REGISTRY_TPU.keys()))
        raise ValueError(f"Unknown TPU workload '{label}'. Available: {available}")
    return WORKLOAD_REGISTRY_TPU[label]


def list_tpu_workloads() -> list[str]:
    return sorted(WORKLOAD_REGISTRY_TPU.keys())


if __name__ == "__main__":
    for _label in list_tpu_workloads():
        print(f"  {_label:28s} {WORKLOAD_REGISTRY_TPU[_label]['description']}")
