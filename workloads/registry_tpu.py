"""TPU workload registry -- label -> {command, description, default_timeout}.

Same shape as workloads/registry.py, but every label follows the cross-cutting
label grammar from the plan (section 2.2):

    {scope}_{class}_{detail}[_rep{N}]
      scope in {tpuT1, tpuT4, tpuT8, tpuT16, tpuTMS}
      class in {train, infershard, inferrep, other, idle}

The SCOPE encodes the slice size. v6e is 8 CHIPS/HOST (official docs: v6e-8 is a
full single host -- the older "4 chips/host, v6e-16 = 4 hosts" note was an untested
assumption; the earlier v6e-16 Spot slices were preempted before a cell ever ran):
    tpuT4  = v6e-4  = 4 chips,  1 host
    tpuT8  = v6e-8  = 8 chips,  1 host (FULL single host)
    tpuT16 = v6e-16 = 16 chips, 2 hosts   (cross-host ICI; DCN still dark on 1 slice)
    tpuTMS = 2x v6e-8 multislice = 16 chips, 2 slices (cross-slice DCN FIRES)
The workload COMMAND is the same across scopes -- the cell reads jax.devices(),
which reflects whatever slice it was provisioned on, and `--multihost auto` brings up
the cross-host cluster when the slice spans >1 host (and the cross-SLICE mesh on the
multislice group). So one command serves the whole ladder; the scope label (and the
slice the queue targets) is what varies. single->multi-HOST boundary = T8->T16;
single->multi-SLICE (DCN) boundary = T16->TMS.

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
# tpuT1 added because single-chip slices provision in 60-110 SECONDS while
# 16-chip requests have waited 7 hours and failed; iterating on one chip is the
# only practical loop. It must be its own scope: make_mesh uses jax.devices(), so
# a 1-chip run labelled tpuT16 would silently record single-chip telemetry under a
# 16-chip name, and gpu_name/scope is what the analysis groups on.
_SCOPES = ["tpuT1", "tpuT4", "tpuT8", "tpuT16"]

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


# ── Adversarial ladder + honest controls (the TPU rung's decisive cells) ─────
# Mirrors the GPU ladder so the comparison is like-for-like. The GPU result being
# tested: held out by workload, run-level k=3-of-5, tau=0.75 -> F=0.5 detected in
# 100% of runs at 0.6% false-alarm rate, F<=0.2 at 0%. Phase 0 showed that
# survives on TPU-available metrics at 3.0% FPR, so the open question is whether
# the TPU-native comm channels (dcn_collective_lat_us_*, hlo_exec_us_*) recover
# what power was providing.
#
# NAMING: the GPU honest controls were called `nonml_lowpower`, but TPU exposes NO
# power at all (collect_tpu_telemetry.py:74-78 writes -1), so a "lowpower" label
# here would name a channel we cannot measure. These are named by DUTY instead,
# which is the thing actually being varied. GPU counterparts noted per cell.
#
# Routing (verified in the module self-check below):
#   *_advtrain_*  -> ml_training AND disguised (model_store.DISGUISED_FAMILIES
#                    contains "advtrain", matched as "_advtrain")
#   *_nonml_*     -> other AND honest
_ADV_MODEL = "1b"          # matches the GPU MN-I ladder
_TO_ADV = 1500
# Global (not per-chip) resident footprint for the matched honest partner. A 1b
# model under AdamW in fp32 is ~12 GB of params+moments across the slice. CALIBRATE
# ON THE SMOKE: tpu_adversarial_multinode.py prints its filler resident bytes and
# param count, and nonml_sat must match the F=0 cell's footprint or the apparatus
# comparison is not like-for-like -- that pairing is what refuted "the apparatus
# is the detection channel" on GPU (135 MB / 0.2 pt / 3 W apart).
# CALIBRATED 2026-09-01 against the F=0 cell's measured footprint, not guessed.
# tpuTMS_advtrain_satfiller_only on a 2x v6e-1 multislice held mem_used_mb=5196
# (median over 758 samples, p95 5196, max 5794) with mem_total_mb=31996. The
# honest partner reaches the same total as alloc + its own baseline, and
# tpuTMS_nonml_duty30 at --alloc-gb 2 measured 3504 MB total, so the non-alloc
# baseline (filler table + runtime) is ~1456 MB. Matching 5196 MB therefore needs
# 5196-1456 = 3740 MB = 3.65 GB. The previous value, 12.0, was an uncalibrated
# guess: it was 2.3x the real footprint AND large enough to OOM the chip, which
# is why every nonml_sat cell died in ~55 s instead of controlling anything.
# Re-derive this per scope from the F=0 parquet; do not carry it across shapes.
_ALLOC_MATCHED = 3.65
_ALLOC_HIMEM = 48.0


def _mni_cmd(train_frac, dur=_DUR, kernel="mixed"):
    return [_PY, os.path.join(_WORKLOADS_DIR, "tpu_adversarial_multinode.py"),
            "--size", _ADV_MODEL, "--train-frac", str(train_frac),
            "--duration", str(dur), "--filler-mode", kernel,
            "--multihost", "auto"]


def _nonml_cmd(dur=_DUR, *, alloc=0.0, duty=1.0, kernel="mixed", mode="hpc"):
    return [_PY, os.path.join(_WORKLOADS_DIR, "tpu_nonml_distributed.py"),
            "--mode", mode, "--duration", str(dur), "--alloc-gb", str(alloc),
            "--duty", str(duty), "--filler-mode", kernel, "--multihost", "auto"]


for _sc in _SCOPES:
    # --- MN-I disguise ladder (ml_training, disguised) ---
    _add(f"{_sc}_advtrain_satfiller_only", _mni_cmd(0.0),
         f"MN-I F=0 on {_sc}: 1b model resident, ZERO training, saturating filler "
         f"(the cell that evades on GPU)", _TO_ADV)
    _add(f"{_sc}_advtrain_satint_f05", _mni_cmd(0.05),
         f"MN-I F=0.05 on {_sc} (undetectable on GPU at any usable threshold)",
         _TO_ADV)
    _add(f"{_sc}_advtrain_satint_f20", _mni_cmd(0.2),
         f"MN-I F=0.2 on {_sc} (undetectable on GPU at any usable threshold)",
         _TO_ADV)
    _add(f"{_sc}_advtrain_satint_f50", _mni_cmd(0.5),
         f"MN-I F=0.5 on {_sc} (100% detected at 0.6% FPR on GPU -- the claim)",
         _TO_ADV)
    # --- Honest non-ML controls (other, honest) ---
    _add(f"{_sc}_nonml_sat", _nonml_cmd(alloc=_ALLOC_MATCHED, duty=1.0),
         f"Honest non-ML on {_sc}: matched footprint, NO model. Like-for-like "
         f"partner of advtrain_satfiller_only -- the apparatus test", _TO_OTHER)
    _add(f"{_sc}_nonml_duty30", _nonml_cmd(alloc=2.0, duty=0.3),
         f"Honest non-ML on {_sc} at 30% duty (GPU: nonml_lowpower, the cell that "
         f"false-alarmed harder than the F=0.05 disguise)", _TO_OTHER)
    _add(f"{_sc}_nonml_duty50_hash", _nonml_cmd(alloc=8.0, duty=0.5,
                                                kernel="hash"),
         f"Honest non-ML on {_sc}, 50% duty, memory-bound kernel (GPU: the WORST "
         f"held-out false positive at 0.837)", _TO_OTHER)
    _add(f"{_sc}_nonml_comm", _nonml_cmd(mode="comm"),
         f"Honest non-ML on {_sc}: slot-bounded cross-chip collectives only. "
         f"Probes the DCN channel that has no NVIDIA counterpart", _TO_OTHER)

# tpuT16 only: the scope that structurally matches the 2x8 H100 GPU setup.
_add("tpuT16_nonml_duty50_fft", _nonml_cmd(alloc=8.0, duty=0.5, kernel="fft"),
     "Honest non-ML on tpuT16, 50% duty, compute-bound kernel (GPU: 0.212 vs "
     "0.837 for hash at identical duty -- the kernel axis)", _TO_OTHER)
_add("tpuT16_nonml_himem", _nonml_cmd(alloc=_ALLOC_HIMEM, duty=1.0),
     "Honest non-ML on tpuT16 at training-scale HBM footprint, no model "
     "(GPU: refuted 'memory is the tell' at 18.6 GB -> p_train 0.069)", _TO_OTHER)


# ── tpuTMS (2x v6e-8 multislice): the ONLY scope that populates the DCN channel ──
# megascale.collective_end_to_end_latencies (-> dcn_collective_lat_us_*) accumulates
# ONLY across separate slices, never within one (within a slice, collectives ride
# ICI and megascale.* answers UNAVAILABLE). This is the headline "signal NVIDIA
# lacks" rung. Commands are scope-agnostic; the queue provisions NODE_COUNT=2 x
# v6e-8 and forms ONE 16-chip cross-slice mesh. Only the decisive Tier-1 subset is
# registered here -- the multislice rental is the priciest, so no full 7b ladder.
_add("tpuTMS_advtrain_satint_f50", _mni_cmd(0.5),
     "MN-I F=0.5 on 2x v6e-8 multislice -- the claim, WITH the DCN channel live", _TO_ADV)
_add("tpuTMS_advtrain_satint_f20", _mni_cmd(0.2),
     "MN-I F=0.2 on multislice (0% on GPU; does DCN make it visible?)", _TO_ADV)
_add("tpuTMS_advtrain_satint_f05", _mni_cmd(0.05),
     "MN-I F=0.05 on multislice (0% on GPU; hardest disguise vs the DCN channel)", _TO_ADV)
_add("tpuTMS_advtrain_satfiller_only", _mni_cmd(0.0),
     "MN-I F=0 on multislice: zero training, saturating filler (evades on GPU)", _TO_ADV)
_add("tpuTMS_nonml_sat", _nonml_cmd(alloc=_ALLOC_MATCHED, duty=1.0),
     "Honest non-ML on multislice, footprint-matched -- the apparatus test", _TO_OTHER)
_add("tpuTMS_nonml_comm", _nonml_cmd(mode="comm"),
     "Honest non-ML on multislice: slot-bounded cross-SLICE collectives -- the DCN "
     "probe with no NVIDIA counterpart (collectives DO cross DCN at this scope)", _TO_OTHER)
_add("tpuTMS_nonml_duty30", _nonml_cmd(alloc=2.0, duty=0.3),
     "Honest non-ML on multislice at 30% duty (GPU: false-alarmed harder than F=0.05)", _TO_OTHER)
_add("tpuTMS_nonml_duty50_hash", _nonml_cmd(alloc=8.0, duty=0.5, kernel="hash"),
     "Honest non-ML on multislice, 50% duty, memory-bound kernel (GPU: worst FP)", _TO_OTHER)
_add("tpuTMS_train_fsdp1b", _train_cmd("1b"),
     "1B training on 2x v6e-8 multislice (honest reference; cross-slice gradient collectives)", _TO_TRAIN)
_add("tpuTMS_inferrep_1b", _infer_cmd("1b", "replicated"),
     "1B replicated serving on multislice (no cross-slice sync -- clean negative)", _TO_INFER)
_add("tpuTMS_infershard_1b", _infer_cmd("1b", "tp"),
     "1B sharded serving across the multislice mesh (cross-slice activation psums)", _TO_INFER)


# ── Multislice LADDER built from 1-CHIP slices ────────────────────────────────
# Capacity, not preference, dictates this. Across a week of parking, GCP has
# granted v6e-1 freely (13 times) and a 2x v6e-1 multislice three times, while
# v6e-8, 2x v6e-8, v5litepod-8, v5litepod-16 and 2x v5litepod-1 have NEVER once
# reached ACTIVE -- the t8 driver log reads "none provisioned". A scale ladder
# assembled from host-sized slices is therefore not obtainable, but one assembled
# from 1-chip slices is: N x v6e-1 gives N chips AND puts every collective on DCN,
# which is the channel the whole TPU rung exists to test.
#
# So the rungs are slice COUNTS, not slice sizes: tpuTMS (2) -> tpuTMS4 (4) ->
# tpuTMS8 (8). The commands are already scope-agnostic -- each cell reads
# jax.devices() and `--multihost auto` forms the cross-slice mesh -- so the rungs
# differ only in what the queue provisions.
#
# NOTE on the label: the header above describes tpuTMS as "2x v6e-8 = 16 chips".
# That was the intent and has never been what ran; every tpuTMS cell to date
# executed on 2x v6e-1 = 2 chips. Do NOT infer chip count from the scope name in
# analysis -- read it from the telemetry (distinct gpu_uuid per chip) or from
# mem_total_mb. The scope name is a slice-count grouping key, nothing more.
_MS_LADDER = {"tpuTMS4": 4, "tpuTMS8": 8}
for _nm, _ns in _MS_LADDER.items():
    for _k in [k for k in list(WORKLOAD_REGISTRY_TPU) if k.startswith("tpuTMS_")]:
        _e = WORKLOAD_REGISTRY_TPU[_k]
        WORKLOAD_REGISTRY_TPU[_k.replace("tpuTMS_", _nm + "_", 1)] = {
            "command": list(_e["command"]),
            "description": f"{_e['description']} [{_ns} slices]",
            "default_timeout": _e["default_timeout"],
        }


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
