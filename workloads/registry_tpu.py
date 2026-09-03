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

# ...and a SECOND matched control, because F=0 turns out not to be the F->0 limit.
# Measured on the same 2x v6e-1 multislice, 2026-09-01, mem_used_mb over the run:
#     F=0   (satfiller_only):  ramps to 5196 MB and stays there
#     F=0.2 (both slices):     5.2 GB until t=124s, then JUMPS to 16495 MB
#     F=0.5:                   same jump to 16495 MB at t=126s
# The jump is the first train step compiling: activations, gradients and the
# optimizer moments get materialised and stay resident for the rest of the run.
# It is independent of F (0.2 and 0.5 both land on exactly 16495 MB), because it
# is a one-off workspace allocation rather than anything per-step.
#
# Consequence for the experiment. A disguised-training run at ANY F>0 holds 16.5
# GB; the filler-only cell holds 5.2 GB. So mem_used_mb alone separates the
# disguise from filler-only by 3.2x -- and on TPU, where power/clocks/temperature
# do not exist, mem_used_mb is one of the few live LEVEL channels. A control
# matched to the F=0 footprint (3.65 GB of dummy) is therefore NOT a matched
# control for the headline F=0.5 comparison: memory alone would separate them, and
# any "detection" would be reading footprint rather than training dynamics --
# exactly the confound a matched control exists to remove.
#
# So there are two honest partners, one per contrast:
#   nonml_sat        3.65 GB  -> matches F=0  (total ~5196 MB)
#   nonml_sat_16g    14.69 GB -> matches F>0  (total ~16495 MB)
#
# NAMING: this cell may NOT be called nonml_sat_train. threeway_label() routes on
# the substring "train", so `tpuTMS_nonml_sat_train` labels an HONEST non-ML
# control as ml_training -- verified, it returns exactly that. The label grammar
# is keyword-based and silently wins over intent.
# using the same ~1456 MB non-alloc baseline measured from duty30.
_ALLOC_MATCHED_TRAIN = 14.69

# The two numbers that actually matter now: MEASURED per-chip footprints of the
# cells the honest controls must match, fed to --match-mem-mb so the cell
# allocates until the runtime allocator agrees rather than trusting arithmetic.
# The open-loop constants above are kept only for the non-matched cells; they
# could not do this job, because the dummy is sharded across the mesh while
# mem_used_mb is per-chip, and the measured baseline is not constant either
# (1344 vs 2480 MB/chip from two cells). The first attempt with --alloc-gb 3.65
# realised 2880 MB/chip against a 5196 MB target -- 45% short, on the one cell
# whose entire purpose is to hold the same footprint as its partner.
_MEM_MATCH_F0 = 5196.0        # tpuTMS_advtrain_satfiller_only, 758 samples
_MEM_MATCH_TRAIN = 16495.0    # tpuTMS_advtrain_satint_f{20,50}, both slices

# PER SCOPE, because a footprint does not transfer between scopes and a single
# constant is guaranteed wrong somewhere. The same 1b model sharded over 8 v5e
# chips holds far less per chip than over 2 v6e chips, and the chips differ too:
#   v6e  mem_total_mb 31996   F>0 footprint 16495 MB/chip   (measured 09-01)
#   v5e  mem_total_mb 16126   F>0 footprint  7013 MB/chip   (measured 09-02)
# so _MEM_MATCH_TRAIN=16495 exceeds an ENTIRE v5e chip. A scope with no measured
# entry falls back to the open-loop --alloc-gb knob, which allocates *something*
# but is explicitly NOT a matched control; the cell logs target vs realised either
# way, so the analysis can tell which cells are usable for the apparatus contrast.
#
# To add a scope: run one F=0 cell and one F>0 cell, read mem_used_mb from the
# parquet (the collector's number, which is what the detector sees), and enter it
# here. The disguise cells print MEM_BYTES_IN_USE_STEADY for cross-checking.
_MEM_MATCH_F0_BY_SCOPE = {
    "tpuTMS@v6e": 5196.0,      # 2x v6e-1, satfiller_only, 758 samples
    "tpuT8@v5e": 1089.0,       # v5litepod-8, satfiller_only rep1, 665 samples
    "tpuTMS@v5e": 5184.0,      # 2x v5litepod-1, satfiller_only rep1, 642 samples
}
_MEM_MATCH_TRAIN_BY_SCOPE = {
    "tpuTMS@v6e": 16495.0,     # 2x v6e-1, f20 and f50, both slices
    "tpuT8@v5e": 7013.0,       # v5litepod-8, f50 rep1, 659 samples
    # 2x v5litepod-1, f50 rep1, 747 samples. NOTE: 15918 of a 16126 MB chip is
    # 98.7% of HBM, so an honest control CANNOT match this footprint -- the 80%
    # cap in tpu_nonml_distributed stops at ~12900 and logs that it did. The
    # apparatus contrast on this scope is therefore bounded at ~81% of the target,
    # which is a property of the hardware (a 1b model's training workspace nearly
    # fills a v5litepod chip), not of the control.
    "tpuTMS@v5e": 15918.0,
}


_GEN_CACHE = [None]


def _generation() -> str:
    """TPU generation of the host this is running on ("v6e", "v5e", "unknown").

    Needed because a SCOPE does not determine a footprint. The campaign maps both
    2x v6e-1 and 2x v5litepod-1 to scope "tpuTMS", and those chips differ in
    capacity (31996 vs 16126 MB) and in how the same model shards, so one target
    under one scope name is wrong for one of them. Concretely: tpuTMS's measured
    _MEM_MATCH_TRAIN of 16495 MB exceeds an ENTIRE v5litepod chip.

    Same metadata attribute the collector derives gpu_name from, cached because
    the registry is imported per cell launch and the call costs a round trip (and
    fails fast off-GCP, where it correctly yields "unknown").
    """
    if _GEN_CACHE[0] is None:
        raw = (os.environ.get("SPAR_TPU_ACCEL") or "").strip().lower()
        if not raw:
            try:
                import urllib.request
                req = urllib.request.Request(
                    "http://metadata.google.internal/computeMetadata/v1/instance/"
                    "attributes/accelerator-type",
                    headers={"Metadata-Flavor": "Google"})
                with urllib.request.urlopen(req, timeout=2) as r:
                    raw = r.read().decode("utf-8", "replace").strip().lower()
            except Exception:
                raw = ""
        gen = "unknown"
        for pre, g in (("v6e", "v6e"), ("v5litepod", "v5e"), ("v5lite", "v5e"),
                       ("v5e", "v5e"), ("v5p", "v5p"), ("v4", "v4")):
            if raw.startswith(pre):
                gen = g
                break
        _GEN_CACHE[0] = gen
    return _GEN_CACHE[0]


def _match_f0(scope):
    """Measured F=0 footprint for this scope ON THIS GENERATION, else 0."""
    g = _generation()
    return _MEM_MATCH_F0_BY_SCOPE.get(f"{scope}@{g}",
                                      _MEM_MATCH_F0_BY_SCOPE.get(scope, 0.0)
                                      if g == "unknown" else 0.0)


def _match_train(scope):
    g = _generation()
    return _MEM_MATCH_TRAIN_BY_SCOPE.get(f"{scope}@{g}",
                                         _MEM_MATCH_TRAIN_BY_SCOPE.get(scope, 0.0)
                                         if g == "unknown" else 0.0)
_ALLOC_HIMEM = 48.0


def _mni_cmd(train_frac, dur=_DUR, kernel="mixed"):
    return [_PY, os.path.join(_WORKLOADS_DIR, "tpu_adversarial_multinode.py"),
            "--size", _ADV_MODEL, "--train-frac", str(train_frac),
            "--duration", str(dur), "--filler-mode", kernel,
            "--multihost", "auto"]


def _typeconf_cmd(dur=_DUR, *, accum=1, tc_fill=0.0, d2h=0, opt="sgd",
                  size=_ADV_MODEL):
    """MN-T: real training shaped to be misread as INFERENCE (Stage 2 attack).

    Each knob targets a channel measured to separate the classes on hardware:
    tensorcore util (training 17 vs inference 88 on v5e), HBM footprint (7013 vs
    2668 MB, via optimizer state), cross-slice collective cadence (every step vs
    none, via local SGD), and the device->host transfer channel inference uses to
    stream tokens.
    """
    return [_PY, os.path.join(_WORKLOADS_DIR, "tpu_adversarial_typeconf.py"),
            "--size", size, "--duration", str(dur), "--optimizer", opt,
            "--accum", str(accum), "--tc-fill", str(tc_fill),
            "--d2h-trickle", str(d2h), "--multihost", "auto"]


def _nonml_cmd(dur=_DUR, *, alloc=0.0, duty=1.0, kernel="mixed", mode="hpc",
               match_mb=0.0):
    """match_mb closes the loop on a MEASURED per-chip footprint; alloc is the
    open-loop knob and cannot match one (the dummy is sharded across the mesh
    while mem_used_mb is per-chip). Any cell whose job is to match another cell
    must use match_mb."""
    cmd = [_PY, os.path.join(_WORKLOADS_DIR, "tpu_nonml_distributed.py"),
           "--mode", mode, "--duration", str(dur), "--alloc-gb", str(alloc),
           "--duty", str(duty), "--filler-mode", kernel, "--multihost", "auto"]
    if match_mb:
        cmd += ["--match-mem-mb", str(match_mb)]
    return cmd


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
    # Partner for the F>0 contrast, not the F=0 one: a disguised run at any F>0
    # carries the training workspace and so sits far above the filler-only cell
    # (v6e 16495 vs 5196 MB/chip; v5e 7013 vs 1089 -- a 6.4x gap there). Matching
    # only F=0 would leave memory alone separating the headline comparison. Named
    # _hi and NOT _16g, because 16 GB is a v6e number that exceeds a whole v5e
    # chip, and NOT anything containing "train", which threeway_label() would
    # route to ml_training.
    _add(f"{_sc}_nonml_sat_hi",
         _nonml_cmd(alloc=(0.0 if _match_train(_sc) else 8.0),
                    match_mb=_match_train(_sc), duty=1.0),
         f"Honest non-ML on {_sc} at the TRAINING footprint -- without it, "
         f"mem_used_mb alone separates the F>0 disguise from its control",
         _TO_OTHER)
    _add(f"{_sc}_nonml_sat",
         _nonml_cmd(alloc=(0.0 if _match_f0(_sc) else _ALLOC_MATCHED),
                    match_mb=_match_f0(_sc), duty=1.0),
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
# Same fallback as the per-scope loop: with no measured target for this scope on
# THIS generation, fall back to the open-loop knob rather than to nothing. Without
# it the cell allocated neither a matched footprint nor any footprint at all --
# caught locally, where the generation reads "unknown", before it could reach a
# slice and silently turn the apparatus control into an empty saturating filler.
_add("tpuTMS_nonml_sat",
     _nonml_cmd(alloc=(0.0 if _match_f0("tpuTMS") else _ALLOC_MATCHED),
                match_mb=_match_f0("tpuTMS"), duty=1.0),
     "Honest non-ML on multislice, footprint-matched to F=0 -- the apparatus test",
     _TO_OTHER)
_add("tpuTMS_nonml_sat_hi",
     _nonml_cmd(alloc=(0.0 if _match_train("tpuTMS") else 8.0),
                match_mb=_match_train("tpuTMS"), duty=1.0),
     "Honest non-ML on multislice, footprint-matched to a TRAINING run -- "
     "without this, mem_used_mb alone separates the F>0 disguise from its control",
     _TO_OTHER)
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
# ── MN-T: the type-confusion attack ladder ───────────────────────────────────
# MN-I attacks Stage 1 (make training look non-ML). This attacks Stage 2: keep the
# run plainly ML, but push the train-vs-infer discriminator toward INFER. Harder
# by construction, because Stage 2 sees only shape features after the WB-O level
# drop, so a level cannot simply be moved.
#   _off  isolates the optimizer choice alone (sgd, no other shaping) -- the
#         control that says how much of any effect is just "no Adam moments"
#   _a8   moderate: sync every 8 steps, half tc fill, light d2h trickle
#   _a32  strongest: sync every 32 steps, near-full tc fill, heavy d2h trickle
# Names keep "train" so threeway_label routes them to ml_training: the GROUND
# TRUTH is training, and the attack succeeds precisely when the detector disagrees.
# Verified: 'advtrain_typeconf_a32' -> ml_training (the "train" rule is checked
# before the "infer" rule, so 'mimicinfer'-style names would also have worked).
for _sc in _SCOPES + ["tpuTMS", "tpuTMS4", "tpuTMS8"]:
    _add(f"{_sc}_advtrain_typeconf_off", _typeconf_cmd(accum=1, tc_fill=0.0, d2h=0),
         f"MN-T control on {_sc}: SGD training, no other shaping -- isolates the "
         f"optimizer's own effect on the footprint channel", _TO_ADV)
    _add(f"{_sc}_advtrain_typeconf_a8", _typeconf_cmd(accum=8, tc_fill=0.5, d2h=64),
         f"MN-T moderate on {_sc}: sync every 8 steps, half tc fill, d2h trickle", _TO_ADV)
    # THE ISOLATING CELL. typeconf_off (plain SGD, no other shaping) evades the
    # detector completely -- p_train 0.264, zero windows above tau, with its
    # family held out -- while train_fsdp1b (AdamW, same model, same topology)
    # scores 0.843. But those two differ in TWO ways: optimizer AND script
    # structure (replicated params with periodic averaging vs sharded FSDP). This
    # cell is byte-identical to typeconf_off except --optimizer adamw, so
    # off-vs-adamw isolates the optimizer and adamw-vs-train_fsdp1b isolates the
    # structure. Without it the section 20 finding stays confounded.
    _add(f"{_sc}_advtrain_typeconf_adamw",
         _typeconf_cmd(accum=1, tc_fill=0.0, d2h=0, opt="adamw"),
         f"MN-T control on {_sc}: identical to typeconf_off but AdamW -- isolates "
         f"the optimizer-state signature from the script structure", _TO_ADV)
    _add(f"{_sc}_advtrain_typeconf_a32", _typeconf_cmd(accum=32, tc_fill=0.9, d2h=256),
         f"MN-T strongest on {_sc}: sync every 32 steps, near-full tc fill, "
         f"heavy d2h trickle -- training shaped hardest toward the inference profile",
         _TO_ADV)

_MS_LADDER = {"tpuTMS4": 4, "tpuTMS8": 8}
for _nm, _ns in _MS_LADDER.items():
    for _k in [k for k in list(WORKLOAD_REGISTRY_TPU) if k.startswith("tpuTMS_")]:
        # The footprint-matched cells must NOT be cloned verbatim: their
        # --match-mem-mb is a 2-slice measurement, and a 4- or 8-slice group
        # shards the same model differently, so copying the number would
        # reintroduce exactly the cross-scope transfer error that
        # _MEM_MATCH_*_BY_SCOPE exists to prevent. Re-register them from the
        # per-scope helpers instead, which fall back to the open-loop knob (and
        # therefore to "not a matched control") until that rung is measured.
        if "nonml_sat" in _k:
            continue
        _e = WORKLOAD_REGISTRY_TPU[_k]
        WORKLOAD_REGISTRY_TPU[_k.replace("tpuTMS_", _nm + "_", 1)] = {
            "command": list(_e["command"]),
            "description": f"{_e['description']} [{_ns} slices]",
            "default_timeout": _e["default_timeout"],
        }
    _add(f"{_nm}_nonml_sat",
         _nonml_cmd(alloc=(0.0 if _match_f0(_nm) else _ALLOC_MATCHED),
                    match_mb=_match_f0(_nm), duty=1.0),
         f"Honest non-ML at the F=0 footprint [{_ns} slices]", _TO_OTHER)
    _add(f"{_nm}_nonml_sat_hi",
         _nonml_cmd(alloc=(0.0 if _match_train(_nm) else 8.0),
                    match_mb=_match_train(_nm), duty=1.0),
         f"Honest non-ML at the TRAINING footprint [{_ns} slices]", _TO_OTHER)


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
