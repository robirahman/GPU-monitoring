#!/usr/bin/env python3
"""Launch a REAL torchtitan training job from the SPAR job table (W03 / W04 / W05).

One command, so the telemetry launcher can wrap it like the vLLM orchestrators::

    python scripts/run_workload_multi_gpu.py --workload moe_ep_train_dsv3_16b_ep8 --num-gpus 8
    python scripts/run_titan_job.py --job moe_ep_train_dsv3_16b_ep8 --steps 60
    python scripts/run_titan_job.py --list
    python scripts/run_titan_job.py --job train_tp2_dp4_qwen3_1p7b_titan --dry-run

Runs ``python -m torch.distributed.run --standalone --nproc_per_node=N -m torchtitan.train
--module <m> --config <fn> <overrides>`` (torchtitan 0.3.0 CLI). Jobs are in
``scripts/titan_jobs.py``; every ``--section.key`` is validated against the installed
torchtitan's source before launch (add ``--no-validate`` to skip).

Logging honesty: torchtitan exposes no collective hooks and none are invented. Outputs are
torchtitan's own step logs (loss / tokens-per-second / MFU) in ``<workdir>/<job>/outputs``, a
``job.json`` manifest (command, versions, wall-clock window), and -- with
``--nccl-collective-log`` -- NCCL's own per-collective debug log, convertible to JSONL with
``scripts/parse_nccl_coll_log.py``. Fabric truth = the NVLink/IB counters the telemetry
launcher records for the same window.

Data / weights: ``c4`` is streamed from the HF hub (needs network); tokenizer files are
downloaded to ``<workdir>/assets/<repo>`` (``--no-download`` to skip). Weights are random-init
unless ``--load-hf-weights`` (downloads safetensors; sets ``checkpoint.initial_load_*`` --
UNVERIFIED path; torchtitan may write a final checkpoint into the workdir).

Status: code (real); not run on GPUs from this repo yet.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from importlib import metadata
from importlib.util import find_spec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from titan_jobs import JOBS, validate_job  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKENIZER_PATTERNS = ["tokenizer.json", "tokenizer_config.json", "tokenizer.model", "vocab.json",
                      "merges.txt", "special_tokens_map.json", "config.json", "generation_config.json"]
WEIGHT_PATTERNS = ["*.safetensors", "model.safetensors.index.json"]

NCCL_COLL_ENV = {
    "NCCL_DEBUG": "INFO",
    "NCCL_DEBUG_SUBSYS": "COLL",
    "NCCL_DEBUG_TIMESTAMP_LEVELS": "INFO",
    "NCCL_DEBUG_TIMESTAMP_FORMAT": "[%F %T.%6f] ",
}


def build_command(job, steps: int, nproc: int, assets: str, dump: str, extra: list[str],
                  load_weights: bool) -> list[str]:
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={nproc}",
           "-m", "torchtitan.train", "--module", job.module, "--config", job.config,
           *job.overrides, "--training.steps", str(steps),
           "--hf_assets_path", assets, "--dump_folder", dump]
    if load_weights:
        cmd += ["--checkpoint.enable", "true", "--checkpoint.initial_load_path", assets,
                "--checkpoint.initial_load_in_hf", "true", "--checkpoint.initial_load_model_only", "true",
                "--checkpoint.interval", "1000000"]
    return cmd + extra


def download_assets(repo: str, dest: str, weights: bool) -> None:
    from huggingface_hub import snapshot_download

    os.makedirs(dest, exist_ok=True)
    snapshot_download(repo_id=repo, local_dir=dest,
                      allow_patterns=TOKENIZER_PATTERNS + (WEIGHT_PATTERNS if weights else []))
    if not os.path.exists(os.path.join(dest, "tokenizer.json")) and \
            not os.path.exists(os.path.join(dest, "tokenizer_config.json")):
        raise SystemExit(f"{repo}: no tokenizer files downloaded to {dest} (gated repo? set HF_TOKEN)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Launch a real torchtitan SPAR training job",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", choices=sorted(JOBS))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--steps", type=int, default=None, help="default: per-job (see --list)")
    ap.add_argument("--nproc", type=int, default=None)
    ap.add_argument("--workdir", default=os.path.join(REPO_ROOT, "data", "titan"))
    ap.add_argument("--load-hf-weights", action="store_true")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--no-validate", action="store_true")
    ap.add_argument("--nccl-collective-log", action="store_true",
                    help="NCCL_DEBUG=INFO COLL log per process -> <workdir>/<job>/nccl.<host>.<pid>.log")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("extra", nargs="*", help="extra torchtitan overrides after '--'")
    args = ap.parse_args()

    if args.list or not args.job:
        for j in JOBS.values():
            print(f"{j.name:38s} {j.wid}  {j.nproc} GPU  steps={j.default_steps:<4d} {j.description}")
        return
    job = JOBS[args.job]
    steps = args.steps or job.default_steps
    nproc = args.nproc or job.nproc
    jobdir = os.path.join(args.workdir, job.name)
    assets = os.path.join(args.workdir, "assets", job.hf_repo.split("/")[-1])
    cmd = build_command(job, steps, nproc, assets, os.path.join(jobdir, "outputs"), args.extra,
                        args.load_hf_weights)
    env_extra = dict(NCCL_COLL_ENV, NCCL_DEBUG_FILE=os.path.join(jobdir, "nccl.%h.%p.log")) \
        if args.nccl_collective_log else {}

    print(f"[titan] {job.name} ({job.wid}): {job.description}")
    print(f"[titan] {job.notes}")
    print("[titan] cmd:", " ".join(cmd))
    if env_extra:
        print("[titan] env:", " ".join(f"{k}={v}" for k, v in env_extra.items()))

    installed = find_spec("torchtitan") is not None
    if installed and not args.no_validate:
        problems = validate_job(job, args.extra)
        if problems:
            print("[titan] SCHEMA PROBLEMS against the installed torchtitan:", *problems, sep="\n  - ", file=sys.stderr)
            raise SystemExit(2)
        print("[titan] overrides validated against the installed torchtitan source")
    elif not installed:
        print("[titan] torchtitan not installed here -> schema NOT validated (pip install torchtitan)")
    if args.dry_run:
        return

    if sys.version_info < (3, 11):
        raise SystemExit("torchtitan needs Python >= 3.11")
    for mod in ("torch", "torchtitan"):
        if find_spec(mod) is None:
            raise SystemExit(f"preflight: '{mod}' not installed (pip install {mod})")
    versions = {m: metadata.version(m) for m in ("torch", "torchtitan")}
    os.makedirs(jobdir, exist_ok=True)
    if not args.no_download:
        download_assets(job.hf_repo, assets, args.load_hf_weights)

    manifest = {"job": job.name, "wid": job.wid, "module": job.module, "config": job.config, "steps": steps,
                "nproc": nproc, "cmd": cmd, "versions": versions, "nccl_collective_log": bool(env_extra),
                "weights": "hf" if args.load_hf_weights else "random-init", "dataset": "c4 (HF stream)",
                "t_start": time.time()}
    with open(os.path.join(jobdir, "job.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    rc = subprocess.run(cmd, env={**os.environ, **env_extra}, cwd=jobdir).returncode
    manifest.update(t_end=time.time(), returncode=rc)
    with open(os.path.join(jobdir, "job.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[titan] rc={rc} window=[{manifest['t_start']:.0f},{manifest['t_end']:.0f}] manifest={jobdir}/job.json")
    sys.exit(rc)


if __name__ == "__main__":
    main()
