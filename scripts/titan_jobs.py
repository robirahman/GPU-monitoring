"""torchtitan job table for the SPAR REAL training workloads (W03 / W04 / W05).

Single source of truth used by ``scripts/run_titan_job.py`` (launcher), the registry entries,
and the offline schema check (``validate_job``). Targets torchtitan 0.3.0, whose CLI is::

    torchrun --nproc_per_node=N -m torchtitan.train --module <model> --config <registry fn> \\
        --<section>.<key> <value> ...

(no more ``.toml`` files: configs are ``models/<module>/config_registry.py`` functions and
every field is overridable from the command line via tyro).

Why torchtitan: it is a real, maintained training framework (Meta/PyTorch) whose MoE and
parallelism collectives are issued by the framework itself -- FSDP2 all-gather/reduce-scatter,
TP (with or without sequence parallel), pipeline P2P via ``torch.distributed.pipelining``,
and expert-parallel all-to-all. Nothing here is a hand-written collective.

Open-weight models only (the Llama-3 configs need gated HF access). Weights are random-init
unless ``--load-hf-weights`` is passed to the launcher: the interconnect fingerprint depends on
architecture / parallelism / shapes, not weight values.

Status: code (real). Flags are validated against the installed torchtitan's source
(``validate_job``, AST-based, no torch/GPU needed); the jobs themselves have NOT been run on
GPUs from this repo.
"""
from __future__ import annotations

import ast
import importlib.util
import os
from dataclasses import dataclass, field

# Trainer.Config sections whose dataclass is defined in torchtitan/config/configs.py and can be
# checked field-by-field; everything else is checked weakly (field name exists somewhere).
STRONG_SECTIONS = {"training": "TrainingConfig", "parallelism": "ParallelismConfig"}
TOP_LEVEL_FIELDS = {"dump_folder", "hf_assets_path"}


@dataclass
class TitanJob:
    name: str
    wid: str
    module: str                   # torchtitan.models.<module>
    config: str                   # function in <module>/config_registry.py
    nproc: int
    hf_repo: str                  # tokenizer (+ weights with --load-hf-weights)
    overrides: list[str]          # "--section.key value" tokens (steps/paths appended by launcher)
    default_steps: int = 60
    description: str = ""
    notes: str = ""
    tags: list[str] = field(default_factory=list)


def _p(**kw) -> list[str]:
    """{'parallelism.tensor_parallel_degree': 2} -> ['--parallelism.tensor_parallel_degree', '2']."""
    out: list[str] = []
    for k, v in kw.items():
        out += [f"--{k.replace('__', '.')}", str(v).lower() if isinstance(v, bool) else str(v)]
    return out


JOBS: dict[str, TitanJob] = {j.name: j for j in [
    # ── W05: real MoE training with framework-issued expert-parallel all-to-all ──
    TitanJob(
        name="moe_ep_train_dsv3_16b_ep8", wid="W05", module="deepseek_v3", config="deepseek_v3_16b",
        nproc=8, hf_repo="deepseek-ai/deepseek-moe-16b-base",
        overrides=_p(parallelism__expert_parallel_degree=8, metrics__log_freq=1),
        default_steps=60,
        description="torchtitan DeepSeek-MoE-16B (64 routed experts) training, EP=8 on 8 GPUs, c4",
        notes="Registry default recipe (EP=8, FSDP over the rest, selective AC). A2A dispatch/combine "
              "fwd+bwd + FSDP AG/RS issued by torchtitan. OLMoE/Mixtral are not in torchtitan 0.3.0.",
        tags=["moe", "ep"]),
    TitanJob(
        name="moe_ep_train_qwen3_30b_a3b_ep8", wid="W05", module="qwen3", config="qwen3_30b_a3b",
        nproc=8, hf_repo="Qwen/Qwen3-30B-A3B",
        overrides=_p(parallelism__expert_parallel_degree=8, metrics__log_freq=1),
        default_steps=40,
        description="torchtitan Qwen3-30B-A3B (128 experts, top-8) training, EP=8 on 8 GPUs, c4",
        notes="30B total params on 8x80GB: memory fit is UNVERIFIED (registry default uses full AC, "
              "batch 2). Try lowering --training.local_batch_size if it OOMs.",
        tags=["moe", "ep"]),
    # ── W03: real TP + DP training ──
    TitanJob(
        name="train_tp2_dp4_qwen3_1p7b_titan", wid="W03", module="qwen3", config="qwen3_1_7b",
        nproc=8, hf_repo="Qwen/Qwen3-1.7B",
        overrides=_p(parallelism__tensor_parallel_degree=2, parallelism__data_parallel_replicate_degree=4,
                     parallelism__data_parallel_shard_degree=1, parallelism__enable_sequence_parallel=False,
                     metrics__log_freq=1),
        default_steps=100,
        description="torchtitan Qwen3-1.7B, TP=2 (sequence-parallel OFF) x replicated DP=4: Megatron-style "
                    "AllReduce on activations (TP) + AllReduce on grads (DP)",
        notes="SP off => TP uses AllReduce (classic Megatron shape); DP replicate => gradient AllReduce. "
              "Whether torchtitan accepts replicate x TP without FSDP is UNVERIFIED on GPU; fall back to "
              "train_tp2_fsdp4_qwen3_14b_titan (FSDP2+TP, the well-trodden recipe).",
        tags=["tp", "dp"]),
    TitanJob(
        name="train_tp2_fsdp4_qwen3_14b_titan", wid="W03", module="qwen3", config="qwen3_14b",
        nproc=8, hf_repo="Qwen/Qwen3-14B",
        overrides=_p(parallelism__tensor_parallel_degree=2, parallelism__data_parallel_shard_degree=4,
                     metrics__log_freq=1),
        default_steps=60,
        description="torchtitan Qwen3-14B, TP=2 (sequence-parallel on, default) x FSDP2 shard=4: "
                    "TP all-gather/reduce-scatter + FSDP all-gather/reduce-scatter",
        notes="Modern default recipe: sequence parallel turns TP AllReduce into AG/RS, so this differs "
              "from Megatron-classic TP by design.",
        tags=["tp", "fsdp"]),
    # ── W04: real pipeline-parallel training (torch.distributed.pipelining under torchtitan) ──
    TitanJob(
        name="train_pp4_1f1b_qwen3_1p7b_titan", wid="W04", module="qwen3", config="qwen3_1_7b",
        nproc=4, hf_repo="Qwen/Qwen3-1.7B",
        overrides=_p(parallelism__pipeline_parallel_degree=4, parallelism__data_parallel_shard_degree=1,
                     parallelism__pipeline_parallel_schedule="1F1B",
                     parallelism__pipeline_parallel_microbatch_size=1,
                     training__local_batch_size=8, metrics__log_freq=1),
        default_steps=100,
        description="torchtitan Qwen3-1.7B, 4-stage 1F1B pipeline (8 microbatches), no DP: P2P activations only",
        notes="Launch with --num-gpus 4. Schedule + stage split are torchtitan's "
              "(torch.distributed.pipelining Schedule1F1B); first/last stages get fewer layers by default.",
        tags=["pp"]),
]}


# ── offline schema validation (AST; no torch, no triton, no GPU) ─────────────

def _titan_root() -> str:
    spec = importlib.util.find_spec("torchtitan")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("torchtitan is not installed (pip install torchtitan)")
    return list(spec.submodule_search_locations)[0]


def _dataclass_fields(tree: ast.AST) -> dict[str, set[str]]:
    """{class name: annotated field names} for every class in a module (incl. nested)."""
    out: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            out.setdefault(node.name, set()).update(
                n.target.id for n in node.body
                if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name))
    return out


def validate_job(job: TitanJob, extra_overrides: list[str] | None = None, root: str | None = None) -> list[str]:
    """Problems found checking ``job`` against the installed torchtitan source ([] = OK).

    Strong checks: module + config function exist; ``training.*`` / ``parallelism.*`` keys exist in
    the real dataclasses; top-level keys exist. Weak checks (other sections): the field name exists
    in *some* torchtitan dataclass. Returns human-readable problems; never imports torchtitan.
    """
    root = root or _titan_root()
    problems: list[str] = []

    reg = os.path.join(root, "models", job.module, "config_registry.py")
    if not os.path.exists(reg):
        return [f"module '{job.module}' has no config_registry.py in this torchtitan"]
    fn_names = {n.name for n in ast.walk(ast.parse(open(reg).read())) if isinstance(n, ast.FunctionDef)}
    if job.config not in fn_names:
        problems.append(f"config '{job.config}' not in models/{job.module}/config_registry.py "
                        f"(has: {sorted(f for f in fn_names if not f.startswith('_'))[:8]}...)")

    all_fields: dict[str, set[str]] = {}
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.endswith(".py"):
                try:
                    for cls, flds in _dataclass_fields(ast.parse(open(os.path.join(dirpath, f)).read())).items():
                        all_fields.setdefault(cls, set()).update(flds)
                except (SyntaxError, UnicodeDecodeError):
                    pass
    every_field = set().union(*all_fields.values()) if all_fields else set()

    toks = job.overrides + (extra_overrides or [])
    for i, tok in enumerate(toks):
        if not tok.startswith("--"):
            continue
        key = tok[2:].split("=")[0]
        if key in TOP_LEVEL_FIELDS:
            if key not in all_fields.get("Config", set()) | every_field:
                problems.append(f"top-level field '{key}' not found")
            continue
        section, _, leaf = key.partition(".")
        if not leaf:
            problems.append(f"override '{tok}' is not '--section.key'")
            continue
        if section in STRONG_SECTIONS:
            cls = STRONG_SECTIONS[section]
            if leaf not in all_fields.get(cls, set()):
                problems.append(f"'{key}': no field '{leaf}' in {cls} (has {sorted(all_fields.get(cls, []))[:6]}...)")
        elif leaf not in every_field:
            problems.append(f"'{key}': no dataclass field named '{leaf}' anywhere in torchtitan")
        if i + 1 >= len(toks) or toks[i + 1].startswith("--"):
            problems.append(f"'{tok}' has no value")
    return problems
