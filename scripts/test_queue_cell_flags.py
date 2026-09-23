#!/usr/bin/env python3
"""Check every queue cell's command against its workload's argparse, statically.

WHY. On 2026-09-21 all seven s16_inferserve_* cells died identically:

    api_server.py: error: unrecognized arguments: --disable-log-requests
    ERROR: vLLM server never became healthy

One removed flag, seven cells, each burning its full 1800 s startup budget before
giving up -- and it was the entire async-serving family, the one collected
specifically to stop long-context inference reading as training. The same class
of bug was caught again on 2026-09-23 while wiring this round: a registry entry
pointed at `llm_inference.py --model qwen25_7b`, which does not exist, because
that file's MODEL_REGISTRY stops at qwen25_3b.

Both are trivially detectable without a GPU, without torch, and without running
anything: the flag either appears in the script's add_argument calls or it does
not. There is no reason to learn it from a rented pair.

WHAT THIS DOES NOT CHECK. Only flags this repo owns. A flag passed through to a
third-party binary (vLLM's api_server, mpirun) is invisible here -- that is what
the `--help` probe in llm_serving_async.py is for. It also cannot check values
that are valid only at runtime (a --grid divisible by the world size, a model
that fits in memory).

    PYTHONPATH=. python3 scripts/test_queue_cell_flags.py
"""
import ast
import os
import re
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUEUE = ROOT / "scripts" / "rebuttal_collection" / "multinode_queue.sh"

# `c=<cell>; <helper> "$c" "$(rid_for ...)" ... -- workloads/<script>.py <args>`
CELL = re.compile(r'^\s*c=([A-Za-z0-9_]+);\s*(\S+)\s+(.*)$')


def _local_dicts(tree):
    """Keys of every module-level dict literal in one file, by variable name."""
    out = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Dict):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    keys = [k.value for k in n.value.keys
                            if isinstance(k, ast.Constant) and isinstance(k.value, str)]
                    if keys:
                        out[t.id] = set(keys)
    return out


def _module_dict_keys(tree, depth=1):
    """Module-level dict keys, FOLLOWING `from <module> import NAME`.

    `choices=list(MODEL_CONFIGS.keys())` is the house idiom, but the dict is
    usually imported rather than defined locally -- train_lora.py takes
    MODEL_CONFIGS from fsdp_training.py, llm_inference_hf_sharded.py takes
    VLLM_MODELS from llm_inference_vllm.py. Without following the import this
    checker resolves nothing for exactly the cells that matter: a mutation test
    showed `--size 99b` sailing through while a misspelled flag was caught.

    Both import spellings used in workloads/ are handled -- `from workloads.X
    import` and the bare `from X import` fallback these files use when launched
    directly by torchrun.
    """
    out = _local_dicts(tree)
    if depth <= 0:
        return out
    for n in tree.body:
        mods = []
        if isinstance(n, ast.ImportFrom) and n.module:
            mods = [n.module]
        elif isinstance(n, ast.Try):
            for sub in ast.walk(n):
                if isinstance(sub, ast.ImportFrom) and sub.module:
                    mods.append(sub.module)
        for mod in mods:
            rel = mod.replace(".", "/") + ".py"
            for cand in (ROOT / rel, ROOT / "workloads" / os.path.basename(rel)):
                if not cand.exists():
                    continue
                try:
                    sub_tree = ast.parse(cand.read_text())
                except SyntaxError:
                    continue
                for k, v in _module_dict_keys(sub_tree, depth - 1).items():
                    out.setdefault(k, v)
                break
    return out


def _resolve_choices(node, dicts):
    """Literal choices, or the keys of a module-level dict, or None."""
    try:
        return set(ast.literal_eval(node))
    except Exception:
        pass
    # list(NAME.keys())
    if (isinstance(node, ast.Call) and getattr(node.func, "id", "") == "list"
            and node.args and isinstance(node.args[0], ast.Call)
            and getattr(node.args[0].func, "attr", "") == "keys"):
        base = getattr(node.args[0].func.value, "id", None)
        if base in dicts:
            return set(dicts[base])
    # list(NAME.keys()) + ["a", "b"]
    if isinstance(node, ast.BinOp):
        vals = set()
        for side in (node.left, node.right):
            got = _resolve_choices(side, dicts)
            if got is None:
                return None          # one half unknown -> do not constrain
            vals |= got
        return vals
    return None


def parser_facts(path):
    """(option strings, {option: choices}) extracted without importing."""
    tree = ast.parse(path.read_text())
    dicts = _module_dict_keys(tree)
    opts, choices = set(), {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "add_argument":
            names = [a.value for a in n.args
                     if isinstance(a, ast.Constant) and isinstance(a.value, str)]
            opts.update(names)
            for kw in n.keywords:
                if kw.arg != "choices":
                    continue
                vals = _resolve_choices(kw.value, dicts)
                if names and vals:
                    choices[names[0]] = vals
    return opts, choices


def main():
    text = QUEUE.read_text()
    problems, checked = [], 0
    facts_cache = {}

    for line in text.splitlines():
        m = CELL.match(line)
        if not m:
            continue
        cell, _helper, rest = m.groups()
        if "--" not in rest:
            continue                       # registry-driven cell; covered elsewhere
        try:
            toks = shlex.split(rest.split(" -- ", 1)[1])
        except ValueError:
            continue                       # unbalanced quotes from a shell var
        script = next((t for t in toks if t.startswith("workloads/")
                       and t.endswith(".py")), None)
        if script is None:
            continue
        p = ROOT / script
        if not p.exists():
            problems.append(f"{cell}: {script} does not exist")
            continue
        if script not in facts_cache:
            facts_cache[script] = parser_facts(p)
        opts, choices = facts_cache[script]
        checked += 1

        args = toks[toks.index(script) + 1:]
        i = 0
        while i < len(args):
            tok = args[i]
            if not tok.startswith("--"):
                i += 1
                continue
            if tok not in opts:
                problems.append(f"{cell}: {os.path.basename(script)} has no {tok}")
            val = args[i + 1] if i + 1 < len(args) and not args[i + 1].startswith("--") else None
            if (val is not None and not val.startswith("$")
                    and tok in choices and val not in choices[tok]):
                problems.append(f"{cell}: {tok}={val} not in "
                                f"{sorted(choices[tok])}")
            i += 2 if val is not None else 1

    print(f"checked {checked} queue cells with an explicit workload command")
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(f"  {p}")
        return 1
    print("OK: every flag the queue passes exists in its workload's argparse, and "
          "every constrained value is valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
