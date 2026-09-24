#!/usr/bin/env python3
"""Catch calls that pass the wrong number of positional arguments.

s16_train_pipe7b ran its full 600 s measurement window on three separate GPU
pairs and then died in teardown, because adversarial_multinode_pipeline.py called
`sync_teardown(device)` while dist_deadline.py defines `sync_teardown()`. Python
only raises on the call, which sat after the training loop, so every static check
and every smoke run passed and it cost a cell on all three tracks.

Scope matters or this is pure noise: `get_cifar10_loader` is defined in several
workloads with different signatures, so a name-only match reports mismatches that
are not real. Each call is resolved against the definition it would ACTUALLY hit
-- the enclosing module's own def, or the def in the module it was imported from.

Deliberately not handled: keyword arguments beyond a count, decorators that
change signatures, methods, and *args/**kwargs targets (skipped, not guessed).
This is a cheap check for the one mistake that has actually bitten.
"""
import ast, os, sys, glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sig(fn):
    a = fn.args
    lo = len(a.posonlyargs) + len(a.args) - len(a.defaults)
    hi = None if a.vararg else len(a.posonlyargs) + len(a.args)
    return lo, hi


def defs_of(tree):
    return {n.name: sig(n) for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def main():
    files = sorted(glob.glob(os.path.join(ROOT, "workloads", "*.py")) +
                   glob.glob(os.path.join(ROOT, "scripts", "*.py")))
    trees, local = {}, {}
    for f in files:
        try:
            trees[f] = ast.parse(open(f).read())
        except SyntaxError:
            continue
        local[f] = defs_of(trees[f])

    by_mod = {os.path.splitext(os.path.basename(f))[0]: local[f] for f in local}
    bad = []
    for f, tree in trees.items():
        # name -> signature it resolves to at this call site
        scope = dict(local[f])
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom) and n.module:
                mod = n.module.split(".")[-1]
                for al in n.names:
                    if mod in by_mod and al.name in by_mod[mod]:
                        scope[al.asname or al.name] = by_mod[mod][al.name]
        # names rebound by assignment or nested def are unresolvable -- drop them
        for n in ast.walk(tree):
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        scope.pop(t.id, None)

        for n in ast.walk(tree):
            if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)):
                continue
            if n.func.id not in scope:
                continue
            if any(isinstance(x, ast.Starred) for x in n.args):
                continue
            lo, hi = scope[n.func.id]
            got = len(n.args)
            kw = {k.arg for k in n.keywords if k.arg}
            if any(k.arg is None for k in n.keywords):
                continue  # **kwargs -- cannot count
            if got > (hi if hi is not None else got):
                bad.append(f"{os.path.relpath(f, ROOT)}:{n.lineno}: "
                           f"{n.func.id}() takes at most {hi} positional, {got} given")
            elif got + len(kw) < lo:
                bad.append(f"{os.path.relpath(f, ROOT)}:{n.lineno}: "
                           f"{n.func.id}() needs {lo} args, {got + len(kw)} given")

    if bad:
        print("WRONG-ARITY CALLS:")
        for b in sorted(set(bad)):
            print("  " + b)
        return 1
    print(f"OK: no wrong-arity calls across {len(trees)} workload/script files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
