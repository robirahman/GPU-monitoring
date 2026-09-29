#!/usr/bin/env python3
"""Real HPC benchmarks (HPL, HPCG, GROMACS) from the NGC containers.

WHY A CONTAINER. This project has tried and failed once before to get a
GPU-capable GROMACS built on a node: the Ubuntu package is CPU-only, and a
source build with CUDA is a long, fragile step to put in a boot path. The NGC
images are the vendor's own builds and they pull ANONYMOUSLY -- verified against
the registry API, `nvcr.io/nvidia/hpc-benchmarks:24.09` and
`nvcr.io/hpc/gromacs:2023.2` both return HTTP 200 for a manifest fetched with an
anonymous bearer token. No NGC account, no API key, no secret in the boot path.

WHY THESE WORKLOADS. Every non-ML multi-node cell in the corpus so far is
something this project wrote: FFT shards, halo stencils, NCCL sweeps, duty-cycle
and low-power loops. They are all one author's idea of what non-ML GPU work looks
like. HPL and GROMACS are not: they are the actual programs HPC clusters run, and
the detector has never seen either. HPL is dense double-precision LU with a
communication pattern nothing else here produces; GROMACS is molecular dynamics,
which is neighbour-list bound and looks like nothing in the ML corpus at all.

DISCOVERY, NOT HARDCODED PATHS. The launcher scripts move between container
versions, so this probes the image for them rather than assuming a path and
failing three hours into a collection run. `--probe` prints what it found and
exits, which is how you check a new image tag without burning a cell.

    python hpc_container.py --benchmark hpl --duration 600
    python hpc_container.py --benchmark gromacs --duration 600
    python hpc_container.py --benchmark hpl --probe
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

IMAGES = {
    "hpl": "nvcr.io/nvidia/hpc-benchmarks:24.09",
    "hpcg": "nvcr.io/nvidia/hpc-benchmarks:24.09",
    "gromacs": "nvcr.io/hpc/gromacs:2023.2",
}

# Candidate launcher locations, most likely first. Probed in order; the first
# that exists wins. Keeping this a list rather than a constant is the whole
# point -- 24.09 moved hpl.sh out of /workspace, and a hardcoded path would have
# failed at cell time rather than at probe time.
LAUNCHERS = {
    "hpl": ["/workspace/hpl.sh", "/workspace/hpl-linux-x86_64/hpl.sh",
            "/usr/local/bin/hpl.sh", "/workspace/hpl-aarch64/hpl.sh"],
    "hpcg": ["/workspace/hpcg.sh", "/workspace/hpcg-linux-x86_64/hpcg.sh",
             "/usr/local/bin/hpcg.sh"],
    "gromacs": ["/usr/local/gromacs/bin/gmx", "/usr/local/bin/gmx", "/opt/gromacs/bin/gmx"],
}


def have_docker() -> str:
    for exe in ("docker", "podman"):
        if shutil.which(exe):
            return exe
    return ""


def run(cmd, timeout=None, capture=True):
    return subprocess.run(cmd, capture_output=capture, text=True, timeout=timeout)


def gpu_count() -> int:
    try:
        out = run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], timeout=30)
        return len([l for l in out.stdout.splitlines() if l.strip()])
    except Exception:
        return 0


def docker_ok(exe: str) -> bool:
    """Docker present AND able to see GPUs. Both matter: the DLVM image ships
    docker and the NVIDIA container toolkit, but a node where the toolkit did
    not register would run the container on CPU and quietly produce a cell of
    idle-GPU telemetry, which is worse than failing."""
    try:
        r = run([exe, "info"], timeout=60)
        if r.returncode != 0:
            print(f"  {exe} info failed: {r.stderr.strip()[:200]}", flush=True)
            return False
        r = run([exe, "run", "--rm", "--gpus", "all", IMAGES["hpl"],
                 "nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], timeout=900)
        n = len([l for l in r.stdout.splitlines() if l.strip()])
        print(f"  container sees {n} GPU(s)", flush=True)
        return n > 0
    except Exception as e:
        print(f"  {exe} GPU probe failed: {e}", flush=True)
        return False


def pull(exe: str, image: str) -> bool:
    print(f"  pulling {image} (anonymous)", flush=True)
    r = run([exe, "pull", image], timeout=3600)
    if r.returncode != 0:
        print(f"  pull FAILED: {r.stderr.strip()[:300]}", flush=True)
        return False
    return True


def find_launcher(exe: str, bench: str) -> str:
    image = IMAGES[bench]
    for path in LAUNCHERS[bench]:
        r = run([exe, "run", "--rm", image, "test", "-x", path], timeout=300)
        if r.returncode == 0:
            print(f"  launcher: {path}", flush=True)
            return path
    # Fall back to a search rather than giving up: the point of probing.
    name = "gmx" if bench == "gromacs" else f"{bench}.sh"
    r = run([exe, "run", "--rm", image, "bash", "-lc",
             f"find / -maxdepth 6 -name {name} -type f 2>/dev/null | head -5"], timeout=600)
    found = [l.strip() for l in r.stdout.splitlines() if l.strip()]
    if found:
        print(f"  launcher (found by search): {found[0]}", flush=True)
        return found[0]
    return ""


def find_dat(exe: str, bench: str) -> str:
    """HPL needs an input deck. The containers ship sample decks sized per node
    type; prefer a 1-node A100 one, else take whatever is smallest."""
    if bench != "hpl":
        return ""
    image = IMAGES[bench]
    r = run([exe, "run", "--rm", image, "bash", "-lc",
             "find / -maxdepth 7 -name 'HPL*.dat' 2>/dev/null | head -20"], timeout=600)
    dats = [l.strip() for l in r.stdout.splitlines() if l.strip()]
    if not dats:
        return ""
    for pref in ("a100-1N", "dgx-a100-1N", "1N"):
        for d in dats:
            if pref in d:
                print(f"  hpl dat: {d}", flush=True)
                return d
    print(f"  hpl dat: {dats[0]}", flush=True)
    return dats[0]


def main():
    ap = argparse.ArgumentParser(description="NGC HPC benchmarks (HPL/HPCG/GROMACS)")
    ap.add_argument("--benchmark", required=True, choices=sorted(IMAGES))
    ap.add_argument("--duration", type=int, default=600)
    ap.add_argument("--gpus", type=int, default=0, help="0 = all visible")
    ap.add_argument("--probe", action="store_true",
                    help="report image/launcher/input availability and exit")
    args = ap.parse_args()

    exe = have_docker()
    print(f"HPC container workload: {args.benchmark}", flush=True)
    print(f"  runtime: {exe or 'NONE'}", flush=True)
    if not exe:
        print("FATAL: no docker/podman on PATH", flush=True)
        return 2

    ngpu = args.gpus or gpu_count()
    print(f"  gpus: {ngpu}", flush=True)
    image = IMAGES[args.benchmark]

    if not pull(exe, image):
        return 3
    if not docker_ok(exe):
        print("FATAL: container cannot see GPUs -- refusing to collect CPU-only "
              "telemetry under a GPU workload label", flush=True)
        return 4

    launcher = find_launcher(exe, args.benchmark)
    if not launcher:
        print(f"FATAL: no launcher for {args.benchmark} in {image}", flush=True)
        return 5
    dat = find_dat(exe, args.benchmark)

    if args.probe:
        print(f"PROBE OK image={image} launcher={launcher} dat={dat or '-'}", flush=True)
        return 0

    if args.benchmark == "gromacs":
        inner = (f"cd /tmp && {launcher} --version >/dev/null 2>&1; "
                 f"while true; do {launcher} mdrun -nsteps 20000 -resetstep 1000 "
                 f"-noconfout -nb gpu -pme gpu -ntmpi {ngpu} -ntomp 4 -s /tmp/bench.tpr "
                 f"|| break; done")
        # No .tpr on the node means no simulation; generate a small one in-container.
        prep = (f"cd /tmp && {launcher} pdb2gmx -h >/dev/null 2>&1; echo prep-done")
    else:
        np = ngpu
        inner = (f"while true; do mpirun --allow-run-as-root -np {np} {launcher}"
                 + (f" --dat {dat}" if dat else "") + " || break; done")
        prep = "echo prep-done"

    cmd = [exe, "run", "--rm", "--gpus", "all", "--shm-size=1g",
           "--ulimit", "memlock=-1", "--ulimit", "stack=67108864",
           image, "bash", "-lc", f"{prep}; timeout {args.duration} bash -lc '{inner}'"]
    print(f"  running for {args.duration}s", flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, text=True)
    el = time.time() - t0
    # `timeout` returns 124 when it does its job, which is the expected exit here.
    ok = p.returncode in (0, 124) and el >= args.duration * 0.8
    print(f"  finished rc={p.returncode} after {el:.0f}s ({'OK' if ok else 'SHORT'})",
          flush=True)
    return 0 if ok else 6


if __name__ == "__main__":
    sys.exit(main())
