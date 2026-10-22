#!/usr/bin/env python3
"""Long-running non-ML GPU workloads (run in ~/venvs/nonml; no torch).

  openmm   molecular dynamics of a solvated box; box size varies per chunk
  rapids   cuDF groupby/join/sort analytics and cuGraph PageRank/BFS over generated tables and graphs
  qiskit   Qiskit Aer GPU state-vector simulation of random and QFT circuits (24-30 qubits)
  cupymc   CuPy Monte Carlo pricing of European/Asian options with varying path counts

Each runs work chunks until --duration elapses and emits chunk events; results are
checked for sanity (finite energy, PageRank sums to 1, Aer on GPU, MC near Black-Scholes).
qiskit needs LD_PRELOAD of Aer's bundled libgomp (session.py sets it).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from h200_campaign.events import EventLog  # noqa: E402


def openmm_chunks(rng):
    import openmm as mm
    import openmm.app as app
    import openmm.unit as u
    ff = app.ForceField("amber14-all.xml", "amber14/tip3p.xml")
    plat = mm.Platform.getPlatformByName("CUDA")
    while True:
        side = rng.choice([4.0, 6.0, 8.0])
        mod = app.Modeller(app.Topology(), [])
        mod.addSolvent(ff, boxSize=mm.Vec3(side, side, side) * u.nanometer)
        sysm = ff.createSystem(mod.topology, nonbondedMethod=app.PME, nonbondedCutoff=1 * u.nanometer,
                               constraints=app.HBonds)
        sim = app.Simulation(mod.topology, sysm, mm.LangevinMiddleIntegrator(300 * u.kelvin, 1 / u.picosecond,
                                                                            0.002 * u.picoseconds), plat)
        sim.context.setPositions(mod.positions)
        sim.minimizeEnergy(maxIterations=200)
        for _ in range(rng.choice([5, 10, 20])):
            sim.step(5000)
            e = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
            yield {"box_nm": side, "atoms": mod.topology.getNumAtoms(), "steps": 5000, "potential": e,
                   "ok": math.isfinite(e)}


def rapids_chunks(rng):
    import cudf
    import cugraph
    import cupy as cp
    while True:
        op = rng.choice(["groupby_join", "sort", "pagerank", "bfs"])
        if op in ("groupby_join", "sort"):
            n = rng.choice([10, 30, 60]) * 1_000_000
            df = cudf.DataFrame({"k": cp.random.randint(0, rng.choice([1000, 100_000, 5_000_000]), n),
                                 "v": cp.random.random(n), "w": cp.random.randint(0, 100, n)})
            if op == "sort":
                r = int(df.sort_values(["w", "v"]).shape[0])
            else:
                g = df.groupby("k").agg({"v": ["mean", "sum"], "w": "max"})
                g.columns = ["vm", "vs", "wm"]
                r = int(df.merge(g.reset_index(), on="k").shape[0])
            yield {"op": op, "rows": n, "result_rows": r, "ok": r == n}
        else:
            nv, ne = rng.choice([1, 4]) * 1_000_000, rng.choice([10, 40]) * 1_000_000
            src = cp.random.randint(0, nv, ne)
            G = cugraph.Graph(directed=True)
            G.from_cudf_edgelist(cudf.DataFrame({"s": src, "d": cp.random.randint(0, nv, ne)}),
                                 source="s", destination="d")
            if op == "pagerank":
                s = float(cugraph.pagerank(G).pagerank.sum())
                yield {"op": op, "vertices": nv, "edges": ne, "pagerank_sum": s, "ok": abs(s - 1) < 1e-3}
            else:
                d = cugraph.bfs(G, start=int(src[0]))      # a vertex that is in the graph
                yield {"op": op, "vertices": nv, "edges": ne, "reached": int((d.distance < 2**31 - 1).sum()), "ok": True}


def qiskit_chunks(rng):
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import QFTGate
    from qiskit_aer import AerSimulator
    sim = AerSimulator(method="statevector", device="GPU")
    while True:
        nq = rng.choice([24, 26, 28, 30])
        qc = QuantumCircuit(nq)
        if rng.random() < 0.5:
            qc.h(range(nq))
            qc.append(QFTGate(nq), range(nq))
            kind = "qft"
        else:
            for _ in range(rng.choice([10, 20])):
                for q in range(nq):
                    qc.rx(rng.random() * math.pi, q)
                for q in range(0, nq - 1, 2):
                    qc.cx(q, q + 1)
            kind = "random_layers"
        qc.measure_all()
        res = sim.run(transpile(qc, sim), shots=256).result()
        dev = res.results[0].metadata.get("device")
        yield {"kind": kind, "qubits": nq, "device": dev, "ok": res.success and dev == "GPU"}


def cupymc_chunks(rng):
    import cupy as cp
    from math import erf, exp, log, sqrt
    N = lambda x: 0.5 * (1 + erf(x / sqrt(2)))
    while True:
        S0, K, r, sig, T = 100.0, rng.choice([90.0, 100.0, 110.0]), 0.03, rng.choice([0.15, 0.25, 0.4]), 1.0
        n = rng.choice([20, 50, 100]) * 1_000_000
        if rng.random() < 0.6:
            z = cp.random.standard_normal(n, dtype=cp.float32)
            ST = S0 * cp.exp((r - 0.5 * sig ** 2) * T + sig * math.sqrt(T) * z)
            price = float(math.exp(-r * T) * cp.maximum(ST - K, 0).mean())
            d1 = (log(S0 / K) + (r + sig ** 2 / 2) * T) / (sig * sqrt(T))
            bs = S0 * N(d1) - K * exp(-r * T) * N(d1 - sig * sqrt(T))
            yield {"kind": "european", "paths": n, "price": price, "black_scholes": bs, "ok": abs(price - bs) < 0.1}
        else:   # arithmetic Asian option, 64 monitoring dates, path-chunked
            steps, paths = 64, n // 64
            dt = T / steps
            S = cp.full(paths, S0, dtype=cp.float32)
            acc = cp.zeros(paths, dtype=cp.float32)
            for _ in range(steps):
                S *= cp.exp((r - 0.5 * sig ** 2) * dt + sig * math.sqrt(dt) * cp.random.standard_normal(paths, dtype=cp.float32))
                acc += S
            price = float(math.exp(-r * T) * cp.maximum(acc / steps - K, 0).mean())
            yield {"kind": "asian", "paths": paths, "price": price, "ok": math.isfinite(price) and price > 0}


APPS = {"openmm": openmm_chunks, "rapids": rapids_chunks, "qiskit": qiskit_chunks, "cupymc": cupymc_chunks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", required=True, choices=sorted(APPS))
    ap.add_argument("--duration", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--events", default=None)
    args = ap.parse_args()
    log = EventLog(args.events, stream="workload")
    log.emit("process_start", argv=sys.argv, app=args.app)
    rng = random.Random(args.seed)
    log.emit("load_start")
    gen = APPS[args.app](rng)
    log.emit("work_start")
    t0, n, bad = time.monotonic(), 0, 0
    first = True
    while time.monotonic() - t0 < args.duration:
        c = next(gen)
        if first:
            log.emit("load_end_first_chunk_done")
            first = False
        n += 1
        bad += 0 if c["ok"] else 1
        log.emit("chunk", ok=c["ok"], **{k: v for k, v in c.items() if k != "ok"})
    log.emit("work_end", chunks=n, failed=bad)
    Path(args.metrics).write_text(json.dumps({"app": args.app, "chunks": n, "failed_chunks": bad,
                                              "wall_s": time.monotonic() - t0, "seed": args.seed}, indent=1))
    log.emit("process_exit")
    log.close()


if __name__ == "__main__":
    main()
