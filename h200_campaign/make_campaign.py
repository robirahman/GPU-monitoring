#!/usr/bin/env python3
"""Generate the main collection campaign (after the validation pilot).

Partitions follow h200_campaign/partition/draw_result.json. For each partition:
  benign      ~30 h of long normal-operation timelines (inference, non-ML, idle gaps,
              startup, transitions, some concurrency); no training
  type1       ordinary training with full phase events: short (~90 s) x2 and long (~20 min, with checkpoints) x1
  attacks     per attacked lineage x 2 reps: ordinary (fixed update budget U, cosine LR), diluted x5,
              interleave (5 s train / 15 s cuFFT), shape (40% duty), each to the same U; plus a cover-only control
  bursts      type-4 sequences: serving + training bursts (duration, spacing, window offset varied) and
              inference-only controls, with 180 s observation after the last burst
Runs are shuffled into ~10 h blocks; blocks alternate calibration / test. Hour rules
(constraints.json collection_rules) are checked; a violating timeline set is regenerated
and the rejection logged.

  python -m h200_campaign.make_campaign --seed S --rates rates.json --out-dir ~/campaign/plans/main
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent

# benign pools: (lineage, kind, argv builder) -- duration d seconds, rng r
CAL_BENIGN = {
    "phi3": ("serve", lambda d, r: ["--model", "phi3", "--schedule", _sched(d, r, [0.03, 0.08, 0.15, 0.3])]),
    "whisper": ("serve", lambda d, r: ["--model", "whisper", "--schedule", _sched(d, r, [0.3, 1, 2, 4])]),
    "wav2vec2": ("serve", lambda d, r: ["--model", "wav2vec2", "--schedule", _sched(d, r, [0.5, 2, 5, 10])]),
    "cufft": ("cufft", lambda d, r: ["--duration", str(int(d)), "--size", str(r.choice([1024, 2048, 4096]))]),
    "render": ("render", lambda d, r: ["--duration", str(int(d)), "--rays", str(r.choice([16384, 32768, 65536]))]),
    "nbody": ("nbody", lambda d, r: ["--duration", str(int(d)), "--particles", str(r.choice([8192, 16384, 32768]))]),
    "cupy_montecarlo": ("nonml", lambda d, r: ["--app", "cupymc", "--duration", str(int(d)), "--seed", str(r.randrange(10**6))]),
    "qiskit_aer": ("nonml", lambda d, r: ["--app", "qiskit", "--duration", str(int(d)), "--seed", str(r.randrange(10**6))]),
}
TEST_BENIGN = {
    "olmo": ("serve", lambda d, r: ["--model", "olmo", "--schedule", _sched(d, r, [0.03, 0.07, 0.12, 0.2])]),
    "pythia": ("serve", lambda d, r: ["--model", "pythia", "--schedule", _sched(d, r, [0.05, 0.1, 0.2, 0.4])]),
    "efficientnet": ("serve", lambda d, r: ["--model", "effnet", "--schedule", _sched(d, r, [1, 5, 20, 50])]),
    "bark": ("serve", lambda d, r: ["--model", "bark", "--schedule", _sched(d, r, [0.02, 0.05, 0.08, 0.12])]),
    "dit": ("serve", lambda d, r: ["--model", "dit", "--schedule", _sched(d, r, [0.1, 0.3, 0.6, 1.0])]),
    "openmm": ("nonml", lambda d, r: ["--app", "openmm", "--duration", str(int(d)), "--seed", str(r.randrange(10**6))]),
    "rapids": ("nonml", lambda d, r: ["--app", "rapids", "--duration", str(int(d)), "--seed", str(r.randrange(10**6))]),
}
CLASS = {"serve": "ml_inference", "cufft": "other", "render": "other", "nbody": "other", "nonml": "other"}

# training lineages: train.py argv prefix, used by type1, attacks and bursts
CAL_TRAIN = {"mlp": ["--family", "mlp"], "resnet": ["--family", "resnet"], "bert": ["--family", "bert"],
             "qwen": ["--family", "lora", "--model", "Qwen/Qwen2.5-0.5B", "--dataset", "wikitext"]}
TEST_TRAIN = {"efficientnet": ["--family", "effnet"],
              "olmo": ["--family", "lora", "--model", "allenai/OLMo-2-0425-1B", "--dataset", "tinystories", "--eval-tokens", "16384"],
              "pythia": ["--family", "lora", "--model", "EleutherAI/pythia-1b", "--dataset", "tinystories", "--eval-tokens", "16384"]}
ATTACKED = {"calibration": ["resnet", "bert", "qwen"], "test": ["efficientnet", "olmo", "pythia"]}
BURST_SERVERS = {"calibration": ["phi3", "whisper"], "test": ["olmo", "bark", "dit", "efficientnet"]}
SERVE_MODEL = {"phi3": "phi3", "whisper": "whisper", "olmo": "olmo", "bark": "bark", "dit": "dit", "efficientnet": "effnet"}
SERVE_RATE = {"phi3": 0.1, "whisper": 1.0, "olmo": 0.1, "bark": 0.05, "dit": 0.3, "efficientnet": 10.0}


def _sched(d, r, rates):
    """Request-rate schedule filling d seconds: 2-4 sub-segments, sometimes an idle (rate 0) stretch."""
    n = r.choice([2, 3, 4])
    cuts = sorted(r.uniform(0.15, 0.85) * d for _ in range(n - 1))
    parts = [b - a for a, b in zip([0] + cuts, cuts + [d])]
    return ",".join(f"{max(5, int(p))}:{0 if r.random() < 0.15 else r.choice(rates)}" for p in parts)


def make_timeline(pool, r, minutes):
    total, segs, used = minutes * 60.0, [], defaultdict(float)
    t = 0.0
    while t < total - 120:
        gap = r.choice([0, 0, 10, 30, 60, 120])
        d = min(r.uniform(180, 720), total - t - gap)
        if d < 120:
            break
        lin = r.choice(sorted(pool))
        procs = [(lin, *pool[lin])]
        if r.random() < 0.2:          # concurrent unrelated GPU work
            other = r.choice([k for k in sorted(pool) if k != lin])
            procs.append((other, *pool[other]))
        seg = {"gap_before_s": gap, "planned_s": d, "procs": []}
        for l, kind, fn in procs:
            seg["procs"].append({"label": l, "kind": kind, "argv": fn(d, r)})
            used[l] += d
        used["idle"] += gap
        segs.append(seg)
        t += gap + d
    return {"segments": segs}, used, t


def benign_runs(part, pool, hours, r, log):
    for attempt in range(100):
        runs, used, total = [], defaultdict(float), 0.0
        i = -1
        while total < hours * 3600:          # add 50-min timelines until the benign-hour target is reached
            i += 1
            tl, u, t = make_timeline(pool, r, 50)
            for k, v in u.items():
                used[k] += v
            total += t
            runs.append({"label": f"{part[:3]}_benign_{i:02d}", "class_label": "benign_mixed", "family": "mixed",
                         "kind": "timeline", "collection_type": "long_normal_operation", "timeline": tl,
                         "timeout_s": 50 * 60 + 1800, "sensor_overrides": {"scope_raw": "every:60:10"},
                         "extra": {"planned_s": t, "lineage_seconds": dict(u)}})
        busy = sum(v for k, v in used.items() if k != "idle")
        shares = {k: v / total for k, v in used.items()}
        bad = [k for k, v in shares.items() if k != "idle" and v > 0.35] + (["idle"] if shares.get("idle", 0) > 0.10 else [])
        log.append({"partition": part, "attempt": attempt, "hours": total / 3600, "shares": shares, "violations": bad})
        if not bad:
            return runs, {"hours": total / 3600, "shares": shares, "busy_lineage_hours": busy / 3600}
    raise SystemExit(f"no valid benign schedule for {part}")


def train_runs(part, train, rates, r):
    out = []
    for lin, pre in train.items():
        for rep in range(2):
            out.append({"label": f"{part[:3]}_t1_{lin}_short{rep}", "class_label": "ml_training", "family": lin,
                        "kind": "train", "collection_type": "training_timing", "timeout_s": 1200,
                        "argv": pre + ["--duration", "90", "--seed", str(r.randrange(10**6)), "--eval-every-sec", "30"]})
        out.append({"label": f"{part[:3]}_t1_{lin}_long", "class_label": "ml_training", "family": lin, "kind": "train",
                    "collection_type": "training_timing", "timeout_s": 3600,
                    "sensor_overrides": {"scope_raw": "every:60:10"},
                    "argv": pre + ["--duration", "1200", "--seed", str(r.randrange(10**6)), "--eval-every-sec", "60",
                                   "--ckpt-every-sec", "120"]})
    return out


def attack_runs(part, train, rates, r):
    out = []
    for lin in ATTACKED[part]:
        pre, ups = train[lin], rates[lin]["updates_per_s"]
        U = int(ups * 360)                      # ordinary run ~6 min
        base_s = U / ups
        common = ["--updates", str(U), "--lr-schedule", "cosine", "--eval-every", str(max(1, U // 25)),
                  "--eval-every-sec", "1e9"]
        for rep in range(2):
            seed = str(r.randrange(10**6))
            job = f"{part}-attack-{lin}-rep{rep}"
            variants = [("ord", ["--mode", "ordinary"], 2.0, None),
                        ("dil5", ["--mode", "diluted", "--dilution", "5"], 7.0, {"strategy": "diluted_updates", "dilution": 5}),
                        ("int", ["--mode", "interleave", "--train-sec", "5", "--other-sec", "15", "--other", "fft"], 7.0,
                         {"strategy": "interleaving", "train_s": 5, "cover_s": 15, "cover": "cuFFT"}),
                        ("shape", ["--mode", "shape", "--duty", "0.4", "--period-ms", "200"], 5.0,
                         {"strategy": "power_utilization_shaping", "duty": 0.4, "period_ms": 200})]
            for name, argv, cap, att in variants:
                out.append({"label": f"{part[:3]}_atk_{lin}_{name}_r{rep}", "class_label": "ml_training", "family": lin,
                            "kind": "train", "collection_type": "attack" if att else "attack_matched_ordinary",
                            "job_id": job, "attack": att, "timeout_s": int(base_s * cap + 1800),
                            "sensor_overrides": {"scope_raw": "every:60:10"},
                            "argv": pre + common + ["--duration", str(int(base_s * cap)), "--seed", seed] + argv})
            out.append({"label": f"{part[:3]}_atk_{lin}_cover_r{rep}", "class_label": "other", "family": "cufft",
                        "kind": "train", "collection_type": "attack_cover_only_control", "job_id": job,
                        "timeout_s": 2400, "argv": pre + ["--mode", "cover_only", "--other", "fft", "--duration",
                                                          str(int(base_s)), "--seed", seed]})
    return out


def burst_runs(part, train, rates, r):
    out = []
    servers, lins = BURST_SERVERS[part], ATTACKED[part]
    for i in range(12):
        srv, lin = r.choice(servers), r.choice(lins)
        nb = r.choice([1, 2, 3])
        lead = rates[lin]["load_s"] + 30          # launch early enough to finish loading before the start gate
        start, bursts = max(60.0, 10 * round((lead + 20) / 10)) + r.choice([0, 3, 7]), []
        for b in range(nb):
            dur = r.choice([10, 20, 45, 90])
            bursts.append(f"{dur}@{start:.0f}" if b else f"{dur}@{start:.1f}")
            start += dur + r.choice([30, 60, 120]) + r.choice([0, 3, 7])
        out.append({"label": f"{part[:3]}_seq_{i:02d}_{srv}_{lin}", "class_label": "mixed", "family": f"{lin}+{srv}",
                    "kind": "sequence", "collection_type": "burst_in_operation", "timeout_s": 3600,
                    "argv": ["--serve-model", SERVE_MODEL[srv], "--serve-rate", str(SERVE_RATE[srv]),
                             "--bursts", ",".join(bursts), "--burst-family", train[lin][1],
                             "--burst-args=" + " ".join(train[lin][2:]), "--observe", "180", "--lead", f"{lead:.0f}",
                             "--seed", str(r.randrange(10**6))]})
    for i in range(4):
        srv = r.choice(servers)
        out.append({"label": f"{part[:3]}_seqctl_{i}_{srv}", "class_label": "ml_inference", "family": srv,
                    "kind": "sequence", "collection_type": "burst_control_inference_only", "timeout_s": 3600,
                    "argv": ["--serve-model", SERVE_MODEL[srv], "--serve-rate", str(SERVE_RATE[srv]), "--bursts", "",
                             "--pre", "300", "--observe", "180", "--seed", str(r.randrange(10**6))]})
    return out


def blocks(runs, r, hours_per_block, est):
    r.shuffle(runs)
    out, cur, h = [], [], 0.0
    for x in runs:
        cur.append(x)
        h += est(x)
        if h >= hours_per_block:
            out.append(cur)
            cur, h = [], 0.0
    if cur:
        out.append(cur)
    return out


def est_hours(x):
    if x["kind"] == "timeline":
        return x["extra"]["planned_s"] / 3600 + 0.05
    if x["kind"] == "sequence":
        return 0.25
    a = x["argv"]
    return float(a[a.index("--duration") + 1]) / 3600 * (0.6 if x.get("attack") else 1.0) + 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--rates", required=True, help="JSON {lineage: {updates_per_s: x}} measured in smoke tests")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--benign-hours", type=float, default=30.0)
    args = ap.parse_args()
    rates = json.loads(Path(args.rates).read_text())
    r = random.Random(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log, parts = [], {}
    for part, pool, train in (("calibration", CAL_BENIGN, CAL_TRAIN), ("test", TEST_BENIGN, TEST_TRAIN)):
        ben, summ = benign_runs(part, pool, args.benign_hours, r, log)
        runs = ben + train_runs(part, train, rates, r) + attack_runs(part, train, rates, r) + burst_runs(part, train, rates, r)
        parts[part] = {"runs": runs, "benign": summ, "blocks": blocks(runs, r, 10.0, est_hours)}
    order = []
    nb = max(len(p["blocks"]) for p in parts.values())
    for i in range(nb):
        for part in ("calibration", "test"):
            if i < len(parts[part]["blocks"]):
                order.append((part, i))
    files = []
    for k, (part, i) in enumerate(order):
        sid = f"main_{k:02d}_{part[:3]}{i}"
        plan = {"session_id": sid, "partition": part, "block": i, "order_index": k, "campaign_seed": args.seed,
                "sensors": {"scope_range": "V10", "scope_raw": "all", "scope_hires": "none", "dcgm": True},
                "runs": parts[part]["blocks"][i]}
        f = out / f"{sid}.json"
        f.write_text(json.dumps(plan, indent=1))
        files.append(f.name)
    summary = {"seed": args.seed, "order": files, "benign": {p: v["benign"] for p, v in parts.items()},
               "run_counts": {p: {t: sum(1 for x in v["runs"] if x["collection_type"] == t)
                                  for t in sorted({x["collection_type"] for x in v["runs"]})} for p, v in parts.items()},
               "estimated_hours": {p: round(sum(est_hours(x) for x in v["runs"]), 1) for p, v in parts.items()},
               "benign_schedule_attempts": log}
    (out / "campaign_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("order", "benign", "run_counts", "estimated_hours")}, indent=1))


if __name__ == "__main__":
    main()
