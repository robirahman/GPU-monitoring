"""Generate a frozen randomized pilot/core panel without contacting a cluster."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from experiments.manifest import SCHEMA_VERSION, atomic_json, canonical_hash, identifier, validate_campaign

# Exactly twelve conditions; labels never depend on substring heuristics.
CONDITIONS = {
    "dense_ddp": ("ml_training", "ddp"),
    "full_fsdp": ("ml_training", "fsdp"),
    "lora_ddp": ("ml_training", "lora"),
    "accum_diluted": ("ml_training", "diluted"),
    "local_sgd": ("ml_training", "local_sgd"),
    "fused_sgd": ("ml_training", "fused_sgd"),
    "matched_prefill": ("ml_inference", "prefill"),
    "matched_decode": ("ml_inference", "decode"),
    "continuous_batch": ("ml_inference", "serving"),
    "backward_no_update": ("ml_inference", "no_update"),
    "distributed_fft": ("other", "fft"),
    "collectives": ("other", "collectives"),
}


def generate(campaign_id, stage="pilot", seed=20261007, fabric="rdma"):
    identifier(campaign_id)
    if stage not in {"pilot", "core", "transfer"} or fabric not in {"rdma", "tcp"}:
        raise ValueError("Invalid stage/fabric")
    node_counts = [1, 2, 4] if stage == "core" else [2]
    reps, duration = (5, 1200) if stage == "core" else (3, 600 if stage == "pilot" else 1200)
    panel = CONDITIONS if stage != "transfer" else {k: CONDITIONS[k] for k in
             ("dense_ddp", "full_fsdp", "matched_prefill", "distributed_fft")}
    rng = random.Random(seed)
    jobs = []
    # Rep-major blocks avoid aliasing temperature drift with condition identity.
    for rep in range(1, reps + 1):
        block = [(name, n) for n in node_counts for name in panel]
        rng.shuffle(block)
        for name, nodes in block:
            label, mode = panel[name]
            split = "pilot" if stage == "pilot" else (
                "test" if stage == "transfer" or rep == 5 else "validation" if rep == 4 else "train")
            job_id = f"{campaign_id}.{name}.n{nodes}.r{rep}"
            jobs.append({"job_id": job_id, "condition": name, "class_label": label,
                         "mode": mode, "nodes": nodes, "gpus_per_node": 8,
                         "duration_sec": duration, "rep": rep, "split": split,
                         "fabric": fabric, "seed": seed + rep,
                         "model": "Qwen/Qwen2.5-1.5B", "batch_size": 4,
                         "seq_len": 512, "dtype": "bfloat16",
                         "accumulation": 4, "dilution": 10, "sync_every": 16})
    doc = {"schema_version": SCHEMA_VERSION, "campaign_id": campaign_id, "stage": stage,
           "seed": seed, "target_gpu": "NVIDIA H100 80GB", "jobs": jobs,
           "observation": ["nvml", "dcgm", "network", "bmc"],
           "false_alert_budget_per_hour": 0.001,
           "window_sec": 30, "stride_sec": 15,
           "visibility_fractions": [0.25, 0.5, 1.0]}
    doc["manifest_sha256"] = canonical_hash(doc)
    return validate_campaign(doc)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--campaign-id", required=True)
    ap.add_argument("--stage", choices=["pilot", "core", "transfer"], default="pilot")
    ap.add_argument("--fabric", choices=["rdma", "tcp"], default="rdma")
    ap.add_argument("--seed", type=int, default=20261007)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        ap.error("Output exists; use a new campaign path")
    doc = generate(args.campaign_id, args.stage, args.seed, args.fabric)
    atomic_json(args.out, doc)
    print(json.dumps({"jobs": len(doc["jobs"]),
                      "cluster_hours": sum(j["duration_sec"] for j in doc["jobs"]) / 3600,
                      "gpu_hours": sum(j["duration_sec"] * j["nodes"] * j["gpus_per_node"]
                                       for j in doc["jobs"]) / 3600,
                      "manifest_sha256": doc["manifest_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
