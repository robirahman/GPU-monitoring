#!/usr/bin/env python3
"""Generate a validation-pilot session plan (COLLECTION_PLAN.md 'First: a small validation pilot').

Contents per session: Phi-3 and Whisper inference, BERT / MLP / ResNet ordinary
training, one ordinary-training / evasion pair (ResNet ordinary vs diluted x5,
same seed and data), non-ML work (cuFFT, path-tracing render), idle, and an
instrumentation-overhead control (ResNet ordinary with all sensors and per-update
GPU marks off). Order is shuffled with --seed, recorded in the plan.

  python -m h200_campaign.make_pilot --session-id pilot_s1 --seed 1 > plan.json
"""
import argparse
import json
import random

RUNS = [
    dict(label="bert_sst2_ord", class_label="ml_training", family="bert", kind="train",
         argv=["--family", "bert", "--duration", "300", "--seed", "0"], timeout_s=900),
    dict(label="mlp_cifar10_ord", class_label="ml_training", family="mlp", kind="train",
         argv=["--family", "mlp", "--duration", "180", "--seed", "0", "--gpu-marks", "1"], timeout_s=600),
    dict(label="resnet18_cifar10_ord", class_label="ml_training", family="resnet", kind="train",
         argv=["--family", "resnet", "--duration", "300", "--seed", "0"], timeout_s=900, job_id_suffix="resnet_pair"),
    dict(label="resnet18_cifar10_dil5", class_label="ml_training", family="resnet", kind="train",
         argv=["--family", "resnet", "--mode", "diluted", "--dilution", "5", "--duration", "300", "--seed", "0"],
         timeout_s=900, job_id_suffix="resnet_pair", attack={"strategy": "diluted_updates", "dilution": 5,
                                                             "matched_control": "resnet18_cifar10_ord"}),
    dict(label="resnet18_cifar10_ord_noinstr", class_label="ml_training", family="resnet", kind="train",
         argv=["--family", "resnet", "--duration", "300", "--seed", "0", "--gpu-marks", "0"], timeout_s=900,
         sensors=False, job_id_suffix="resnet_pair", purpose="instrumentation overhead control"),
    dict(label="phi3_serve", class_label="ml_inference", family="phi3", kind="serve",
         argv=["--model", "phi3", "--schedule", "60:0.5,30:0,60:2,30:0.2", "--seed", "0"], timeout_s=900),
    dict(label="whisper_serve", class_label="ml_inference", family="whisper", kind="serve",
         argv=["--model", "whisper", "--schedule", "60:1,30:0,60:3", "--seed", "0"], timeout_s=900),
    dict(label="cufft", class_label="other", family="cufft", kind="nonml", workload="cufft",
         argv=["--duration", "120"], timeout_s=300),
    dict(label="render_pathtrace", class_label="other", family="render", kind="nonml", workload="render",
         argv=["--duration", "120"], timeout_s=300),
    dict(label="idle", class_label="other", family="idle", kind="nonml", workload="idle",
         argv=["--duration", "60"], timeout_s=200),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session-id", required=True)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    runs = [dict(r) for r in RUNS]
    random.Random(args.seed).shuffle(runs)
    for r in runs:
        suf = r.pop("job_id_suffix", None)
        if suf:
            r["job_id"] = f"{args.session_id}-{suf}"
    plan = {"session_id": args.session_id, "partition": "pilot", "order_seed": args.seed,
            "notes": "validation pilot; order shuffled; resnet ord/dil5/noinstr share a job_id (same partition group)",
            "sensors": {"scope_range": "V10", "scope_raw": "all", "scope_hires": "every:60:1:20", "dcgm": True},
            "runs": runs}
    print(json.dumps(plan, indent=1))


if __name__ == "__main__":
    main()
