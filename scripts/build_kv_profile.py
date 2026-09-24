#!/usr/bin/env python3
"""Build the empirical KV-handoff profile W18 mimics from a REAL disagg run's proxy log.

    python scripts/build_kv_profile.py \\
        --handoffs data/disagg_kv_vllm_qwen7b_handoffs.jsonl \\
        --out data/profiles/kv_handoff_disagg_kv_vllm_qwen7b.json

Input is the ``--handoff-log`` written by ``workloads/disagg_kv_vllm.py`` (one JSON record
per prefill->decode handoff). Sizes are the proxy's KV-block estimates; the profile says so
in its ``provenance`` field. Then::

    torchrun --nproc_per_node=8 workloads/train_kv_disguise.py --mode p2p \\
        --schedule-from data/profiles/kv_handoff_disagg_kv_vllm_qwen7b.json ...
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "workloads"))
from kv_profile import build_profile  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--handoffs", nargs="+", required=True, help="proxy handoff JSONL file(s)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", default="", help="provenance label (default: the input file names)")
    ap.add_argument("--min-bytes", type=int, default=1)
    args = ap.parse_args()

    records = []
    for path in args.handoffs:
        with open(path) as f:
            records += [json.loads(line) for line in f if line.strip()]
    profile = build_profile(records, source=args.source or ",".join(os.path.basename(p) for p in args.handoffs),
                            min_bytes=args.min_bytes)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(profile, f, indent=1)
    sz = profile["size_bytes"]
    print(f"profile: {profile['n_handoffs']} handoffs, size p10/p50/p90 = "
          f"{sz['p10'] / 2**20:.1f}/{sz['p50'] / 2**20:.1f}/{sz['p90'] / 2**20:.1f} MiB, "
          f"{profile['duty']['handoffs_per_s']:.2f} handoffs/s -> {args.out}")


if __name__ == "__main__":
    main()
