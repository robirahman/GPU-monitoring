#!/usr/bin/env python3
"""Build the paper corpus extended with the Round 6 collection campaign.

cache/paper_corpus.parquet is tier A only, and the whole Round 6 campaign
(symmetric-GEMM ladder, PPO-RLHF, 16-rank distributed RL) sits in B_rebuttal /
C_followup, so it is absent from every tier-A cache. The arms-race simulation
needs it in the same frame as rounds 1-5, restricted to the same nine GPUs and
built through the same substitution path, or the Round 6 row is not comparable
with the rows above it.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, ".")
import pandas as pd

from classifier.data_sources import restrict_to_paper_corpus
from classifier.threeway_improved import load_all_data

ROOT = Path(".").resolve()
OUT = ROOT / "cache" / "paper_corpus_r6.parquet"
# Every cell collected outside tier A that the arms-race analysis needs. The
# tiers are frozen by collection date, so the whole Round 6 campaign and its
# later controls sit in B_rebuttal / C_followup and no tier-A cache contains
# them. ppo_rlhf matches ppo_rlhf_accum* too; s\\d+ covers the 8- and 16-rank
# distributed RL cells (the 16-rank one is then dropped by the single-node
# scope filter, not here, so the exclusion stays visible in one place).
PAT = r"^(symgemm_|ppo_rlhf|bestofn_rm|s\d+_train_ppodist)"

t0 = time.time()
man = pd.read_parquet("docs/camera_ready_manifest_20261022/streams.parquet")
r6 = man[man.base.str.match(PAT, na=False)]
files = sorted({str((ROOT / f).resolve()) for f in r6.file})
print(f"Round 6: {len(r6)} streams -> {len(files)} files", flush=True)

raw = load_all_data(tier="A_camera_ready", extra_files=files)
print(f"  [{time.time()-t0:.0f}s] raw rows {len(raw):,}", flush=True)
paper = restrict_to_paper_corpus(raw)
del raw
print(f"  [{time.time()-t0:.0f}s] restricted rows {len(paper):,} "
      f"runs {paper.run_id.nunique():,} workloads {paper.workload_label.nunique()} "
      f"gpus {paper.paper_gpu.nunique()}", flush=True)

is_r6 = paper.workload_label.str.match(PAT, na=False)
print(f"  Round 6 rows kept: {int(is_r6.sum()):,} over "
      f"{paper.loc[is_r6, 'run_id'].nunique()} runs, "
      f"{paper.loc[is_r6, 'workload_label'].nunique()} labels", flush=True)
print("  Round 6 label -> threeway_label:")
tw = (paper[is_r6].groupby(["workload_label", "threeway_label"]).run_id.nunique()
      .reset_index().sort_values("workload_label"))
for _, r in tw.iterrows():
    print(f"    {r.workload_label:36s} {r.threeway_label:14s} runs={r.run_id}")

paper.to_parquet(OUT, index=False)
print(f"[{time.time()-t0:.0f}s] wrote {OUT} ({OUT.stat().st_size/1e6:.1f} MB)", flush=True)
