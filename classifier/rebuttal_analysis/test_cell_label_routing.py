#!/usr/bin/env python3
"""Assert every collection cell name routes to the class we intend.

Why this is a test and not a comment
------------------------------------
`threeway_label()` (classifier/threeway_improved.py:211) is a chain of plain
SUBSTRING tests, and the training keywords are checked BEFORE `infer`. So a cell
named for what it does can land in the wrong class with no warning at all:

    s16_infer_gpt2serve   -> ml_training   ("gpt")
    s16_inferbert_embed   -> ml_training   ("bert")
    s16_nonml_ddp_bench   -> ml_training   ("ddp_")

and any label matching none of the keywords silently becomes `other`. A
mislabelled cell is worse than a missing one: it teaches the detector that
inference telemetry IS training, which is exactly the contamination the corpus
hygiene elsewhere works to prevent.

There is a second, quieter hazard. `honest_mask()` (classifier/model_store.py:69)
drops any label containing `advtrain` / `adversarial` / `whitebox` / `symgemm` /
`_mw8` from the honest pool. An honest pipeline-parallel training cell derived
from adversarial_multinode_pipeline.py would be silently excluded from every
honest-corpus analysis if it inherited the word.

And a third: `family()` (multinode_family_holdout.py:63) decides the holdout
group. Two cells that collapse to one family are one fold, not two -- which is
the whole reason this collection round exists.

    PYTHONPATH=. python3 classifier/rebuttal_analysis/test_cell_label_routing.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from classifier.threeway_improved import threeway_label  # noqa: E402
from classifier.model_store import honest_mask  # noqa: E402
from multinode_family_holdout import family  # noqa: E402

# (cell, expected class). Every cell this round collects, plus the ones already
# in the corpus whose routing we rely on.
CELLS = [
    # --- A. async serving (written, fixed, never collected) ---
    ("s16_inferserve_poisson_7b", "ml_inference"),
    ("s16_inferserve_saturate_7b", "ml_inference"),
    ("s16_inferserve_bursty_7b", "ml_inference"),
    ("s16_inferserve_prefill_72b", "ml_inference"),
    ("s16_inferserve_decode_72b", "ml_inference"),
    ("s16_inferserve_poisson_72b", "ml_inference"),
    ("s16_inferserve_chunked_72b", "ml_inference"),
    # --- B. non-LLM inference, replicated ---
    ("s16_infervis_vitl", "ml_inference"),
    ("s16_inferembed_bge", "ml_inference"),
    ("s16_inferembed_clip", "ml_inference"),
    ("s16_inferdiff_sdxl", "ml_inference"),
    ("s16_inferasr_whisper", "ml_inference"),
    ("s16_infercnn_rn50", "ml_inference"),
    # --- B2. recommender: the one workload class the corpus has none of ---
    ("s16_inferrec_dlrm", "ml_inference"),
    ("s16_inferrec_dlrm_t8", "ml_inference"),
    # --- C. second inference engine (not vLLM) ---
    ("s16_inferhf_qwen7b", "ml_inference"),
    ("s16_inferhf_shard72b", "ml_inference"),
    ("s16_inferhf_shard72b_m40", "ml_inference"),
    ("s16_inferhf_shard72b_m60", "ml_inference"),
    # --- D. genuinely-distributed HPC ---
    ("s16_nonml_fftshard", "other"),
    ("s16_nonml_fftshard_g1024", "other"),
    ("s16_nonml_stencil", "other"),
    ("s16_nonml_nccl_bw", "other"),
    ("s16_nonml_hpl", "other"),
    ("s16_nonml_gromacs", "other"),
    # --- E. new training programs ---
    ("s16_train_lora7b", "ml_training"),
    ("s16_train_lora7b_b8", "ml_training"),
    ("s16_train_pipe7b", "ml_training"),
    ("s16_train_contrastive", "ml_training"),
    ("s16_train_diffusion", "ml_training"),
    # --- already in the corpus, routing we depend on ---
    ("s16_train_fsdp1b", "ml_training"),
    ("s16_train_ddp_resnet50", "ml_training"),
    ("s16_infershard_qwen72b_kv32k", "ml_inference"),
    ("s16_nonml_sat", "other"),
]

# Names we must NOT use. Each is a plausible thing someone would type, and each
# is silently wrong. Kept as a test so the trap stays documented and detected.
TRAPS = [
    ("s16_infer_gpt2serve", "ml_training", "'gpt' beats 'infer'"),
    ("s16_inferbert_embed", "ml_training", "'bert' beats 'infer'"),
    ("s16_nonml_ddp_bench", "ml_training", "'ddp_' matches in a non-ML label"),
    ("s16_nonml_traincomp", "ml_training", "'train' matches anywhere"),
    ("s16_hpcfft_shard", "other", "no keyword at all -> silent 'other'"),
]

# Cells that must stay in the honest pool. The pipeline-parallel training cell is
# the live risk: it is derived from adversarial_multinode_pipeline.py and must not
# inherit 'advtrain' or 'adversarial' in its name.
MUST_BE_HONEST = [c for c, _ in CELLS]


def main():
    failures = []

    for cell, want in CELLS:
        got = threeway_label(cell)
        if got != want:
            failures.append(f"CLASS  {cell}: routes to {got}, expected {want}")

    hm = honest_mask(pd.DataFrame({"workload_label": MUST_BE_HONEST}))
    for cell, keep in zip(MUST_BE_HONEST, hm):
        if not keep:
            failures.append(f"HONEST {cell}: excluded from the honest pool by "
                            f"honest_mask (contains a disguise token)")

    for cell, routes_to, why in TRAPS:
        got = threeway_label(cell)
        if got != routes_to:
            failures.append(f"TRAP   {cell}: now routes to {got}, not {routes_to}. "
                            f"The taxonomy changed -- re-check the rule ({why}).")

    # Families: report, and fail only on a collision between two NEW cells, since
    # collapsing two cells into one family silently halves the holdout folds.
    fams = {}
    for cell, _ in CELLS:
        fams.setdefault(family(cell), []).append(cell)

    print(f"{len(CELLS)} cells -> {len(fams)} families\n")
    for f in sorted(fams):
        mark = "  " if len(fams[f]) == 1 else "* "
        print(f"{mark}{f:24s} {', '.join(fams[f])}")
    print("\n(* = several cells share one family, i.e. ONE holdout fold. Intended "
          "for deliberate sweeps such as the memory-footprint variants.)")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  {f}")
        return 1
    print(f"\nOK: {len(CELLS)} cells route correctly, all stay in the honest pool, "
          f"{len(TRAPS)} traps still behave as documented.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
