# Pre-registration: held-out test lineages for the H200 follow-up campaign

Committed before any new calibration or test data is collected, before the feasibility results exist, and
before the draw is run on the real pool. The commit that adds this file, `lineages.json` and
`constraints.json` (including the seed) is pushed to GitHub before `draw.py` is executed; its timestamp is the
record of order.

## Why a constrained random draw

Hand-picking test families invites the suspicion that they were chosen because the detector handles them.
A plain lottery over a handful of families can produce an infeasible or lopsided test set. So the test
lineages are drawn at random, under constraints written down here first.

## Procedure

1. **Lineages, not models** (`lineages.json` `lineage_rule`). Size variants, fine-tunes, distillations,
   shared pretrained components and declared architecture/tokenizer adoption tie models into one lineage. All
   uses of a lineage land in one partition.
2. **Seen lineages are excluded from test.** Every lineage with any file in this repository's data, on any GPU
   or campaign, counts as seen (`lineages.json` `seen`, with file counts and example labels), because that
   data may train or calibrate the detector. This includes ViT, TinyLlama (so Llama), Mistral/Mixtral, Qwen,
   Phi-3, SmolLM2, GPT-2, BERT, Whisper, SDXL, CLIP, YOLO, DLRM, N-body, GROMACS, Blender, HPL/HPCG, cuFFT,
   rendering, mining and video encoding.
3. **Candidate pool** of unseen, openly licensed lineages per class (`lineages.json` `candidates`):
   9 training-capable lineages in 4 strata (LLM decoder: Pythia, OLMo, Falcon, Granite, Mamba; vision: Swin,
   EfficientNet; seq2seq: T5; graph: GIN), 3 inference-only lineages (wav2vec2, DiT, Bark) and 4 non-ML
   application families (OpenMM, RAPIDS, Qiskit Aer, CuPy Monte Carlo). Exclusions and reasons are listed in
   `excluded_before_draw` (gated models such as Gemma and Llama; lineages tied to seen ones).
4. **Feasibility first.** `feasibility.py` runs each candidate for a few real steps on the H200 (real data,
   training steps or a real inference request, or a short non-ML GPU run). Failures are excluded before the
   draw with the recorded error. No sensor data is collected and nothing from these checks is used for
   calibration.
5. **Constraints** (`constraints.json`): the test partition gets 3 training-capable lineages (the attacked test
   lineages), 2 inference-only lineages and 2 non-ML lineages. The 3 training lineages must include an LLM
   decoder and at least one of vision / seq2seq / graph, spanning ≥2 strata. Collection-time rules (30 + 30
   benign hours, ≤35% of a partition's benign hours per lineage, ≤10% idle, ordinary training in calibration,
   separate sessions and days) are checked when the schedules are built.
6. **Draw** (`draw.py`): attempt *i* uses `random.Random(f"{seed}:{i}")` on the sorted eligible ids of each
   pool. An attempt that breaks a constraint is logged with the violation and the next attempt is used. The
   first valid attempt is the result. Constraints are never relaxed afterwards; if no valid draw exists, the
   draw stops and is reported.

## Seed

`constraints.json` → `seed` = 4390937826454204550, from `secrets.randbits(63)`, generated once.
