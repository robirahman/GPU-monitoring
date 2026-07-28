# Rebuttal analysis (NeurIPS 2026) — code and computed results

Analyses written during the rebuttal collection push, kept so every number in
the response can be regenerated. Run from the repo root with
`PYTHONPATH=/root/GPU-monitoring` (they import `classifier.threeway_improved`).

The matching collection queues are in `scripts/rebuttal_collection/`; the
per-stage outcomes are in `logs/queue_master.log`.

## Scripts

| file | what it answers |
|---|---|
| `q3q4.py` | M7wU Q3: ROC/PR per evasion family, family held out (unseen). Also scores every 7B operating point. |
| `lora_in_training.py` | Does putting LoRA examples in the training corpus fix detection? Four conditions, strictly grouped splits. |
| `scale_ablation.py` | **SUPERSEDED — do not cite.** Earlier version of the above; splits one inference run across train/test, so its AUC = 1.000 is within-run leakage. Kept only to document the error. |
| `q2_confusion.py` | Per-window confusion vs STEP-log ground truth (does the classifier clear genuinely idle windows?). |
| `analyze_correlation.py` | Window autocorrelation / two-state Markov transitions, overlapping vs disjoint windows. |
| `select_dilution.py` | Scores the 7B dilution pilots to pick a mid-band operating point; also reports the continuous-`p_train` ACF of the 8 h run. |
| `check_ptrain.py` | `p_train` distribution check — confirms the 7B negative result is real, not a degenerate all-zero scoring path. |

## Results (JSON)

`q3q4.json`, `lora_in_training.json`, `q2_confusion.json`, `scale_ablation.json`
(superseded), `pilot_scores.json`.

## Methodological notes that changed conclusions

1. **Overlapping windows inflate autocorrelation.** At 30 s windows / 15 s
   stride adjacent windows share 50% of their samples, giving lag-1 ρ ≈ +0.4
   from the overlap alone. Between disjoint windows it is ≈ 0. Any lag-1 figure
   computed at the default stride is an upper bound, not a measurement.

2. **Pooling transitions across runs measures between-run heterogeneity, not
   within-run structure.** 89% of training runs have a constant alert flag, so
   pooled P(alert|prev alert) = 0.993 says only that runs differ. The
   within-run quantity is ICC(1) = 0.757 (training class) / 0.985
   unconditionally, flat from 30 s to 4 h of separation — i.e. an 8 h run at
   1 Hz carries ≈1.3 independent windows' worth of information.

3. **Grouped splits are mandatory here.** Because ICC ≈ 0.98, letting any part
   of a run into training makes that run trivially classifiable. This is the
   same failure mode as the paper's own within-run leak; `scale_ablation.py`
   reproduced it and `lora_in_training.py` fixes it.

4. **Choose negatives at the same scale.** Per-family ROC AUC against the full
   benign pool is 0.999–1.000, but that pool is dominated by small workloads.
   Against same-scale controls, 7B LoRA is 0.65 and 1B LoRA 0.67.

5. **`run_level_detection.py:127` is still buggy.** Its "persistence" term
   reduces algebraically to `p01`, so it prints P(alert|prev clear) twice under
   two names; its own output contradicts itself (marginal alert rate 0.953
   alongside "persistence" 0.257). `P_detect_markov` uses `p01` correctly, so
   the extrapolation stands — only the persistence figure is wrong. NOT FIXED.

6. **`load_all_data` label coverage.** `whitebox_lora_7b_*` and
   `whitebox_oracle_7b_long` fell through to `other` until the fix in
   `threeway_improved.py`; predictions dumped before that fix carry the old
   labels.
