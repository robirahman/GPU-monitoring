# Class Balancing for Classifier Training and Evaluation

## The Problem

Our dataset has severe class imbalance: ~766 training runs, ~172 other runs,
and only ~49 inference runs. This is inherent to the project — we collected
many adversarial training variants (each round adds new evasion strategies,
all labeled `ml_training`) while the set of clean inference and non-ML
workloads grew much more slowly.

Without correction, XGBoost optimizes raw accuracy. When 78% of windows are
`ml_training`, a classifier that predicts "training" for every input achieves
78% accuracy. In our case this is exactly what happened: the hardened
classifier trained on unbalanced data labeled nearly all inference windows as
training. Detection rates appeared high (near 100%) because the classifier
was not distinguishing training from anything — it was calling everything
training.

This was discovered when the hardened classifier showed 0% inference recall:
every inference window was misclassified as training. A classifier that cannot
identify inference is useless for deployment, regardless of its training
detection rate.

## Why Subsampling Is Wrong

The first fix was `balanced_subsample()`: downsample majority classes by
randomly dropping entire runs until each class has the same number of windows
as the minority class (inference, ~49 runs). This produces a balanced 1/3 /
1/3 / 1/3 dataset.

The problem: with 766 training runs and a target of ~49, this discards **93%
of training data**. The classifier never sees most adversarial strategies
during training. Since the whole point of hardening is to learn from diverse
evasion approaches, throwing away 93% of them defeats the purpose.

With subsampling, leave-one-strategy-out (LOO) mean detection was 68.3%, with
4 of 18 strategy families evading at below 50%. These numbers were honest in
the sense that the classifier wasn't cheating, but they were artificially low
because the classifier was learning from a tiny fraction of available data.

## The Correct Approach: Class Weights

Instead of discarding data, we assign each training sample a weight inversely
proportional to its class frequency:

```
weight(class c) = total_samples / (n_classes * count(class c))
```

This is the same formula scikit-learn uses for `class_weight="balanced"`. The
effect is that each class contributes equally to the total weighted loss,
regardless of how many samples it has. A single inference window has ~15x the
weight of a single training window, so the classifier is equally penalized for
misclassifying inference as it is for misclassifying training.

In XGBoost, this is passed via the `sample_weight` parameter to `clf.fit()`.
Every sample participates in training — the classifier sees every adversarial
strategy, every evasion variant, every clean workload — but the loss function
treats the three classes equally.

## Implementation

The function `compute_sample_weights(y)` in `classifier/threeway_improved.py`
computes per-sample weights from label frequencies. It is used in all
classifier training across the codebase:

- `classifier/adversary_defender_iteration.py` — baseline, hardened, and R3
  classifiers, plus their CV folds
- `classifier/eval_dilution_sweep.py` — baseline, LOO, and hardened
  classifiers at each window size
- `classifier/leave_one_strategy_out.py` — per-family LOO classifiers
- `classifier/classifier_oracle.py` — real-time oracle classifier

All evaluation is done on unweighted predictions. The weights affect training
only; test-time classification uses the standard `clf.predict()` with no
weighting. Per-class metrics (detection rate per strategy, false positive rate
on inference) are reported separately so that no class can inflate the
aggregate numbers.

## Results

| Metric | Subsampling (old) | Class weights (current) |
|--------|-------------------|-------------------------|
| LOO mean detection | 68.3% | 96.8% |
| Families evading (<50%) | 4 / 18 | 0 / 18 |
| Worst LOO family | L_diluted (3.7%) | E_memory_minimal (71.4%) |
| Hardened composite detection | 50-100% | 100% across all idle fractions |
| Training data retained | ~7% | 100% |

The improvement is entirely from retaining all training data. The classifier
architecture, hyperparameters, features, and evaluation methodology are
unchanged. The class weights ensure the classifier does not exploit the
imbalance, while keeping every adversarial example available for learning.
