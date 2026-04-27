# Detecting Hidden ML Training With Zero-Overhead Telemetry

**Supervised Program for Alignment Research (SPAR), S26 cohort** — mentored by Robi Rahman (MIRI), with Sabiha Tajdari (UVA).

This repository accompanies an anonymous submission under review at **ICML 2026**. It contains the workload corpus, classifier training pipeline, adversarial-evaluation framework, and figure-generation scripts described in the paper.

- 📄 [`report/adversarial_gpu_monitoring.pdf`](report/adversarial_gpu_monitoring.pdf) — full paper
- 🌐 [`index.html`](index.html) — project landing page

## What's in scope

We evaluate whether **always-available NVML telemetry** can reliably distinguish ML training from other GPU workloads, even when an operator actively tries to disguise the workload. Across 5 rounds of monitor-evader iteration, we evaluate **20 evasion strategy families on 9 GPU models** spanning Ampere through Blackwell.

Headline results:

- **98.58% ± 0.57%** pooled binary accuracy (training vs. rest), macro-F1 = 0.986, on 30 s windows with 5-fold grouped CV — 26,496 windows across 1,404 runs.
- **97.76% at 5 s** detection latency; 98.35% at 30 s. Real-time monitoring is feasible.
- **98–100% on multi-GPU DDP evasion** in leave-one-strategy-out evaluation. The governance-relevant case is the most robust.
- **4.8% mean evasion rate** across 18 non-white-box single-GPU strategies; only one family (white-box LoRA) drops below 50% detection.
- Cross-validated against external wall-power measurements (TP-Link Tapo P115, n = 13,571 paired readings).

## Repository layout

| Directory | Contents |
|---|---|
| `workloads/` | All workload scripts. Real ML training (`bert_finetune.py`, `gpt2_finetune.py`, FSDP runners), inference, HPC, mining, rendering, and adversarial evasion strategies (`adversarial_round2.py` … `adversarial_whitebox.py`). |
| `classifier/` | Two-stage RandomForest training (`twostage.py`), leave-one-strategy-out evaluation (`leave_one_strategy_out.py`), dilution sweep (`eval_dilution_sweep.py`), adversary-defender iteration loop (`adversary_defender_iteration.py`), and physics-based feature extraction. |
| `analysis/` | Exploratory analysis scripts and one-off plots. |
| `data/` | Telemetry traces (parquet) and processed feature CSVs. |
| `report/` | Paper source (`midterm_report.tex`, anonymized PDF), final figures, and figure-regen scripts (`regen_*.py`, `plot_*.py`). |
| `scripts/` | CLI entry points (telemetry collection, batch run dispatch). |
| `notebooks/` | Jupyter notebooks for interactive analysis. |
| `docs/` | Round-by-round notes (`Round_5_notes.md`, `class_balancing.md`, `wall_power_results.md`, etc.). |

## Setup

```bash
# Linux/macOS
python -m venv venv && source venv/bin/activate
pip install -r requirements_v2.txt

# Windows
python -m venv venv
venv\Scripts\activate
pip install -r requirements_v2.txt
```

`requirements_v2.txt` is the current pinned set (CUDA 12.x, PyTorch 2.x, pynvml, pandas, scikit-learn, xgboost). The older `requirements.txt` is from the initial Phase 1 literature-review phase and is kept only for archival reasons.

A working NVIDIA GPU with NVML access is required for telemetry collection. Classifier training and figure regeneration run on CPU.

## Common operations

### Collect telemetry while running a workload

```bash
python -m workloads.bert_finetune --gpu 0 --duration 60
# Telemetry is logged to data/<run_id>.parquet alongside the workload run.
```

### Run an adversarial evasion strategy

```bash
python -m workloads.adversarial_round4 --strategy composite_memmin --gpu 0
python -m workloads.adversarial_whitebox --strategy lora_diluted --gpu 0
```

### Train the two-stage classifier

```bash
python -m classifier.twostage --windows 30 --features full
```

### Leave-one-strategy-out evaluation

```bash
python -m classifier.leave_one_strategy_out --hold-out whitebox_lora
```

### Regenerate paper figures

```bash
python regen_figures.py            # core figures
python regen_crossgpu_heatmap.py   # cross-GPU transfer
python regen_wall_power_scatter.py # Tapo validation
python regen_roofline.py           # roofline corner-case sweep
```

Figures land in `report/figures/`.

## Citing this work

The paper is currently anonymous under ICML 2026 review. Please cite the report PDF directly if referencing results before deanonymization.

## Threat model and limitations

See `report/adversarial_gpu_monitoring.pdf` §2 (threat model) and §6.5 (HEM requirements). Notable limitations: single-node only (≤ 8 GPUs per node), framework-level evasion only (no custom CUDA-kernel evaders), 1 Hz sampling, NVIDIA hardware only, and only one round of white-box adversarial iteration.
