# SPAR: Adversarial Detection of ML Training on Monitored GPUs

**Supervised Program for Alignment Research (SPAR)** — mentored by Robi Rahman (MIRI)

## Research Question

Can we distinguish ML training from other GPU workloads (scientific computing, crypto mining, etc.) — even when the user adversarially disguises their workload? This supports international AI compute governance.

## Phase 1: Literature Review Tooling

This repository contains scripts and notebooks for systematically searching, organizing, and annotating academic papers relevant to GPU workload classification, hardware side-channels, ML training fingerprinting, and compute governance.

## Setup

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Copy the API keys template and add your keys (optional)
copy config\api_keys.yaml.example config\api_keys.yaml
```

## Usage

### Batch search all configured queries
```bash
python scripts/run_full_search.py
```

### Export bibliography
```bash
python scripts/export_bibliography.py
```

### Generate summary report
```bash
python scripts/generate_summary.py
```

### Interactive exploration
```bash
jupyter notebook notebooks/01_search_exploration.ipynb
```

## Project Structure

- `src/search/` — API wrappers for arXiv and Semantic Scholar
- `src/storage/` — JSON-based paper database and BibTeX export
- `src/analysis/` — Auto-tagging, deduplication, and summarization
- `scripts/` — CLI entry points for batch operations
- `notebooks/` — Interactive Jupyter notebooks for exploration
- `config/` — Search queries and API key configuration
- `data/` — Paper database and generated outputs
- `tests/` — Unit tests

## Research Categories

1. GPU workload classification
2. Hardware performance counters
3. GPU side-channels
4. ML training fingerprinting
5. Compute governance
6. Adversarial evasion

## Phase 2 (Future)

Once GPU access is available: set up CUPTI monitoring on Linux, collect workload traces, build classifiers, test adversarial evasion.
