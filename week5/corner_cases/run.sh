#!/usr/bin/env bash
# week5/corner_cases/run.sh
# Run all corner cases, generate roofline plots, and push to git.

set -euo pipefail
PYTHON=/venv/main/bin/python
export PYTHONPATH=/tmp/pynvml_pkg
cd "$(dirname "$0")"

echo "============================================================"
echo " Week 5 Corner Cases"
echo " $(date)"
echo "============================================================"

echo ""
echo "[1/3] Running corner cases (CNN / LLM / ViT / quantisation)..."
$PYTHON corner_cases.py 2>&1 | tee /tmp/corner_cases_run.log
echo ""
echo "[2/3] Generating roofline plots..."
$PYTHON plot_rooflines.py 2>&1 | tee /tmp/corner_cases_plots.log

echo ""
echo "[3/3] Git commit & push..."
cd ../../..
git add week5/corner_cases/
git commit -m "Add Week 5 corner cases: CNN/LLM/ViT/quantisation roofline sweep

Six groups of real-application parameter sweeps showing where inference
configurations overlap with training on the roofline:
  CC-A: ResNet-18/50/101/152 inference, batch 1-1024, img 224/512, FP16/BF16
  CC-B: GPT-2/medium/large prefill sweep, batch 1-64, seq_len 32-2048
  CC-C: GPT-2 decode (memory-bound), batch 1-256, context 128-1024
  CC-D: FP32/FP16/BF16 quantisation comparison (ResNet50 + GPT-2)
  CC-E: Training variants (forward-only vs full train, bs=1-256)
  CC-F: ViT-S/B/L/H inference, patch 8/14/16, batch 1-512

Plots: roofline_all, roofline_vs_training, roofline_cnn, roofline_llm,
       roofline_quant, roofline_vit, regime_scatter

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>" && git push

echo ""
echo "============================================================"
echo " Done: $(date)"
echo "============================================================"
