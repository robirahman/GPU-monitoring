#!/usr/bin/env bash
# week5/scripts/run_week5.sh
# Full week 5 pipeline: feature engineering → classifier training → git push

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="python3"
PYTHONPATH_EXTRA="/root/xgb_pkg"
export PYTHONPATH="${PYTHONPATH_EXTRA}:${PYTHONPATH:-}"

echo "============================================================"
echo " Week 5 Pipeline"
echo " $(date)"
echo "============================================================"

echo ""
echo "[1/2] Feature engineering (sliding windows: 30s, 60s, 120s)"
$PYTHON scripts/feature_engineering.py

echo ""
echo "[2/2] Training classifiers (RF / XGBoost / SVM / LR)"
$PYTHON scripts/train_classifiers.py

echo ""
echo "============================================================"
echo " Results:"
echo "============================================================"
ls -lh results/*.csv results/*.parquet results/*.json 2>/dev/null | awk '{print $5, $9}'
echo ""
echo " Plots:"
ls plots/*.png 2>/dev/null | wc -l
echo " plot files generated"

echo ""
echo "[3/3] Git commit & push"
cd ..
git add week5/ || true
git commit -m "Add Week 5: sliding-window feature engineering + RF/XGBoost/SVM/LR classifiers

- 122 features per window (9 signals × 13 stats + 5 derived)
- Window sizes: 30s, 60s, 120s with 50% stride overlap
- Stratified Group K-Fold (k=5, grouped by run_id, no leakage)
- Tasks: binary, 3-way, multi-class
- Target: >85% binary accuracy

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>" 2>/dev/null || echo "Nothing new to commit"

git push 2>/dev/null || echo "Push failed or nothing to push"

echo ""
echo "============================================================"
echo " Done: $(date)"
echo "============================================================"
