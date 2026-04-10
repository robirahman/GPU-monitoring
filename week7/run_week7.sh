#!/usr/bin/env bash
# ============================================================
# Week 7 — Full Test Suite on 2× B200 (NVLink 5)
# Runs same experiments as weeks 4 & 5 with B200 hardware.
#
# Execution order:
#   Step 1: NVLink characterization (T1-T6)
#   Step 2: DDP training characterization
#   Step 3: Edge cases (adversarial workloads)
#   Step 4: Model width scaling (accuracy + roofline)
#   Step 5: Dataset size scaling
#   Step 6: EDA (exploratory data analysis)
#   Step 7: Classifier suite (RF/SVM/LR, binary/3-way/multi)
#   Step 8: Feature engineering (week5 style, sliding windows)
#   Step 9: Classifier training on windows (week5 style)
#   Step 10: Generate reports
# ============================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

cd "$REPO_ROOT"

echo "========================================================"
echo " Week 7 Full Test Suite — 2× NVIDIA B200"
echo " $(date)"
echo "========================================================"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'GPUs: {torch.cuda.device_count()}x {torch.cuda.get_device_name(0)}')"
echo ""

# ── Step 1: NVLink ──────────────────────────────────────────
echo "[1/10] NVLink Characterization..."
python3 week7/run_nvlink_tests.py 2>&1 | tee "$LOG_DIR/nvlink.log"
echo "Done: NVLink"

# ── Step 2: DDP Training ────────────────────────────────────
echo "[2/10] DDP Training Characterization..."
torchrun --nproc_per_node=2 week7/ddp_training_characterize.py 2>&1 | tee "$LOG_DIR/ddp.log"
echo "Done: DDP"

# ── Step 3: Edge Cases ──────────────────────────────────────
echo "[3/10] Edge Cases..."
# edge_cases.py uses torchrun internally for BASELINE_TRAIN etc.
python3 week7/edge_cases.py 2>&1 | tee "$LOG_DIR/edge_cases.log"
echo "Done: Edge Cases"

# ── Step 4: Model Scaling ───────────────────────────────────
echo "[4/10] Model Width Scaling (accuracy + roofline)..."
python3 week7/scale_model_accuracy.py 2>&1 | tee "$LOG_DIR/scale_model.log"
echo "Done: Model Scaling"

# ── Step 5: Dataset Scaling ─────────────────────────────────
echo "[5/10] Dataset Size Scaling..."
torchrun --nproc_per_node=2 week7/scale_dataset.py 2>&1 | tee "$LOG_DIR/scale_dataset.log"
echo "Done: Dataset Scaling"

# ── Step 6: EDA ─────────────────────────────────────────────
echo "[6/10] Exploratory Data Analysis..."
python3 week7/run_eda.py 2>&1 | tee "$LOG_DIR/eda.log"
echo "Done: EDA"

# ── Step 7: Classifiers (week4 style) ───────────────────────
echo "[7/10] Classifier Suite (week4 style)..."
python3 week7/run_classifiers.py 2>&1 | tee "$LOG_DIR/classifiers_w4.log"
echo "Done: Classifiers (week4)"

# ── Step 8: Feature Engineering (week5 style) ───────────────
echo "[8/10] Feature Engineering (week5 style)..."
python3 week7/scripts/feature_engineering.py 2>&1 | tee "$LOG_DIR/feature_eng.log"
echo "Done: Feature Engineering"

# ── Step 9: Classifier Training (week5 style) ───────────────
echo "[9/10] Classifier Training (week5 style)..."
python3 week7/scripts/train_classifiers.py 2>&1 | tee "$LOG_DIR/classifiers_w5.log"
echo "Done: Classifiers (week5)"

# ── Step 10: Comparison Report ──────────────────────────────
echo "[10/10] Generating comparison report..."
python3 week7/generate_roofline_tier1.py 2>&1 | tee "$LOG_DIR/roofline.log"
python3 week7/generate_comparison.py 2>&1 | tee "$LOG_DIR/comparison.log"
echo "Done: Reports"

echo ""
echo "========================================================"
echo " Week 7 complete. $(date)"
echo " Results: $SCRIPT_DIR/results/"
echo " Plots:   $SCRIPT_DIR/plots/"
echo " Reports: $SCRIPT_DIR/reports/"
echo "========================================================"
