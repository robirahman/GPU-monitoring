#!/bin/bash
# Queue v10 — replicates for leakage-free grouped analysis, plus the missing
# cells of the model-size x method grid.
#
# Why replicates. The coverage ablation (does same-scale training data fix the
# 7B failure?) currently splits ONE inference run in half, putting half in
# training and half in test. With ICC ~= 0.98 between windows of the same run,
# those halves are near-identical, so its AUC = 1.000 is contaminated by
# within-run leakage — the very error the paper's own leak fix was about. A
# clean version needs INDEPENDENT runs on each side of the split.
#
# Why the grid. Under the unseen-family protocol, undiluted LoRA is missed at
# 1.1B (0% alerts), 3B (0%) and 7B (14.8%, below the 16.6% benign control),
# while undiluted FULL fine-tuning at 7B does show signal (22.4%). So the
# failure looks specific to LoRA's tiny backward pass rather than to scale.
# 3B full-FT completes the 2x3 grid; the 3B AUC of 0.032 (inverted) needs a
# replicate before it can be reported.

REPO=/root/GPU-monitoring; cd "$REPO" || exit 1
export HF_HOME=/workspace/hf-cache PYTHONPATH=/root/GPU-monitoring
PY=python3
M7=NousResearch/Llama-2-7b-hf; M3=unsloth/Llama-3.2-3B; M1=TinyLlama/TinyLlama-1.1B-Chat-v1.0
CKPT=300; MASTER=logs/queue_master.log
mkdir -p logs logs/.done data
stamp(){ date -u '+%Y-%m-%dT%H:%M:%SZ'; }; say(){ echo "[$(stamp)] $*" >> "$MASTER"; }
run_stage(){ local n="$1"; shift; local l="$1"; shift; local m="logs/.done/${n}.done"
  [ -f "$m" ] && { say "SKIP  $n"; return 0; }
  say "START $n -> $l"; local t0=$SECONDS; "$@" > "$l" 2>&1; local rc=$?; local d=$((SECONDS-t0))
  if [ $rc -eq 0 ]; then echo "$(stamp) rc=0 dur=${d}s" > "$m"; say "OK    $n (${d}s)"
  else echo "$(stamp) rc=$rc dur=${d}s" > "logs/.done/${n}.failed"; say "FAIL  $n rc=$rc (${d}s)"; fi; return 0; }

say "===== queue v10 waiting for v9 ====="
while pgrep -f 'rebuttal_queue9\.sh' >/dev/null 2>&1; do sleep 30; done
say "===== queue v10 starting (pid $$) ====="

# missing grid cell: full fine-tune at 3B
run_stage wb_diluted_3b_N1_1h logs/wb_diluted_3b_N1_1h.log \
  $PY scripts/run_workload.py --workload whitebox_diluted_3b_N1_1h --timeout 4500 --checkpoint-sec $CKPT -- \
  $PY workloads/adversarial_whitebox.py --strategy WB-D --model "$M3" \
  --batch-size 6 --seq-len 512 --dilution 1 --grad-ckpt --duration 3600

# independent replicates -> grouped train/test splits with no shared run
run_stage wb_inference_3b_rep2 logs/wb_inference_3b_rep2.log \
  $PY scripts/run_workload.py --workload whitebox_inference_3b_rep2 --timeout 4500 --checkpoint-sec $CKPT -- \
  $PY workloads/adversarial_whitebox.py --strategy WB-I --model "$M3" \
  --batch-size 6 --seq-len 512 --duration 3600

run_stage wb_lora_3b_N1_rep2 logs/wb_lora_3b_N1_rep2.log \
  $PY scripts/run_workload.py --workload whitebox_lora_3b_N1_rep2 --timeout 4500 --checkpoint-sec $CKPT -- \
  $PY workloads/adversarial_whitebox.py --strategy WB-L --model "$M3" \
  --batch-size 6 --seq-len 512 --dilution 1 --duration 3600

run_stage wb_lora_7b_N10_rep2 logs/wb_lora_7b_N10_rep2.log \
  $PY scripts/run_workload.py --workload whitebox_lora_7b_N10_rep2 --timeout 4500 --checkpoint-sec $CKPT -- \
  $PY workloads/adversarial_whitebox.py --strategy WB-L --model "$M7" \
  --batch-size 6 --seq-len 512 --dilution 10 --duration 3600

run_stage wb_inference_7b_rep3 logs/wb_inference_7b_rep3.log \
  $PY scripts/run_workload.py --workload whitebox_inference_7b_rep3 --timeout 4500 --checkpoint-sec $CKPT -- \
  $PY workloads/adversarial_whitebox.py --strategy WB-I --model "$M7" \
  --batch-size 6 --seq-len 512 --duration 3600

# matched-settings benign control for the 1B LoRA run (batch 2 / seq 256)
run_stage wb_inference_1b_matched logs/wb_inference_1b_matched.log \
  $PY scripts/run_workload.py --workload whitebox_inference_1b_matched --timeout 4500 --checkpoint-sec $CKPT -- \
  $PY workloads/adversarial_whitebox.py --strategy WB-I --model "$M1" \
  --batch-size 2 --seq-len 256 --duration 3600

say "===== queue v10 complete ====="
