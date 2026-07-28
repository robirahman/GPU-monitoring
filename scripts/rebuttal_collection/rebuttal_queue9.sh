#!/bin/bash
# Queue v9 — the 7B results turned the rebuttal's open question from "does
# dilution evade?" into "does this detector work at scale at all?".
#
# What we now know at 7B (30.4 GB, in the corpus's training band):
#   - every LoRA dilution N=1..500 alerts on <=15% of windows
#   - full-model diluted fine-tuning (WB-D N=10) alerts on 0%
#   - the BENIGN 7B inference control alerts on 16.6% — more than N=1 training
#   - vs that same-scale control, ROC AUC is 0.64-0.83, and 0.419 at N=500
# So at 7B the classifier neither detects training nor stays quiet on benign
# inference. These runs test how far that generalises.
#
#  A  7B WB-D N=1  — undiluted FULL fine-tune, the maximum-signal training
#     workload at 7B. If even this is missed, the failure is not about LoRA's
#     small backward pass; if it is caught, the failure is LoRA-specific.
#  B  3B WB-L N=1  — undiluted LoRA at 3B, to locate the scale transition
#     between 1.1B (where detection works) and 7B (where it does not).
#  C  3B WB-I      — same-scale benign control for B. Without a same-scale
#     negative, an AUC at 3B would repeat the mistake that made the 7B AUC
#     look like 0.994 against a 1B-dominated negative pool.
#  D  7B WB-I 2h   — more same-scale 7B negatives, so a retrained classifier
#     has enough 7B inference data to define the boundary.
#  E  1B WB-D N=1  — undiluted full fine-tune at 1B, the symmetry check for A.

REPO=/root/GPU-monitoring; cd "$REPO" || exit 1
export HF_HOME=/workspace/hf-cache PYTHONPATH=/root/GPU-monitoring
PY=python3
M7=NousResearch/Llama-2-7b-hf
M3=unsloth/Llama-3.2-3B
M1=TinyLlama/TinyLlama-1.1B-Chat-v1.0
CKPT=300; MASTER=logs/queue_master.log
mkdir -p logs logs/.done data
stamp(){ date -u '+%Y-%m-%dT%H:%M:%SZ'; }; say(){ echo "[$(stamp)] $*" >> "$MASTER"; }
run_stage(){ local n="$1"; shift; local l="$1"; shift; local m="logs/.done/${n}.done"
  [ -f "$m" ] && { say "SKIP  $n"; return 0; }
  say "START $n -> $l"; local t0=$SECONDS; "$@" > "$l" 2>&1; local rc=$?; local d=$((SECONDS-t0))
  if [ $rc -eq 0 ]; then echo "$(stamp) rc=0 dur=${d}s" > "$m"; say "OK    $n (${d}s)"
  else echo "$(stamp) rc=$rc dur=${d}s" > "logs/.done/${n}.failed"; say "FAIL  $n rc=$rc (${d}s)"; fi; return 0; }

say "===== queue v9 waiting for v8 ====="
while pgrep -f 'rebuttal_queue8\.sh' >/dev/null 2>&1; do sleep 30; done
say "===== queue v9 starting (pid $$) ====="

run_stage wb_diluted_7b_N1_1h logs/wb_diluted_7b_N1_1h.log \
  $PY scripts/run_workload.py --workload whitebox_diluted_7b_N1_1h --timeout 4500 --checkpoint-sec $CKPT -- \
  $PY workloads/adversarial_whitebox.py --strategy WB-D --model "$M7" \
  --batch-size 6 --seq-len 512 --dilution 1 --grad-ckpt --duration 3600

run_stage wb_lora_3b_N1_1h logs/wb_lora_3b_N1_1h.log \
  $PY scripts/run_workload.py --workload whitebox_lora_3b_N1_1h --timeout 5400 --checkpoint-sec $CKPT -- \
  $PY workloads/adversarial_whitebox.py --strategy WB-L --model "$M3" \
  --batch-size 6 --seq-len 512 --dilution 1 --duration 3600

run_stage wb_inference_3b_1h logs/wb_inference_3b_1h.log \
  $PY scripts/run_workload.py --workload whitebox_inference_3b_1h --timeout 4500 --checkpoint-sec $CKPT -- \
  $PY workloads/adversarial_whitebox.py --strategy WB-I --model "$M3" \
  --batch-size 6 --seq-len 512 --duration 3600

run_stage wb_inference_7b_2h logs/wb_inference_7b_2h.log \
  $PY scripts/run_workload.py --workload whitebox_inference_7b_2h --timeout 8100 --checkpoint-sec $CKPT -- \
  $PY workloads/adversarial_whitebox.py --strategy WB-I --model "$M7" \
  --batch-size 6 --seq-len 512 --duration 7200

run_stage wb_diluted_1b_N1_1h logs/wb_diluted_1b_N1_1h.log \
  $PY scripts/run_workload.py --workload whitebox_diluted_1b_N1_1h --timeout 4500 --checkpoint-sec $CKPT -- \
  $PY workloads/adversarial_whitebox.py --strategy WB-D --model "$M1" \
  --batch-size 2 --seq-len 256 --dilution 1 --grad-ckpt --duration 3600

say "===== queue v9 complete ====="
