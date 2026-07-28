#!/bin/bash
# Queue v7 — deadline-day run, ordered most-decisive-first so that whatever is
# finished at submission time is the most useful subset.
#
# All stages are 1 h and use 7B at batch 6 / seq 512 (30.4 GB, inside the
# corpus's 25-35 GB band) so every result is directly comparable to the 8 h
# 7B WB-L N=10 run, which scored p_train mean 0.328 / max 0.537 and 0/1920
# alerts.
#
#  1  WB-I 7B  — THE MISSING CONTROL. Pure inference, no training at all. If
#     benign 7B inference also scores ~0.33, the classifier has no signal at 7B
#     and the "LoRA evades" claim is really "the detector is blind at this
#     scale". If inference scores much lower, there IS sub-threshold signal and
#     the negative result is the more interesting kind. Without this, a reviewer
#     can dismiss the headline result outright.
#  2  WB-L 7B dilution=1 — undiluted LoRA: every batch trains. Separates
#     "dilution evades detection" from "LoRA evades detection". If even 100%
#     LoRA training is missed, dilution is irrelevant to the finding.
#  3  WB-D 7B dilution=10 — diluted FULL fine-tuning. The paper reports full-FT
#     dilution stays 75-100% detected; if that holds at 7B while LoRA does not,
#     it isolates LoRA as the mechanism. May OOM (AdamW states on 6.7B params
#     are large even at dilution 10); it fails fast and the queue continues.
#  4  WB-L 7B dilution=3 — the pilot peak (p_train mean 0.584, the highest of
#     the whole sweep) held for a full hour rather than 15 min.

REPO=/root/GPU-monitoring
cd "$REPO" || exit 1
export HF_HOME=/workspace/hf-cache
export PYTHONPATH=/root/GPU-monitoring

PY=python3
M7=NousResearch/Llama-2-7b-hf
CKPT=300
DUR=3600
MASTER=logs/queue_master.log
mkdir -p logs logs/.done data

stamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
say()   { echo "[$(stamp)] $*" >> "$MASTER"; }

run_stage() {
  local name="$1"; shift; local logf="$1"; shift
  local marker="logs/.done/${name}.done"
  if [ -f "$marker" ]; then say "SKIP  $name"; return 0; fi
  say "START $name -> $logf"
  local t0=$SECONDS
  "$@" > "$logf" 2>&1
  local rc=$?; local dur=$((SECONDS - t0))
  if [ $rc -eq 0 ]; then
    echo "$(stamp) rc=0 dur=${dur}s" > "$marker"; say "OK    $name (${dur}s)"
  else
    echo "$(stamp) rc=$rc dur=${dur}s" > "logs/.done/${name}.failed"
    say "FAIL  $name rc=$rc (${dur}s) -- continuing"
  fi
  return 0
}

say "===== queue v7 (deadline day, most-decisive-first) starting (pid $$) ====="

run_stage "wb_inference_7b_1h" "logs/wb_inference_7b_1h.log" \
  $PY scripts/run_workload.py --workload "whitebox_inference_7b_1h" \
    --timeout 4500 --checkpoint-sec $CKPT -- \
    $PY workloads/adversarial_whitebox.py --strategy WB-I --model "$M7" \
    --batch-size 6 --seq-len 512 --duration $DUR

run_stage "wb_lora_7b_N1_1h" "logs/wb_lora_7b_N1_1h.log" \
  $PY scripts/run_workload.py --workload "whitebox_lora_7b_N1_1h" \
    --timeout 4500 --checkpoint-sec $CKPT -- \
    $PY workloads/adversarial_whitebox.py --strategy WB-L --model "$M7" \
    --batch-size 6 --seq-len 512 --dilution 1 --duration $DUR

run_stage "wb_diluted_7b_N10_1h" "logs/wb_diluted_7b_N10_1h.log" \
  $PY scripts/run_workload.py --workload "whitebox_diluted_7b_N10_1h" \
    --timeout 4500 --checkpoint-sec $CKPT -- \
    $PY workloads/adversarial_whitebox.py --strategy WB-D --model "$M7" \
    --batch-size 6 --seq-len 512 --dilution 10 --grad-ckpt --duration $DUR

run_stage "wb_lora_7b_N3_1h" "logs/wb_lora_7b_N3_1h.log" \
  $PY scripts/run_workload.py --workload "whitebox_lora_7b_N3_1h" \
    --timeout 4500 --checkpoint-sec $CKPT -- \
    $PY workloads/adversarial_whitebox.py --strategy WB-L --model "$M7" \
    --batch-size 6 --seq-len 512 --dilution 3 --duration $DUR

say "===== queue v7 complete ====="
