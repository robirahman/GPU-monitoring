#!/usr/bin/env bash
# Collect multi-GPU and other-than-GPU0 training (+ matching non-training) so the
# physical classifier can detect training on ANY GPU, not just GPU0.
#
# Prereq: export SAIGE_USER / SAIGE_PASS first (the sensor logger reads them):
#   export SAIGE_USER=... SAIGE_PASS=...
# These are NOT stored in the repo.
#
# NOTE: run on the host under nohup/tmux (NOT a bare SSH session) so a dropped
# connection can't orphan a workload into the NVIDIA UVM driver (D-state).
set -u
cd "$(dirname "$0")/.."
PY=/opt/conda/bin/python
: "${SAIGE_USER:?set SAIGE_USER}"; : "${SAIGE_PASS:?set SAIGE_PASS}"

runw () { local label=$1 gi=$2 cd=$3 to=$4; shift 4
  echo "===== $label (cuda=$cd) $(date -u +%H:%M:%S) ====="
  $PY scripts/run_workload_with_sensors.py --workload "$label" --gpu-index "$gi" --cuda-devices "$cd" \
    --no-dcgm --timeout "$to" -- "$@" 2>&1 | grep -E "Sensor samples|multigpu]|Workload finished|visible GPU"
}

# TRAINING on other GPUs / multi-GPU (the cases a GPU0-only detector misses)
runw mg_gpu1_resnet     1 1     120 $PY workloads/pytorch_training.py --model resnet18 --dataset cifar10 --epochs 10 --batch-size 256
runw mg_gpu2_resnet     2 2     120 $PY workloads/pytorch_training.py --model resnet18 --dataset cifar10 --epochs 10 --batch-size 256
runw mg_gpu1_gpt2       1 1     120 $PY workloads/gpt2_finetune.py --steps 400 --batch-size 8
runw mg_multi012_resnet 0 0,1,2 130 $PY workloads/multigpu_training.py --duration 120 --batch-size 384
runw mg_multi12_resnet  1 1,2   130 $PY workloads/multigpu_training.py --duration 120 --batch-size 256
# NON-training on other GPUs (so the classifier doesn't equate "GPU1/2 busy" with training)
runw mg_gpu1_cufft      1 1     120 $PY workloads/scientific_hpc.py --workload cufft --duration 110
runw mg_gpu2_cufft      2 2     120 $PY workloads/scientific_hpc.py --workload cufft --duration 110
runw mg_gpu1_idle       1 1      90 $PY workloads/idle.py
echo "MULTIGPU COLLECTION DONE"
