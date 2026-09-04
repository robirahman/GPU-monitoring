#!/usr/bin/env bash
# env.sh -- single source of truth for the GCP multi-node (2x a3-highgpu-8g H100)
# + TPU (v6e-16) telemetry-collection runbook. SOURCED by every other script in
# scripts/gcp/. It only DEFINES variables/helpers -- it deliberately does NOT run
# `set -euo pipefail` itself, so it is safe to `source` from an interactive shell
# without turning the shell into one that exits on the next typo. The wrapper
# scripts each set their own strict mode.
#
#   Facts baked in below are from the 2026-08-18 provisioning session
#   (see "gcp provisioning test.md"): project is Owner+paid; 2x a3-highgpu-8g
#   H100 Spot works in us-central1-a same-zone (~$38.86/node-h, ~$78/h pair);
#   v6e-16 Spot only had a live entitlement in us-east1-d; there is NO on-demand
#   H100 quota metric (Committed + Spot only).
#
# Override any value by exporting it before sourcing, e.g.
#   SESSION=20260820_run1 source scripts/gcp/env.sh
#
# shellcheck disable=SC2155  # export+assign on one line is intentional here.

# --- Project / location -------------------------------------------------------
export PROJECT="${PROJECT:-focus-surfer-494820-g0}"
export ZONE="${ZONE:-us-central1-a}"     # a3 H100 Spot verified here; same-zone => NCCL RTT is low
export REGION="${REGION:-us-central1}"
export TPU_ZONE="${TPU_ZONE:-us-east1-d}" # the only zone with a live v6e-16 Spot entitlement in the session
export NETWORK="${NETWORK:-default}"      # auto-mode "default" network => instances get *.internal DNS

# --- Stable node identity -----------------------------------------------------
# STABLE names => stable internal DNS (spar-gpu-0.<zone>.c.<project>.internal), so
# MASTER_ADDR keeps resolving after a re-provision under the same name. This is
# why the plan pins names instead of date-stamping them: the rendezvous endpoint
# must survive a preemption+recreate with no coordination channel.
export NODE0="${NODE0:-spar-gpu-0}"
export NODE1="${NODE1:-spar-gpu-1}"
# shellcheck disable=SC2034  # NODE_NAMES is consumed by provision/teardown, which source this file.
NODE_NAMES=("$NODE0" "$NODE1")            # array: sourced (arrays can't be exported) so provision/teardown can iterate
export MASTER_ADDR="${MASTER_ADDR:-$NODE0}"   # torchrun rendezvous host = node0's internal DNS name
export MASTER_PORT="${MASTER_PORT:-29500}"

# --- Machine / image ----------------------------------------------------------
export MACHINE_TYPE="${MACHINE_TYPE:-a3-highgpu-8g}"   # 8x H100 80GB SXM baked into the machine type (no --accelerator flag)
# SPOT or STANDARD. Was hardcoded SPOT because "the only H100 quota that exists
# here is Spot (=64)" -- that is STALE: as of 2026-09-04 this project has NO H100
# quota of any kind in any region, and the A100 quotas differ by provisioning
# model in a way that decides the machine type:
#     NVIDIA_A100_80GB_GPUS             16   (a2-ultragpu-8g, on-demand only)
#     PREEMPTIBLE_NVIDIA_A100_80GB_GPUS  0   <-- so Spot 80GB is impossible
#     NVIDIA_A100_GPUS                  16   (a2-highgpu-8g 40GB, on-demand)
#     PREEMPTIBLE_NVIDIA_A100_GPUS      64   (40GB Spot is ample)
# Picking 80GB to match the existing H100 corpus therefore FORCES on-demand,
# which is why this is a knob rather than a constant.
export PROVISIONING_MODEL="${PROVISIONING_MODEL:-SPOT}"
export BOOT_DISK_SIZE="${BOOT_DISK_SIZE:-500GB}"
export BOOT_DISK_TYPE="${BOOT_DISK_TYPE:-pd-balanced}"
# IMAGE_MODE=ubuntu  -> pristine Ubuntu 22.04, drivers installed by bootstrap_node.sh
#                       (this is the path the provisioning session exercised).
# IMAGE_MODE=dlvm    -> Deep-Learning-VM: drivers/CUDA preinstalled, so bootstrap's
#                       driver phase SKIPS (nvidia-smi already lists 8 GPUs). Skips the
#                       whole kernel-module/reboot failure class -- but STILL needs
#                       fabric-manager enabled for a3 (bootstrap does that regardless).
export IMAGE_MODE="${IMAGE_MODE:-ubuntu}"
export DLVM_FAMILY="${DLVM_FAMILY:-common-cu123-ubuntu-2204-py310}"  # VERIFY: gcloud compute images list --project deeplearning-platform-release --filter="family~cu12"
# shellcheck disable=SC2034  # IMAGE_FLAGS is consumed by provision_gpu_pair.sh, which sources this file.
case "$IMAGE_MODE" in
  ubuntu) IMAGE_FLAGS=(--image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud) ;;
  dlvm)   IMAGE_FLAGS=(--image-family="$DLVM_FAMILY" --image-project=deeplearning-platform-release) ;;
  *)      echo "env.sh: unknown IMAGE_MODE=$IMAGE_MODE (want ubuntu|dlvm)" >&2 ; IMAGE_FLAGS=() ;;
esac
# IMAGE_FLAGS stays a plain (non-exported) array; provision sources env.sh so it's in scope.

# --- TPU identity -------------------------------------------------------------
export TPU_NAME="${TPU_NAME:-spar-tpu-0}"
export TPU_ACCEL="${TPU_ACCEL:-v6e-16}"
export TPU_RUNTIME="${TPU_RUNTIME:-v2-alpha-tpuv6e}"
export TPU_SA="${TPU_SA:-tpu-service-account@${PROJECT}.iam.gserviceaccount.com}"

# --- Storage (telemetry egress) ----------------------------------------------
# Per-node/-worker prefixes keep two writers off one git branch (contract 2.7):
#   $BUCKET/runs/$SESSION/<node>/{data,logs,meta}   and  .../control/  for .done markers.
export BUCKET="${BUCKET:-gs://${PROJECT}-telemetry}"

# --- Session ------------------------------------------------------------------
# Date-stamped by default so a same-day provision->teardown shares one prefix.
# PIN IT for anything spanning a UTC midnight or for reproducibility:
#   export SESSION=20260820_scope_ladder    (do this once, in your shell, before provisioning)
export SESSION="${SESSION:-$(date -u +%Y%m%d)}"
export GCS_SESSION="${BUCKET}/runs/${SESSION}"
export GCS_CONTROL="${GCS_SESSION}/control"

# --- Repo / venvs (paths on the provisioned node) -----------------------------
# REPO_URL is the PRIVATE canonical clone URL. bootstrap_node.sh injects a
# fine-grained PAT at clone time (read -s, cached root-only) -- the URL here never
# contains the token, and the token is NEVER put in instance metadata.
export REPO_URL="${REPO_URL:-https://github.com/robirahman/GPU-monitoring.git}"
export REPO_BRANCH="${REPO_BRANCH:-multinode-tpu-experiments}"
# Stable clone path so it survives on a STOP-preserved boot disk across a preemption.
export REPO_DIR="${REPO_DIR:-/opt/spar/repo}"
export VENV_MAIN="${VENV_MAIN:-${REPO_DIR}/venv-main}"   # requirements_v2 + requirements_gcp_extra (torch 2.5.1+cu121)
export VENV_VLLM="${VENV_VLLM:-${REPO_DIR}/venv-vllm}"   # vllm + ray ONLY (vllm repins torch, must not clobber venv-main)

# --- Firewall -----------------------------------------------------------------
# We reuse the auto-created "default-allow-internal" rule (all TCP/UDP/ICMP between
# instances on the default network) -- it already covers :29500 rendezvous AND the
# dynamic high ports NCCL opens. Nothing is ever opened to the public internet.
export FW_INTERNAL="${FW_INTERNAL:-default-allow-internal}"

# --- On a provisioned node, /etc/spar/node_env is AUTHORITATIVE ---------------
# bootstrap_node.sh writes it (NODE_INDEX/NODE_NAME/SESSION/BUCKET/MASTER_ADDR/
# MASTER_PORT/REPO_DIR/...). Sourcing it last lets sync_data.sh / preempt_watch.sh /
# the collection queue pick up this node's real identity instead of the defaults.
if [[ -r /etc/spar/node_env ]]; then
  # shellcheck disable=SC1091
  . /etc/spar/node_env
fi

# --- Helpers shared by the wrappers ------------------------------------------
spar_ts()  { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
spar_say() { echo "[$(spar_ts)] $*"; }
# GCS prefix for a node (defaults to THIS node's name / hostname).
gcs_node_prefix() { echo "${GCS_SESSION}/${1:-${NODE_NAME:-$(hostname -s)}}"; }
# Fail fast if a required command is missing.
require_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "MISSING required command: $1" >&2; return 1; }; }

# If run directly (not sourced) print the resolved config -- handy for eyeballing.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "PROJECT=$PROJECT ZONE=$ZONE TPU_ZONE=$TPU_ZONE"
  echo "NODE0=$NODE0 NODE1=$NODE1 MASTER_ADDR=$MASTER_ADDR:$MASTER_PORT"
  echo "MACHINE_TYPE=$MACHINE_TYPE IMAGE_MODE=$IMAGE_MODE"
  echo "BUCKET=$BUCKET SESSION=$SESSION"
  echo "GCS_SESSION=$GCS_SESSION"
  echo "REPO_DIR=$REPO_DIR (branch $REPO_BRANCH)"
fi
