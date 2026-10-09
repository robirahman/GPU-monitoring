#!/bin/bash
# Per-host TPU bootstrap -- runs on EVERY worker of the slice (via
# `gcloud compute tpus tpu-vm ssh --worker=all --command=...` or scp'd and run).
# Idempotent: safe to re-run after a preemption restart.
#
# Installs the JAX TPU stack + collector deps, points the XLA compilation cache
# at GCS (so the 5-10 min first-step compile is paid once per slice, not per
# cell), verifies the clock is NTP-synced to <50 ms (the STEP<->telemetry epoch
# join depends on it), and probes the live TPU metric channels so V1 can confirm
# what is actually exposed before any real collection.
#
# Usage (on a worker):
#   REPO_URL=https://github.com/you/repo.git BUCKET=gs://my-telemetry \
#     bash scripts/tpu_collection/setup_tpu_host.sh
set -uo pipefail

REPO_URL=${REPO_URL:-https://github.com/robirahman/SPAR-GPU-monitoring.git}
REPO_DIR=${REPO_DIR:-$HOME/SPAR-GPU-monitoring}
BRANCH=${BRANCH:-multinode-tpu-experiments}
BUCKET=${BUCKET:-gs://spar-tpu-telemetry}
JAX_CACHE=${JAX_CACHE:-${BUCKET}/jax-compilation-cache}

echo "===== TPU host bootstrap ($(hostname)) ====="

# ── 1. Python deps ───────────────────────────────────────────────────────────
# jax[tpu] pulls libtpu (which ships libtpu.sdk.tpumonitoring, our richest
# metric source). tpu-info is the fallback source + CLI.
echo "[1/5] installing python deps..."
pip install -q -U pip
# PINNED via requirements_tpu.txt. The previous form was `-U "jax[tpu]"` with no
# constraint, against a v2-alpha-tpuv6e runtime -- so jax/jaxlib/libtpu floated
# independently of the runtime image. libtpu METRIC NAMES drift between releases
# and collect_tpu_telemetry.py classifies them by substring for exactly that
# reason, which means an unpinned upgrade could change which telemetry columns
# exist between two cells of the same experiment.
REQ="${REPO_DIR:-$HOME/GPU-monitoring}/requirements_tpu.txt"
if [ ! -f "$REQ" ]; then
  echo "FATAL: $REQ not found; refusing to install unpinned deps" >&2
  exit 1
fi
pip install -q -r "$REQ" \
  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
# scikit-learn is only needed if a host also scores locally; not pinned with the
# collection set because it does not affect what telemetry is produced.
pip install -q scikit-learn

# ── 2. Env: PJRT + GCS compilation cache ─────────────────────────────────────
echo "[2/5] configuring PJRT + XLA cache env..."
PROFILE=$HOME/.spar_tpu_env
cat > "$PROFILE" <<EOF
export PJRT_DEVICE=TPU
export JAX_PLATFORMS=tpu
export JAX_COMPILATION_CACHE_DIR=${JAX_CACHE}
export TPU_STDERR_LOG_LEVEL=1
EOF
# shellcheck disable=SC1090
source "$PROFILE"
grep -qF ".spar_tpu_env" "$HOME/.bashrc" 2>/dev/null || \
  echo "source $PROFILE" >> "$HOME/.bashrc"

# ── 3. Repo ──────────────────────────────────────────────────────────────────
echo "[3/5] syncing repo ($BRANCH)..."
if [ -d "$REPO_DIR/.git" ]; then
  git -C "$REPO_DIR" fetch --quiet origin "$BRANCH" && \
  git -C "$REPO_DIR" checkout --quiet "$BRANCH" && \
  git -C "$REPO_DIR" reset --hard --quiet "origin/$BRANCH"
else
  git clone --quiet --branch "$BRANCH" "$REPO_URL" "$REPO_DIR" || \
  git clone --quiet "$REPO_URL" "$REPO_DIR"
fi

# Scrub any injected PAT from the SAVED remote. REPO_URL is passed in with a
# token so the clone can authenticate, but git persists whatever URL it cloned
# from into .git/config -- which would leave the credential sitting on all four
# TPU workers. Rewrite the remote to the tokenless form (bootstrap_node.sh does
# the same on the GPU nodes). Verify with: git -C "$REPO_DIR" remote -v
TOKENLESS_URL="$(printf '%s' "$REPO_URL" | sed -E 's#https://[^@/]*@#https://#')"
if [ -d "$REPO_DIR/.git" ]; then
  git -C "$REPO_DIR" remote set-url origin "$TOKENLESS_URL" 2>/dev/null || true
  case "$(git -C "$REPO_DIR" remote get-url origin 2>/dev/null)" in
    *@github.com*) echo "  WARN: token may still be present in the saved remote" ;;
    *) echo "  saved remote is tokenless" ;;
  esac
fi

# ── 4. Clock discipline (STEP<->telemetry join needs <50 ms) ─────────────────
echo "[4/5] checking clock sync..."
if ! command -v chronyc >/dev/null 2>&1; then
  sudo apt-get update -qq && sudo apt-get install -y -qq chrony || \
    echo "  WARN: could not install chrony"
fi
if command -v chronyc >/dev/null 2>&1; then
  chronyc waitsync 12 0.05 >/dev/null 2>&1 || echo "  WARN: chrony not yet synced"
  OFFSET=$(chronyc tracking 2>/dev/null | awk '/Last offset/ {print ($4<0?-$4:$4)}')
  echo "  clock |offset| = ${OFFSET:-unknown}s (want < 0.05)"
else
  echo "  WARN: chronyc unavailable; cannot assert <50 ms clock skew"
fi

# ── 5. Metric-liveness probe ─────────────────────────────────────────────────
# Print which sources are live + what libtpu monitoring actually supports, then
# run the collector for a few seconds so V1 can inspect a real parquet schema.
echo "[5/5] probing TPU telemetry sources..."
cd "$REPO_DIR"
PYTHONPATH="$REPO_DIR" python3 - <<'PY' || echo "  probe raised (non-fatal)"
def try_import(name, fn):
    try:
        fn(); print(f"  [ok]  {name}")
    except Exception as e:
        print(f"  [--]  {name}: {e}")

def libtpu():
    from libtpu.sdk import tpumonitoring
    names = list(tpumonitoring.list_supported_metrics())
    print("        libtpu metrics:", [str(getattr(n,'name',n)) for n in names])

def tpuinfo():
    from tpu_info import device
    print("        tpu_info local chips:", device.get_local_chips())

try_import("libtpu.sdk.tpumonitoring", libtpu)
try_import("tpu_info", tpuinfo)
try_import("psutil", lambda: __import__("psutil").cpu_percent())
PY

echo "  short collector probe -> logs/smoke_data/ ..."
PYTHONPATH="$REPO_DIR" python3 scripts/collect_tpu_telemetry.py \
  --duration 8 --label tpu_probe --worker-id "${TPU_WORKER_ID:-0}" \
  --output-dir logs/smoke_data --checkpoint-sec 0 || echo "  collector probe failed"

echo "===== bootstrap complete on $(hostname) ====="
