#!/usr/bin/env bash
# bootstrap_node.sh -- idempotent, phase-markered node bring-up for an
# a3-highgpu-8g H100 node. Run FIRST, over SSH, as root:
#
#   sudo NODE_INDEX=0 SPAR_AUTO_REBOOT=1 bash /tmp/bootstrap_node.sh
#
# Phase markers live in /var/lib/spar/phase.<name>.done so a re-run resumes.
# On a kernel/driver module mismatch this reboots (or exits 42 if you didn't set
# SPAR_AUTO_REBOOT) and RESUMES from the marker after the reboot -- exactly the
# failure the provisioning session hit (kernel-pinned nvidia modules + reboot).
#
# Exit codes:  0 = done   42 = reboot required, re-run after reboot   !=0 = hard fail
#
# WHY a systemd oneshot (spar-bootstrap.service) is installed EARLY: if this
# reboots for kernel modules, or a Spot preemption STOP->start cycles the node,
# the service re-runs this script (After=cloud-final.service) and self-heals in
# the right order, until /var/lib/spar/bootstrap.done exists.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

[[ "$(id -u)" -eq 0 ]] || { echo "must run as root (sudo)"; exit 1; }

export DEBIAN_FRONTEND=noninteractive
STATE=/var/lib/spar
GCPDIR=/opt/spar/gcp            # stable copy of these scripts (survives on the STOP boot disk)
install -d "$STATE" /etc/spar "$GCPDIR"

# --- phase-marker helpers ----------------------------------------------------
have_phase() { [[ -f "${STATE}/phase.$1.done" ]]; }
mark_phase() { spar_ts > "${STATE}/phase.$1.done"; }
log() { echo "[$(spar_ts)] [bootstrap] $*"; }
# apt with the lock timeout that fixes the cloud-init apt-lock RACE from the session.
apt_get() { apt-get -o DPkg::Lock::Timeout=600 -y "$@"; }
# read a non-secret instance metadata attribute (empty on failure)
meta() { curl -sf -H "Metadata-Flavor: Google" \
  "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1" 2>/dev/null || true; }

# --- resolve THIS node's identity (env > metadata > sensible default) --------
NODE_INDEX="${NODE_INDEX:-$(meta spar-node-index)}"
NODE_INDEX="${NODE_INDEX:-0}"
NODE_NAME="${NODE_NAME:-$(meta spar-node-name)}"
NODE_NAME="${NODE_NAME:-$(hostname -s)}"
SESSION="${SESSION:-$(meta spar-session)}"; SESSION="${SESSION:-$(date -u +%Y%m%d)}"
MASTER_ADDR="${MASTER_ADDR:-$(meta spar-master-addr)}"; MASTER_ADDR="${MASTER_ADDR:-$NODE0}"
MASTER_PORT="${MASTER_PORT:-$(meta spar-master-port)}"; MASTER_PORT="${MASTER_PORT:-29500}"
BUCKET="${BUCKET:-$(meta spar-bucket)}"; BUCKET="${BUCKET:-gs://${PROJECT}-telemetry}"
NODE_RANK="$NODE_INDEX"
GCS_NODE="${BUCKET}/runs/${SESSION}/${NODE_NAME}"
log "identity: node_index=$NODE_INDEX name=$NODE_NAME session=$SESSION master=$MASTER_ADDR:$MASTER_PORT"

# --- stabilise the runbook scripts + install the self-heal unit EARLY --------
# (so a reboot mid-bootstrap, or a preemption restart, resumes correctly).
if [[ "$SCRIPT_DIR" != "$GCPDIR" ]]; then
  cp -f "$SCRIPT_DIR"/*.sh "$GCPDIR"/ 2>/dev/null || true
fi
SELF="${GCPDIR}/bootstrap_node.sh"

install_bootstrap_service() {
  cat > /etc/systemd/system/spar-bootstrap.service <<EOF
[Unit]
Description=SPAR node self-heal bootstrap (resumes until /var/lib/spar/bootstrap.done)
After=cloud-final.service network-online.target
Wants=network-online.target
ConditionPathExists=!/var/lib/spar/bootstrap.done

[Service]
Type=oneshot
Environment=SPAR_AUTO_REBOOT=1
ExecStart=/usr/bin/env bash ${SELF}
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF
  systemctl daemon-reload
  systemctl enable spar-bootstrap.service >/dev/null 2>&1 || true
}
install_bootstrap_service

# reboot-or-signal helper for the kernel-module case ---------------------------
reboot_or_exit42() {
  local n; n=$(( $(cat "${STATE}/driver-reboots" 2>/dev/null || echo 0) + 1 ))
  echo "$n" > "${STATE}/driver-reboots"
  if [[ "$n" -gt 3 ]]; then
    log "FATAL: still no working driver after $((n-1)) reboots -- stopping to avoid a reboot loop."
    exit 1
  fi
  if [[ "${SPAR_AUTO_REBOOT:-0}" == "1" ]]; then
    log "rebooting to load freshly-installed NVIDIA modules (attempt $n); will resume via spar-bootstrap.service"
    systemctl reboot
    sleep 60   # give systemd time to actually reboot before the script would otherwise continue
    exit 0
  fi
  log "NVIDIA modules installed but a REBOOT is required. Reboot this node, then re-run me (I resume)."
  exit 42
}

gpu_count() { nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true; }

########################################################################
# Phase (a): wait for cloud-init, then apt update with the lock timeout
########################################################################
if ! have_phase apt; then
  log "(a) waiting for cloud-init to finish (avoids the apt-lock race)"
  cloud-init status --wait || log "cloud-init reported non-OK; proceeding anyway"
  apt_get update
  apt_get install curl ca-certificates gnupg jq chrony netcat-openbsd
  # chrony: the STEP<->telemetry join is by epoch clock, so ms-accurate NTP is mandatory.
  systemctl enable --now chrony >/dev/null 2>&1 || systemctl enable --now chronyd >/dev/null 2>&1 || true
  mark_phase apt
fi

########################################################################
# Phase (b): NVIDIA driver + fabric-manager (SKIP if 8 GPUs already up)
########################################################################
if ! have_phase driver; then
  if [[ "$(gpu_count)" == "8" ]] && systemctl is-active --quiet nvidia-fabricmanager; then
    log "(b) 8 GPUs already visible and fabric-manager active (DLVM image?) -- skipping driver install"
    mark_phase driver
  else
    log "(b) installing NVIDIA datacenter driver + fabric-manager"
    if ! dpkg -l cuda-keyring >/dev/null 2>&1; then
      tmp="$(mktemp -d)"
      curl -fsSL -o "$tmp/cuda-keyring.deb" \
        https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
      dpkg -i "$tmp/cuda-keyring.deb"; rm -rf "$tmp"
      apt_get update
    fi
    # cuda-drivers = the datacenter driver metapackage.
    # nvidia-fabricmanager is MANDATORY on a3 (SXM/NVSwitch) -- without it the NVSwitch
    # fabric never initialises and CUDA sees the GPUs but NVLink collectives fail.
    apt_get install cuda-drivers nvidia-fabricmanager || true

    # fabric-manager version MUST match the driver; enable+start it.
    systemctl enable --now nvidia-fabricmanager || log "WARN: nvidia-fabricmanager did not start (will re-check after reboot)"

    if [[ "$(gpu_count)" == "8" ]] && nvidia-smi >/dev/null 2>&1; then
      log "(b) driver OK: 8 GPUs enumerated"
      rm -f "${STATE}/driver-reboots"
      mark_phase driver
    else
      # The session's exact failure: userspace installed but the kernel module for the
      # RUNNING kernel isn't loaded (version mismatch). Pin the GCP-flavoured precompiled
      # module to this kernel, then reboot+resume. (NVIDIA_BRANCH lets you force a branch,
      # e.g. NVIDIA_BRANCH=570; otherwise we try to derive it and fall back to a reboot,
      # which alone often clears a freshly-installed-but-unloaded module.)
      KREL="$(uname -r)"
      BRANCH="${NVIDIA_BRANCH:-$(dpkg -l 'nvidia-driver-*' 2>/dev/null | awk '/^ii/{print $2}' | grep -oE '[0-9]+' | head -1)}"
      log "(b) driver not live yet (kernel=$KREL branch=${BRANCH:-unknown}); installing kernel-pinned modules"
      if [[ -n "$BRANCH" ]]; then
        apt_get install "linux-modules-nvidia-${BRANCH}-gcp" \
          || apt_get install "linux-modules-nvidia-${BRANCH}-${KREL}" \
          || log "WARN: could not install linux-modules-nvidia-${BRANCH}-* (VERIFY exact pkg name: apt-cache search linux-modules-nvidia)"
      fi
      reboot_or_exit42
    fi
  fi
fi

########################################################################
# Phase (c): DCGM -- prove NVLink counters (fields 1011/1012) are LIVE
########################################################################
if ! have_phase dcgm; then
  log "(c) installing + starting DCGM (NVLink TX/RX counters must populate)"
  apt_get install datacenter-gpu-manager || log "WARN: datacenter-gpu-manager install failed"
  # Prefer the packaged service; fall back to a raw host engine daemon like setup.sh does.
  systemctl enable --now nvidia-dcgm 2>/dev/null \
    || { pgrep -x nv-hostengine >/dev/null 2>&1 || nv-hostengine -d || true; }
  sleep 3
  if ! dcgmi discovery -l 2>/dev/null | grep -qE '8 GPUs|GPU 7'; then
    log "WARN: dcgmi discovery did not clearly report 8 GPUs -- check DCGM"
  fi
  # 1011 = NVLink TX bytes, 1012 = NVLink RX bytes. If this errors, the round's
  # whole cross-rank NVLink feature family would be blank, so treat it as important.
  if dcgmi dmon -e 1011,1012 -c 2 >/dev/null 2>&1; then
    log "(c) DCGM NVLink counters live (1011/1012 responding)"
  else
    log "WARN: 'dcgmi dmon -e 1011,1012' failed -- NVLink telemetry may be unavailable"
  fi
  mark_phase dcgm
fi

########################################################################
# Phase (d): private repo clone + the TWO venvs
########################################################################
if ! have_phase repo; then
  log "(d) cloning repo + building venvs"
  apt_get install git python3 python3-venv python3-pip
  # --- resolve a git credential for the PRIVATE clone, without ever using metadata ---
  PAT=""
  if [[ -n "${GIT_PAT:-}" ]]; then
    PAT="$GIT_PAT"
  elif [[ -r /etc/spar/git_pat ]]; then
    PAT="$(cat /etc/spar/git_pat)"          # cached from a previous interactive run (root-only 0600)
  elif [[ -t 0 ]]; then
    read -r -s -p "Fine-grained GitHub PAT (repo:read; blank = try anonymous): " PAT; echo
    if [[ -n "$PAT" ]]; then
      umask 077; printf '%s' "$PAT" > /etc/spar/git_pat; chmod 600 /etc/spar/git_pat
    fi
  fi
  CLONE_URL="$REPO_URL"
  if [[ -n "$PAT" ]]; then
    # inject token only into the transient clone URL; the stored remote is rewritten to tokenless below
    CLONE_URL="$(printf '%s' "$REPO_URL" | sed -E "s#https://#https://x-access-token:${PAT}@#")"
  fi
  if [[ ! -d "${REPO_DIR}/.git" ]]; then
    install -d "$(dirname "$REPO_DIR")"
    if ! git clone --branch "$REPO_BRANCH" "$CLONE_URL" "$REPO_DIR"; then
      log "FATAL: git clone failed. If the repo is private, re-run interactively so I can read a PAT,"
      log "       or set GIT_PAT / write /etc/spar/git_pat (0600)."
      exit 1
    fi
    git -C "$REPO_DIR" remote set-url origin "$REPO_URL"   # scrub any token from the saved remote
  else
    git -C "$REPO_DIR" fetch --depth=1 origin "$REPO_BRANCH" || true
    git -C "$REPO_DIR" checkout "$REPO_BRANCH" || true
  fi

  # venv-main: telemetry+classifier+FSDP/DDP stack (torch 2.5.1+cu121) + model libs.
  if [[ ! -x "${VENV_MAIN}/bin/python" ]]; then
    python3 -m venv "$VENV_MAIN"
    "${VENV_MAIN}/bin/pip" install --upgrade pip
    "${VENV_MAIN}/bin/pip" install \
      -r "${REPO_DIR}/requirements_v2.txt" \
      -r "${REPO_DIR}/requirements_gcp_extra.txt"
  fi
  # venv-vllm: vllm+ray ONLY. Separate venv because vllm repins torch and would
  # otherwise clobber venv-main's 2.5.1+cu121 pin (contract 2 / requirements_gcp_extra note).
  if [[ ! -x "${VENV_VLLM}/bin/python" ]]; then
    python3 -m venv "$VENV_VLLM"
    "${VENV_VLLM}/bin/pip" install --upgrade pip
    "${VENV_VLLM}/bin/pip" install vllm ray
  fi
  # DCGM python bindings come from the apt package, not pip -- expose them in venv-main
  # (mirrors setup.sh's dcgm.pth trick) so the collector's DCGM tier can import them.
  if [[ -d /usr/lib/python3/dist-packages ]]; then
    SITE="$("${VENV_MAIN}/bin/python" -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null || true)"
    [[ -n "$SITE" && ! -f "${SITE}/dcgm.pth" ]] && echo /usr/lib/python3/dist-packages > "${SITE}/dcgm.pth"
  fi
  mark_phase repo
fi

########################################################################
# Phase (e): /etc/spar/node_env + chrony offset gate (<50 ms) + record it
########################################################################
if ! have_phase clock; then
  log "(e) writing node_env + verifying NTP offset < 50 ms"
  # Wait up to ~30 iterations for chrony to converge to within 0.05 s (=50 ms).
  chronyc waitsync 30 0.05 || log "WARN: chronyc waitsync did not fully converge"
  OFF="$(chronyc tracking 2>/dev/null | awk '/Last offset/ {v=$4; if (v<0) v=-v; print v; exit}')"
  OFF="${OFF:-9.999}"
  cat > /etc/spar/node_env <<EOF
# Authoritative per-node identity. Sourced by env.sh on this node.
NODE_INDEX=${NODE_INDEX}
NODE_RANK=${NODE_RANK}
NODE_NAME=${NODE_NAME}
SESSION=${SESSION}
BUCKET=${BUCKET}
MASTER_ADDR=${MASTER_ADDR}
MASTER_PORT=${MASTER_PORT}
REPO_DIR=${REPO_DIR}
VENV_MAIN=${VENV_MAIN}
VENV_VLLM=${VENV_VLLM}
CHRONY_OFFSET_S=${OFF}
EOF
  chmod 644 /etc/spar/node_env
  # HARD GATE: a >50 ms clock error corrupts the epoch join, so fail loudly.
  if awk "BEGIN{exit !(${OFF} > 0.050)}"; then
    log "FATAL: NTP offset ${OFF}s > 50 ms -- the STEP<->telemetry epoch join would be wrong."
    log "       Check chrony ('chronyc sources -v'); do NOT collect until this is under 50 ms."
    exit 1
  fi
  log "(e) NTP offset ${OFF}s (< 50 ms) OK"
  mark_phase clock
fi

########################################################################
# Phase (f): systemd units -- final-sync on shutdown + preemption watcher
########################################################################
if ! have_phase services; then
  log "(f) installing spar-final-sync.service + spar-preempt-watch.service"
  # spar-final-sync: its ExecStop fires during the ACPI shutdown that a Spot STOP
  # triggers, flushing deltas inside the ~30 s window. TimeoutStopSec keeps systemd
  # from killing it early.
  cat > /etc/systemd/system/spar-final-sync.service <<EOF
[Unit]
Description=SPAR final telemetry sync on shutdown/preemption
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/bin/true
ExecStop=/usr/bin/env bash ${GCPDIR}/sync_data.sh --final
TimeoutStopSec=25

[Install]
WantedBy=multi-user.target
EOF

  # spar-preempt-watch: polls the metadata preempted flag (see preempt_watch.sh) and
  # triggers a graceful kill + final sync as soon as the ~30 s preemption notice lands.
  cat > /etc/systemd/system/spar-preempt-watch.service <<EOF
[Unit]
Description=SPAR Spot-preemption watcher
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=/usr/bin/env bash ${GCPDIR}/preempt_watch.sh
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

  systemctl daemon-reload
  systemctl enable --now spar-final-sync.service  >/dev/null 2>&1 || true
  systemctl enable --now spar-preempt-watch.service >/dev/null 2>&1 || true
  mark_phase services
fi

########################################################################
# Phase (g): verify + emit bootstrap_report.json (local + GCS)
########################################################################
log "(g) verifying node"
NGPU="$(gpu_count)"
TORCH_NGPU="$("${VENV_MAIN}/bin/python" -c 'import torch;print(torch.cuda.device_count())' 2>/dev/null || echo 0)"
NVLINK_OK=false; nvidia-smi topo -m 2>/dev/null | grep -q 'NV[0-9]' && NVLINK_OK=true
# GCS write probe -- proves the SA/scope can egress telemetry before we collect any.
GCS_OK=false
if printf 'ok %s\n' "$(spar_ts)" | gcloud storage cp - "${GCS_NODE}/meta/bootstrap_probe.txt" >/dev/null 2>&1; then
  GCS_OK=true
fi
DCGM_OK=false; dcgmi dmon -e 1011,1012 -c 1 >/dev/null 2>&1 && DCGM_OK=true

REPORT="${REPO_DIR}/logs/bootstrap_report_${NODE_NAME}.json"
install -d "${REPO_DIR}/logs"
cat > "$REPORT" <<EOF
{
  "node_name": "${NODE_NAME}",
  "node_index": ${NODE_INDEX},
  "session": "${SESSION}",
  "timestamp": "$(spar_ts)",
  "gpu_count": ${NGPU:-0},
  "torch_device_count": ${TORCH_NGPU:-0},
  "fabric_manager_active": $(systemctl is-active --quiet nvidia-fabricmanager && echo true || echo false),
  "nvlink_topo_ok": ${NVLINK_OK},
  "dcgm_nvlink_ok": ${DCGM_OK},
  "gcs_write_ok": ${GCS_OK},
  "chrony_offset_s": "$(awk -F= '/^CHRONY_OFFSET_S=/{print $2}' /etc/spar/node_env 2>/dev/null || echo unknown)",
  "master_addr": "${MASTER_ADDR}:${MASTER_PORT}"
}
EOF
gcloud storage cp "$REPORT" "${GCS_NODE}/meta/$(basename "$REPORT")" >/dev/null 2>&1 || true
cat "$REPORT"

# Hard asserts on the load-bearing invariants.
FAIL=0
[[ "${NGPU:-0}" == "8" ]]        || { log "ASSERT FAIL: gpu_count=$NGPU (want 8)"; FAIL=1; }
[[ "${TORCH_NGPU:-0}" == "8" ]]  || { log "ASSERT FAIL: torch.cuda.device_count()=$TORCH_NGPU (want 8)"; FAIL=1; }
$NVLINK_OK                        || log "WARN: nvidia-smi topo -m shows no NVLink (NV#) pairs"
$GCS_OK                           || log "WARN: GCS write probe failed -- telemetry egress will not work"
if [[ "$FAIL" != "0" ]]; then
  log "bootstrap verification FAILED -- not marking bootstrap.done"
  exit 1
fi

touch "${STATE}/bootstrap.done"   # spar-bootstrap.service ConditionPathExists=! stops re-running now
log "BOOTSTRAP COMPLETE on ${NODE_NAME}. From node0 run: bash ${GCPDIR}/verify_pair.sh"
