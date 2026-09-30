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

# --- resolve THIS node's identity -------------------------------------------
# ORDER MATTERS AND USED TO BE WRONG. env.sh is sourced above and exports
# BUCKET, MASTER_ADDR, MASTER_PORT, NODE0, NODE1, PROJECT, SESSION and ZONE with
# hand-run defaults, so the idiom `X="${X:-$(meta spar-x)}"` can never reach the
# metadata: X is already set. Four separate failures in two days came from this
# one shape (see docs/node_env_audit.md), so the metadata is now FORCED for
# everything the instance actually knows about itself.
#
# zone and project come from the metadata server's own endpoints rather than a
# spar-* key, because those are always right and need no provisioner cooperation.
_force() {  # _force <VAR> <spar-metadata-key>
  local v; v="$(meta "$2")"
  [[ -n "$v" ]] && printf -v "$1" '%s' "$v"
  return 0
}
_inst_meta() {  # bare instance metadata, not an attribute
  curl -sf -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/$1" 2>/dev/null || true
}
_z="$(_inst_meta instance/zone)"; _z="${_z##*/}"          # projects/N/zones/ZONE
[[ -n "$_z" ]] && ZONE="$_z"
_p="$(_inst_meta project/project-id)"
[[ -n "$_p" ]] && PROJECT="$_p"
_force SESSION     spar-session
_force MASTER_PORT spar-master-port
_force BUCKET      spar-bucket

# --- resolve THIS node's identity (env > metadata > sensible default) --------
NODE_INDEX="${NODE_INDEX:-$(meta spar-node-index)}"
NODE_INDEX="${NODE_INDEX:-0}"
NODE_NAME="${NODE_NAME:-$(meta spar-node-name)}"
NODE_NAME="${NODE_NAME:-$(hostname -s)}"
SESSION="${SESSION:-$(meta spar-session)}"; SESSION="${SESSION:-$(date -u +%Y%m%d)}"
# INSTANCE METADATA MUST WIN over env.sh's default here. env.sh is sourced above
# and unconditionally exports MASTER_ADDR="${MASTER_ADDR:-$NODE0}" with NODE0
# defaulting to spar-gpu-0, so `${MASTER_ADDR:-$(meta ...)}` never reached the
# metadata -- it was always already set. Measured 2026-09-21 on the A100-80GB
# pair: the nodes were spar-gpua-0/1, the template metadata said
# spar-master-addr=spar-gpua-0, and /etc/spar/node_env still got
# MASTER_ADDR=spar-gpu-0. Ray then could not form a cluster against a host that
# does not exist in that project region, and every 2-node cell failed
# ray-cluster-incomplete. The node's own metadata is the authority for who the
# master is; env.sh's value is only a fallback for hand-driven runs.
_meta_master="$(meta spar-master-addr)"
[[ -n "$_meta_master" ]] && MASTER_ADDR="$_meta_master"
MASTER_ADDR="${MASTER_ADDR:-$NODE0}"
MASTER_PORT="${MASTER_PORT:-$(meta spar-master-port)}"; MASTER_PORT="${MASTER_PORT:-29500}"
BUCKET="${BUCKET:-$(meta spar-bucket)}"; BUCKET="${BUCKET:-gs://${PROJECT}-telemetry}"
NODE_RANK="$NODE_INDEX"
# NODE0/NODE1 likewise: env.sh's spar-gpu-{0,1} defaults were being written into
# node_env verbatim, and every peer-facing consumer then pointed at a host that
# is not part of this experiment -- cell_barrier's peer, launch_queue's
# STOP_WHEN_DONE rendezvous, and verify_pair's describe/ssh. Derive them from
# this node's own name and its peer's, ordered by index.
# A MIG template is shared by both nodes, so a per-node spar-peer-name cannot
# exist on that path; the pair's two names are shipped as spar-node0-name /
# spar-node1-name instead, and NODE_INDEX says which one is me. spar-peer-name
# remains supported for the per-instance provisioners.
_n0="$(meta spar-node0-name)"; _n1="$(meta spar-node1-name)"
if [[ -n "$_n0" && -n "$_n1" ]]; then
  NODE0="$_n0"; NODE1="$_n1"
else
  _peer_name="$(meta spar-peer-name)"
  if [[ -n "$_peer_name" ]]; then
    if [[ "$NODE_INDEX" == "0" ]]; then NODE0="$NODE_NAME"; NODE1="$_peer_name"
    else                                NODE0="$_peer_name"; NODE1="$NODE_NAME"; fi
  fi
fi
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

# NVSwitch presence decides whether fabric-manager is REQUIRED or a no-op.
# a3-highgpu-8g (H100 SXM) has NVSwitches -> fabric-manager is mandatory.
# a2-highgpu-8g / a2-ultragpu-8g (A100) have NO NVSwitch: the GPUs are wired
# point-to-point (NV12), nvidia-fabricmanager.service exits 1 by design, and
# demanding it would (and did) send a perfectly good DLVM node down the
# driver-install path and break its working driver. See docs/gcp_a100.md.
nvswitch_count() { lspci -d 10de: 2>/dev/null | grep -ci nvswitch || true; }

# NVIDIA's apt repo (cuda-keyring). Phase (b) used to add this as a side effect,
# so on a DLVM image -- where (b) is skipped because the driver is already good --
# phase (c) found NO candidate for any DCGM package and "install failed". Both
# phases now call this explicitly.
ensure_cuda_repo() {
  dpkg -l cuda-keyring >/dev/null 2>&1 && return 0
  local rel tmp
  . /etc/os-release
  rel="${ID}${VERSION_ID//./}"            # ubuntu22.04 -> ubuntu2204
  tmp="$(mktemp -d)"
  if curl -fsSL -o "$tmp/cuda-keyring.deb" \
      "https://developer.download.nvidia.com/compute/cuda/repos/${rel}/$(uname -m)/cuda-keyring_1.1-1_all.deb"; then
    dpkg -i "$tmp/cuda-keyring.deb" && apt_get update
  else
    log "WARN: could not fetch cuda-keyring for ${rel}/$(uname -m)"
  fi
  rm -rf "$tmp"
}

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
  _nvsw="$(nvswitch_count)"
  if [[ "$(gpu_count)" == "8" ]] && nvidia-smi >/dev/null 2>&1 \
     && { [[ "$_nvsw" == "0" ]] || systemctl is-active --quiet nvidia-fabricmanager; }; then
    log "(b) 8 GPUs already live on driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1) (nvswitch=$_nvsw) -- skipping driver install"
    mark_phase driver
  else
    log "(b) installing NVIDIA datacenter driver + fabric-manager"
    ensure_cuda_repo
    # cuda-drivers = the datacenter driver metapackage.
    # nvidia-fabricmanager is MANDATORY on a3 (SXM/NVSwitch) -- without it the NVSwitch
    # fabric never initialises and CUDA sees the GPUs but NVLink collectives fail.
    apt_get install cuda-drivers nvidia-fabricmanager || true

    # fabric-manager version MUST match the driver; enable+start it.
    if [[ "$(nvswitch_count)" == "0" ]]; then
      log "(b) no NVSwitch on this machine type -- skipping fabric-manager (expected on a2/A100)"
    else
      systemctl enable --now nvidia-fabricmanager || log "WARN: nvidia-fabricmanager did not start (will re-check after reboot)"
    fi

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
  ensure_cuda_repo
  # NVIDIA retired the DCGM 3.x "datacenter-gpu-manager" name; DCGM 4 ships
  # -4-cuda12 (full) and -4-core. Try newest-first and stop at the first hit.
  _dcgm_ok=0
  for _p in datacenter-gpu-manager-4-cuda12 datacenter-gpu-manager-4-core datacenter-gpu-manager; do
    if [[ -n "$(apt-cache policy "$_p" 2>/dev/null | awk '/Candidate:/{print $2}' | grep -v '(none)')" ]]; then
      if apt_get install "$_p"; then log "(c) installed $_p"; _dcgm_ok=1; break; fi
    fi
  done
  [[ "$_dcgm_ok" == "1" ]] || log "WARN: no installable DCGM package found (NVLink counters will be blank)"
  # Prefer the packaged service; fall back to a raw host engine daemon like setup.sh does.
  systemctl enable --now nvidia-dcgm 2>/dev/null \
    || { pgrep -x nv-hostengine >/dev/null 2>&1 || nv-hostengine -d || true; }
  # nv-hostengine needs a moment after install; retry instead of a single sleep 3.
  _disc=""
  for _i in $(seq 1 10); do
    _disc="$(dcgmi discovery -l 2>/dev/null || true)"
    printf '%s' "$_disc" | grep -qE '8 GPUs|GPU 7' && break
    sleep 2
  done
  if printf '%s' "$_disc" | grep -qE '8 GPUs|GPU 7'; then
    log "(c) DCGM sees 8 GPUs"
  else
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
# Phase (c2): container runtime for the NGC HPC cells
########################################################################
# s16_nonml_hpl and s16_nonml_gromacs run real HPC benchmarks out of the NGC
# containers. On 2026-09-30 all four attempts died two seconds in with
#   FATAL: no docker/podman on PATH
# because the image family we provision (common-cu129-ubuntu-2204-nvidia-580)
# does not ship a container runtime. The workload had been validated on a
# machine that happened to have docker, which is exactly the kind of check that
# proves nothing about the target.
#
# This phase is best-effort on purpose: every other cell in the queue runs fine
# without a container runtime, so a failure here must cost us two cells, never
# the round.
if ! have_phase container; then
  log "(c2) installing container runtime + NVIDIA container toolkit"
  if command -v docker >/dev/null 2>&1; then
    log "(c2) docker already present at $(command -v docker)"
  else
    apt_get install docker.io || log "(c2) WARN: docker.io install failed"
  fi
  # The toolkit is what makes `docker run --gpus all` work; docker alone is not
  # enough and fails only at run time, inside the cell.
  if ! dpkg -s nvidia-container-toolkit >/dev/null 2>&1; then
    _nvctk_list=/etc/apt/sources.list.d/nvidia-container-toolkit.list
    if curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
         | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg 2>/dev/null; then
      curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        > "$_nvctk_list" 2>/dev/null || true
      apt_get update || true
      apt_get install nvidia-container-toolkit || log "(c2) WARN: toolkit install failed"
    else
      log "(c2) WARN: could not fetch the NVIDIA container-toolkit key"
    fi
  fi
  if command -v nvidia-ctk >/dev/null 2>&1; then
    nvidia-ctk runtime configure --runtime=docker >/dev/null 2>&1 || true
    systemctl restart docker >/dev/null 2>&1 || true
  fi
  # Prove it end to end rather than trusting the installs. `docker run --gpus
  # all` seeing zero GPUs is the failure hpc_container.py already refuses on,
  # and it is far cheaper to learn about it here than 6 hours into a queue.
  if command -v docker >/dev/null 2>&1 \
     && timeout 300 docker run --rm --gpus all ubuntu:22.04 true >/dev/null 2>&1; then
    log "(c2) container runtime OK (docker + --gpus all)"
    mark_phase container
  else
    log "(c2) WARN: no working GPU container runtime -- the HPC cells will skip"
  fi
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
  if [[ -f "${REPO_DIR}/requirements_v2.txt" && ! -d "${REPO_DIR}/.git" ]]; then
    # Pre-staged by `git archive | gcloud compute scp` -- no credential needed, and
    # nothing to fetch. scripts/gcp/stage_repo.sh writes REPO_DIR/.spar_staged_commit.
    log "(d) using pre-staged repo tree at ${REPO_DIR} (commit $(cat "${REPO_DIR}/.spar_staged_commit" 2>/dev/null || echo unknown)) -- skipping clone"
  elif [[ ! -d "${REPO_DIR}/.git" ]]; then
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
  # Remove flashinfer, which vllm pulls in as an OPTIONAL accelerated-allreduce
  # backend. Two reasons it cannot stay on these nodes:
  #   1. flashinfer 0.6.x annotates `array.array[int]` at module scope in
  #      comm/fd_exchange.py. array.array only became subscriptable in Python
  #      3.13, so under this venv's 3.10 the import raises TypeError -- and vllm
  #      guards that import with `except ImportError`, which does NOT catch it.
  #      Every tensor-parallel worker then dies in init_device().
  #   2. Patching that annotation only gets you further in: flashinfer then
  #      JIT-compiles and needs nvcc, but these images ship the driver without
  #      the CUDA toolkit, so workers die with "Could not find nvcc" -- and that
  #      failure mode HANGS the engine instead of exiting, burning the cell's
  #      whole timeout (~26 min of idle H100s per cell, observed 2026-08-20).
  # Uninstalling makes the import a clean ImportError, which vllm's guard does
  # catch: fi_ar_available=False and it falls back to custom allreduce / pynccl.
  # Do NOT "fix" this by installing the CUDA toolkit: flashinfer would then JIT
  # at cell start, inside the measurement window.
  if "${VENV_VLLM}/bin/pip" show flashinfer-python >/dev/null 2>&1; then
    "${VENV_VLLM}/bin/pip" uninstall -y flashinfer-python >/dev/null 2>&1 || true
    log "    removed flashinfer-python (vllm falls back to custom allreduce)"
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
# The PEER names must be recorded too, or every peer-facing script (verify_pair.sh
# above all) silently falls back to env.sh's spar-gpu-{0,1} defaults. On a renamed
# pair that is not merely wrong, it is dangerously wrong: the old spar-gpu-1 still
# existed (stopped), so `instances describe` succeeded, HAVE_NODE1=1, and
# verify_pair reported dns/port/nccl/clock FAIL against a host that was not
# part of this experiment at all.
NODE0=${NODE0}
NODE1=${NODE1}
# ZONE/PROJECT likewise: without them a node falls back to env.sh's default zone
# and every `gcloud compute ... describe` about its own peer looks in the wrong
# place and reports "not found" -- which verify_pair.sh renders as the far more
# innocent-sounding "node1 not provisioned (single-node mode)".
ZONE=${ZONE}
PROJECT=${PROJECT}
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
_topo="$(nvidia-smi topo -m 2>/dev/null || true)"
NVLINK_PAIRS="$(printf '%s' "$_topo" | grep -oE '\bNV[0-9]+\b' | wc -l | tr -d ' ')"
NVLINK_OK=false; [[ "${NVLINK_PAIRS:-0}" -gt 0 ]] && NVLINK_OK=true
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
  "nvswitch_count": $(nvswitch_count),
  "nvlink_topo_ok": ${NVLINK_OK},
  "nvlink_pair_entries": ${NVLINK_PAIRS:-0},
  "dcgm_nvlink_ok": ${DCGM_OK},
  "gcs_write_ok": ${GCS_OK},
  "chrony_offset_s": "$(awk -F= '/^CHRONY_OFFSET_S=/{print $2}' /etc/spar/node_env 2>/dev/null || echo unknown)",
  "master_addr": "${MASTER_ADDR}:${MASTER_PORT}"
}
EOF
gcloud storage cp "$REPORT" "${GCS_NODE}/meta/$(basename "$REPORT")" >/dev/null 2>&1 || true
cat "$REPORT"

# --- stage CIFAR-10 from GCS ------------------------------------------------
# Every freshly provisioned node used to fetch this from cave.cs.toronto.edu at
# cell time, and that source is slow enough to blow a 900 s cell timeout on its
# own: s16_infercnn_rn50 died at 933 s on 2026-09-23 and again on 2026-09-26
# with a single in-flight download, after an flock had already removed the
# sixteen-way race. s16_train_ddp_resnet50_b512 then failed outright, because
# pytorch_training_ddp.py opens the dataset with download=False and got
# "Dataset not found or corrupted".
#
# scripts/gcp/preload_vllm_models.sh does stage it, but it is a documented manual
# step that nothing in the boot path calls, so it never ran on a flex node. Pull
# it from our own bucket instead: same region, ~170 MB, seconds rather than
# minutes, and no dependency on a university mirror being reachable or fast.
CIFAR_DIR="${REPO_DIR}/data/cifar10"
CIFAR_SRC="${BUCKET}/staging/cifar10/cifar-10-python.tar.gz"
if [[ -d "${CIFAR_DIR}/cifar-10-batches-py" ]]; then
  log "cifar10 already extracted at ${CIFAR_DIR}"
else
  mkdir -p "$CIFAR_DIR"
  if gcloud storage cp "$CIFAR_SRC" "${CIFAR_DIR}/cifar-10-python.tar.gz" >/dev/null 2>&1; then
    tar xzf "${CIFAR_DIR}/cifar-10-python.tar.gz" -C "$CIFAR_DIR" \
      && log "cifar10 staged from ${CIFAR_SRC}" \
      || log "WARN: cifar10 archive fetched but did not extract"
  else
    # Not fatal: the workloads can still fall back to downloading it themselves,
    # just slowly. Worth a loud line because that fallback is what times out.
    log "WARN: could not fetch ${CIFAR_SRC} -- CIFAR cells will fall back to the"
    log "      upstream mirror and may exceed their cell timeout"
  fi
fi

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
