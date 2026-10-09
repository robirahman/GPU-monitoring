#!/usr/bin/env bash
# gpuslow.sh -- make a subset of this node's GPUs genuinely slower, for one cell.
#
# Why this exists
# ---------------
# Decoupled DiLoCo (arXiv 2604.21428) exists to absorb heterogeneous hardware,
# and its own experiment mixes TPUv5p with TPUv6e plus 10% artificial speed
# variance. We cannot provision a mixed pair -- a DWS Flex Start resize request
# is all-or-nothing per machine type, so two machine types means two independent
# requests with no atomicity between them -- so the heterogeneity is emulated.
#
# HOW, AND WHY NOT THE OBVIOUS WAY. The telemetry records gpu_utilization_pct,
# power_draw_w, sm_clock_mhz, mem_used_mb and temperature_c, so the mechanism has
# to produce a realistic slow-chip signature rather than an artefact that gives
# the game away:
#
#   injected per-step sleeps  -- REJECTED. Utilisation drops to zero in bursts at
#       the step period. No real slow chip idles between steps; it runs the same
#       kernels for longer. And this project already found that a fixed-period
#       duty cycle writes itself into acf10/acf20, which is precisely the feature
#       family the cross-rank block computes.
#   smaller per-rank batch    -- REJECTED. It makes the rank FASTER, and the only
#       way to slow it is a bigger batch, which moves mem_used_mb -- the feature
#       the model leans on hardest. That is a different job, not a slower chip.
#   -lgc alone                -- a hard clock lock pins sm_clock_mhz to a flat
#       line with near-zero variance, which a real A100 under load never shows.
#   -pl plus an -lgc BAND     -- CHOSEN. The power cap lets the chip's own DVFS
#       pick the clock, so sm_clock_mhz varies the way a thermally or
#       power-constrained GPU's does, while the band keeps the slowdown
#       deterministic and reproducible across reps.
#
#   gpuslow.sh up   --gpus 0,1,2,3 [--level slow|veryslow] [--ttl N] [--json DIR]
#   gpuslow.sh down [--json DIR]
#   gpuslow.sh verify
#   gpuslow.sh status
#   gpuslow.sh reap
#
# SELF-CLEANING, for the same reason netshape.sh is: a clock lock that outlives
# its cell makes every later cell in the queue 25% slower with a wrong power
# profile, and nothing in the telemetry would say so. Same four layers --
# unconditional reset before every cell, a trap, a detached TTL reaper, and the
# round-level clear.
#
# Precedent: workloads/adversarial_advanced.py::strategy_g_clock_throttled
# already does --lock-gpu-clocks/--reset-gpu-clocks with a finally: restore.
# This lifts that out of the workload so the cleanup is owned by the queue.
set -uo pipefail

ACTION="${1:-status}"; shift || true
GPUS=""
LEVEL="${GPUSLOW_LEVEL:-slow}"
TTL=""
JSONDIR=""
while [ $# -gt 0 ]; do
  case "$1" in
    --gpus) GPUS="$2"; shift 2;;
    --level) LEVEL="$2"; shift 2;;
    --ttl) TTL="$2"; shift 2;;
    --json) JSONDIR="$2"; shift 2;;
    *) shift;;
  esac
done

STATE=/run/spar
mkdir -p "$STATE" 2>/dev/null || STATE=/tmp
GPUS_F="$STATE/gpuslow.gpus"
DEADLINE_F="$STATE/gpuslow.deadline"
say() { echo "[gpuslow] $*" >&2; }
have_smi() { command -v nvidia-smi >/dev/null 2>&1; }

# A100-40GB SXM: 400 W default, 1410 MHz boost.
case "$LEVEL" in
  slow)     PL=250; LGC="900,1200";;
  veryslow) PL=180; LGC="600,800";;
  *)        say "unknown level $LEVEL"; exit 2;;
esac

default_pl() {  # the card's own default, so `down` restores rather than guesses
  nvidia-smi -i "$1" --query-gpu=power.default_limit --format=csv,noheader,nounits 2>/dev/null \
    | tr -d ' ' | head -1
}

write_json() {
  [ -z "$JSONDIR" ] && return 0
  mkdir -p "$JSONDIR" 2>/dev/null || return 0
  local f="$JSONDIR/gpuslow_${GPUSLOW_CELL:-unknown}_node${NODE_RANK:-0}_${GPUSLOW_RID:-norid}.json"
  printf '{"applied": %s, "level": "%s", "gpus": "%s", "power_limit_w": %s, "lgc": "%s", "cell": "%s", "run_id": "%s", "reason": %s}\n' \
    "$1" "$LEVEL" "${GPUS:-}" "$PL" "$LGC" "${GPUSLOW_CELL:-}" "${GPUSLOW_RID:-}" "$2" \
    > "$f" 2>/dev/null || true
}

do_down() {
  local list="${1:-}"
  [ -z "$list" ] && list="$(cat "$GPUS_F" 2>/dev/null)"
  if [ -n "$list" ] && have_smi; then
    for g in ${list//,/ }; do
      nvidia-smi -i "$g" -rgc >/dev/null 2>&1 || true          # release clock lock
      d="$(default_pl "$g")"
      [ -n "$d" ] && nvidia-smi -i "$g" -pl "$d" >/dev/null 2>&1 || true
    done
  fi
  rm -f "$GPUS_F" "$DEADLINE_F" 2>/dev/null || true
  return 0
}

case "$ACTION" in

up)
  do_down                                     # idempotent: start from default
  if ! have_smi; then
    say "WARN: no nvidia-smi -- running at FULL SPEED"
    write_json false '"no nvidia-smi"'; exit 0
  fi
  [ -n "$GPUS" ] || { say "WARN: --gpus not given -- nothing slowed"
                      write_json false '"no gpus selected"'; exit 0; }
  ok=1
  for g in ${GPUS//,/ }; do
    nvidia-smi -i "$g" -pm 1 >/dev/null 2>&1 || true
    nvidia-smi -i "$g" -pl "$PL" >/dev/null 2>&1 || ok=0
    nvidia-smi -i "$g" -lgc "$LGC" >/dev/null 2>&1 || ok=0
  done
  if [ "$ok" != 1 ]; then
    # Some SKUs and driver builds refuse power limits outside a narrow band.
    # Running a het cell at full speed under a _het name is a mislabelled cell,
    # so the caller is told and skips rather than collecting a lie.
    say "WARN: could not apply the power/clock cap -- reverting"
    do_down "$GPUS"; write_json false '"nvidia-smi refused"'; exit 1
  fi
  echo "$GPUS" > "$GPUS_F"
  if [ -n "$TTL" ]; then
    echo "$(( $(date +%s) + TTL ))" > "$DEADLINE_F"
    setsid "$0" reap </dev/null >/dev/null 2>&1 & disown
  fi
  say "slowed GPUs $GPUS to ${PL}W / ${LGC}MHz ($LEVEL)"
  write_json true 'null'
  ;;

down)
  GPUS="$(cat "$GPUS_F" 2>/dev/null)"
  do_down "$GPUS"
  write_json false '"restored"'
  say "restored"
  ;;

verify)
  have_smi || { say "no nvidia-smi"; exit 1; }
  # Measure, do not assume. A cap that silently failed looks identical to one
  # that worked until you read the clocks back.
  nvidia-smi --query-gpu=index,clocks.sm,clocks.max.sm,power.draw,power.limit \
             --format=csv,noheader
  ;;

reap)
  while :; do
    d="$(cat "$DEADLINE_F" 2>/dev/null)" || exit 0
    [ -z "$d" ] && exit 0
    [ "$(date +%s)" -ge "$d" ] && { say "TTL expired -- restoring"; do_down; exit 0; }
    sleep 30
  done
  ;;

status)
  g="$(cat "$GPUS_F" 2>/dev/null)"
  if [ -n "$g" ]; then echo "slowed: $g"; else echo "all GPUs at default"; fi
  ;;

*)
  say "usage: gpuslow.sh {up|down|verify|status|reap} [--gpus 0,1] [--level slow|veryslow] [--ttl S] [--json DIR]"
  exit 2
  ;;
esac
