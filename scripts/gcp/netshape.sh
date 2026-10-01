#!/usr/bin/env bash
# netshape.sh -- cap the link to the PEER NODE only, for one collection cell.
#
# Why this exists
# ---------------
# Every cell in this corpus has run on a datacenter NIC. The threat model we are
# built against (arXiv 2605.29359) assumes a 100 Mbps consumer link with 100 ms
# round-trip latency. A 5.2 GB DiLoCo sync takes ~0.4 s on our link and ~416 s at
# 100 Mbps, with the GPUs idle throughout -- and GPU utilisation is most of what
# the classifier sees. So telemetry collected unshaped is from the wrong regime,
# and a shaped twin of each cell is what makes bandwidth a controlled variable
# instead of a confound.
#
#   netshape.sh up   [--rate 100mbit] [--delay-ms 50] [--ttl N] [--json DIR]
#   netshape.sh down [--json DIR]
#   netshape.sh verify [--rate 100mbit]
#   netshape.sh status
#   netshape.sh reap                       # internal: TTL watchdog
#
# SMOKED ON THE REAL IMAGE, 2026-09-30. Two e2-standard-2 Spot VMs booted from
# common-cu129-ubuntu-2204-nvidia-580 (kernel 6.8.0-1066-gcp): tc present,
# sch_htb and sch_netem both loadable, peer link 4201 Mbps -> 96.2 Mbps under a
# 100mbit cap (-3.8%), ping +50.7 ms, HTB overlimits 3558. Class 1:20 recorded
# 40,261,270 bytes against a 40 MB test transfer, i.e. ONLY the peer's traffic
# entered the shaped class -- a gcloud storage call completed normally while the
# cap was on. Teardown restored 4632 Mbps and left `qdisc mq` behind.
#
# UNITS. --delay-ms is ONE WAY. Both nodes shape egress, so 50 gives a 100 ms
# RTT, which is what the paper means. And 100 Mbps is 11.92 MiB/s in the NIC
# sidecar's columns -- those are named nic_*_mbps but hold MEBIBYTES per second
# (scripts/collect_host_telemetry.py). A threshold of "100" against them would
# be 839 Mbps, 8.4x off.
#
# WHAT IS NEVER SHAPED. The filter matches the peer's /32 and nothing else, so
# the default class carries GCS telemetry sync, SSH, the GCS-file cell barrier,
# the metadata server and NTP at line rate. If the filter fails to install, the
# node runs UNSHAPED rather than broken -- the failure mode has to be "no
# shaping", never "no control plane".
#
# SELF-CLEANING. Remote kills are blocked in this project and scripts must clean
# up after themselves. Shaping that survives its cell would silently corrupt
# every later cell in the queue, which is worse than any single failure this
# project has had. Four layers: `down` runs unconditionally before every cell
# (shaped or not), a trap covers the cell, a detached TTL reaper survives even
# SIGKILL of the queue, and run_stage_and_stop.sh clears before the final sync.
set -uo pipefail

ACTION="${1:-status}"; shift || true
RATE="${NETSHAPE_RATE:-100mbit}"
DELAY_MS="${NETSHAPE_DELAY_MS:-50}"
TTL=""
JSONDIR=""
while [ $# -gt 0 ]; do
  case "$1" in
    --rate) RATE="$2"; shift 2;;
    --delay-ms) DELAY_MS="$2"; shift 2;;
    --ttl) TTL="$2"; shift 2;;
    --json) JSONDIR="$2"; shift 2;;
    *) shift;;
  esac
done

STATE=/run/spar
mkdir -p "$STATE" 2>/dev/null || STATE=/tmp
IFACE_F="$STATE/netshape.iface"
DEADLINE_F="$STATE/netshape.deadline"
say() { echo "[netshape] $*" >&2; }

# BDP at 100 Mbps x 100 ms RTT is 1.25 MB ~= 860 packets. A 1500-packet limit is
# a little under 2x that: deep enough that a bulk NCCL flow is not tail-dropped
# into collapse, shallow enough that we are not building a ten-second
# bufferbloat queue and calling it latency.
LIMIT="${NETSHAPE_LIMIT:-1500}"

resolve_peer() {
  [ -n "${PEER_IP:-}" ] && { echo "$PEER_IP"; return 0; }
  local name=""
  # MASTER_ADDR is node0's OWN name on node0, so it is not "the peer" there.
  # Getting this backwards shapes nothing and reports success.
  if [ "${NODE_RANK:-0}" = "0" ]; then name="${PEER:-${NODE1:-}}"
  else                                 name="${MASTER_ADDR:-${NODE0:-}}"; fi
  [ -n "$name" ] || return 1
  local ip; ip="$(getent hosts "$name" 2>/dev/null | awk '{print $1; exit}')"
  if [ -z "$ip" ] && command -v gcloud >/dev/null 2>&1; then
    ip="$(timeout 30 gcloud compute instances describe "$name" \
            --project="${PROJECT:-}" --zone="${ZONE:-}" \
            --format='value(networkInterfaces[0].networkIP)' 2>/dev/null)"
  fi
  [ -n "$ip" ] || return 1          # never guess: a wrong IP shapes nothing
  printf '%s' "$ip"
}

resolve_iface() {  # the iface that actually routes to the peer, not "the default"
  # Phase (c2) of bootstrap installs docker, which creates docker0. Picking the
  # default route blind can shape an interface NCCL is not using.
  ip -o route get "$1" 2>/dev/null \
    | awk '{for(i=1;i<=NF;i++) if($i=="dev"){print $(i+1); exit}}'
}

have_tc() { command -v tc >/dev/null 2>&1; }

do_down() {
  local iface="${1:-}"
  [ -z "$iface" ] && iface="$(cat "$IFACE_F" 2>/dev/null)"
  [ -z "$iface" ] && iface="$(ip -o -4 route show to default 2>/dev/null | awk '{print $5; exit}')"
  if [ -n "$iface" ] && have_tc; then
    tc qdisc del dev "$iface" root >/dev/null 2>&1 || true
  fi
  rm -f "$IFACE_F" "$DEADLINE_F" 2>/dev/null || true
  return 0
}

write_json() {  # write_json <applied> <reason> <extra-json>
  [ -z "$JSONDIR" ] && return 0
  mkdir -p "$JSONDIR" 2>/dev/null || return 0
  local f="$JSONDIR/netshape_${NETSHAPE_CELL:-unknown}_node${NODE_RANK:-0}_${NETSHAPE_RID:-norid}.json"
  printf '{"applied": %s, "rate": "%s", "delay_ms": %s, "iface": "%s", "peer_ip": "%s", "cell": "%s", "run_id": "%s", "reason": %s%s}\n' \
    "$1" "$RATE" "$DELAY_MS" "${IFACE:-}" "${PEER_IP_R:-}" \
    "${NETSHAPE_CELL:-}" "${NETSHAPE_RID:-}" "$2" "${3:-}" > "$f" 2>/dev/null || true
}

case "$ACTION" in

up)
  do_down                                   # idempotent: always start clean
  if ! have_tc; then
    say "WARN: tc not present -- running UNSHAPED"
    write_json false '"no tc"'; exit 0
  fi
  PEER_IP_R="$(resolve_peer)" || {
    say "WARN: cannot resolve the peer -- running UNSHAPED (a wrong IP would"
    say "      shape nothing while reporting success)"
    write_json false '"peer unresolved"'; exit 0; }
  IFACE="$(resolve_iface "$PEER_IP_R")"
  [ -n "$IFACE" ] || { say "WARN: no route to $PEER_IP_R -- UNSHAPED"
                       write_json false '"no route"'; exit 0; }

  # 1:10 is EVERYTHING ELSE at line rate -- GCS, SSH, the barrier, metadata.
  # 1:20 is the peer, and only the peer.
  ok=1
  tc qdisc add dev "$IFACE" root handle 1: htb default 10 r2q 1000 2>/dev/null || ok=0
  tc class add dev "$IFACE" parent 1: classid 1:10 htb rate 100gbit ceil 100gbit 2>/dev/null || ok=0
  tc class add dev "$IFACE" parent 1: classid 1:20 htb rate "$RATE" ceil "$RATE" \
       burst 15k cburst 15k 2>/dev/null || ok=0
  tc qdisc add dev "$IFACE" parent 1:20 handle 20: netem \
       delay "${DELAY_MS}ms" limit "$LIMIT" 2>/dev/null || ok=0
  tc filter add dev "$IFACE" protocol ip parent 1: prio 1 u32 \
       match ip dst "${PEER_IP_R}/32" flowid 1:20 2>/dev/null || ok=0

  if [ "$ok" != 1 ]; then
    say "WARN: tc stack incomplete -- tearing down and running UNSHAPED"
    do_down "$IFACE"; write_json false '"tc install failed"'; exit 0
  fi
  echo "$IFACE" > "$IFACE_F"

  if [ -n "$TTL" ]; then
    # Survives SIGKILL of the queue, which the trap does not.
    echo "$(( $(date +%s) + TTL ))" > "$DEADLINE_F"
    setsid "$0" reap </dev/null >/dev/null 2>&1 & disown
  fi
  say "shaped $IFACE -> $PEER_IP_R at $RATE, +${DELAY_MS}ms one-way"
  write_json true 'null'
  ;;

down)
  IFACE="$(cat "$IFACE_F" 2>/dev/null)"
  PEER_IP_R="$(resolve_peer 2>/dev/null)" || PEER_IP_R=""   # for the sidecar only
  # Record the post-hoc counters BEFORE tearing the qdisc down: `overlimits`
  # is the only proof that HTB actually throttled rather than merely being
  # installed on traffic that never reached the cap.
  extra=""
  if [ -n "$IFACE" ] && have_tc; then
    stats="$(tc -s class show dev "$IFACE" classid 1:20 2>/dev/null | tr '\n' ' ')"
    sent="$(printf '%s' "$stats" | grep -oE 'Sent [0-9]+' | awk '{print $2}' | head -1)"
    over="$(printf '%s' "$stats" | grep -oE 'overlimits [0-9]+' | awk '{print $2}' | head -1)"
    extra=", \"class_sent_bytes\": ${sent:-0}, \"class_overlimits\": ${over:-0}"
    [ "${over:-0}" = "0" ] && say "WARN: class 1:20 never hit its ceiling -- the"
    [ "${over:-0}" = "0" ] && say "      cell may not be evidence of the shaped regime"
  fi
  do_down "$IFACE"
  write_json false '"torn down"' "$extra"
  say "cleared"
  ;;

verify)
  IFACE="$(cat "$IFACE_F" 2>/dev/null)"
  PEER_IP_R="$(resolve_peer)" || { say "no peer"; exit 1; }
  # Measure, do not assume. This project has been bitten repeatedly by reading
  # "no output" as evidence of absence.
  rtt="$(ping -n -c 5 -i 0.2 -q "$PEER_IP_R" 2>/dev/null \
          | awk -F/ '/rtt|round-trip/{print $5}')"
  say "rtt to peer: ${rtt:-unknown} ms (expect >= $(echo "$DELAY_MS" | awk '{print $1*0.8}'))"
  if [ -n "$IFACE" ] && have_tc; then
    tc -s class show dev "$IFACE" classid 1:20 2>/dev/null | head -4
  else
    say "no qdisc installed"; exit 1
  fi
  ;;

reap)
  # Detached watchdog: tear down if the deadline passes without a refresh.
  while :; do
    d="$(cat "$DEADLINE_F" 2>/dev/null)" || exit 0
    [ -z "$d" ] && exit 0
    now="$(date +%s)"
    [ "$now" -ge "$d" ] && { say "TTL expired -- clearing"; do_down; exit 0; }
    sleep 30
  done
  ;;

status)
  IFACE="$(cat "$IFACE_F" 2>/dev/null)"
  if [ -n "$IFACE" ]; then
    echo "shaped on $IFACE"; have_tc && tc qdisc show dev "$IFACE" | head -3
  else
    echo "not shaped"
  fi
  ;;

*)
  say "usage: netshape.sh {up|down|verify|status|reap} [--rate R] [--delay-ms N] [--ttl S] [--json DIR]"
  exit 2
  ;;
esac
