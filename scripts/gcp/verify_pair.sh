#!/usr/bin/env bash
# verify_pair.sh -- prove the two nodes can actually run a 16-GPU job before you
# spend Spot time on the real queue. Intended to run ON node0 (it drives node1 via
# `gcloud compute ssh`, which works from the node thanks to --scopes=cloud-platform).
#
#   bash /opt/spar/gcp/verify_pair.sh
#
# Checks (prints a PASS/FAIL table):
#   1. internal DNS: node0 can resolve spar-gpu-1
#   2. port :29500 reachable on node1 (transient listener + probe)
#   3. 2-node x 8-GPU NCCL all-reduce over torchrun rendezvous (60 s budget)
#   4. cross-node chrony offset both < 50 ms
#
# If node1 doesn't exist yet (single-node V1 gate), it degrades to a 1-node NCCL
# smoke and skips the cross-node rows.
set -uo pipefail   # NOT -e: we want every row to run and report, not abort on the first FAIL
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

require_cmd gcloud || true
# NB: the port/clock probes use the system python3 (trivial stdlib socket calls);
# only the NCCL smoke needs venv-main's torchrun (torch is only in venv-main).
TORCHRUN="${VENV_MAIN}/bin/torchrun"
[[ -x "$TORCHRUN" ]] || TORCHRUN="$(command -v torchrun || echo torchrun)"

# run a command on node1 (prefer plain internal ssh; fall back to gcloud)
remote1() {
  if ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=8 "$NODE1" true 2>/dev/null; then
    ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$NODE1" "$@"
  else
    gcloud compute ssh "$NODE1" --zone="$ZONE" --project="$PROJECT" --command="$*" 2>/dev/null
  fi
}

declare -a ROWS
row() { ROWS+=("$1|$2|$3"); }   # name | PASS/FAIL | detail

HAVE_NODE1=1
gcloud compute instances describe "$NODE1" --zone="$ZONE" --project="$PROJECT" >/dev/null 2>&1 || HAVE_NODE1=0

# --- 1. internal DNS ---------------------------------------------------------
if [[ "$HAVE_NODE1" == "1" ]]; then
  if getent hosts "$NODE1" >/dev/null 2>&1; then
    row "dns:${NODE1}" PASS "$(getent hosts "$NODE1" | awk '{print $1}')"
  else
    row "dns:${NODE1}" FAIL "getent hosts $NODE1 returned nothing"
  fi
else
  row "dns:${NODE1}" SKIP "node1 not provisioned (single-node mode)"
fi

# --- 2. port :29500 reachable (transient listener on node1) ------------------
if [[ "$HAVE_NODE1" == "1" ]]; then
  # Start a throwaway listener on node1, probe from node0, then it self-times-out.
  remote1 "timeout 20 python3 -c 'import socket; s=socket.socket(); s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1); s.bind((\"0.0.0.0\",${MASTER_PORT})); s.listen(1); import time; time.sleep(18)'" >/dev/null 2>&1 &
  LPID=$!
  sleep 3
  if command -v nc >/dev/null 2>&1 && nc -z -w5 "$NODE1" "$MASTER_PORT" 2>/dev/null; then
    row "port:${MASTER_PORT}" PASS "nc -z $NODE1 $MASTER_PORT ok"
  elif python3 -c "import socket,sys; sys.exit(0 if socket.socket().connect_ex(('${NODE1}',${MASTER_PORT}))==0 else 1)" 2>/dev/null; then
    row "port:${MASTER_PORT}" PASS "socket connect ok"
  else
    row "port:${MASTER_PORT}" FAIL "cannot reach ${NODE1}:${MASTER_PORT} (firewall? listener?)"
  fi
  wait "$LPID" 2>/dev/null || true
else
  row "port:${MASTER_PORT}" SKIP "node1 not provisioned"
fi

# --- 3. NCCL all-reduce smoke ------------------------------------------------
# Tiny all-reduce: every rank contributes its rank; the sum must equal
# world_size*(world_size-1)/2. Proves NCCL init + collective across all ranks.
SMOKE=/tmp/_spar_allreduce_smoke.py
cat > "$SMOKE" <<'PY'
import os, torch, torch.distributed as dist
dist.init_process_group("nccl")
r, w = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
t = torch.full((1024, 1024), float(r), device="cuda")
dist.all_reduce(t)
expected = float(w * (w - 1) // 2)
ok = abs(t[0, 0].item() - expected) < 1e-3
if r == 0:
    print(f"[smoke] world={w} allreduce={t[0,0].item():.1f} expected={expected:.1f} -> {'OK' if ok else 'MISMATCH'}", flush=True)
dist.barrier(); dist.destroy_process_group()
raise SystemExit(0 if ok else 2)
PY

RDZV="${MASTER_ADDR}:${MASTER_PORT}"
if [[ "$HAVE_NODE1" == "1" ]]; then
  # push the smoke script to node1 at the same path, then launch rank1 there in the
  # background and rank0 here in the foreground; both use the c10d rendezvous.
  gcloud compute scp --zone="$ZONE" --project="$PROJECT" "$SMOKE" "${NODE1}:${SMOKE}" >/dev/null 2>&1 \
    || remote1 "cat > ${SMOKE}" < "$SMOKE" >/dev/null 2>&1 || true
  remote1 "${TORCHRUN} --nnodes=2 --node-rank=1 --nproc-per-node=8 --rdzv-backend=c10d --rdzv-endpoint=${RDZV} --rdzv-conf=join_timeout=120 ${SMOKE}" >/tmp/_spar_smoke_node1.log 2>&1 &
  R1=$!
  if timeout 120 "$TORCHRUN" --nnodes=2 --node-rank=0 --nproc-per-node=8 \
       --rdzv-backend=c10d --rdzv-endpoint="$RDZV" --rdzv-conf=join_timeout=120 "$SMOKE" 2>/tmp/_spar_smoke_node0.log; then
    row "nccl:2x8" PASS "16-rank all-reduce correct"
  else
    row "nccl:2x8" FAIL "see /tmp/_spar_smoke_node0.log (+ node1:/tmp/_spar_smoke_node1.log)"
  fi
  wait "$R1" 2>/dev/null || true
else
  # single-node fallback (V1 gate): standalone 8-GPU all-reduce
  if timeout 120 "$TORCHRUN" --standalone --nnodes=1 --nproc-per-node=8 "$SMOKE" >/tmp/_spar_smoke_node0.log 2>&1; then
    row "nccl:1x8" PASS "8-rank all-reduce correct (single-node)"
  else
    row "nccl:1x8" FAIL "see /tmp/_spar_smoke_node0.log"
  fi
fi

# --- 4. cross-node chrony offset --------------------------------------------
node_offset() {  # $1 = "local" | node name
  if [[ "$1" == "local" ]]; then
    chronyc tracking 2>/dev/null | awk '/Last offset/ {v=$4; if (v<0) v=-v; print v; exit}'
  else
    remote1 "chronyc tracking" 2>/dev/null | awk '/Last offset/ {v=$4; if (v<0) v=-v; print v; exit}'
  fi
}
OFF0="$(node_offset local)"; OFF0="${OFF0:-9.999}"
if [[ "$HAVE_NODE1" == "1" ]]; then
  OFF1="$(node_offset "$NODE1")"; OFF1="${OFF1:-9.999}"
  if awk "BEGIN{exit !(${OFF0} < 0.050 && ${OFF1} < 0.050)}"; then
    row "clock:<50ms" PASS "node0=${OFF0}s node1=${OFF1}s"
  else
    row "clock:<50ms" FAIL "node0=${OFF0}s node1=${OFF1}s (epoch join at risk)"
  fi
else
  if awk "BEGIN{exit !(${OFF0} < 0.050)}"; then
    row "clock:<50ms" PASS "node0=${OFF0}s"
  else
    row "clock:<50ms" FAIL "node0=${OFF0}s"
  fi
fi

# --- report ------------------------------------------------------------------
echo
echo "=================== verify_pair (session ${SESSION}) ==================="
printf "%-16s %-6s %s\n" "CHECK" "RESULT" "DETAIL"
printf "%-16s %-6s %s\n" "-----" "------" "------"
NFAIL=0
for r in "${ROWS[@]}"; do
  IFS='|' read -r n res det <<< "$r"
  printf "%-16s %-6s %s\n" "$n" "$res" "$det"
  [[ "$res" == "FAIL" ]] && NFAIL=$((NFAIL+1))
done
echo "======================================================================="
if [[ "$NFAIL" -eq 0 ]]; then
  echo "ALL CHECKS PASSED -- safe to start the collection queue."
  exit 0
else
  echo "$NFAIL CHECK(S) FAILED -- fix before collecting."
  exit 1
fi
