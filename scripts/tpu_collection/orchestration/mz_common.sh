PROJECT=focus-surfer-494820-g0
RESOURCES=(
  "us-east5-a:spar-p6a-qr:spar-p6a:v6e-16/v2-alpha-tpuv6e"
  "us-east5-b:spar-p6b-qr:spar-p6b:v6e-16/v2-alpha-tpuv6e"
  "europe-west4-a:spar-p6e-qr:spar-p6e:v6e-16/v2-alpha-tpuv6e"
  "us-central1-a:spar-p5c-9760-qr:spar-p5c:v5litepod-16/v2-alpha-tpuv5-lite"
)
qr_state(){ gcloud compute tpus queued-resources describe "$2" --project=$PROJECT --zone="$1" --format='value(state.state)' 2>/dev/null; }
kill_one(){
  gcloud compute tpus tpu-vm delete "$3" --project=$PROJECT --zone="$1" --quiet >/dev/null 2>&1 || true
  gcloud compute tpus queued-resources delete "$2" --project=$PROJECT --zone="$1" --force --quiet >/dev/null 2>&1 || true
}
kill_all(){ for r in "${RESOURCES[@]}"; do IFS=: read -r z q n a <<<"$r"; echo "  deleting $q ($z)"; kill_one "$z" "$q" "$n"; done; }
