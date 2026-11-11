#!/bin/bash
# Upload campaign data from amodo to Google Drive (rclone remote gdrive:, scope drive.file), then verify by MD5.
# amodo's uplink is capped at ~1.2-1.5 MB/s, so data the analysis reads first goes first:
#   phase 1: everything except raw waveform segments (*.npy)   ~36 GB
#   phase 2: the raw waveform segments                         ~129 GB
# Re-runnable: rclone copy only sends what is missing or changed. Status: ~/campaign/upload_status.json
set -u
DEST="gdrive:H200-followup-raw-2026-10"
LOG=~/campaign/logs/upload.log
ST=~/campaign/upload_status.json
# filter file: rules apply in order, first match wins; the upload's own logs are never part of the data set
FILTER=~/campaign/upload_filter.txt
printf "%s\n" "- /logs/upload*" "+ /sessions/**" "+ /logs/**" "+ /exports/**" "+ /plans/**" "+ /FINAL_STATUS.json" "- *" > "$FILTER"
SEL=(--filter-from "$FILTER")
TUNE=(--transfers 8 --checkers 16 --drive-chunk-size 128M --fast-list --retries 5 --low-level-retries 20
      --log-file "$LOG" --log-level INFO --stats 60s --stats-one-line)
status() { echo "{\"state\": \"$1\", \"t\": \"$(date -u +%FT%TZ)\"$2}" > "$ST"; }

phase() {   # phase NAME [extra rclone filter args...]
  local name=$1; shift
  for try in $(seq 1 10); do
    status "uploading_$name" ", \"try\": $try"
    rclone copy ~/campaign "$DEST" "$@" "${SEL[@]}" "${TUNE[@]}" && return 0
    echo "$name try $try failed" >> "$LOG"
    sleep 300
  done
  status "${name}_failed" ""
  exit 1
}

cd ~/campaign || exit 1
phase phase1_no_waveforms --exclude "*.npy"   # given before the filter file, so it takes precedence
status phase1_done ""
phase phase2_waveforms
status verifying ""
rclone check ~/campaign "$DEST" "${SEL[@]}" --one-way --checkers 16 --fast-list \
  --log-file ~/campaign/logs/upload_check.log --log-level NOTICE \
  --differ ~/campaign/logs/upload_differ.txt --missing-on-dst ~/campaign/logs/upload_missing.txt
crc=$?
n=$(find ~/campaign/sessions ~/campaign/logs ~/campaign/exports ~/campaign/plans -type f ! -name 'upload*' | wc -l)
if [ $crc -eq 0 ]; then
  status verified_complete ", \"local_files\": $n"
else
  status verify_mismatch ", \"rc\": $crc, \"see\": \"logs/upload_missing.txt, logs/upload_differ.txt\""
fi
