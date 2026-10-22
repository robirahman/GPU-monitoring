#!/bin/bash
# Runs every session plan of the main campaign in order; safe to restart (session.py resumes).
#   campaign_driver.sh PLAN_DIR
# Stop cleanly after the current session: touch ~/campaign/STOP
set -u
PLANS=$1
STATUS=~/campaign/driver_status.json
cd ~/gpu-monitoring
PY=~/venvs/main/bin/python
for f in $($PY -c "import json,sys; print(' '.join(json.load(open('$PLANS/campaign_summary.json'))['order']))"); do
  sid=${f%.json}
  [ -f ~/campaign/sessions/$sid/SESSION_COMPLETE ] && continue
  if [ -f ~/campaign/STOP ]; then echo "{\"state\": \"stopped_by_file\", \"next\": \"$sid\", \"t\": \"$(date -u +%FT%TZ)\"}" > $STATUS; exit 0; fi
  echo "{\"state\": \"running\", \"session\": \"$sid\", \"t\": \"$(date -u +%FT%TZ)\"}" > $STATUS
  for try in 1 2; do   # an abnormal session exit is retried once; completed runs are skipped on resume
    $PY -m h200_campaign.session $PLANS/$f >> ~/campaign/logs/$sid.log 2>&1
    rc=$?
    if [ $rc -eq 5 ]; then echo "{\"state\": \"stopped_disk\", \"session\": \"$sid\", \"t\": \"$(date -u +%FT%TZ)\"}" > $STATUS; exit 0; fi
    if [ $rc -eq 0 ]; then date -u +%FT%TZ > ~/campaign/sessions/$sid/SESSION_COMPLETE; break; fi
    echo "session $sid exited rc=$rc (try $try)" >> ~/campaign/logs/driver.log
  done
done
echo "{\"state\": \"campaign_done\", \"t\": \"$(date -u +%FT%TZ)\"}" > $STATUS
