#!/usr/bin/env bash
# K10 progress watcher (Mac): one BatchMode ssh pass every 5 min via progress_mac.py, which rewrites
# pilots/high_k/data/progress.md. Alerts (rc 10) and the finish (rc 12) are appended to alerts.log;
# scheduled reports (rc 11) are appended to progress_history.md. Exits when all six K10 jobs have finished.
cd /Users/nickydesai/Documents/manyifeval_claude_code_starter || exit 1
D=pilots/high_k/code/mac_loops; DATA=pilots/high_k/data
while true; do
  out=$(.venv/bin/python $D/progress_mac.py "$PWD/$D"); rc=$?
  case $rc in
    10) { echo "$(date '+%F %T') ALERT"; echo "$out" | grep -E "ALERT|failed|FAILED|TIMEOUT|WAITING|Space guard|Progress pass"; } >> $DATA/alerts.log ;;
    11) { echo; echo "$out"; } >> $DATA/progress_history.md ;;
    12) echo "$(date '+%F %T') DONE all six K10 jobs have left the queue" >> $DATA/alerts.log; exit 0 ;;
  esac
  sleep 300
done
