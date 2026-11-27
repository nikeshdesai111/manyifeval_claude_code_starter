#!/usr/bin/env bash
# K10 copy-back loop (Mac): stage.sh sync for each model every 5 min; stops on a checksum mismatch.
# Run detached:  nohup bash pilots/high_k/code/mac_loops/sync_loop.sh >> pilots/high_k/data/sync_loop.log 2>&1 &
cd /Users/nickydesai/Documents/manyifeval_claude_code_starter || exit 1
LOG=pilots/high_k/data/sync.log; ALERT=pilots/high_k/data/alerts.log; T=$(mktemp)
while true; do
  for m in gemma2b llama gemma9b; do
    S=$(date +%s); echo "=== $(date '+%F %T') sync $m" >> $LOG
    bash pilots/high_k/code/stage.sh sync $m > "$T" 2>&1; rc=$?
    grep -vE "^\||^-{5}|Storage Quotas|check your|storage usage|vast/parcc|^\s*$|parcc_" "$T" >> $LOG
    echo "exit $rc ($(( $(date +%s)-S ))s)" >> $LOG
    if grep -q "CHECKSUM MISMATCH" "$T"; then
      echo "$(date '+%F %T') ALERT sync $m CHECKSUM MISMATCH - copy-back loop stopped (see sync.log)" >> $ALERT; exit 1
    fi
    [ $rc -ne 0 ] && echo "$(date '+%F %T') ALERT sync $m failed rc=$rc: $(tail -n 2 "$T" | tr '\n' ' ')" >> $ALERT
  done
  sleep 90
done
