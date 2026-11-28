#!/usr/bin/env bash
# Watchdog for the Mac fit runners (k10fit_local, and the gemma9b fallback runner once it starts):
# alert if a runner exits without its DONE line, or if it is alive but has made no progress for 30 min.
cd /Users/nickydesai/Documents/manyifeval_claude_code_starter || exit 1
D=pilots/high_k/data; A=$D/alerts.log; idle_alerted=""
while true; do
  R=$(awk '/^k10fit_local /{print $2}' $D/loops.pid)
  if [ -n "$R" ] && ! ps -p "$R" >/dev/null 2>&1; then
    if ! grep -q "all models DONE" $D/k10fit.log; then
      echo "$(date '+%F %T') ALERT watchdog: Mac fit runner $R exited without 'all models DONE' (see k10fit.log)" >> $A
    fi
    grep -v '^k10fit_local ' $D/loops.pid > $D/loops.pid.tmp; mv $D/loops.pid.tmp $D/loops.pid
  fi
  if [ -n "$R" ] && ps -p "$R" >/dev/null 2>&1; then
    newest=$(stat -f %m $D/k10fit.log $D/k10fit_workers.log 2>/dev/null | sort -n | tail -1)
    age=$(( $(date +%s) - ${newest:-0} ))
    if [ $age -gt 1800 ] && [ "$idle_alerted" != "$newest" ]; then
      echo "$(date '+%F %T') ALERT watchdog: Mac fit runner $R alive but no progress for $((age/60)) min" >> $A
      idle_alerted=$newest
    fi
  fi
  sleep 300
done
