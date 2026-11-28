#!/usr/bin/env bash
# Wait until gemma9b generation (G3 11275185) has no pending/running tasks, then start the gemma9b Betty stager (8 GB cap).
cd /Users/nickydesai/Documents/manyifeval_claude_code_starter || exit 1
L=pilots/high_k/data/k10fit_g9.log
while true; do
  n=$(ssh -o BatchMode=yes -o ConnectTimeout=30 desai7@login.betty.parcc.upenn.edu 'bash -lc "squeue -h -j 11275185 -o %i | wc -l"' 2>/dev/null | tail -1)
  [ "$n" = "0" ] && break
  echo "$(date '+%F %T') waiter: G3 tasks left in queue: ${n:-ssh?}" >> $L
  sleep 300
done
echo "$(date '+%F %T') waiter: G3 finished; starting the gemma9b stager" >> $L
bash pilots/high_k/code/k10fit/stage_fit.sh gemma9b both 01:00:00 >> $L 2>&1
