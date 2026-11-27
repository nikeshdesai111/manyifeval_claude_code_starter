#!/usr/bin/env bash
# Waits for the gemma2b chunk 0 redo (11276179) to leave the queue, then writes its completeness check to
# pilots/high_k/data/chunk0.log and a one-line result to alerts.log.
cd /Users/nickydesai/Documents/manyifeval_claude_code_starter || exit 1
B=desai7@login.betty.parcc.upenn.edu; D=pilots/high_k/code/mac_loops; DATA=pilots/high_k/data
while true; do
  s=$(ssh -o BatchMode=yes -o ConnectTimeout=20 $B 'bash -lc "squeue -h -j 11276179 -o %T"' 2>/dev/null)
  rc=$?
  [ $rc -eq 0 ] && ! echo "$s" | grep -qE "PENDING|RUNNING|CONFIG|COMPLETING" && break
  sleep 300
done
{ echo "=== $(date '+%F %T') chunk 0 redo finished"
  ssh -o BatchMode=yes $B 'bash -lc "sacct -j 11276179 -X -n -o JobID%14,State,ExitCode,Elapsed; tail -n 3 ~/mife/repo/logs/mife-k10_11276179_0.out | cut -c1-200"' 2>/dev/null | grep -vE "^\|"
  ssh -o BatchMode=yes $B 'bash -lc "cd ~/mife/repo && source scripts/betty/env.sh && mife_activate >/dev/null 2>&1 && python -"' < $D/chunk0_check.py 2>/dev/null | grep CHUNK0
} >> $DATA/chunk0.log
echo "$(date '+%F %T') CHUNK0 $(grep CHUNK0 $DATA/chunk0.log | tail -1)" >> $DATA/alerts.log
