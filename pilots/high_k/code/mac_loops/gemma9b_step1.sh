#!/usr/bin/env bash
# Detached waiter: when every gemma9b TRAIN response is on the Mac (copied by stage.sh sync, md5-verified before the
# Betty copy is removed), run verify.py (TRAIN only), analyze.py (2,000 group draws) and matched_check.py for gemma9b.
# Log: pilots/high_k/data/gemma9b_step1.log; done marker: pilots/high_k/data/gemma9b_step1.done; one line to alerts.log.
cd /Users/nickydesai/Documents/manyifeval_claude_code_starter || exit 1
DATA=pilots/high_k/data; LOG=$DATA/gemma9b_step1.log; PY=.venv/bin/python
while true; do
  r=$($PY pilots/high_k/code/mac_loops/train_ready.py gemma9b 2>/dev/null)
  echo "$(date '+%F %T') $r" >> $LOG
  set -- $r
  [ "$2" = "$3" ] && [ "$4" = "0" ] && [ -n "$3" ] && break
  sleep 300
done
grep -q "MISMATCH" $DATA/sync.log && { echo "$(date '+%F %T') ALERT gemma9b step1 not run: sync.log shows a mismatch" >> $DATA/alerts.log; exit 1; }
{ echo "=== $(date '+%F %T') gemma9b TRAIN complete on the Mac; verify"
  S=$(date +%s); PYTHONPATH=src $PY pilots/high_k/code/verify.py gemma9b --workers 10 2>&1 | tail -3; echo "verify_s $(( $(date +%s)-S )) rc=${PIPESTATUS[0]}"
  echo "=== analyze"; S=$(date +%s); PYTHONPATH=src $PY pilots/high_k/code/analyze.py gemma9b 2>&1 | tail -20; echo "analyze_s $(( $(date +%s)-S ))"
  echo "=== matched"; S=$(date +%s); PYTHONPATH=src $PY pilots/high_k/code/matched_check.py gemma9b 2>&1 | tail -40; echo "matched_s $(( $(date +%s)-S ))"
} >> $LOG 2>&1
date '+%F %T' > $DATA/gemma9b_step1.done
echo "$(date '+%F %T') DONE gemma9b step1 (verify, analyze, matched) - see gemma9b_step1.log" >> $DATA/alerts.log
