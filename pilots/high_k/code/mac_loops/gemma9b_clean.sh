#!/usr/bin/env bash
# gemma9b step 1 after the end-of-turn cleaning (PRESPEC "Generation issues"): verify (TRAIN), analyze, matched, by_k_ci.
cd /Users/nickydesai/Documents/manyifeval_claude_code_starter || exit 1
L=pilots/high_k/data/gemma9b_clean.log; PY=.venv/bin/python
{ echo "=== $(date '+%F %T') verify (cleaned)"; S=$(date +%s)
  PYTHONPATH=src $PY pilots/high_k/code/verify.py gemma9b --workers 10 2>&1 | tail -2; echo "verify_s $(( $(date +%s)-S ))"
  echo "=== analyze"; S=$(date +%s); PYTHONPATH=src $PY pilots/high_k/code/analyze.py gemma9b 2>&1 | tail -16; echo "analyze_s $(( $(date +%s)-S ))"
  echo "=== matched"; PYTHONPATH=src $PY pilots/high_k/code/matched_check.py gemma9b 2>&1 | tail -30
  echo "=== by_k_ci"; $PY pilots/high_k/code/by_k_ci.py gemma9b
} >> $L 2>&1
date '+%F %T' > pilots/high_k/data/gemma9b_clean.done
echo "$(date '+%F %T') DONE gemma9b cleaned step 1 - see gemma9b_clean.log" >> pilots/high_k/data/alerts.log
