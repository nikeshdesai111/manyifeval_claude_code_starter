#!/usr/bin/env bash
# gemma9b fallback: after the Mac runner (gemma2b -> llama) finishes, stop the Betty stager and its arrays, pull back
# any finished gemma9b units (md5-checked), then fit the rest of gemma9b on the Mac with the same planner/point cache.
cd /Users/nickydesai/Documents/manyifeval_claude_code_starter || exit 1
D=pilots/high_k/data; L=$D/k10fit_g9.log; A=$D/alerts.log
RUNNER=$(awk '/^k10fit_local /{print $2}' $D/loops.pid)
while ps -p "$RUNNER" >/dev/null 2>&1; do sleep 300; done
if ! grep -q "all models DONE" $D/k10fit.log; then
  echo "$(date '+%F %T') ALERT g9 fallback: Mac runner exited without 'all models DONE'; gemma9b Mac fallback NOT started" >> $A; exit 1
fi
echo "$(date '+%F %T') g9 fallback: llama done on the Mac; stopping the Betty stager and taking over gemma9b" >> $L
S=$(awk '/^k10fit_stager /{print $2}' $D/loops.pid); [ -n "$S" ] && kill "$S" 2>/dev/null
B=desai7@login.betty.parcc.upenn.edu
ssh -o BatchMode=yes $B 'bash -lc "squeue -u desai7 -h -n k10fit -o %A | sort -u | xargs -r scancel"' >> $L 2>&1
sleep 60
# pull every unit file Betty finished for any gemma9b request, md5-checked, into the Mac unit dirs
PYTHONPATH=src .venv/bin/python - >> $L 2>&1 <<'PY'
import sys; sys.path.insert(0, "pilots/high_k/code/k10fit")
import stager as S
dirs = [d for d in S.ssh(f"ls ~/{S.RK}/units", check=False).split() if d.startswith("gemma9b__")]
for u in dirs:
    S.pull_units("gemma9b", u, u)
    print("pulled", u)
S.ssh(f"rm -rf ~/{S.RK}/in/gemma9b__*", check=False)
print("Betty gemma9b packs removed")
PY
echo "$(date '+%F %T') g9 fallback: starting Mac fits for gemma9b" >> $L
PYTHONPATH=src exec .venv/bin/python pilots/high_k/code/k10fit/local_fit.py --models gemma9b
