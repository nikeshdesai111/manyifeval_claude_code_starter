#!/usr/bin/env bash
# Run on the MAC from the difficulty worktree root. Stages the difficulty calibration pilot to Betty and fetches results.
# Touches only ~/mife/repo/pilots/difficulty_pilot/ and ~/mife/repo/data/difficulty/pilot/round<R>/ on Betty.
#   bash pilots/high_k/code/difficulty_pilot_stage.sh push 1      # code + round inputs, sha256-checked
#   bash pilots/high_k/code/difficulty_pilot_stage.sh fetch 1     # generations back to the Mac, sha256-checked
set -euo pipefail
B="${BETTY:-desai7@login.betty.parcc.upenn.edu}"
R="${BETTY_REPO:-mife/repo}"
N="${2:?round}"
IN="data/difficulty/pilot/round${N}/inputs"
OUTB="data/difficulty/pilot/round${N}/betty"
SHA="$(mktemp)"
case "${1:?push|fetch}" in
push)
  [[ -f "$IN/build_summary.json" ]] || { echo "missing $IN/build_summary.json"; exit 2; }
  ssh "$B" "mkdir -p $R/pilots/difficulty_pilot/code $R/$IN $R/logs"
  rsync -a pilots/high_k/code/generate.py pilots/high_k/code/difficulty_pilot.sbatch "$B:$R/pilots/difficulty_pilot/code/"
  rsync -a "$IN/" "$B:$R/$IN/"
  (cd "$IN" && shasum -a 256 *) > "$SHA"
  ssh "$B" "cd $R/$IN && sha256sum -c -" < "$SHA"
  (cd pilots/high_k/code && shasum -a 256 generate.py difficulty_pilot.sbatch) > "$SHA"
  ssh "$B" "cd $R/pilots/difficulty_pilot/code && sha256sum -c -" < "$SHA"
  ;;
fetch)
  mkdir -p "$OUTB"
  rsync -a "$B:$R/$OUTB/" "$OUTB/"
  ssh "$B" "cd $R/$OUTB && find . -type f -name '*.parquet' -o -type f -name '*.json' | sort | xargs sha256sum" > "$SHA"
  (cd "$OUTB" && shasum -a 256 -c "$SHA")
  ;;
esac
rm -f "$SHA"
