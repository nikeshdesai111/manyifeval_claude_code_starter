#!/usr/bin/env bash
# Run on the MAC from the repository root.
#   bash pilots/high_k/code/stage.sh push                    # code + inputs to Betty ~/mife/repo/pilots/high_k, sha256-checked
#   BETTY_HF=/project/path/hf/hub bash pilots/high_k/code/stage.sh push-weights gemma2b|llama|gemma9b
#                                                             # pinned snapshot from the Mac HF cache to Betty non-home storage
#   bash pilots/high_k/code/stage.sh fetch gemma2b|llama|gemma9b   # generations back to the Mac, md5-checked
set -euo pipefail
B="${BETTY:-desai7@login.betty.parcc.upenn.edu}"
R="${BETTY_REPO:-mife/repo}"
D=pilots/high_k
case "${1:-push}" in
push)
  ssh "$B" "mkdir -p $R/$D/code $R/$D/data/inputs $R/logs"
  rsync -a $D/PRESPEC.md "$B:$R/$D/"
  rsync -a $D/code/ "$B:$R/$D/code/"
  rsync -a $D/data/inputs/ "$B:$R/$D/data/inputs/"
  (cd $D/data/inputs && shasum -a 256 *) > /tmp/highk_sha.txt
  ssh "$B" "cd $R/$D/data/inputs && sha256sum -c -" < /tmp/highk_sha.txt
  ;;
push-weights)
  M="${2:?model}"; H="${BETTY_HF:?set BETTY_HF to a non-home directory on Betty}"
  read -r HFID REV < <(python3 -c "import json;s=json.load(open('$D/data/inputs/build_summary.json'))['models']['$M'];print(s['hf_id'],s['revision'])")
  python3 -c "from huggingface_hub import snapshot_download as s; print(s('$HFID', revision='$REV', allow_patterns=['*.json','*.safetensors','tokenizer*','LICENSE*','USE_POLICY.md']))"
  C="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"
  P="models--${HFID//\//--}"
  ssh "$B" "mkdir -p $H/$P/snapshots $H/$P/refs"
  rsync -aL "$C/$P/snapshots/$REV" "$B:$H/$P/snapshots/"
  ssh "$B" "echo -n $REV > $H/$P/refs/main"
  (cd "$C/$P/snapshots/$REV" && shasum -a 256 *.safetensors) > "/tmp/highk_w_$M.txt"
  ssh "$B" "cd $H/$P/snapshots/$REV && sha256sum -c -" < "/tmp/highk_w_$M.txt"
  ;;
fetch)
  M="${2:?model}"
  mkdir -p "$D/data/generations/$M"
  rsync -a "$B:$R/$D/data/generations/$M/" "$D/data/generations/$M/"
  ssh "$B" "cd $R/$D/data/generations/$M && md5sum *" > "/tmp/highk_${M}_md5.txt"
  (cd "$D/data/generations/$M" && md5 -r * | awk '{print $1"  "$2}') | diff - "/tmp/highk_${M}_md5.txt" && echo "fetch $M: all files identical"
  ;;
*) echo "usage: $0 [push | push-weights MODEL | fetch MODEL]"; exit 2 ;;
esac
