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
sync)
  # Copy finished generation and feature shards back, verify md5 on both sides, and only then delete the heavy files
  # on Betty (shard_*, hidden_*, attn_*). Ledgers (done_*, index_*) and run summaries stay on Betty for resumption.
  # Only files older than 2 minutes are touched, so shards still being written are left alone.
  M="${2:?model}"
  for K in generations features; do
    L="$D/data/$K/$M"; mkdir -p "$L"
    ssh "$B" "mkdir -p $R/$L && cd $R/$L && find . -maxdepth 1 -type f -mmin +2 \( -name 'shard_*' -o -name 'hidden_*' -o -name 'attn_*' -o -name 'done_*' -o -name 'index_*' -o -name '*.json' \) -printf '%f\n'" > "/tmp/hk_${M}_${K}_list.txt"
    [[ -s "/tmp/hk_${M}_${K}_list.txt" ]] || continue
    ssh "$B" "cd $R/$L && xargs md5sum" < "/tmp/hk_${M}_${K}_list.txt" | sort -k2 > "/tmp/hk_${M}_${K}_remote.md5"
    # Never overwrite a Mac copy with different content: any listed file that already exists here must match Betty.
    while read -r F; do [[ -f "$L/$F" ]] && echo "$F"; done < "/tmp/hk_${M}_${K}_list.txt" > "/tmp/hk_${M}_${K}_exist.txt"
    if [[ -s "/tmp/hk_${M}_${K}_exist.txt" ]]; then
      (cd "$L" && xargs md5 -r < "/tmp/hk_${M}_${K}_exist.txt" | awk '{print $1"  "$2}' | sort -k2) > "/tmp/hk_${M}_${K}_exist.md5"
      if ! join -1 2 -2 2 "/tmp/hk_${M}_${K}_exist.md5" "/tmp/hk_${M}_${K}_remote.md5" | awk '$2!=$3{print; bad=1} END{exit bad}'; then
        echo "sync $M $K: OVERWRITE REFUSED - Mac copies above differ from Betty; CHECKSUM MISMATCH, nothing copied or deleted"; exit 1
      fi
    fi
    rsync -a --files-from="/tmp/hk_${M}_${K}_list.txt" "$B:$R/$L/" "$L/"
    (cd "$L" && xargs md5 -r < "/tmp/hk_${M}_${K}_list.txt" | awk '{print $1"  "$2}' | sort -k2) > "/tmp/hk_${M}_${K}_local.md5"
    if diff -q "/tmp/hk_${M}_${K}_local.md5" "/tmp/hk_${M}_${K}_remote.md5" >/dev/null; then
      grep -E '^(shard_|hidden_|attn_)' "/tmp/hk_${M}_${K}_list.txt" > "/tmp/hk_${M}_${K}_del.txt" || true
      [[ -s "/tmp/hk_${M}_${K}_del.txt" ]] && ssh "$B" "cd $R/$L && xargs rm -f" < "/tmp/hk_${M}_${K}_del.txt"
      echo "sync $M $K: $(wc -l < /tmp/hk_${M}_${K}_list.txt) files verified, $(wc -l < /tmp/hk_${M}_${K}_del.txt 2>/dev/null || echo 0) removed on Betty"
    else
      echo "sync $M $K: CHECKSUM MISMATCH, nothing deleted"; diff "/tmp/hk_${M}_${K}_local.md5" "/tmp/hk_${M}_${K}_remote.md5" | head; exit 1
    fi
  done
  ssh "$B" "du -sh $R/$D/data 2>/dev/null"
  ;;
*) echo "usage: $0 [push | push-weights MODEL | fetch MODEL | sync MODEL]"; exit 2 ;;
esac
