#!/usr/bin/env bash
# Run on the MAC from the repository root. Copies the S3 expansion code and inputs to Betty's ~/mife/repo, then
# verifies checksums there. Copies only new/needed files; never touches Betty's data/generations/*/main.
#   bash scripts/betty/expansion_s3_stage.sh            # code + split + worklists/tokenized inputs
#   bash scripts/betty/expansion_s3_stage.sh weights llama|qwen   # pinned BF16 snapshot from the Mac HF cache
#   bash scripts/betty/expansion_s3_stage.sh fetch llama|qwen     # copy finished expansion shards back to the Mac
set -euo pipefail
B="${BETTY:-desai7@login.betty.parcc.upenn.edu}"
R="${BETTY_REPO:-mife/repo}"
REV_llama=meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
REV_qwen=Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
case "${1:-code}" in
code)
  ssh "$B" "mkdir -p $R/data/evaluation_expansion_s3 $R/results/split_s3 $R/logs"
  rsync -a src/mife/expansion_s3.py src/mife/splits.py "$B:$R/src/mife/"
  rsync -a configs/active_split.yaml "$B:$R/configs/"
  rsync -a results/split_s3/proposed_split.json "$B:$R/results/split_s3/"
  rsync -a scripts/betty/expansion_s3_*.sbatch "$B:$R/scripts/betty/"
  rsync -a tests/test_expansion_s3.py "$B:$R/tests/"
  rsync -a data/evaluation_expansion_s3/{manifest_summary.json,tokenized_llama.parquet,tokenized_qwen.parquet,worklist_llama.parquet,worklist_llama.json,worklist_qwen.parquet,worklist_qwen.json} "$B:$R/data/evaluation_expansion_s3/"
  (cd data/evaluation_expansion_s3 && shasum -a 256 tokenized_*.parquet worklist_*.parquet) > /tmp/exp_s3_sha.txt
  ssh "$B" "cd $R/data/evaluation_expansion_s3 && sha256sum -c -" < /tmp/exp_s3_sha.txt
  ;;
weights)
  M="${2:?model}"; V="REV_$M"; P="models--${!V}"
  H="${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"
  ssh "$B" "mkdir -p mife/hf/hub/$P"
  rsync -aL --partial "$H/$P/" "$B:mife/hf/hub/$P/"
  ssh "$B" "cd mife/hf/hub && grep '${P}/model-' ../../model_sha256.txt | sha256sum -c -"
  ;;
fetch)
  M="${2:?model}"
  mkdir -p "data/generations/$M/expansion_s3_20261123"
  rsync -a "$B:$R/data/generations/$M/expansion_s3_20261123/" "data/generations/$M/expansion_s3_20261123/"
  ssh "$B" "cd $R/data/generations/$M/expansion_s3_20261123 && md5sum *" > "/tmp/exp_s3_${M}_md5.txt"
  (cd "data/generations/$M/expansion_s3_20261123" && md5 -r * | awk '{print $1"  "$2}') | diff - "/tmp/exp_s3_${M}_md5.txt" && echo "fetch $M: all files identical"
  ;;
*) echo "usage: $0 [code | weights llama|qwen | fetch llama|qwen]"; exit 2 ;;
esac
