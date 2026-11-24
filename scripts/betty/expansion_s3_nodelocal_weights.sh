# Sourced by expansion_s3_{array,preflight}.sbatch when MIFE_NODELOCAL_WEIGHTS=1 (home quota cannot hold both
# generators' BF16 weights). Downloads the PINNED snapshot of model $M to node-local disk, verifies every safetensors
# shard against ~/mife/model_sha256.txt (the hashes of the snapshots used for run main), points HF_HUB_CACHE at it,
# then goes back offline. Removed when the job exits. Fails before any generation if anything does not match.
set -euo pipefail
LOCAL="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/mife_hf_${SLURM_JOB_ID:-$$}"
mkdir -p "$LOCAL/hub"
trap 'rm -rf "$LOCAL"' EXIT
echo "node-local weights: $LOCAL ($(df -h "$LOCAL" | tail -1))"
export HF_HUB_CACHE="$LOCAL/hub"
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python - "$M" <<'P'
import sys, yaml
from huggingface_hub import snapshot_download
m = yaml.safe_load(open("configs/experiment.yaml"))["models"][sys.argv[1]]
allow = ["*.json", "*.safetensors", "tokenizer*", "merges.txt", "vocab.json", "LICENSE", "README.md", "USE_POLICY.md"]
print(snapshot_download(m["hf_id"], revision=m["revision"], allow_patterns=allow), flush=True)
P
P=$(python -c "import yaml,sys; m=yaml.safe_load(open('configs/experiment.yaml'))['models']['$M']; print('models--'+m['hf_id'].replace('/','--')+'/snapshots/'+m['revision'])")
N=$(grep -c "$P/model-" "$PROJ/model_sha256.txt")
[ "$N" -gt 0 ] || { echo "no recorded hashes for $P"; exit 3; }
(cd "$HF_HUB_CACHE" && grep "$P/model-" "$PROJ/model_sha256.txt" | sha256sum -c -)
python -c "from mife.generate_mlx import model_path; print('weights ok:', model_path('$M'))"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
