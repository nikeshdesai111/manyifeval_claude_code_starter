#!/usr/bin/env bash
# One-time setup on a Betty LOGIN node: conda env with vLLM + Transformers for CUDA, then download both
# pinned model snapshots (safetensors only; Llama's 16 GB original/*.pth is skipped).
# Usage (from the repo root):  bash scripts/betty/setup.sh [env|models|all]   (default: all)
# Prerequisite for Llama (gated): `hf auth login` once, run inside the activated env; the token is
# stored under $HF_HOME (never in this repo).
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/betty/env.sh
STEP="${1:-all}"

# Pinned stack. vLLM 0.25.0 is the newest release with a CUDA 12.x (cu129) build: newer releases
# require torch >= 2.13 (CUDA 13 wheels), which needs an R580+ driver. cu129 runs on a CUDA 12.8 (R570)
# driver through CUDA minor-version compatibility. Override any of these via the environment.
VLLM_VERSION="${VLLM_VERSION:-0.25.0}"
TORCH_CUDA="${TORCH_CUDA:-cu129}"
TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.26.0}"
VLLM_WHEEL="${VLLM_WHEEL:-https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+${TORCH_CUDA}-cp38-abi3-manylinux_2_28_x86_64.whl}"
# Match the Mac's transformers so chat templates / tokenization / eager attention are the same code.
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-5.22.2}"
PIP="python -m pip install --no-cache-dir"

if [[ "$STEP" == env || "$STEP" == all ]]; then
  set +u; module load anaconda3; source "$(conda info --base)/etc/profile.d/conda.sh"; set -u
  [[ -d "$MIFE_ENV" ]] || conda create -y -p "$MIFE_ENV" python=3.12
  mife_activate
  $PIP --upgrade pip
  $PIP "torch==${TORCH_VERSION}+${TORCH_CUDA}" "torchvision==${TORCHVISION_VERSION}+${TORCH_CUDA}" \
       "torchaudio==${TORCH_VERSION}+${TORCH_CUDA}" --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"
  $PIP "$VLLM_WHEEL"
  # torchcodec (video decoding, unused for text) needs system FFmpeg libraries that Betty lacks; installed but
  # unloadable it crashes `import vllm`. vLLM falls back to a placeholder when it is absent, so remove it.
  python -m pip uninstall -y torchcodec || true
  $PIP "transformers==${TRANSFORMERS_VERSION}" || echo "WARNING: could not pin transformers ${TRANSFORMERS_VERSION}; keeping vLLM's"
  # Project dependencies at the Mac's versions where they exist (verifier stack must match).
  $PIP "pandas==3.1.1" "pyarrow==26.0.1" pyyaml "langdetect==1.0.9" "nltk==3.10.4" "absl-py==2.5.1" \
       "immutabledict==4.3.1" pytest
  python -m pip check || echo "WARNING: pip check reported conflicts (recorded in env/betty_pip_check.txt)"
  mkdir -p env
  python -m pip check > env/betty_pip_check.txt 2>&1 || true
  python -m pip freeze > env/betty_requirements.lock
  python - <<'EOF'
import torch, vllm, transformers
print("torch", torch.__version__, "cuda runtime", torch.version.cuda)
print("vllm", vllm.__version__, "transformers", transformers.__version__)
EOF
  conda clean -a -y >/dev/null
  rm -rf "$PIP_CACHE_DIR" "$UV_CACHE_DIR"
fi

if [[ "$STEP" == models || "$STEP" == all ]]; then
  mife_activate
  python - <<'EOF'
import yaml
from huggingface_hub import snapshot_download, whoami
from mife.generate_mlx import model_path
cfg = yaml.safe_load(open("configs/experiment.yaml"))
missing = []
for key in cfg["models"]:
    try:
        print(key, "already present:", model_path(key))
    except Exception:
        missing.append(key)
if not missing:
    raise SystemExit(0)  # e.g. snapshots copied from the Mac's HF cache; nothing to download
try:
    print("HF account:", whoami().get("name"))
except Exception as e:
    raise SystemExit(f"Not logged in to Hugging Face ({e}). Run `hf auth login` first (Llama-3.1 is gated).")
cfg["models"] = {k: v for k, v in cfg["models"].items() if k in missing}
allow = ["*.json", "*.safetensors", "tokenizer*", "merges.txt", "vocab.json", "LICENSE", "README.md", "USE_POLICY.md"]
for key, m in cfg["models"].items():
    p = snapshot_download(m["hf_id"], revision=m["revision"], allow_patterns=allow)
    print(key, "->", p)
from mife.generate_mlx import model_path  # verifies every safetensors shard in the index is present
for key in cfg["models"]:
    print(key, "ok:", model_path(key))
EOF
fi

echo "--- disk usage under $PROJ (limit ~45 GB) ---"
du -sh "$PROJ"/* 2>/dev/null || true
du -sh "$PROJ"
