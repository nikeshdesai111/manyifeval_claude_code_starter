# Source me (login node and jobs). Every large path lives under one configurable root, $PROJ.
# Home quota is 50 GB; keep total usage under ~45 GB. Move PROJ to project storage when it exists.
export PROJ="${PROJ:-$HOME/mife}"
export MIFE_ENV="${MIFE_ENV:-$PROJ/envs/mife}"            # conda env prefix
export HF_HOME="$PROJ/hf"                                  # model weights + HF token
export XDG_CACHE_HOME="$PROJ/cache"
export PIP_CACHE_DIR="$PROJ/cache/pip"
export UV_CACHE_DIR="$PROJ/cache/uv"
export CONDA_PKGS_DIRS="$PROJ/cache/conda-pkgs"
export VLLM_CACHE_ROOT="$PROJ/cache/vllm"                  # torch.compile / CUDA-graph artefacts
export TRITON_CACHE_DIR="$PROJ/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$PROJ/cache/inductor"
export FLASHINFER_WORKSPACE_BASE="$PROJ/cache"
export CUDA_CACHE_PATH="$PROJ/cache/nv"
export TOKENIZERS_PARALLELISM=false
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1                # no telemetry from compute nodes
export PYTHONPATH="${MIFE_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/src"
mkdir -p "$PROJ" "$XDG_CACHE_HOME"

mife_activate() {  # Lmod and conda activation scripts are not `set -u` safe
  local had_u=0; [[ $- == *u* ]] && had_u=1; set +u
  module load anaconda3 >/dev/null 2>&1 || true
  # FlashInfer JIT-compiles Blackwell attention/sampling kernels on first use and needs nvcc + CUDA_HOME.
  # cuda/12.9.0 matches the torch/vLLM cu129 build (cuda/12.8.1 also exists on Betty).
  module load "${MIFE_CUDA_MODULE:-cuda/12.9.0}" >/dev/null 2>&1 || echo "WARNING: could not load ${MIFE_CUDA_MODULE:-cuda/12.9.0}"
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$MIFE_ENV"
  (( had_u )) && set -u; return 0
}
