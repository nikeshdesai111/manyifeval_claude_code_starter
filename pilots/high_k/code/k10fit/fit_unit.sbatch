#!/usr/bin/env bash
# K10 head fits on Betty: run mife.k10paper.unitfit fit on one TRAIN-only pack (copied to node-local disk first).
#   sbatch [--partition/--gpus/--cpus-per-task/--time ...] fit_unit.sbatch <pack_name> <units_name> <dev: cuda|cpu> <request: path under ~/mife/k10fit> [--types i,j] [--folds a,b]
# Pack:  ~/mife/k10fit/in/<pack_name>/   (pushed by stage_fit.sh, MD5SUMS checked here before use)
# Units: ~/mife/k10fit/units/<units_name>/  (one npz per (fold, type); existing units are skipped, so a rerun resumes)
#SBATCH --job-name=k10fit
#SBATCH --account=zives-cis-academic-curri
#SBATCH --output=/vast/home/d/desai7/mife/k10fit/logs/%x_%j.out
set -euo pipefail
PACK="${1:?pack}"; UNITS="${2:?units}"; DEV="${3:?cuda|cpu}"; REQ="${4:?request}"; shift 4
K=$HOME/mife/k10fit
LOCAL="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/k10fit_${SLURM_JOB_ID:-$$}"
mkdir -p "$LOCAL"; trap 'rm -rf "$LOCAL"' EXIT
cd "$HOME/mife/repo"; source scripts/betty/env.sh; mife_activate >/dev/null 2>&1
t0=$(date +%s)
cp -r "$K/in/$PACK" "$LOCAL/pack"
(cd "$LOCAL/pack" && md5sum -c --quiet MD5SUMS)
echo "staged $(du -sh "$LOCAL/pack" | cut -f1) in $(( $(date +%s)-t0 ))s on $(hostname); dev=$DEV"
export MIFE_V2_HESS="$DEV"
N="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS=$N OPENBLAS_NUM_THREADS=$N MKL_NUM_THREADS=$N
[ "$DEV" = cuda ] && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd "$K"
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ] && [ -n "${K10_TYPES:-}" ]; then  # array: task i fits the i-th type of K10_TYPES
  IFS=, read -ra TL <<< "$K10_TYPES"; set -- --types "${TL[$SLURM_ARRAY_TASK_ID]}" "$@"
fi
PYTHONPATH="$K/pylib:$K/src" python -m mife.k10paper.unitfit fit "$LOCAL/pack" "$K/units/$UNITS" "$K/$REQ" "$@"
echo "DONE $PACK -> $UNITS in $(( $(date +%s)-t0 ))s"
