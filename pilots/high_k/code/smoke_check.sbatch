#!/usr/bin/env bash
# K10 smoke-test gate (CPU). Submit with --dependency=afterok:<smoke gen>:<smoke features>; exits non-zero if any
# check fails, so jobs depending on it with afterok + --kill-on-invalid-dep=yes cancel themselves.
#   sbatch --dependency=afterok:G:F --kill-on-invalid-dep=yes pilots/high_k/code/smoke_check.sbatch gemma2b G F
#SBATCH --job-name=mife-k10chk
#SBATCH --account=zives-cis-academic-curri
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
M="${1:?model}"; GJ="${2:?smoke generation job id}"; FJ="${3:?smoke features job id}"
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export MIFE_REPO="$(pwd)"
source scripts/betty/env.sh
mife_activate
PYTHONPATH=src:pilots/high_k/code python - "$M" "$GJ" "$FJ" <<'P'
import glob, json, sys
import numpy as np, pandas as pd
m, gj, fj = sys.argv[1:4]
G, F = f"pilots/high_k/data/generations/{m}", f"pilots/high_k/data/features/{m}"
s = json.load(open("pilots/high_k/data/inputs/build_summary.json"))["models"][m]
fails, r = [], {}
def need(ok, msg):
    if not ok: fails.append(msg)
gs, gd = sorted(glob.glob(f"{G}/shard_c00_j{gj}_*.parquet")), sorted(glob.glob(f"{G}/done_c00_j{gj}_*.parquet"))
need(gs and len(gs) == len(gd), f"generation shards/ledgers: {len(gs)} shards, {len(gd)} done ledgers")
need(glob.glob(f"{G}/run_summary_c00_j{gj}_*.json"), "no generation run_summary")
if gs:
    d = pd.concat(map(pd.read_parquet, gs))
    r.update(gen_records=len(d), gen_empty=int((d.n_output_tokens == 0).sum()),
             gen_blank_text=int((d.response_text.str.strip() == "").sum()),
             gen_hit_max_tokens_frac=float(d.hit_max_tokens.mean()), gen_median_tokens=float(d.n_output_tokens.median()),
             gen_finish_reasons=d.finish_reason.value_counts().to_dict(), gen_duplicates=int(d.duplicated(["prompt_id", "sample_index"]).sum()))
    need(180 <= len(d) <= 220, f"gen records {len(d)} not ~200")
    need(r["gen_empty"] == 0 and r["gen_blank_text"] == 0, "empty responses")
    need(r["gen_hit_max_tokens_frac"] <= 0.10, f"truncation rate {r['gen_hit_max_tokens_frac']:.3f} > 0.10")
    need(r["gen_duplicates"] == 0, "duplicate (prompt, sample) records")
    w = pd.read_parquet(f"pilots/high_k/data/inputs/worklist_{m}.parquet")
    seeds = dict(zip(w.prompt_id + "#" + w.sample_index.astype(str), w.seed))
    need(all(seeds.get(f"{p}#{i}") == b for p, i, b in zip(d.prompt_id, d.sample_index, d.batch_seed)), "seed mismatch")
fh, fa, fi = (sorted(glob.glob(f"{F}/{k}_c00_j{fj}_*")) for k in ("hidden", "attn", "index"))
need(fh and len(fh) == len(fa) == len(fi), f"feature files: {len(fh)} hidden, {len(fa)} attn, {len(fi)} index")
need(glob.glob(f"{F}/meta_c00_j{fj}_*.json"), "no features meta")
if fh:
    H = np.concatenate([np.load(f) for f in fh]); A = [np.load(f) for f in fa]
    MX = np.concatenate([a["max_bits"] for a in A]); MK = np.concatenate([a["mask"] for a in A])
    I = pd.concat(map(pd.read_parquet, fi))
    meta = json.load(open(sorted(glob.glob(f"{F}/meta_c00_j{fj}_*.json"))[-1]))
    exp_h = {"gemma2b": (5, 2304), "llama": (5, 4096), "gemma9b": (5, 3584)}[m]
    exp_a = {"gemma2b": (26, 8, 10), "llama": (32, 32, 10), "gemma9b": (42, 16, 10)}[m]
    r.update(hidden_shape=list(H.shape), attn_max_bits_shape=list(MX.shape), index_rows=len(I),
             hidden_rows=meta.get("hidden_rows"), per_prompt_s_median=meta.get("per_prompt_s_median"),
             hidden_nonzero_frac=float((H != 0).mean()), mask_rules_per_prompt=MK.sum(1).tolist()[:5])
    need(H.shape[1:] == exp_h, f"hidden shape {H.shape}")
    need(MX.shape[1:] == exp_a, f"attn shape {MX.shape}")
    need(180 <= H.shape[0] <= 220 and H.shape[0] == len(I) == MX.shape[0], f"feature n {H.shape[0]} / index {len(I)}")
    need(I.prompt_id.is_unique, "duplicate feature prompts")
    need(r["hidden_nonzero_frac"] > 0.5, "hidden states mostly zero")
    if m == "gemma2b":
        need(meta.get("hidden_rows") == [7, 10, 13, 20, 27], f"hidden_rows {meta.get('hidden_rows')}")
from features import check as fcheck
fc = fcheck(m)
r["features_check"] = fc
need(fc["duplicates"] == 0 and fc["committed"] == r.get("index_rows"), "features.py check: duplicates or count mismatch")
r["fails"] = fails
print("SMOKE_CHECK", json.dumps(r, default=str))
sys.exit(1 if fails else 0)
P
echo "SMOKE_CHECK PASSED"
