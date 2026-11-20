"""Cross-backend consistency: a CUDA pilot run vs the Mac (MLX generation / MPS introspection) pilot.

Checks, all on the SAME 100 pre-selected training prompts:
  1. tokens: re-render and re-tokenize with this machine's tokenizer/chat template; compare input IDs,
     rule/task token spans and rendered-text hashes to the frozen data/tokenized/{model}.parquet; EOS ids.
  2. hidden states / attention: per layer, cosine similarity and max-abs difference between the Mac and
     CUDA final-prompt-token hidden states; absolute differences of rule-span attention sums/means.
  3. outcomes: verifier violation rates per k and per instruction type, with a prompt-level bootstrap CI
     for the CUDA - Mac difference (samples are stochastic on both sides, so only distributions compare).
Different kernels (and here, a different sampler order) mean representations and samples are NOT
identical; this module quantifies the differences and flags the material ones. It never claims identity.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import integrity
from .config import DATA_DIR
from .manifest import MANIFEST_DIR
from .storage import write_json

COS_FLAG = 0.995       # flag a layer whose minimum cosine similarity falls below this
ATTN_FLAG = 0.02       # flag if max |delta| of a rule-span attention SUM exceeds this (probability mass)


# ---------- 1. tokens ----------
def _norm(x):
    """Nested arrays/lists of ints -> nested Python lists; strings unchanged."""
    if isinstance(x, str):
        return x
    if hasattr(x, "__len__"):
        return [_norm(v) for v in x]
    return int(x)


def check_tokens(model_key: str, prompt_ids: list[str]) -> dict:
    from . import chat
    from .generate_mlx import eos_token_ids
    from .storage import sha256_text

    tokdf = pd.read_parquet(DATA_DIR / "tokenized" / f"{model_key}.parquet").set_index("prompt_id")
    prompts = pd.read_parquet(MANIFEST_DIR / "prompts.parquet").set_index("prompt_id")
    rules = pd.read_parquet(MANIFEST_DIR / "prompt_rules.parquet").sort_values(["prompt_id", "position"])
    bad = []
    for pid in prompt_ids:
        rr = rules[rules.prompt_id == pid]
        e = chat.encode(model_key, prompts.loc[pid, "prompt_text"], list(zip(rr.char_start, rr.char_end)),
                        tuple(prompts.loc[pid, "task_char_span"]))
        ref = tokdf.loc[pid]
        diffs = [n for n, x, y in [("input_ids", e["input_ids"], ref.input_ids),
                                   ("rule_token_spans", e["rule_token_spans"], ref.rule_token_spans),
                                   ("task_token_span", e["task_token_span"], ref.task_token_span),
                                   ("rendered_sha256", sha256_text(e["rendered"]), ref.rendered_sha256)]
                 if _norm(x) != _norm(y)]
        if diffs:
            bad.append({"prompt_id": pid, "differs": diffs})
    mac_eos = None
    sums = sorted((DATA_DIR / "generations" / model_key / "pilot").glob("run_summary_*.json"))
    if sums:
        mac_eos = json.loads(sums[-1].read_text())["model_info"]["eos_token_ids"]
    here_eos = eos_token_ids(model_key)
    tinfo = chat.template_info(model_key)
    return {"n_prompts": len(prompt_ids), "n_mismatched": len(bad), "mismatches": bad[:20],
            "identical_token_ids_and_spans": not bad, "eos_here": here_eos, "eos_mac_pilot": mac_eos,
            "eos_identical": mac_eos is None or sorted(mac_eos) == here_eos,
            "chat_template_sha256_here": tinfo["chat_template_sha256"],
            "transformers_version_here": __import__("transformers").__version__}


# ---------- 2. hidden states and attention ----------
def _load_features(model_key: str, run: str, prompt_ids: list[str]):
    from .introspect import bits_to_float32

    d = DATA_DIR / "features" / model_key / run
    idx = pd.read_parquet(d / "index.parquet")
    integrity.check_prompt(idx, str(d))
    integrity.check_rendered(model_key, idx, str(d))
    idx = idx.set_index("prompt_id").loc[prompt_ids]
    H, A = {}, {}
    hs = {s: np.load(d / f"hidden_{s:05d}.npy", mmap_mode="r") for s in idx["shard"].unique()}
    at = {s: np.load(d / f"attn_{s:05d}.npz") for s in idx["shard"].unique()}
    H = np.stack([bits_to_float32(np.asarray(hs[r.shard][r.row])) for r in idx.itertuples()])
    A = {k: np.stack([at[r.shard][k][r.row] for r in idx.itertuples()]) for k in ("rule_sum", "rule_mean", "task_sum")}
    meta = json.loads((d / "features_meta.json").read_text())
    return H, A, idx, meta


def compare_hidden(Ha: np.ndarray, Hb: np.ndarray) -> pd.DataFrame:
    """Ha, Hb: (n_prompts, n_rows, d) float32. Returns per-row (layer) summaries."""
    num = (Ha * Hb).sum(-1)
    den = np.linalg.norm(Ha, axis=-1) * np.linalg.norm(Hb, axis=-1)
    cos = num / np.maximum(den, 1e-30)
    mad = np.abs(Ha - Hb).max(-1)
    rel = np.linalg.norm(Ha - Hb, axis=-1) / np.maximum(np.linalg.norm(Ha, axis=-1), 1e-30)
    return pd.DataFrame({"row": np.arange(Ha.shape[1]), "cos_min": cos.min(0), "cos_median": np.median(cos, 0),
                         "max_abs_diff_max": mad.max(0), "max_abs_diff_median": np.median(mad, 0),
                         "rel_l2_median": np.median(rel, 0), "rel_l2_max": rel.max(0),
                         "ref_abs_max": np.abs(Ha).max((0, 2))})


def compare_attention(Aa: dict, Ab: dict) -> dict:
    out = {}
    for k in ("rule_sum", "rule_mean", "task_sum"):
        d = np.abs(Aa[k] - Ab[k])
        d = d[~np.isnan(d)]
        out[k] = {"max_abs_diff": float(d.max()), "p99_abs_diff": float(np.quantile(d, .99)),
                  "median_abs_diff": float(np.median(d))}
    out["nan_pattern_identical"] = bool(np.array_equal(np.isnan(Aa["rule_sum"]), np.isnan(Ab["rule_sum"])))
    return out


# ---------- 3. verifier outcomes ----------
def bootstrap_diff(a: pd.DataFrame, b: pd.DataFrame, by: str, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """a, b: prompt-rule label tables (prompt_id, `by`, v, n). Rates = sum v / sum n per group.
    CI for rate_b - rate_a by resampling PROMPTS (the independent unit) with replacement, jointly."""
    rng = np.random.default_rng(seed)
    pids = np.array(sorted(set(a.prompt_id) & set(b.prompt_id)))
    groups = sorted(set(a[by]) | set(b[by]))

    def agg(df):
        g = df.groupby(["prompt_id", by])[["v", "n"]].sum()
        return g

    ga, gb = agg(a), agg(b)

    def rates(g, sample):
        sub = g.loc[g.index.get_level_values(0).isin(set(sample))]
        # weight duplicated prompts
        cnt = pd.Series(sample).value_counts()
        w = sub.index.get_level_values(0).map(cnt).to_numpy()
        tot = (sub.mul(w, axis=0)).groupby(level=1).sum()
        return (tot["v"] / tot["n"]).reindex(groups)

    ra, rb = rates(ga, pids), rates(gb, pids)
    boots = []
    for _ in range(n_boot):
        s = rng.choice(pids, size=len(pids), replace=True)
        boots.append((rates(gb, s) - rates(ga, s)).to_numpy())
    boots = np.array(boots)
    lo, hi = np.nanquantile(boots, [.025, .975], axis=0)
    df = pd.DataFrame({by: groups, "rate_mac": ra.to_numpy(), "rate_cuda": rb.to_numpy(),
                       "diff": (rb - ra).to_numpy(), "ci_lo": lo, "ci_hi": hi})
    df["ci_excludes_0"] = (df.ci_lo > 0) | (df.ci_hi < 0)
    return df


def run(model_key: str, run: str, mac_run: str = "pilot") -> dict:
    pilot = list(pd.read_parquet(DATA_DIR / "pilot" / "pilot_prompts.parquet").prompt_id)
    flags = []
    rep = {"model_key": model_key, "cuda_run": run, "mac_run": mac_run, "n_prompts": len(pilot)}

    rep["tokens"] = check_tokens(model_key, pilot)
    if not rep["tokens"]["identical_token_ids_and_spans"]:
        flags.append("TOKENS: re-tokenization on this machine differs from the frozen Mac tokenization")
    if not rep["tokens"]["eos_identical"]:
        flags.append("TOKENS: EOS ids differ from the Mac pilot")

    fm, fc = DATA_DIR / "features" / model_key / mac_run, DATA_DIR / "features" / model_key / run
    if (fm / "index.parquet").exists() and (fc / "index.parquet").exists():
        Ha, Aa, ia, ma = _load_features(model_key, mac_run, pilot)
        Hb, Ab, ib, mb = _load_features(model_key, run, pilot)
        same_spans = (ia["rule_token_spans"].map(lambda x: [list(y) for y in x]).tolist() ==
                      ib["rule_token_spans"].map(lambda x: [list(y) for y in x]).tolist())
        hd = compare_hidden(Ha, Hb)
        rep["hidden_states"] = {"mac_backend": ma.get("backend"), "cuda_backend": mb.get("backend"),
                                "spans_identical_in_feature_index": bool(same_spans),
                                "convention_mac": ma.get("hidden_states_convention"),
                                "convention_cuda": mb.get("hidden_states_convention"),
                                "per_row": hd.to_dict("records"),
                                "worst_cos_min": float(hd.cos_min.min()),
                                "worst_row": int(hd.cos_min.idxmin())}
        rep["attention"] = compare_attention(Aa, Ab)
        low = hd[hd.cos_min < COS_FLAG]
        if len(low):
            flags.append(f"HIDDEN: {len(low)} hidden-state rows have min cosine < {COS_FLAG} "
                         f"(worst {hd.cos_min.min():.4f} at row {int(hd.cos_min.idxmin())})")
        if rep["attention"]["rule_sum"]["max_abs_diff"] > ATTN_FLAG:
            flags.append(f"ATTENTION: max |delta rule_sum| {rep['attention']['rule_sum']['max_abs_diff']:.3f} > {ATTN_FLAG}")
        if ma.get("hidden_states_convention") != mb.get("hidden_states_convention"):
            flags.append("HIDDEN: hidden_states[-1] convention differs between transformers versions")
    else:
        flags.append("HIDDEN: features missing for one side; comparison skipped")

    vm, vc = DATA_DIR / "verification" / model_key / mac_run, DATA_DIR / "verification" / model_key / run
    if (vm / "labels_prompt_rule.parquet").exists() and (vc / "labels_prompt_rule.parquet").exists():
        a = integrity.read_verification(model_key, mac_run, "labels_prompt_rule.parquet")
        b = integrity.read_verification(model_key, run, "labels_prompt_rule.parquet")
        k_of = pd.read_parquet(MANIFEST_DIR / "prompts.parquet").set_index("prompt_id")["k"]
        a["k"], b["k"] = a.prompt_id.map(k_of), b.prompt_id.map(k_of)
        a["all"], b["all"] = "all", "all"
        res = {by: bootstrap_diff(a, b, by).to_dict("records") for by in ("all", "k", "type_id")}
        rep["verifier"] = res
        for by in ("all", "k"):
            for r in res[by]:
                if r["ci_excludes_0"]:
                    flags.append(f"VERIFIER: violation rate differs at {by}={r[by]}: mac {r['rate_mac']:.3f} vs "
                                 f"cuda {r['rate_cuda']:.3f} (95% CI of diff [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}])")
    else:
        flags.append("VERIFIER: labels missing for one side; comparison skipped")

    rep["flags"] = flags
    rep["statement"] = ("Representations and samples are NOT claimed identical across backends: kernels differ "
                        "(MLX/MPS vs CUDA) and the samplers apply temperature and top-p/top-k in a different order.")
    write_json(rep, DATA_DIR / "consistency" / f"{model_key}_{run}_vs_{mac_run}.json")
    return rep
