"""Phase 1: descriptive results on TRAIN only (57 groups), 95% CIs from the group bootstrap (stats.py).

Violation rates are sum(v) / sum(n) over prompt-rule labels (n = 20 responses each), i.e. the share of
(response, included rule) checks that were violated. Primary labels keep truncated responses as they are.

Repetition loop (prespecified, both models): a response whose last 256 output tokens are periodic, i.e. for some
period p in 1..128 at least 90% of those positions i satisfy t[i] == t[i - p]. Only responses with >= 384 output
tokens are examined (shorter ones are counted as not looping). Loops are reported, never dropped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds

from ..config import DATA_DIR, load_config
from .. import integrity
from . import data, stats
from .data import MODELS, RESULTS

LOOP_WINDOW, LOOP_MAX_P, LOOP_MIN_LEN, LOOP_THRESH = 256, 128, 384, 0.90
EC = "change_case:english_capital"


def _cfg():
    c = load_config()["probing"]
    return c["bootstrap"]["descriptive_B"], c["seeds"]["bootstrap"]


def _save(df: pd.DataFrame, name: str) -> pd.DataFrame:
    (RESULTS / "tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "tables" / f"{name}.csv", index=False)
    return df


def loop_flags(model: str) -> pd.DataFrame:
    """Per TRAIN response: n_output_tokens, hit_max_tokens, loop flag and detected period."""
    ids = data.prompt_ids("train")
    out = []
    for shard in sorted((DATA_DIR / "generations" / model / data.RUN).glob("shard_*.parquet")):
        t = ds.dataset(shard, format="parquet").to_table(
            columns=["response_id", "prompt_id", "rendered_sha256", "n_output_tokens", "hit_max_tokens", "output_token_ids"],
            filter=ds.field("prompt_id").isin(ids))
        if t.num_rows == 0:
            continue
        meta = t.select(["response_id", "prompt_id", "rendered_sha256", "n_output_tokens", "hit_max_tokens"]).to_pandas()
        integrity.check_rendered(model, meta, f"{model} {shard.name}")
        col = t.column("output_token_ids").combine_chunks()
        offs = col.offsets.to_numpy()
        vals = col.values.to_numpy()
        lens = np.diff(offs)
        loop = np.zeros(len(lens), dtype=bool)
        period = np.zeros(len(lens), dtype=np.int16)
        cand = np.nonzero(lens >= LOOP_MIN_LEN)[0]
        for s in range(0, len(cand), 50000):
            c = cand[s:s + 50000]
            A = vals[offs[c + 1][:, None] - LOOP_MIN_LEN + np.arange(LOOP_MIN_LEN)]  # last 384 tokens
            tail = A[:, LOOP_MIN_LEN - LOOP_WINDOW:]
            best = np.zeros(len(c))
            bestp = np.zeros(len(c), dtype=np.int16)
            for p in range(1, LOOP_MAX_P + 1):
                m = (tail == A[:, LOOP_MIN_LEN - LOOP_WINDOW - p:LOOP_MIN_LEN - p]).mean(1)
                better = m > best + 1e-12
                best[better], bestp[better] = m[better], p
            hit = best >= LOOP_THRESH
            loop[c[hit]] = True
            period[c[hit]] = bestp[hit]
        meta["loop"], meta["loop_period"] = loop, period
        out.append(meta.drop(columns="rendered_sha256"))
    return pd.concat(out, ignore_index=True)


def run() -> dict:
    B, seed = _cfg()
    P = data.prompts("train")[["prompt_id", "group_id", "k", "task_family", "config_id"]]
    PR = data.prompt_rules("train")[["prompt_id", "position", "rule_id", "type_id", "category"]]
    tabs = {n: [] for n in ("by_k", "by_type", "by_cat", "by_pos", "by_fam", "type_k", "dist", "trunc_k", "trunc_type",
                            "ec", "loops", "sens_trunc", "sens_ec")}
    per_model = {}
    for m in MODELS:
        L = data.labels_prompt_rule(m, "train")[["prompt_id", "position", "rule_id", "type_id", "n", "v", "p_hat"]]
        L = L.merge(P, on="prompt_id").merge(PR[["prompt_id", "position", "category"]], on=["prompt_id", "position"])
        assert len(L) == len(PR)
        per_model[m] = L
        for key, by in (("by_k", ["k"]), ("by_type", ["type_id"]), ("by_cat", ["category"]), ("by_pos", ["k", "position"]),
                        ("by_fam", ["task_family"]), ("type_k", ["type_id", "k"])):
            r = stats.ratio_ci(L, "v", "n", by, B=B, seed=seed)
            tabs[key].append(r.assign(model=m))
        tabs["by_k"].append(stats.ratio_ci(L, "v", "n", [], B=B, seed=seed).assign(model=m, k="all"))
        # label distributions
        Q = data.labels_prompt(m, "train").merge(P, on="prompt_id")
        L["v0"], L["v20"], L["one"] = (L["v"] == 0).astype(float), (L["v"] == L["n"]).astype(float), 1.0
        Q["one"], Q["q0"] = 1.0, (Q["q_hat"] == 0).astype(float)
        rows = []
        for lab, d, num in (("share of prompt-rule rows with v = 0", L, "v0"), ("share of prompt-rule rows with v = 20", L, "v20"),
                            ("mean p_hat", L, "p_hat"), ("mean q_hat", Q, "q_hat"), ("share of prompts with q_hat = 0", Q, "q0")):
            r = stats.ratio_ci(d, num, "one", [], B=B, seed=seed)
            rows.append({"model": m, "statistic": lab, "est": r["est"][0], "lo": r["lo"][0], "hi": r["hi"][0]})
        for qq in (0.1, 0.25, 0.5, 0.75, 0.9):
            rows.append({"model": m, "statistic": f"q_hat quantile {qq}", "est": float(Q["q_hat"].quantile(qq)), "lo": np.nan, "hi": np.nan})
        tabs["dist"].append(pd.DataFrame(rows))
        per_model[m + "_Q"] = Q
        # responses: truncation, loops, english_capital diagnostics
        R = loop_flags(m).merge(P[["prompt_id", "group_id", "k"]], on="prompt_id")
        R["one"], R["trunc"], R["loop_f"] = 1.0, R["hit_max_tokens"].astype(float), R["loop"].astype(float)
        tk = stats.ratio_ci(R, "trunc", "one", ["k"], B=B, seed=seed)
        tk_all = stats.ratio_ci(R, "trunc", "one", [], B=B, seed=seed).assign(k="all")
        tabs["trunc_k"].append(pd.concat([tk, tk_all]).assign(model=m))
        RT = R[["response_id", "prompt_id", "group_id", "trunc", "one"]].merge(PR[["prompt_id", "type_id"]], on="prompt_id")
        tabs["trunc_type"].append(stats.ratio_ci(RT, "trunc", "one", ["type_id"], B=B, seed=seed).assign(model=m))
        lp = []
        for lab, d, num, den in (("loop share of all responses", R, "loop_f", "one"),
                                 ("loop share among truncated responses", R[R.trunc == 1], "loop_f", "one"),
                                 ("truncated share among looping responses", R[R.loop], "trunc", "one")):
            if len(d) == 0:
                lp.append({"model": m, "statistic": lab, "est": np.nan, "lo": np.nan, "hi": np.nan, "count": 0})
                continue
            r = stats.ratio_ci(d, num, den, [], B=B, seed=seed)
            lp.append({"model": m, "statistic": lab, "est": r["est"][0], "lo": r["lo"][0], "hi": r["hi"][0], "count": int(d[num].sum())})
        lp.append({"model": m, "statistic": "looping responses (count)", "est": float(R["loop"].sum()), "lo": np.nan, "hi": np.nan,
                   "count": int(R["loop"].sum())})
        tabs["loops"].append(pd.DataFrame(lp))
        # long table for sensitivity analyses
        V = data.verification(m, "verification_long.parquet", "train",
                              columns=["response_id", "position", "type_id", "violated"])
        V = V.merge(R[["response_id", "trunc", "group_id", "k"]], on="response_id")
        V["one"] = 1.0
        prim = stats.ratio_ci(V, "violated", "one", ["k"], B=B, seed=seed).assign(labels="primary (all responses)")
        excl = stats.ratio_ci(V[V.trunc == 0], "violated", "one", ["k"], B=B, seed=seed).assign(labels="excluding truncated responses")
        tabs["sens_trunc"].append(pd.concat([prim, excl]).assign(model=m))
        noec = stats.ratio_ci(V[V.type_id != EC], "violated", "one", ["k"], B=B, seed=seed).assign(labels="excluding english_capital")
        tabs["sens_ec"].append(pd.concat([prim, noec]).assign(model=m))
        # english_capital: violations that come only from the langdetect step
        D = data.verification(m, "verification_responses.parquet", "train", columns=["response_id", "diag_is_upper", "diag_langdetect"])
        E = V[V.type_id == EC].merge(D[["response_id", "diag_is_upper", "diag_langdetect"]], on="response_id")
        E["lang_only"] = (E["violated"] == 1) & E["diag_is_upper"] & (E["diag_langdetect"] != "en")
        E["viol"] = E["violated"].astype(float)
        E["lang_only_f"] = E["lang_only"].astype(float)
        a = stats.ratio_ci(E, "viol", "one", [], B=B, seed=seed)
        b = stats.ratio_ci(E[E.violated == 1], "lang_only_f", "one", [], B=B, seed=seed)
        E["viol_case_only"] = E["viol"] - E["lang_only_f"]
        c = stats.ratio_ci(E, "viol_case_only", "one", [], B=B, seed=seed)
        tabs["ec"].append(pd.DataFrame([
            {"model": m, "statistic": "english_capital checks (response x rule)", "est": float(len(E)), "lo": np.nan, "hi": np.nan},
            {"model": m, "statistic": "violation rate (official checker)", "est": a["est"][0], "lo": a["lo"][0], "hi": a["hi"][0]},
            {"model": m, "statistic": "share of violations caused only by langdetect", "est": b["est"][0], "lo": b["lo"][0], "hi": b["hi"][0]},
            {"model": m, "statistic": "violation rate if langdetect-only failures counted as followed (sensitivity)",
             "est": c["est"][0], "lo": c["lo"][0], "hi": c["hi"][0]}]))
    # ---------------------------------------------------------------------------------------- write tables
    cat = lambda k: pd.concat(tabs[k], ignore_index=True)
    first = lambda d, c: d[c + [x for x in d.columns if x not in c]]
    _save(first(cat("by_k"), ["model", "k"]), "desc_by_k")
    _save(first(cat("by_type"), ["model", "type_id"]), "desc_by_type")
    _save(first(cat("by_cat"), ["model", "category"]), "desc_by_category")
    _save(first(cat("by_pos"), ["model", "k", "position"]), "desc_by_position")
    _save(first(cat("by_fam"), ["model", "task_family"]), "desc_by_family")
    tk = first(cat("type_k"), ["model", "type_id", "k"])
    _save(tk, "desc_type_by_k_long")
    _save(tk.pivot_table(index=["model", "type_id"], columns="k", values="est").reset_index().rename(
        columns={i: f"k={i}" for i in range(1, 6)}), "desc_type_by_k")
    _save(cat("dist"), "desc_label_distribution")
    _save(first(cat("trunc_k"), ["model", "k"]), "desc_truncation")
    _save(first(cat("trunc_type"), ["model", "type_id"]), "desc_truncation_by_type")
    _save(cat("ec"), "desc_english_capital")
    _save(cat("loops"), "desc_loops")
    _save(first(cat("sens_trunc"), ["model", "labels", "k"]), "sens_truncation")
    _save(first(cat("sens_ec"), ["model", "labels", "k"]), "sens_english_capital")
    # cross-model
    a, b = per_model["llama"].sort_values(["prompt_id", "position"]), per_model["qwen"].sort_values(["prompt_id", "position"])
    assert (a["prompt_id"].values == b["prompt_id"].values).all()
    d = stats.paired_ratio_diff_ci(a, b, "v", "n", ["type_id"], B=B, seed=seed)
    ra = stats.ratio_ci(a, "v", "n", ["type_id"], B=B, seed=seed)[["type_id", "est"]].rename(columns={"est": "llama"})
    rb = stats.ratio_ci(b, "v", "n", ["type_id"], B=B, seed=seed)[["type_id", "est"]].rename(columns={"est": "qwen"})
    _save(ra.merge(rb, on="type_id").merge(d.rename(columns={"est": "llama_minus_qwen", "lo": "diff_lo", "hi": "diff_hi"}),
                                           on="type_id"), "desc_cross_model_type")
    qa, qb = per_model["llama_Q"][["prompt_id", "group_id", "q_hat"]], per_model["qwen_Q"][["prompt_id", "q_hat"]]
    Q = qa.merge(qb, on="prompt_id", suffixes=("_llama", "_qwen"))
    from scipy.stats import spearmanr
    est, lo, hi = stats.statistic_ci(Q, lambda x: spearmanr(x["q_hat_llama"], x["q_hat_qwen"]).statistic, B=B, seed=seed)
    _save(pd.DataFrame([{"statistic": "Spearman rho of q_hat (per prompt)", "est": est, "lo": lo, "hi": hi, "n_prompts": len(Q)}]),
          "desc_cross_model_corr")
    figures()
    return {"tables": sorted(p.name for p in (RESULTS / "tables").glob("desc_*.csv"))}


def figures() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    F = RESULTS / "figures"
    F.mkdir(parents=True, exist_ok=True)
    t = pd.read_csv(RESULTS / "tables" / "desc_by_k.csv")
    t = t[t.k != "all"].assign(k=lambda d: d.k.astype(int))
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for m, d in t.groupby("model"):
        ax.errorbar(d.k, d.est, yerr=[d.est - d.lo, d.hi - d.est], marker="o", capsize=3, label=m)
    ax.set_xlabel("k (rules in prompt)"); ax.set_ylabel("violation rate"); ax.set_title("TRAIN violation rate by k"); ax.legend()
    fig.tight_layout(); fig.savefig(F / "desc_by_k.png", dpi=150); plt.close(fig)
    t = pd.read_csv(RESULTS / "tables" / "desc_by_type.csv")
    fig, ax = plt.subplots(figsize=(7, 5))
    order = t[t.model == "llama"].sort_values("est").type_id.tolist()
    for j, (m, d) in enumerate(t.groupby("model")):
        d = d.set_index("type_id").loc[order]
        y = np.arange(len(order)) + (j - 0.5) * 0.35
        ax.errorbar(d.est, y, xerr=[d.est - d.lo, d.hi - d.est], fmt="o", capsize=2, label=m)
    ax.set_yticks(np.arange(len(order))); ax.set_yticklabels([o.split(":")[1] for o in order], fontsize=8)
    ax.set_xlabel("violation rate"); ax.set_title("TRAIN violation rate by rule type"); ax.legend()
    fig.tight_layout(); fig.savefig(F / "desc_by_type.png", dpi=150); plt.close(fig)
    t = pd.read_csv(RESULTS / "tables" / "desc_type_by_k_long.csv")
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    for ax, (m, d) in zip(axs, t.groupby("model")):
        pv = d.pivot(index="type_id", columns="k", values="est").loc[order]
        im = ax.imshow(pv.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(5)); ax.set_xticklabels(range(1, 6)); ax.set_title(m); ax.set_xlabel("k")
        ax.set_yticks(range(len(order))); ax.set_yticklabels([o.split(":")[1] for o in order], fontsize=8)
    fig.colorbar(im, ax=axs, label="violation rate"); fig.suptitle("TRAIN violation rate, type × k")
    fig.savefig(F / "desc_type_by_k.png", dpi=150, bbox_inches="tight"); plt.close(fig)
