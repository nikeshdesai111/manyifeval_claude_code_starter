"""PRELIMINARY S3 TRAIN, Objective 2 (whole-prompt expected fraction of rules violated, q).

Evaluation 1: RMSE, MAE, MSE, Pearson, Spearman and ECE of the V2 grouped-CV OOF Objective 2 predictions over the
36,900 TRAIN prompts (prompts weighted equally), with the seven fixed comparisons as paired contrasts and two references:
the training-fold mean (for each fold, the mean q of the other four folds' prompts) and the average of each predictor's
Objective 1 OOF predictions (results/v2_s3/predictions/<gen>/obj2/avg_obj1). Uncertainty: paired 50-pool bootstrap
(B = 2,000 here, because Spearman needs a re-rank per replicate; seed 20261203), percentile intervals.

Evaluation 2 (preliminary design choices, NOT approved in the plan): prompt-level matched pairs from the same metadata
manifests as Objective 1: position (every pair of COA orders of the same rule set and task), strict append (k -> k+1,
new rule included, so the denominator changes), strict one-companion replacement, and task (same configuration, other
task of the pool). Observed: signed mean change (append) and noise-adjusted RMS (others; the variance of q-hat is the
response-level variance of the per-response violated fraction / n, which keeps the within-response correlation between
rules). Predictive: paired Delta-MSE as for Objective 1, plus the zero-change reference. Weights: equal per stratum
(k or transition), then pool -> task -> context -> pair, as in eval2.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from . import matches
from .common import COMPS, GENS, OOF_DIR, PREDS, oof
from .paths import EVAL_DIR, N_POOLS
from .stats import Boot, estimate, group_table

T = EVAL_DIR / "tables"
B2 = 2000
SEED = 20261203


def data(gen: str) -> pd.DataFrame:
    from ..probing import data as pdata
    P = pdata.prompts("train")[["prompt_id", "group_id", "task_key", "config_id", "subset_key", "k", "ordered_rule_ids"]]
    L = pdata.labels_prompt(gen, "train")[["prompt_id", "q_hat", "n"]]
    R = pdata.verification(gen, "verification_responses.parquet", "train", columns=["prompt_id", "fraction_violated"])
    var = R.groupby("prompt_id").fraction_violated.agg(lambda s: s.var(ddof=1) / len(s)).rename("var_q")
    d = P.merge(L, on="prompt_id", validate="one_to_one").merge(var.reset_index(), on="prompt_id", validate="one_to_one")
    O = oof(gen, 2)
    d = d.merge(O, on="prompt_id", how="left", validate="one_to_one")
    assert len(d) == 36900 and d[PREDS].notna().all().all() and d.group_id.nunique() == N_POOLS
    import json
    from .. import splits
    fog = json.loads((splits.cv_dir() / "folds.json").read_text())["fold_of_group"]
    d["fold"] = d.group_id.map(fog)
    for p in PREDS + ["REF_structure"]:
        assert (d[f"fold_{p}"] == d.fold).all(), p
    # references
    fm = {f: d.q_hat[d.fold != f].mean() for f in d.fold.unique()}
    d["REF_trainmean"] = d.fold.map(fm)
    for p in PREDS:
        a = pd.read_parquet(OOF_DIR / gen / "obj2" / "avg_obj1" / f"{p}.parquet", columns=["prompt_id", "pred", "fold"])
        a = a.rename(columns={"pred": f"AVG1_{p}", "fold": "_f"})
        d = d.merge(a, on="prompt_id", how="left", validate="one_to_one")
        assert (d._f == d.fold).all() and d[f"AVG1_{p}"].notna().all()
        d = d.drop(columns="_f")
    return d


def _metrics(y, p, w):
    """Weighted metrics for one replicate (w = row multiplicities)."""
    sw = w.sum()
    e = p - y
    mse = (w * e ** 2).sum() / sw
    mae = (w * np.abs(e)).sum() / sw
    def wcorr(a, b):
        ma, mb = (w * a).sum() / sw, (w * b).sum() / sw
        return (w * (a - ma) * (b - mb)).sum() / np.sqrt((w * (a - ma) ** 2).sum() * (w * (b - mb) ** 2).sum())
    return {"mse": mse, "rmse": np.sqrt(mse), "mae": mae, "pearson": wcorr(p, y)}


def eval1(gen: str, d: pd.DataFrame):
    groups = sorted(d.group_id.unique())
    gi = d.group_id.map({g: i for i, g in enumerate(groups)}).values
    rng = np.random.default_rng(SEED)
    M = np.zeros((B2, len(groups)))
    idx = rng.integers(0, len(groups), size=(B2, len(groups)))
    np.add.at(M, (np.repeat(np.arange(B2), len(groups)), idx.ravel()), 1.0)
    y = d.q_hat.values
    cols = PREDS + ["REF_structure", "REF_trainmean"] + [f"AVG1_{p}" for p in PREDS]
    # ECE bins: deciles of each predictor's own OOF predictions (TRAIN analogue of the plan's TRAIN-fixed bins)
    res = {c: {m: np.empty(B2 + 1) for m in ("mse", "rmse", "mae", "pearson", "spearman", "ece")} for c in cols}
    reps = [np.ones(len(groups))] + list(M)
    pre = {}
    for c in cols + ["__y"]:
        p = y if c == "__y" else d[c].values
        u, inv = np.unique(p, return_inverse=True)
        edges = np.unique(np.quantile(p, np.linspace(0, 1, 11)[1:-1]))
        pre[c] = (p, inv, len(u), np.digitize(p, edges), len(edges) + 1)

    def wrank(c, w):  # weighted mid-ranks (ties averaged) under row multiplicities w
        _, inv, nu, _, _ = pre[c]
        wu = np.bincount(inv, weights=w, minlength=nu)
        cw = np.cumsum(wu)
        return (cw - wu / 2)[inv]

    for b, wg in enumerate(reps):
        w = wg[gi]
        sw = w.sum()
        ry = wrank("__y", w)
        for c in cols:
            p, _, _, bins, nb = pre[c]
            m = _metrics(y, p, w)
            rp = wrank(c, w)
            ma, mb = (w * rp).sum() / sw, (w * ry).sum() / sw
            m["spearman"] = (w * (rp - ma) * (ry - mb)).sum() / np.sqrt((w * (rp - ma) ** 2).sum() * (w * (ry - mb) ** 2).sum())
            sy = np.bincount(bins, weights=w * y, minlength=nb)
            sp = np.bincount(bins, weights=w * p, minlength=nb)
            m["ece"] = np.abs(sy - sp).sum() / sw
            for k, v in m.items():
                res[c][k][b] = v
        if b % 200 == 0:
            print(gen, "obj2 eval1 replicate", b, flush=True)
    rows = []
    for c in cols:
        r = {"generator": gen, "predictor": c}
        for k in ("rmse", "mse", "mae", "pearson", "spearman", "ece"):
            bt = res[c][k][1:]
            r.update({k: res[c][k][0], f"{k}_lo": np.quantile(bt, .025), f"{k}_hi": np.quantile(bt, .975)})
        rows.append(r)
    scores = pd.DataFrame(rows)
    crow = []
    for cid, c, k, lab, prim in COMPS + [("refA", "REF_trainmean", "H2", "train-mean -> H2 (reference)", False),
                                          ("refB", "AVG1_H2", "H2", "avg Obj1 H2 -> Obj2 H2 (reference)", False)]:
        r = {"generator": gen, "comparison": cid, "label": lab, "primary": prim, "comparator": c, "candidate": k}
        for met, sign, name in (("rmse", 1, "rmse_reduction"), ("mse", 1, "mse_reduction"), ("mae", 1, "mae_reduction"),
                                ("spearman", -1, "spearman_gain"), ("pearson", -1, "pearson_gain")):
            diff = sign * (res[c][met] - res[k][met])
            r.update({name: diff[0], f"{name}_lo": np.quantile(diff[1:], .025), f"{name}_hi": np.quantile(diff[1:], .975)})
        crow.append(r)
    # calibration curve points (point estimate) for the appendix
    cal = []
    for c in PREDS + ["REF_trainmean"]:
        p = d[c].values
        edges = np.unique(np.quantile(p, np.linspace(0, 1, 11)[1:-1]))
        bins = np.digitize(p, edges)
        for b_ in range(len(edges) + 1):
            m = bins == b_
            cal.append({"generator": gen, "predictor": c, "bin": b_, "n_prompts": int(m.sum()),
                        "mean_pred": float(p[m].mean()) if m.any() else np.nan,
                        "mean_obs": float(y[m].mean()) if m.any() else np.nan})
    return scores, pd.DataFrame(crow), pd.DataFrame(cal)


# --------------------------------------------------------------------------------------------- Evaluation 2
def pairs(d: pd.DataFrame) -> dict:
    """Prompt-level matched pairs (a, b) from metadata only."""
    out = {}
    rows = []
    for (g, t, sk), x in d[d.k >= 2].groupby(["group_id", "task_key", "subset_key"]):
        ids = sorted(x.prompt_id)
        k = int(x.k.iloc[0])
        ctx = f"{g}|{t}|{sk}"
        for a, b in combinations(ids, 2):
            rows.append((f"{a}~{b}", ctx, g, t, k, a, b))
    out["position"] = pd.DataFrame(rows, columns=["pair_id", "ctx", "group_id", "task_key", "k", "prompt_a", "prompt_b"])
    a = matches.load("append").drop_duplicates("pair_id")
    out["append"] = pd.DataFrame({"pair_id": a.pair_id, "ctx": a.pair_id, "group_id": a.group_id, "task_key": a.task_key,
                                  "k": a.transition, "prompt_a": a.prompt_before, "prompt_b": a.prompt_after})
    c = matches.load("companion_pairs")
    c = c.assign(pp=c.prompt_a + "~" + c.prompt_b).drop_duplicates("pp")
    out["companion"] = pd.DataFrame({"pair_id": c.pp, "ctx": c.pp, "group_id": c.group_id, "task_key": c.task_key,
                                     "k": c.k, "prompt_a": c.prompt_a, "prompt_b": c.prompt_b})
    t = matches.load("task_pairs")
    t = t.assign(pp=t.prompt_a + "~" + t.prompt_b).drop_duplicates("pp")
    out["task"] = pd.DataFrame({"pair_id": t.pp, "ctx": t.config_id, "group_id": t.group_id, "task_key": t.config_id,
                                "k": t.k, "prompt_a": t.prompt_a, "prompt_b": t.prompt_b})
    return out


def eval2(gen: str, d: pd.DataFrame, boot: Boot):
    x = d.set_index("prompt_id")
    rows, obs = [], []
    for fac, pr in pairs(d).items():
        a, b = x.loc[pr.prompt_a], x.loc[pr.prompt_b]
        pr = pr.copy()
        pr["dobs"] = b.q_hat.values - a.q_hat.values
        pr["V"] = b.var_q.values + a.var_q.values
        pr["dsq"] = pr.dobs ** 2
        pr["csq"] = pr.dsq - pr.V
        for p in PREDS:
            pr[f"se_{p}"] = (b[p].values - a[p].values - pr.dobs) ** 2
        pr["se_zero"] = pr.dobs ** 2
        for cid, c, k, _, _ in COMPS:
            pr[f"dmse_{cid}"] = pr[f"se_{c}"] - pr[f"se_{k}"]
        lv = ["task_key", "ctx", "pair_id"]
        vals = [f"dmse_{c[0]}" for c in COMPS] + [f"se_{p}" for p in PREDS] + ["se_zero", "dobs", "csq", "dsq", "V"]
        tabs = group_table(pr, vals, ["k"], lv)
        for v in vals:
            one = {v: tabs[v]}
            if v == "csq":
                r = estimate(one, boot, lambda m: m[v], post=lambda r: 100 * np.sqrt(np.maximum(r, 0)))
                q = "rms_pp_noise_adjusted"
            else:
                r = estimate(one, boot, lambda m: m[v])
                q = v
            rows.append({"generator": gen, "factor": fac, "stratum": "macro", "quantity": q, **r})
            if v in ("dobs", "csq") or v.startswith("dmse"):
                per = estimate(one, boot, lambda m: m[v], strata_mean=False, with_loo=False,
                               post=(lambda r: 100 * np.sqrt(np.maximum(r, 0))) if v == "csq" else None)
                for s, rr in zip(list(tabs[v].index), per):
                    rows.append({"generator": gen, "factor": fac, "stratum": str(s), "quantity": q, **rr})
        obs.append({"generator": gen, "factor": fac, "n_prompt_pairs": len(pr), "n_pools": pr.group_id.nunique(),
                    "strata": ";".join(f"{s}:{n}" for s, n in pr.groupby("k").size().items())})
    return pd.DataFrame(rows), pd.DataFrame(obs)


def main():
    T.mkdir(parents=True, exist_ok=True)
    acc = {}
    for gen in GENS:
        d = data(gen)
        boot = Boot(sorted(d.group_id.unique()))
        e2, sup = eval2(gen, d, boot)
        acc.setdefault("obj2_eval2", []).append(e2)
        acc.setdefault("obj2_eval2_support", []).append(sup)
        print(gen, "obj2 eval2 done", flush=True)
        s, c, cal = eval1(gen, d)
        acc.setdefault("obj2_eval1_scores", []).append(s)
        acc.setdefault("obj2_eval1_contrasts", []).append(c)
        acc.setdefault("obj2_eval1_calibration", []).append(cal)
        print(gen, "obj2 eval1 done", flush=True)
    for k, v in acc.items():
        pd.concat(v, ignore_index=True).to_csv(T / f"{k}.csv", index=False)


if __name__ == "__main__":
    main()
