"""TRAIN-only variance components for the evaluation-expansion audit (reads TRAIN labels only; changes nothing).

The precision scenarios in `evaluation_expansion_audit.py` depend on two unknowns: how much a factor's effect varies
between independent rule-pool groups (tau) and how strongly rule violations co-occur within one response (rho).
This script estimates both on the 57 TRAIN groups of the frozen 246-configuration design, using the train-locked
loaders in `mife.probing.data` (they raise TestSetLocked for any TEST prompt).

For each contrast and group g, D_g = sum_c w_c * p_hat_c (an average matched effect within the group), and the
generation-noise variance of D_g is estimated exactly from the responses: V_g = sum_prompt Var_r(sum_t w_pt x_rt) / n.
Then tau_eff^2 = Var_g(D_g) - mean_g(V_g) (method of moments, floored at 0). tau_eff includes any idiosyncratic
prompt-level deviation that matching does not remove, so it is the quantity that limits across-group precision.

Run on the machine that holds the frozen data (from the repository root):
    PYTHONPATH=src .venv/bin/python audits/evaluation_expansion_train_variance.py
Output: results/audits/evaluation_expansion/train_variance_components.csv
"""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "audits" / "evaluation_expansion"


def contrast_weights(P: pd.DataFrame) -> dict:
    """{contrast: {group: {(prompt_id, rule_id): w}}} on the current design (TRAIN prompts only)."""
    W = defaultdict(lambda: defaultdict(dict))
    for g, gp in P.groupby("group_id"):
        tasks = sorted(gp.task_key.unique())
        nt = len(tasks)
        for t_key, tp in gp.groupby("task_key"):
            pid = {tuple(o): p for o, p in zip(tp.ordered_rule_ids.map(tuple), tp.prompt_id)}
            orders = list(pid)
            for k in range(2, 6):
                by_sub = defaultdict(list)
                for o in orders:
                    if len(o) == k:
                        by_sub[frozenset(o)].append(o)
                cells = [(s, r) for s in by_sub for r in s]
                w = W[f"A_k{k}_last_minus_first"][g]
                for s, r in cells:
                    last = [o for o in by_sub[s] if o[-1] == r]; first = [o for o in by_sub[s] if o[0] == r]
                    for o in last:
                        w[(pid[o], r)] = w.get((pid[o], r), 0) + 1 / (len(last) * len(cells) * nt)
                    for o in first:
                        w[(pid[o], r)] = w.get((pid[o], r), 0) - 1 / (len(first) * len(cells) * nt)
            S = set(orders)
            for k in range(1, 5):
                pairs = [(X, Y) for Y in S if len(Y) == k + 1 for X in [Y[:k]] if X in S]
                if not pairs:
                    continue
                w = W[f"B1_{k}to{k + 1}_strict_append"][g]
                m = len(pairs) * k * nt
                for X, Y in pairs:
                    for r in X:
                        w[(pid[Y], r)] = w.get((pid[Y], r), 0) + 1 / m
                        w[(pid[X], r)] = w.get((pid[X], r), 0) - 1 / m
            # marginal composition at k = 5 (all six 5-subsets are present in the current design), one companion at a time
            k5 = [o for o in orders if len(o) == 5]
            rules = sorted({r for o in k5 for r in o})
            for c in rules:
                w = W["C_k5_marginal_single_companion (unit = group x companion)"][f"{g}|{c}"]
                cells = [(t, p) for t in rules if t != c for p in range(5)]
                for t, p in cells:
                    at = [o for o in k5 if o[p] == t]
                    pr = [o for o in at if c in o]; ab = [o for o in at if c not in o]
                    for o in pr:
                        w[(pid[o], t)] = w.get((pid[o], t), 0) + 1 / (len(pr) * len(cells) * nt)
                    for o in ab:
                        w[(pid[o], t)] = w.get((pid[o], t), 0) - 1 / (len(ab) * len(cells) * nt)
    return W


def estimate(W: dict, long: pd.DataFrame, n_resp: int) -> pd.DataFrame:
    rows = []
    x = long.set_index(["prompt_id", "rule_id"])
    for name, per_group in W.items():
        D, V = [], []
        for g, w in per_group.items():
            ws = pd.Series(w, name="w")
            ws.index = pd.MultiIndex.from_tuples(ws.index, names=["prompt_id", "rule_id"])
            j = x.join(ws, how="inner")
            j["z"] = j["violated"].astype(float) * j["w"]
            z = j.groupby(["prompt_id", "sample_index"])["z"].sum()
            per_prompt = z.groupby(level=0)
            D.append(float(per_prompt.mean().sum()))
            V.append(float((per_prompt.var(ddof=1) / n_resp).sum()))
        D, V = np.array(D), np.array(V)
        G = len(D)
        between = D.var(ddof=1)
        tau2 = max(between - V.mean(), 0.0)
        rows.append({"contrast": name, "units": G, "mean_effect_pp": round(100 * D.mean(), 3),
                     "t_ci95_lo_pp": round(100 * (D.mean() - stats_t(G) * D.std(ddof=1) / math.sqrt(G)), 3),
                     "t_ci95_hi_pp": round(100 * (D.mean() + stats_t(G) * D.std(ddof=1) / math.sqrt(G)), 3),
                     "between_group_sd_of_D_pp": round(100 * math.sqrt(between), 3),
                     "mean_within_group_noise_se_pp": round(100 * math.sqrt(V.mean()), 3),
                     "tau_eff_pp": round(100 * math.sqrt(tau2), 3),
                     "share_of_between_group_variance_from_generation_noise": round(V.mean() / between, 3) if between > 0 else float("nan")})
    return pd.DataFrame(rows)


def stats_t(G: int) -> float:
    from scipy.stats import t
    return float(t.ppf(0.975, G - 1))


def within_response_correlation(long: pd.DataFrame) -> float:
    """Pooled correlation of two rules' violation indicators within one response, after removing each prompt-rule mean."""
    d = long[["prompt_id", "sample_index", "rule_id", "violated"]].copy()
    d["x"] = d.violated.astype(float)
    d["r"] = d.x - d.groupby(["prompt_id", "rule_id"]).x.transform("mean")
    num = 0.0
    for _, grp in d.groupby("prompt_id"):
        M = grp.pivot(index="sample_index", columns="rule_id", values="r").values
        if M.shape[1] < 2:
            continue
        s = M.sum(1)
        num += float((s ** 2).sum() - (M ** 2).sum(0).sum())  # sum over t != t' of products
    pair_var = 0.0
    for _, grp in d.groupby("prompt_id"):
        M = grp.pivot(index="sample_index", columns="rule_id", values="r").values
        if M.shape[1] < 2:
            continue
        v = (M ** 2).sum(0)
        pair_var += float(np.sum(np.sqrt(np.outer(v, v))) - v.sum())
    return num / pair_var if pair_var > 0 else float("nan")


def main():
    from mife.probing import data  # train-locked loaders
    P = data.prompts("train")
    assert set(P.split) == {"train"}
    W = contrast_weights(P)
    out = []
    for model in data.MODELS:
        long = data.verification(model, "verification_long.parquet", "train",
                                 columns=["prompt_id", "sample_index", "rule_id", "violated"])
        assert set(long.prompt_id) <= set(P.prompt_id)
        n = int(long.groupby(["prompt_id", "rule_id"]).size().min())
        est = estimate(W, long, n)
        est.insert(0, "model", model)
        est["rho_within_response"] = round(within_response_correlation(long), 4)
        out.append(est)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(out).to_csv(OUT / "train_variance_components.csv", index=False)
    print(pd.concat(out).to_string())


if __name__ == "__main__":
    main()
