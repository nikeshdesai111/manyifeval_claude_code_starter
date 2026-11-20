"""Evaluation 2: controlled comparisons (docx §11). Pairs are built from manifest metadata only (task, rule IDs,
positions, chains, COA rows), never from outcomes, so the matching is fixed before the test set is read.

A position (Objective 1): fix task, exact rule set (subset), k >= 2 and target rule; for each pair of target
   positions (j < j'), compare COA-balanced means over the orderings that put the target at j vs j'.
B count: fix task, chain and the chain-prefix order (COA row 0 is the chain order, and the k-prefix nests in the
   (k+1)-prefix); compare k -> k+1. Objective 1: each of the k shared targets (same position). Objective 2: the prompt pair.
C composition: fix task, k >= 2, target and target position; the two rule sets differ by exactly one companion, which
   occupies the same position, and every shared companion keeps its position (strict matches only).
D task: identical ordered configuration (config_id) on two different tasks of the group (3 task pairs per config).
E difficulty: TRAIN-only k = 1 violation rate per type -> terciles of the 15 types (5 each) -> Evaluation 1 by tercile.

Scores per configuration: mean |pred_delta - obs_delta| over pairs, and sign accuracy over RELIABLE pairs only.
Reliable (fixed): Objective 1 - the Agresti-Caffo 95% CI of the difference of the two violation proportions (counts
v out of n responses, aggregated over the orderings for A) excludes 0. Objective 2 - the Welch 95% CI of the difference
in mean per-response fraction violated (20 responses per prompt) excludes 0.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

Z = 1.959963984540054


def agresti_caffo_reliable(v1, n1, v2, n2) -> np.ndarray:
    p1, p2 = (v1 + 1) / (n1 + 2), (v2 + 1) / (n2 + 2)
    se = np.sqrt(p1 * (1 - p1) / (n1 + 2) + p2 * (1 - p2) / (n2 + 2))
    d = p2 - p1
    return (d - Z * se > 0) | (d + Z * se < 0)


def welch_reliable(m1, s1, n1, m2, s2, n2) -> np.ndarray:
    se = np.sqrt(s1 ** 2 / n1 + s2 ** 2 / n2)
    d = m2 - m1
    with np.errstate(divide="ignore", invalid="ignore"):
        from scipy.stats import t as tdist
        df = se ** 4 / ((s1 ** 2 / n1) ** 2 / (n1 - 1) + (s2 ** 2 / n2) ** 2 / (n2 - 1))
        q = np.where(np.isfinite(df), tdist.ppf(0.975, np.where(np.isfinite(df), df, 1)), np.inf)
    return np.where(se > 0, (d - q * se > 0) | (d + q * se < 0), d != 0)


# ------------------------------------------------------------------------------------------------- pair builders
def _rule_rows(P: pd.DataFrame, PR: pd.DataFrame) -> pd.DataFrame:
    return PR[["prompt_id", "position", "rule_id", "type_id"]].merge(
        P[["prompt_id", "group_id", "task_key", "subset_id", "chain_index", "k", "coa_row", "config_id"]], on="prompt_id")


def cells_A(P, PR) -> pd.DataFrame:
    """Position cells: (task, subset, target, position) -> prompt_ids; pairs are all position pairs per target."""
    R = _rule_rows(P, PR)
    R = R[R.k >= 2]
    c = R.groupby(["group_id", "task_key", "subset_id", "k", "rule_id", "type_id", "position"])["prompt_id"].agg(list).reset_index()
    out = []
    for key, g in c.groupby(["group_id", "task_key", "subset_id", "k", "rule_id", "type_id"]):
        g = g.set_index("position")
        for a, b in combinations(sorted(g.index), 2):
            out.append(dict(zip(["group_id", "task_key", "subset_id", "k", "rule_id", "type_id"], key),
                            pos_a=a, pos_b=b, prompts_a=g.loc[a, "prompt_id"], prompts_b=g.loc[b, "prompt_id"]))
    return pd.DataFrame(out)


def pairs_B(P, PR, objective: int) -> pd.DataFrame:
    o0 = P[P.coa_row == 0].set_index(["group_id", "task_key", "chain_index", "k"])["prompt_id"]
    rows = []
    for (g, t, c, k), pid in o0.items():
        if k < 5:
            rows.append({"group_id": g, "task_key": t, "chain_index": c, "k_a": k, "k_b": k + 1,
                         "prompt_a": pid, "prompt_b": o0[(g, t, c, k + 1)]})
    pairs = pd.DataFrame(rows)
    if objective == 2:
        return pairs
    R = _rule_rows(P, PR)[["prompt_id", "rule_id", "type_id", "position"]]
    m = pairs.merge(R.rename(columns={"prompt_id": "prompt_a"}), on="prompt_a")
    chk = m.merge(R.rename(columns={"prompt_id": "prompt_b", "position": "position_b"})[["prompt_b", "rule_id", "position_b"]],
                  on=["prompt_b", "rule_id"])
    assert (chk.position == chk.position_b).all() and len(chk) == len(m)
    return m


def pairs_C(P, PR, objective: int) -> pd.DataFrame:
    """Strict composition matches (see module docstring)."""
    R = _rule_rows(P, PR)
    order = R.sort_values(["prompt_id", "position"]).groupby("prompt_id")["rule_id"].agg(tuple)
    Q = P[P.k >= 2][["prompt_id", "group_id", "task_key", "k"]].assign(order=lambda d: d.prompt_id.map(order))
    rows = []
    for (g, t, k), grp in Q.groupby(["group_id", "task_key", "k"]):
        recs = list(grp[["prompt_id", "order"]].itertuples(index=False))
        for (pa, oa), (pb, ob) in combinations(recs, 2):
            diff = [i for i in range(k) if oa[i] != ob[i]]
            if len(diff) == 1 and len(set(oa) ^ set(ob)) == 2:  # one position differs, by a rule absent from the other
                j = diff[0]
                base = {"group_id": g, "task_key": t, "k": k, "prompt_a": pa, "prompt_b": pb, "swap_position": j + 1,
                        "swapped_out": oa[j], "swapped_in": ob[j]}
                if objective == 2:
                    rows.append(base)
                else:
                    for i in range(k):
                        if i != j:
                            rows.append({**base, "rule_id": oa[i], "position": i + 1})
    out = pd.DataFrame(rows)
    if objective == 1 and len(out):
        out = out.merge(R[["rule_id", "type_id"]].drop_duplicates(), on="rule_id")
    return out


def pairs_D(P, PR, objective: int) -> pd.DataFrame:
    rows = []
    for (g, cfg), grp in P.groupby(["group_id", "config_id"]):
        recs = sorted(grp[["task_key", "prompt_id"]].itertuples(index=False))
        for (ta, pa), (tb, pb) in combinations(recs, 2):
            rows.append({"group_id": g, "config_id": cfg, "task_a": ta, "task_b": tb, "prompt_a": pa, "prompt_b": pb})
    pairs = pd.DataFrame(rows)
    if objective == 2:
        return pairs
    R = _rule_rows(P, PR)[["prompt_id", "rule_id", "type_id", "position"]]
    return pairs.merge(R.rename(columns={"prompt_id": "prompt_a"}), on="prompt_a")


def type_terciles(train_labels: pd.DataFrame, P_train: pd.DataFrame) -> pd.DataFrame:
    """TRAIN-only k = 1 violation rate per type -> easy / medium / hard (5 types each, ties by type name)."""
    L = train_labels.merge(P_train[["prompt_id", "k"]], on="prompt_id")
    L = L[L.k == 1]
    r = (L.groupby("type_id").v.sum() / L.groupby("type_id").n.sum()).rename("train_k1_rate").reset_index()
    r = r.sort_values(["train_k1_rate", "type_id"]).reset_index(drop=True)
    r["tercile"] = np.repeat(["easy", "medium", "hard"], 5)
    return r


def pair_counts(P, PR) -> pd.DataFrame:
    """Metadata-only pair counts per analysis (and per group; the design is identical in every group)."""
    ng = P.group_id.nunique()
    rows = []
    a = cells_A(P, PR)
    rows.append({"analysis": "A position", "objective": 1, "pairs": len(a)})
    for o in (1, 2):
        for name, fn in (("B count", pairs_B), ("C composition (strict)", pairs_C), ("D task", pairs_D)):
            rows.append({"analysis": name, "objective": o, "pairs": len(fn(P, PR, o))})
    out = pd.DataFrame(rows)
    out["pairs_per_group"] = out["pairs"] / ng
    return out


# ------------------------------------------------------------------------------------------------------ scoring
def score_obj1(pairs: pd.DataFrame, lab: pd.DataFrame, pred: pd.Series, analysis: str) -> pd.DataFrame:
    """lab: prompt_id, rule_id, v, n; pred: Series indexed by (prompt_id, rule_id). One row per pair with
    obs/pred deltas, abs error and reliability."""
    L = lab.set_index(["prompt_id", "rule_id"])
    if analysis == "A":
        def agg(col_prompts, rule):
            idx = [(p, r) for ps, r in zip(col_prompts, rule) for p in ps]
            return idx
        out = []
        for side in ("a", "b"):
            idx = pd.MultiIndex.from_tuples([(p, r) for ps, r in zip(pairs[f"prompts_{side}"], pairs.rule_id) for p in ps])
            owner = np.repeat(np.arange(len(pairs)), [len(ps) for ps in pairs[f"prompts_{side}"]])
            v = pd.Series(L.v.reindex(idx).values).groupby(owner).sum().values
            n = pd.Series(L.n.reindex(idx).values).groupby(owner).sum().values
            pm = pd.Series(pred.reindex(idx).values).groupby(owner).mean().values
            out.append((v, n, pm))
        (va, na, pa), (vb, nb, pb) = out
    else:
        ia = pd.MultiIndex.from_arrays([pairs.prompt_a, pairs.rule_id])
        ib = pd.MultiIndex.from_arrays([pairs.prompt_b, pairs.rule_id])
        va, na, pa = L.v.reindex(ia).values, L.n.reindex(ia).values, pred.reindex(ia).values
        vb, nb, pb = L.v.reindex(ib).values, L.n.reindex(ib).values, pred.reindex(ib).values
    obs = vb / nb - va / na
    prd = pb - pa
    return pd.DataFrame({"group_id": pairs.group_id.values, "obs_delta": obs, "pred_delta": prd,
                         "abs_err": np.abs(prd - obs), "reliable": agresti_caffo_reliable(va, na, vb, nb),
                         "sign_correct": np.sign(prd) == np.sign(obs)})


def score_obj2(pairs: pd.DataFrame, resp: pd.DataFrame, pred: pd.Series) -> pd.DataFrame:
    """resp: per-prompt mean, sd and count of the per-response fraction violated; pred indexed by prompt_id."""
    R = resp.set_index("prompt_id")
    a, b = R.reindex(pairs.prompt_a), R.reindex(pairs.prompt_b)
    obs = b["mean"].values - a["mean"].values
    prd = pred.reindex(pairs.prompt_b).values - pred.reindex(pairs.prompt_a).values
    rel = welch_reliable(a["mean"].values, a["sd"].values, a["count"].values, b["mean"].values, b["sd"].values, b["count"].values)
    return pd.DataFrame({"group_id": pairs.group_id.values, "obs_delta": obs, "pred_delta": prd,
                         "abs_err": np.abs(prd - obs), "reliable": rel, "sign_correct": np.sign(prd) == np.sign(obs)})
