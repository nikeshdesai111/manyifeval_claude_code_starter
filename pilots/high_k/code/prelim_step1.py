"""PRELIMINARY K10 step-1 check for one model, TRAIN only, before the chunk 0 redo has landed.

Same estimators as analyze.py (pooled rule-check means; delta = k 8-10 minus k 1-3 in pp; OLS slope of the per-k
rate), but the group bootstrap is computed from per-group sufficient statistics (sum and count by k, type and origin),
which gives the same resampled statistics as re-concatenating rows, with a reduced number of draws.

    PYTHONPATH=src .venv/bin/python pilots/high_k/code/prelim_step1.py gemma2b --draws 200
Writes pilots/high_k/tables/<model>_PRELIM_step1_*.csv and prints them. Gates nothing; superseded by analyze.py.
"""
from __future__ import annotations

import argparse
import glob
import json

import numpy as np
import pandas as pd

from mife.config import ROOT

PILOT = ROOT / "pilots" / "high_k"
LOW, HIGH = (1, 2, 3), (8, 9, 10)
SEED = 20261127
ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--draws", type=int, default=200)
ap.add_argument("--verification", default=None)
a = ap.parse_args()
m = a.model
V = pd.read_parquet(a.verification or PILOT / f"data/verification/{m}_train.parquet",
                    columns=["response_id", "group_id", "k", "type_id", "origin", "violated", "hit_max_tokens",
                             "n_output_tokens", "position"])
TAB = PILOT / "tables"
TAB.mkdir(exist_ok=True)

# missing TRAIN records per k (expected from the worklist vs verified responses)
P = pd.read_parquet(PILOT / "data/inputs/prompts.parquet", columns=["prompt_id", "split", "k"])
P = P[P.split == "train"]
w = pd.read_parquet(PILOT / f"data/inputs/worklist_{m}.parquet", columns=["prompt_id", "sample_index"])
w = w[w.prompt_id.isin(set(P.prompt_id))].merge(P[["prompt_id", "k"]], on="prompt_id")
w["response_id"] = m + ":" + w.prompt_id + ":s" + w.sample_index.map(lambda s: f"{s:02d}")
got = set(V.response_id.unique())
w["have"] = w.response_id.isin(got)
miss = w.groupby("k").agg(expected=("have", "size"), verified=("have", "sum")).reset_index()
miss["missing"] = miss.expected - miss.verified
miss["missing_prompts"] = w[~w.have].groupby("k").prompt_id.nunique().reindex(miss.k).fillna(0).astype(int).values
miss["missing_frac"] = miss.missing / miss.expected

# by-k table (point estimates)
resp = V.groupby("response_id").agg(k=("k", "first"), any_v=("violated", "max"), hit=("hit_max_tokens", "first"),
                                    tok=("n_output_tokens", "first"))
byk = V.groupby("k").violated.agg(rule_violation_rate="mean", n_checks="size").reset_index()
byk["frozen_rules_rate"] = V[V.origin == "frozen"].groupby("k").violated.mean().reindex(byk.k).values
byk["new_rules_rate"] = V[V.origin == "new"].groupby("k").violated.mean().reindex(byk.k).values
r = resp.groupby("k").agg(prompt_all_followed=("any_v", lambda x: 1 - x.mean()), hit_max_tokens=("hit", "mean"),
                          mean_output_tokens=("tok", "mean"), n_responses=("any_v", "size")).reset_index()
byk = byk.merge(r, on="k").merge(miss[["k", "missing", "missing_prompts"]], on="k")

# sufficient statistics per group x k (and per type)
def suff(df):
    s = df.groupby(["group_id", "k"]).violated.agg(["sum", "count"]).unstack("k", fill_value=0)
    return s["sum"], s["count"]


groups = np.array(sorted(V.group_id.unique()))
rng = np.random.default_rng(SEED)
picks = [rng.choice(len(groups), len(groups), replace=True) for _ in range(a.draws)]


def stats_from(S, N, idx=None):
    s = S.values if idx is None else S.values[idx]
    n = N.values if idx is None else N.values[idx]
    ks = np.array(S.columns)
    st, nt = s.sum(0), n.sum(0)
    rate = np.where(nt > 0, st / np.maximum(nt, 1), np.nan)
    hi, lo = np.isin(ks, HIGH), np.isin(ks, LOW)
    delta = 100 * (st[hi].sum() / nt[hi].sum() - st[lo].sum() / nt[lo].sum())
    ok = ~np.isnan(rate)
    slope = float(np.polyfit(ks[ok], 100 * rate[ok], 1)[0])
    return delta, slope


def boot(df):
    S, N = suff(df)
    S, N = S.reindex(groups, fill_value=0), N.reindex(groups, fill_value=0)
    est = stats_from(S, N)
    dr = np.array([stats_from(S, N, p) for p in picks])
    return est, np.nanpercentile(dr, [2.5, 97.5], axis=0)


(e_d, e_s), ci = boot(V)
prim = pd.DataFrame([{"model": m, "status": "PRELIMINARY (chunk 0 redo pending)", "delta_pp_high_minus_low": e_d,
                      "lo": ci[0, 0], "hi": ci[1, 0], "slope_pp_per_rule": e_s, "slope_lo": ci[0, 1],
                      "slope_hi": ci[1, 1], "pass_rule_lo_gt0_and_ge3pp": bool(ci[0, 0] > 0 and e_d >= 3.0),
                      "bootstrap_draws": a.draws, "n_groups": len(groups), "n_checks": len(V),
                      "n_responses": V.response_id.nunique(), "missing_train_responses": int(miss.missing.sum())}])
rows = []
for t, d in V.groupby("type_id"):
    ng = d.group_id.nunique()
    (de, _), c = boot(d) if ng >= 5 else ((np.nan, np.nan), np.full((2, 2), np.nan))
    lo_r, hi_r = d[d.k.isin(LOW)].violated.mean(), d[d.k.isin(HIGH)].violated.mean()
    rows.append({"type_id": t, "n_groups": ng, "low_rate": lo_r, "high_rate": hi_r,
                 "delta_pp": 100 * (hi_r - lo_r), "lo": c[0, 0], "hi": c[1, 0], "n_checks": len(d)})
bytype = pd.DataFrame(rows).sort_values("delta_pp", ascending=False)
tag = f"{m}_PRELIM"
prim.to_csv(TAB / f"{tag}_step1_primary.csv", index=False)
byk.to_csv(TAB / f"{tag}_step1_by_k.csv", index=False)
bytype.to_csv(TAB / f"{tag}_step1_by_type.csv", index=False)
miss.to_csv(TAB / f"{tag}_step1_missing_by_k.csv", index=False)
pd.set_option("display.width", 200)
print(prim.round(4).T.to_string(header=False))
print(byk.round(4).to_string(index=False))
print(bytype.round(4).to_string(index=False))
print(miss.round(4).to_string(index=False))
