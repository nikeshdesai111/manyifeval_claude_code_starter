"""Per-k 95% CIs for the step-1 by-k rates, TRAIN only: group bootstrap as in analyze.py (2,000 draws over the S3 TRAIN
groups, seed 20261127, whole groups resampled, pooled means).

    .venv/bin/python pilots/high_k/code/by_k_ci.py gemma9b
Writes pilots/high_k/tables/<model>_step1_by_k_ci.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PILOT = Path(__file__).resolve().parents[1]
B, SEED = 2000, 20261127


def run(m: str) -> pd.DataFrame:
    V = pd.read_parquet(PILOT / f"data/verification/{m}_train.parquet", columns=["response_id", "group_id", "k", "violated"])
    assert V.group_id.nunique() <= 50
    groups = np.array(sorted(V.group_id.unique()))
    gi = {g: i for i, g in enumerate(groups)}
    ks = sorted(V.k.unique())
    S = V.groupby(["group_id", "k"]).violated.agg(["sum", "count"]).unstack("k", fill_value=0)
    s, n = S["sum"].reindex(groups).values, S["count"].reindex(groups).values
    R = V.groupby("response_id").agg(g=("group_id", "first"), k=("k", "first"), v=("violated", "max"))
    R["ok"] = 1 - R.v
    T = R.groupby(["g", "k"]).ok.agg(["sum", "count"]).unstack("k", fill_value=0)
    ts, tn = T["sum"].reindex(groups).values, T["count"].reindex(groups).values
    rng = np.random.default_rng(SEED)
    rr, pr = [], []
    for _ in range(B):
        pick = np.array([gi[g] for g in rng.choice(groups, len(groups), replace=True)])
        rr.append(s[pick].sum(0) / n[pick].sum(0))
        pr.append(ts[pick].sum(0) / tn[pick].sum(0))
    rr, pr = np.array(rr), np.array(pr)
    out = pd.DataFrame({"model": m, "k": ks, "rule_violation_rate": s.sum(0) / n.sum(0),
                        "rule_violation_rate_ci95_lo": np.percentile(rr, 2.5, axis=0),
                        "rule_violation_rate_ci95_hi": np.percentile(rr, 97.5, axis=0),
                        "prompt_all_followed": ts.sum(0) / tn.sum(0),
                        "prompt_all_followed_ci95_lo": np.percentile(pr, 2.5, axis=0),
                        "prompt_all_followed_ci95_hi": np.percentile(pr, 97.5, axis=0),
                        "n_checks": n.sum(0), "n_responses": tn.sum(0), "n_groups": len(groups), "bootstrap_draws": B})
    out.to_csv(PILOT / f"tables/{m}_step1_by_k_ci.csv", index=False)
    return out


if __name__ == "__main__":
    for model in sys.argv[1:]:
        run(model)
        print((PILOT / f"tables/{model}_step1_by_k_ci.csv").read_text())
