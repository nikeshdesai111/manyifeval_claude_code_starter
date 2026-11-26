"""K10 early descriptive check, TRAIN only (pilots/high_k/PRESPEC.md "Gemma 2B early check"; gates nothing).

    PYTHONPATH=src python pilots/high_k/code/analyze.py gemma2b     # needs data/verification/<model>_train.parquet
All statistics within one model. Group bootstrap over the 50 S3 TRAIN groups, 2,000 draws, seed 20261127.
Writes pilots/high_k/tables/.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PILOT = Path(__file__).resolve().parents[1]
TAB = PILOT / "tables"
B, SEED = 2000, 20261127
LOW, HIGH = (1, 2, 3), (8, 9, 10)
MIN_PP = 3.0


def boot(V: pd.DataFrame, stat) -> tuple[float, float, float]:
    """stat(df) on the data and on 2,000 group-resampled copies -> (estimate, lo, hi)."""
    groups = np.array(sorted(V.group_id.unique()))
    parts = {g: d for g, d in V.groupby("group_id")}
    rng = np.random.default_rng(SEED)
    est = stat(V)
    draws = []
    for _ in range(B):
        pick = rng.choice(groups, len(groups), replace=True)
        draws.append(stat(pd.concat([parts[g] for g in pick], ignore_index=True)))
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return est, lo, hi


def delta(d: pd.DataFrame) -> float:
    return 100 * (d[d.k.isin(HIGH)].violated.mean() - d[d.k.isin(LOW)].violated.mean())


def slope(d: pd.DataFrame) -> float:
    r = d.groupby("k").violated.mean() * 100
    return float(np.polyfit(r.index.values, r.values, 1)[0])


def main(models: list[str]) -> None:
    tag = "_".join(models)
    TAB.mkdir(exist_ok=True)
    prim, byk, bytype, prompt_level, pos10 = [], [], [], [], []
    for m in models:
        V = pd.read_parquet(PILOT / "data" / "verification" / f"{m}_train.parquet")
        assert V.group_id.nunique() <= 50
        e, lo, hi = boot(V, delta)
        s, slo, shi = boot(V, slope)
        prim.append({"model": m, "delta_pp_high_minus_low": e, "lo": lo, "hi": hi, "slope_pp_per_rule": s,
                     "slope_lo": slo, "slope_hi": shi, "pass": bool(lo > 0 and e >= MIN_PP),
                     "n_checks": len(V), "n_responses": V.response_id.nunique()})
        for k, d in V.groupby("k"):
            pr = d.groupby("response_id").violated.max()
            byk.append({"model": m, "k": k, "rule_violation_rate": d.violated.mean(), "n_checks": len(d),
                        "prompt_all_followed": 1 - pr.mean(), "hit_max_tokens": d.groupby("response_id").hit_max_tokens.first().mean(),
                        "mean_output_tokens": d.groupby("response_id").n_output_tokens.first().mean(),
                        "frozen_rules_rate": d[d.origin == "frozen"].violated.mean(),
                        "new_rules_rate": d[d.origin == "new"].violated.mean()})
        for t, d in V.groupby("type_id"):
            n_groups = d.group_id.nunique()
            e, lo, hi = boot(d, delta) if n_groups >= 5 else (delta(d), np.nan, np.nan)
            bytype.append({"model": m, "type_id": t, "n_groups": n_groups, "low_rate": d[d.k.isin(LOW)].violated.mean(),
                           "high_rate": d[d.k.isin(HIGH)].violated.mean(), "delta_pp": e, "lo": lo, "hi": hi})
        d10 = V[V.k == 10]
        for p, d in d10.groupby("position"):
            pos10.append({"model": m, "position": p, "rule_violation_rate": d.violated.mean(), "n_checks": len(d)})
    pd.DataFrame(prim).to_csv(TAB / f"{tag}_step1_primary.csv", index=False)
    pd.DataFrame(byk).to_csv(TAB / f"{tag}_step1_by_k.csv", index=False)
    pd.DataFrame(bytype).to_csv(TAB / f"{tag}_step1_by_type.csv", index=False)
    pd.DataFrame(pos10).to_csv(TAB / f"{tag}_step1_k10_by_position.csv", index=False)
    print(pd.DataFrame(prim).round(3).to_string(index=False))
    print(pd.DataFrame(byk).round(3).to_string(index=False))


if __name__ == "__main__":
    import sys
    main(sys.argv[1:] or ["gemma2b"])
