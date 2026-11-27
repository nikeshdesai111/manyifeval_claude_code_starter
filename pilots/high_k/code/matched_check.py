"""Matched within-rule check, TRAIN only: is the SAME rule violated more when more other rules are present?

    PYTHONPATH=src .venv/bin/python pilots/high_k/code/matched_check.py gemma2b llama
Reads data/verification/<model>_train.parquet (the analyze.py input; S3 TRAIN groups only) and the prompts table for
chain structure. Writes pilots/high_k/tables/<model>_matched_{cells,summary,by_type,position1,balance}.csv.

(A) cell = (group, task, exact rule instance). Low = violation rate over responses where the rule is present at
    k in 1-3; high = the same at k in 8-10. Cells with both. Mean over cells of high - low (pp), group bootstrap.
(B) Chain prompts: chain c's head rule (its first chain element) is the whole prompt at k=1. Compare its violation
    when it sits at position 1 at k=1 vs k=10 (full pool), and at k=3 vs k=8 (chain c, COA rows with the head at
    position 1), matched by (group, task, chain).
(C) Per instruction type, the (A) difference.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from mife.config import ROOT

PILOT = ROOT / "pilots" / "high_k"
TAB = PILOT / "tables"
B, SEED = 2000, 20261127
LOW, HIGH = (1, 2, 3), (8, 9, 10)


def boot_mean(df: pd.DataFrame, col: str) -> tuple[float, float, float, int]:
    """Mean of df[col] over rows (cells) with a bootstrap over groups (2,000 draws) from per-group sums/counts."""
    g = df.groupby("group_id")[col].agg(["sum", "count"])
    s, n = g["sum"].values, g["count"].values
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(g), size=(B, len(g)))
    draws = s[idx].sum(1) / n[idx].sum(1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(s.sum() / n.sum()), float(lo), float(hi), int(n.sum())


def run(m: str) -> None:
    V = pd.read_parquet(PILOT / f"data/verification/{m}_train.parquet",
                        columns=["prompt_id", "group_id", "task_key", "k", "position", "rule_id", "type_id", "origin",
                                 "violated"])
    assert V.group_id.nunique() <= 50
    P = pd.read_parquet(PILOT / "data/inputs/prompts.parquet",
                        columns=["prompt_id", "split", "chain_index", "coa_row", "is_original", "ordered_rule_ids"])
    P = P[P.split == "train"]
    assert set(V.prompt_id) <= set(P.prompt_id)

    # (A) matched cells
    V["band"] = np.where(V.k.isin(LOW), "low", np.where(V.k.isin(HIGH), "high", "mid"))
    c = (V[V.band != "mid"].groupby(["group_id", "task_key", "rule_id", "type_id", "origin", "band"]).violated
         .agg(["mean", "size"]).unstack("band"))
    c = c.dropna(subset=[("mean", "low"), ("mean", "high")])
    cells = pd.DataFrame({"low": c[("mean", "low")], "high": c[("mean", "high")],
                          "n_low": c[("size", "low")], "n_high": c[("size", "high")]}).reset_index()
    cells["diff_pp"] = 100 * (cells.high - cells.low)
    cells.to_csv(TAB / f"{m}_matched_cells.csv", index=False)
    rows = []
    for name, d in (("all", cells), ("frozen", cells[cells.origin == "frozen"]), ("new", cells[cells.origin == "new"])):
        e, lo, hi, n = boot_mean(d, "diff_pp")
        rows.append({"model": m, "rules": name, "mean_diff_pp": e, "lo": lo, "hi": hi, "n_cells": n,
                     "pct_cells_high_gt_low": 100 * (d.diff_pp > 0).mean(), "pct_cells_equal": 100 * (d.diff_pp == 0).mean(),
                     "mean_low": 100 * d.low.mean(), "mean_high": 100 * d.high.mean()})
    summ = pd.DataFrame(rows)
    summ.to_csv(TAB / f"{m}_matched_summary.csv", index=False)

    # (C) per type
    rows = []
    for t, d in cells.groupby("type_id"):
        e, lo, hi, n = boot_mean(d, "diff_pp") if d.group_id.nunique() >= 5 else (d.diff_pp.mean(), np.nan, np.nan, len(d))
        rows.append({"model": m, "type_id": t, "n_groups": d.group_id.nunique(), "n_cells": n, "mean_diff_pp": e,
                     "lo": lo, "hi": hi, "pct_cells_high_gt_low": 100 * (d.diff_pp > 0).mean(),
                     "mean_low": 100 * d.low.mean(), "mean_high": 100 * d.high.mean()})
    bytype = pd.DataFrame(rows).sort_values("mean_diff_pp", ascending=False)
    bytype.to_csv(TAB / f"{m}_matched_by_type.csv", index=False)

    # (B) position-1 head rule, chain prompts
    heads = (P[(P.chain_index > 0) & (P.ordered_rule_ids.map(len) == 1)]
             .assign(head_rule=lambda d: d.ordered_rule_ids.str[0])[["prompt_id", "chain_index", "head_rule"]])
    head_of = {(pid.split("-")[1], c): h for pid, c, h in zip(heads.prompt_id, heads.chain_index, heads.head_rule)}
    p1 = V[V.position == 1].merge(P[["prompt_id", "chain_index"]], on="prompt_id")
    out = []
    for (ka, kb) in ((1, 10), (3, 8)):
        a = p1[(p1.k == ka) & (p1.chain_index > 0)].copy()
        a["chain"] = a.chain_index
        a = a[[r == head_of.get((g, c)) for g, c, r in zip(a.group_id, a.chain, a.rule_id)]]
        b = p1[p1.k == kb].copy()
        if kb == 10:  # the full pool is chain 0; match by the head rule sitting at position 1
            inv = {(g, h): c for (g, c), h in head_of.items()}
            b["chain"] = [inv.get((g, r)) for g, r in zip(b.group_id, b.rule_id)]
            b = b.dropna(subset=["chain"])
        else:
            b["chain"] = b.chain_index
            b = b[[r == head_of.get((g, c)) for g, c, r in zip(b.group_id, b.chain, b.rule_id)]]
        key = ["group_id", "task_key", "chain", "rule_id", "type_id"]
        ra = a.groupby(key).violated.agg(["mean", "size"]).rename(columns={"mean": "rate_a", "size": "n_a"})
        rb = b.groupby(key).violated.agg(["mean", "size"]).rename(columns={"mean": "rate_b", "size": "n_b"})
        j = ra.join(rb, how="inner").reset_index()
        j["diff_pp"] = 100 * (j.rate_b - j.rate_a)
        e, lo, hi, n = boot_mean(j, "diff_pp")
        out.append({"model": m, "comparison": f"k={kb} minus k={ka}", "mean_diff_pp": e, "lo": lo, "hi": hi,
                    "n_cells": n, "responses_low_k": int(j.n_a.sum()), "responses_high_k": int(j.n_b.sum()),
                    "mean_rate_low_k": 100 * j.rate_a.mean(), "mean_rate_high_k": 100 * j.rate_b.mean(),
                    "pct_cells_high_gt_low": 100 * (j.diff_pp > 0).mean()})
    pos1 = pd.DataFrame(out)
    pos1.to_csv(TAB / f"{m}_matched_position1.csv", index=False)

    # composition balance: prompts per (group, task, rule, k), each rule within a group
    cnt = V.groupby(["group_id", "task_key", "k", "rule_id"]).prompt_id.nunique().reset_index(name="n_prompts")
    bal = cnt.groupby(["group_id", "task_key", "k"]).n_prompts.agg(["min", "max"]).reset_index()
    balance = pd.DataFrame([{"model": m, "group_task_k_cells": len(bal),
                             "cells_where_every_rule_has_equal_prompts": int((bal["min"] == bal["max"]).sum()),
                             "rules_present_per_group_task_k_min": int(cnt.groupby(["group_id", "task_key", "k"]).size().min()),
                             "rules_present_per_group_task_k_max": int(cnt.groupby(["group_id", "task_key", "k"]).size().max())}])
    balance.to_csv(TAB / f"{m}_matched_balance.csv", index=False)

    pd.set_option("display.width", 220)
    for t in (summ, pos1, bytype, balance):
        print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    for model in sys.argv[1:] or ["gemma2b", "llama"]:
        run(model)
