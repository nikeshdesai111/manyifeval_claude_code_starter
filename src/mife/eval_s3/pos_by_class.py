"""Pos-A2: observed target violation risk by slot and k, split into fixed and parameterised rule types (S3 TEST, Obj. 1).

Added on 2026-11-25 at the project owner's request, after the test results were written (descriptive, observed
outcomes only; no predictor is involved). Same cells, same equal-split hierarchy (k, slot -> pool -> task -> context)
and the same paired 22-pool bootstrap as Pos-A in eval2.position, so the "all" rows reproduce
eval2_position_slot_risk.csv exactly. Writes results/eval_s3/tables/eval2_position_slot_risk_by_class.csv.
"""
from __future__ import annotations

import pandas as pd

from . import matches, verify
from .common import GENS
from .matches import FIXED_TYPES
from .paths import EVAL_DIR
from .stats import Boot, estimate, group_table

PP = 100.0


def run() -> pd.DataFrame:
    cells0 = matches.load("position_cells")
    rows = []
    for gen in GENS:
        L = verify.labels(gen)
        L["p"] = L.v / L.n
        boot = Boot(sorted(L.group_id.unique()))
        x = L.set_index(["prompt_id", "rule_id"]).p
        cells = cells0.copy()
        cells["p"] = x.reindex(pd.MultiIndex.from_arrays([cells.prompt_id, cells.rule_id])).values
        assert cells.p.notna().all()
        for cls, sel in (("all", None), ("fixed", True), ("parameterized", False)):
            c = cells if sel is None else cells[cells.type_id.isin(FIXED_TYPES) == sel]
            cr = c.groupby(["ctx", "slot", "group_id", "task_key", "k"]).p.mean().reset_index()
            tabs = group_table(cr, ["p"], ["k", "slot"], ["task_key", "ctx"])
            res = estimate(tabs, boot, lambda m: PP * m["p"], strata_mean=False, with_loo=False)
            for (k, j), r in zip(tabs["p"].index, res):
                rows.append({"generator": gen, "rule_class": cls, "k": k, "slot": str(j), "risk_pct": r["est"],
                             "lo": r["lo"], "hi": r["hi"], "n_pools": r["n_groups"]})
            # paired last - first within each matched context
            ends = c[(c.slot == 1) | (c.slot == c.k)]
            w = ends.groupby(["ctx", "group_id", "task_key", "k", "slot"]).p.mean().unstack("slot")
            w["d"] = [rec[k] - rec[1] for k, rec in zip(w.index.get_level_values("k"), w.to_dict("records"))]
            w = w.reset_index()[["ctx", "group_id", "task_key", "k", "d"]]
            tabs = group_table(w, ["d"], ["k"], ["task_key", "ctx"])
            res = estimate(tabs, boot, lambda m: PP * m["d"], strata_mean=False, with_loo=False)
            for k, r in zip(tabs["d"].index, res):
                k = k[0] if isinstance(k, tuple) else k
                rows.append({"generator": gen, "rule_class": cls, "k": k, "slot": "last-first", "risk_pct": r["est"],
                             "lo": r["lo"], "hi": r["hi"], "n_pools": r["n_groups"]})
    out = pd.DataFrame(rows)
    out.to_csv(EVAL_DIR / "tables" / "eval2_position_slot_risk_by_class.csv", index=False)
    return out


if __name__ == "__main__":
    run()
