"""Predeclared TRAIN-only pilot selection: 100 prompts, 20 per k.

Algorithm (fixed before any generation; uses manifest metadata only, never outcomes):
for each k in 1..5, among TRAIN prompts at that k:
  1. take the 2 longest prompts (by characters; ties by prompt_id) to include worst-case lengths;
  2. fill to 20 greedily in a seeded random order, scoring each candidate by the number of
     not-yet-covered (group, task family, instruction type) items it adds (first max wins).
Test-group prompts are excluded before any computation.
"""
from __future__ import annotations

import random

import pandas as pd

from .config import DATA_DIR, load_config
from .manifest import MANIFEST_DIR
from .storage import write_json, write_parquet

PILOT_DIR = DATA_DIR / "pilot"


def select() -> pd.DataFrame:
    cfg = load_config()
    prompts = pd.read_parquet(MANIFEST_DIR / "prompts.parquet")
    train = prompts[prompts["split"] == "train"]
    assert (train["split"] == "train").all()
    rng = random.Random(cfg["seeds"]["pilot_selection"])
    chosen = []
    for k in range(1, 6):
        cand = train[train["k"] == k].sort_values("prompt_id")
        longest = cand.sort_values(["n_chars", "prompt_id"], ascending=[False, True]).head(2)
        picks = list(longest["prompt_id"])
        cov_g, cov_f, cov_t = set(), set(), set()
        rowmap = cand.set_index("prompt_id")
        for pid in picks:
            r = rowmap.loc[pid]
            cov_g.add(r.group_id); cov_f.add(r.task_family); cov_t.update(r.ordered_type_ids)
        order = list(cand["prompt_id"])
        rng.shuffle(order)
        while len(picks) < cfg["pilot"]["per_k"]:
            best, best_s = None, -1
            for pid in order:
                if pid in picks:
                    continue
                r = rowmap.loc[pid]
                s = (r.group_id not in cov_g) + (r.task_family not in cov_f) + len(set(r.ordered_type_ids) - cov_t)
                if s > best_s:
                    best, best_s = pid, s
            r = rowmap.loc[best]
            picks.append(best)
            cov_g.add(r.group_id); cov_f.add(r.task_family); cov_t.update(r.ordered_type_ids)
        chosen += picks
    sel = prompts.set_index("prompt_id").loc[chosen].reset_index()
    assert len(sel) == cfg["pilot"]["n_prompts"] and sel["prompt_id"].is_unique
    assert (sel["split"] == "train").all() and (sel.groupby("k").size() == cfg["pilot"]["per_k"]).all()
    write_parquet(sel[["prompt_id", "group_id", "task_key", "task_family", "k", "n_chars"]],
                  PILOT_DIR / "pilot_prompts.parquet")
    types = sorted({t for ts in sel["ordered_type_ids"] for t in ts})
    write_json({
        "seed": cfg["seeds"]["pilot_selection"], "n": len(sel),
        "per_k": sel.groupby("k").size().to_dict(),
        "n_groups": int(sel["group_id"].nunique()), "families": sel["task_family"].value_counts().to_dict(),
        "n_types_covered": len(types), "types": types,
        "n_chars_quantiles": sel["n_chars"].quantile([0, .5, 1]).to_dict(),
        "includes_global_train_max_len_per_k": True, "uses_test_groups": False,
    }, PILOT_DIR / "pilot_selection.json")
    return sel
