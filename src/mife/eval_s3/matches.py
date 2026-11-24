"""Metadata-only Evaluation 2, Objective 1 comparison manifests for the S3 TEST design (plan §§5.1-5.8, 9.1).

Built from data/evaluation_expansion_s3/{prompts,prompt_rules}.parquet only: no response, label, feature or prediction
is read. Every table is keyed by (prompt_id, rule_id) targets so outcomes and predictions are joined later.

  position  cells: (ctx, k, slot) -> the k-1 COA prompts placing the exact target in that slot (weight 1/(k-1))
            pairs: every i<j slot pair of each exact (group, task, subset, target) context, k = 2..5
  append    strict k -> k+1 appends: after.ordered_rule_ids[:k] == before.ordered_rule_ids, every retained rule
  companion strict one-companion replacements (same group/task/k, every slot equal except one; the target is any
            unchanged slot), unordered prompt pairs oriented by prompt_id; and C1 contexts (fixed background, all
            available companions at the replacement slot)
  task      the three unordered task pairs of each (group, config), oriented by task_key, every target rule
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from ..config import ROOT
from .paths import EXP_DIR

MATCH_DIR = ROOT / "data" / "eval_s3" / "matches"
FAMILIES = ROOT / "configs" / "task_families_v1.csv"
FIXED_TYPES = ("change_case:english_capital", "change_case:english_lowercase", "detectable_format:title",
               "punctuation:no_comma", "startend:quotation")


def manifest():
    P = pd.read_parquet(EXP_DIR / "prompts.parquet")
    PR = pd.read_parquet(EXP_DIR / "prompt_rules.parquet")
    assert P["split"].eq("test").all() and P["group_id"].nunique() == 22 and len(P) == 30096
    P["rules"] = P["ordered_rule_ids"].map(tuple)
    P["types"] = P["ordered_type_ids"].map(tuple)
    fam = pd.read_csv(FAMILIES).set_index("task_key")["family"]
    P["family"] = P["task_key"].map(fam)
    assert P["family"].notna().all()
    return P, PR


def type_of(PR) -> dict:
    return PR.drop_duplicates("rule_id").set_index("rule_id")["type_id"].to_dict()


def position(P, T):
    cells, pairs = [], []
    for (g, t, sk), d in P[P.k >= 2].groupby(["group_id", "task_key", "subset_key"]):
        k = int(d.k.iloc[0])
        for tr in d.rules.iloc[0]:
            ctx = f"{g}|{t}|{sk}|{tr}"
            slot_prompts = {}
            for pid, rules in zip(d.prompt_id, d.rules):
                slot_prompts.setdefault(rules.index(tr) + 1, []).append(pid)
            assert sorted(slot_prompts) == list(range(1, k + 1)), (ctx, slot_prompts)
            assert all(len(v) == k - 1 for v in slot_prompts.values()), ctx  # COA: k-1 orders per slot
            for j, pids in slot_prompts.items():
                for pid in pids:
                    cells.append((ctx, g, t, k, j, tr, T[tr], pid, 1.0 / (k - 1)))
            for i, j in combinations(range(1, k + 1), 2):
                pairs.append((f"{ctx}|{i}>{j}", ctx, g, t, k, i, j, tr, T[tr]))
    cells = pd.DataFrame(cells, columns=["ctx", "group_id", "task_key", "k", "slot", "rule_id", "type_id", "prompt_id", "a"])
    pairs = pd.DataFrame(pairs, columns=["pair_id", "ctx", "group_id", "task_key", "k", "i", "j", "rule_id", "type_id"])
    return cells, pairs


def append(P, T):
    key = {(g, t, r): pid for g, t, r, pid in zip(P.group_id, P.task_key, P.rules, P.prompt_id)}
    rows = []
    for g, t, r, pid in zip(P.group_id, P.task_key, P.rules, P.prompt_id):
        k = len(r)
        if k < 2:
            continue
        before = key.get((g, t, r[:-1]))
        if before is None:
            continue
        for j, rid in enumerate(r[:-1], start=1):
            rows.append((f"{before}>{pid}", before, pid, g, t, f"{k - 1}->{k}", k - 1, j, rid, T[rid], r[-1], T[r[-1]]))
    return pd.DataFrame(rows, columns=["pair_id", "prompt_before", "prompt_after", "group_id", "task_key", "transition",
                                       "k_before", "slot", "rule_id", "type_id", "added_rule_id", "added_type_id"])


def companion(P, T):
    """C2 strict pairs (one row per retained target) and C1 contexts (one row per available companion)."""
    pairs, ctx_rows = [], []
    for (g, t, k), d in P[P.k >= 2].groupby(["group_id", "task_key", "k"]):
        for s in range(k):
            buckets = {}
            for pid, r in zip(d.prompt_id, d.rules):
                buckets.setdefault(r[:s] + ("*",) + r[s + 1:], []).append((pid, r[s]))
            for bg, items in buckets.items():
                if len(items) < 2:
                    continue
                items = sorted(items)
                for tslot in range(k):
                    if tslot == s:
                        continue
                    tr = bg[tslot]
                    h = f"{g}|{t}|k{k}|t{tslot + 1}|s{s + 1}|{'/'.join(bg)}"
                    rel = "earlier" if s < tslot else "later"
                    for pid, comp in items:
                        ctx_rows.append((h, g, t, k, tslot + 1, s + 1, rel, tr, T[tr], pid, comp, T[comp], len(items) - 1))
                    for (pa, ca), (pb, cb) in combinations(items, 2):
                        pairs.append((f"{pa}~{pb}|t{tslot + 1}", h, pa, pb, g, t, k, tslot + 1, s + 1, rel, tr, T[tr],
                                      ca, T[ca], cb, T[cb]))
    pairs = pd.DataFrame(pairs, columns=["cmp_id", "ctx", "prompt_a", "prompt_b", "group_id", "task_key", "k",
                                         "target_slot", "replacement_slot", "relative_order", "rule_id", "type_id",
                                         "comp_a", "comp_a_type", "comp_b", "comp_b_type"])
    ctx = pd.DataFrame(ctx_rows, columns=["ctx", "group_id", "task_key", "k", "target_slot", "replacement_slot",
                                          "relative_order", "rule_id", "type_id", "prompt_id", "comp_rule_id",
                                          "comp_type_id", "n_alternatives"])
    return pairs, ctx


def task(P, T):
    rows, fam_rows = [], []
    for (g, c), d in P.groupby(["group_id", "config_id"]):
        assert len(d) == 3 and d.family.nunique() == 3, (g, c)
        d = d.sort_values("task_key")
        rules = d.rules.iloc[0]
        assert all(r == rules for r in d.rules)
        k = len(rules)
        recs = list(zip(d.prompt_id, d.task_key, d.family))
        for j, rid in enumerate(rules, start=1):
            for (pa, ta, fa), (pb, tb, fb) in combinations(recs, 2):
                rows.append((f"{pa}~{pb}|{rid}", g, c, k, j, rid, T[rid], pa, pb, ta, tb, fa, fb))
            for x in range(3):
                others = [recs[y] for y in range(3) if y != x]
                fam_rows.append((g, c, k, j, rid, T[rid], recs[x][2], recs[x][0], others[0][0], others[1][0]))
    pairs = pd.DataFrame(rows, columns=["cmp_id", "group_id", "config_id", "k", "slot", "rule_id", "type_id", "prompt_a",
                                        "prompt_b", "task_a", "task_b", "family_a", "family_b"])
    fam = pd.DataFrame(fam_rows, columns=["group_id", "config_id", "k", "slot", "rule_id", "type_id", "family",
                                          "prompt_f", "prompt_o1", "prompt_o2"])
    return pairs, fam


def build() -> dict:
    MATCH_DIR.mkdir(parents=True, exist_ok=True)
    P, PR = manifest()
    T = type_of(PR)
    cells, ppairs = position(P, T)
    app = append(P, T)
    cp, cctx = companion(P, T)
    tp, tf = task(P, T)
    for name, df in {"position_cells": cells, "position_pairs": ppairs, "append": app, "companion_pairs": cp,
                     "companion_ctx": cctx, "task_pairs": tp, "task_family": tf}.items():
        df.to_parquet(MATCH_DIR / f"{name}.parquet")
    counts = {
        "prompts": len(P), "prompt_rules": len(PR),
        "position_contexts": ppairs.ctx.nunique(), "position_pairs": len(ppairs),
        "position_pairs_by_k": ppairs.groupby("k").size().to_dict(),
        "append_prompt_pairs_by_transition": app.drop_duplicates("pair_id").groupby("transition").size().to_dict(),
        "append_targets_by_transition": app.groupby("transition").size().to_dict(),
        "append_prompt_pairs": app.pair_id.nunique(), "append_targets": len(app),
        "companion_prompt_pairs_by_k": cp.assign(pp=cp.prompt_a + cp.prompt_b).drop_duplicates("pp").groupby("k").size().to_dict(),
        "companion_targets_by_k": cp.groupby("k").size().to_dict(), "companion_targets": len(cp),
        "task_prompt_pairs": tp.assign(pp=tp.prompt_a + tp.prompt_b).pp.nunique(), "task_targets": len(tp),
    }
    return counts


EXPECTED = {"position_pairs": 57420, "position_contexts": 11880, "append_prompt_pairs": 25740, "append_targets": 69300,
            "companion_targets": 177408, "task_prompt_pairs": 30096, "task_targets": 115236, "prompt_rules": 115236}


def load(name: str) -> pd.DataFrame:
    return pd.read_parquet(MATCH_DIR / f"{name}.parquet")


if __name__ == "__main__":
    import json
    from .paths import EVAL_DIR
    c = build()
    (EVAL_DIR / "tables").mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / "tables" / "match_coverage.json").write_text(json.dumps(c, indent=1, default=str))
    print(json.dumps(c, indent=1, default=str))
    for k, v in EXPECTED.items():
        print(k, c[k], v, "OK" if c[k] == v else "MISMATCH")
