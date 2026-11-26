"""K10 split check from pool metadata only (pools.jsonl, prompts/prompt_rules for split and tasks; no generations,
labels or features). Writes pilots/high_k/data/split_check_k10.csv and .md.

    PYTHONPATH=src .venv/bin/python pilots/high_k/code/split_check.py
"""
from __future__ import annotations

import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import pandas as pd

from mife import source as S
from mife.split_v3 import compatible_type_pairs, related_units
from mife.config import ROOT

INP = ROOT / "pilots" / "high_k" / "data" / "inputs"
OUT = ROOT / "pilots" / "high_k" / "data"
S3_SCORE_6 = 0.0871

pools = [json.loads(l) for l in (INP / "pools.jsonl").read_text().splitlines() if l.strip()]
pr = pd.read_parquet(INP / "prompts.parquet", columns=["group_id", "split", "task_key"]).drop_duplicates()
split_of = pr.drop_duplicates("group_id").set_index("group_id").split.to_dict()
s3 = json.loads((ROOT / "results" / "split_s3" / "proposed_split.json").read_text())
assert set(s3["test_groups"]) == {g for g, s in split_of.items() if s == "test"}, "K10 split differs from S3 assignment"
fam_of = (pd.read_parquet(ROOT / "data" / "manifest" / "prompts.parquet", columns=["task_key", "task_family"])
          .drop_duplicates().set_index("task_key").task_family.to_dict())

G = {p["group_id"]: p for p in pools}
test = sorted(g for g in G if split_of[g] == "test")
train = sorted(g for g in G if split_of[g] == "train")
types = sorted(S.TYPES)
n_rules = {len(p["rules"]) for p in pools}
gtypes = {g: sorted({r["type_id"] for r in p["rules"]}) for g, p in G.items()}
gexact = {g: sorted({r["exact_rule_hash"] for r in p["rules"]}) for g, p in G.items()}
rows, md = [], []

# (a) types
ta = {t: (sum(t in gtypes[g] for g in test), sum(t in gtypes[g] for g in train)) for t in types}
for t, (a, b) in ta.items():
    rows.append({"section": "a_type", "item": t, "test_groups": a, "train_groups": b, "all_groups": a + b})

# (b) compatible type pairs; also any within-pool conflict
pairs = compatible_type_pairs(types)
tp = {}
for a, b in pairs:
    tp[(a, b)] = (sum(a in gtypes[g] and b in gtypes[g] for g in test), sum(a in gtypes[g] and b in gtypes[g] for g in train))
    rows.append({"section": "b_type_pair", "item": f"{a}|{b}", "test_groups": tp[(a, b)][0], "train_groups": tp[(a, b)][1],
                 "all_groups": sum(tp[(a, b)])})
conflict_in_pool = [(g, a, b) for g in G for a, b in combinations(gtypes[g], 2) if S.conflicts(a, b)]

# (c) exact instances and exact pairs
ex_test = Counter(h for g in test for h in gexact[g]); ex_train = Counter(h for g in train for h in gexact[g])
both = sorted(set(ex_test) & set(ex_train))
ep_test = Counter(p for g in test for p in combinations(gexact[g], 2))
ep_train = Counter(p for g in train for p in combinations(gexact[g], 2))
ep_all = set(ep_test) | set(ep_train)
ep_both = set(ep_test) & set(ep_train)
desc = {}
for g, p in G.items():
    for r in p["rules"]:
        desc[r["exact_rule_hash"]] = f"{r['type_id']} {json.dumps(r['kwargs'], sort_keys=True)}"
for h in both:
    rows.append({"section": "c_exact_rule_in_both", "item": desc[h], "test_groups": ex_test[h], "train_groups": ex_train[h],
                 "all_groups": ex_test[h] + ex_train[h]})

# (d) balance score for 10-rule pools (same formula as split_v3, 10 types per pool)
fam_all = Counter(fam_of[t] for g in G for t in G[g]["task_keys"])
fam_test = Counter(fam_of[t] for g in test for t in G[g]["task_keys"])
typ_all = Counter(t for g in G for t in gtypes[g]); typ_test = Counter(t for g in test for t in gtypes[g])
nT, nA = len(test), len(G)
per_pool = 10
fam_gap = sum(abs(fam_test[f] / (3 * nT) - fam_all[f] / (3 * nA)) for f in fam_all)
typ_gap = sum(abs(typ_test[t] / (per_pool * nT) - typ_all[t] / (per_pool * nA)) for t in types)

# (e) tasks and related units
task_split = pr.groupby("task_key").split.nunique()
units = related_units([{"group_id": g, "task_keys": G[g]["task_keys"]} for g in G])
units_split = [u for u in units if len(u) > 1 and 0 < sum(g in test for g in u) < len(u)]

summ = {
    "groups": {"test": nT, "train": len(train)}, "rules_per_pool": sorted(n_rules),
    "split_matches_s3_assignment": True,
    "a_type_test_min_max": [min(v[0] for v in ta.values()), max(v[0] for v in ta.values())],
    "a_type_train_min_max": [min(v[1] for v in ta.values()), max(v[1] for v in ta.values())],
    "b_n_compatible_pairs": len(pairs), "b_pairs_in_any_pool": sum(sum(v) > 0 for v in tp.values()),
    "b_test_min": min(v[0] for v in tp.values()), "b_train_min": min(v[1] for v in tp.values()),
    "b_pairs_test_lt2": ["|".join(p) + f" (test {v[0]}, train {v[1]})" for p, v in tp.items() if v[0] < 2],
    "b_pairs_train_lt1": ["|".join(p) + f" (test {v[0]}, train {v[1]})" for p, v in tp.items() if v[1] < 1],
    "b_pairs_test_ge3": sum(v[0] >= 3 for v in tp.values()),
    "b_conflicting_pairs_inside_a_pool": len(conflict_in_pool),
    "c_exact_rules": {"distinct": len(set(ex_test) | set(ex_train)), "test": len(ex_test), "train": len(ex_train),
                      "in_both_splits": len(both)},
    "c_exact_pairs": {"distinct": len(ep_all), "in_test": len(ep_test), "in_train": len(ep_train),
                      "in_both_splits": len(ep_both), "max_groups_per_exact_pair": max((ep_test + ep_train).values())},
    "d_family_L1_gap": round(fam_gap, 4), "d_type_L1_gap": round(typ_gap, 4), "d_score_10rule": round(fam_gap + typ_gap, 4),
    "d_score_6rule_s3": S3_SCORE_6,
    "e_tasks_in_both_splits": int((task_split > 1).sum()), "e_related_units_split": units_split,
}
pd.DataFrame(rows).to_csv(OUT / "split_check_k10.csv", index=False)
md += ["# K10 split check (pool metadata only)", "",
       f"Split: S3, {nT} TEST / {len(train)} TRAIN groups; K10 pools have {sorted(n_rules)} rules. "
       "Source: pilots/high_k/data/inputs/pools.jsonl and prompts.parquet (split, tasks); task families from the "
       "frozen manifest. No generations, labels or features were read.", "",
       "```json", json.dumps(summ, indent=1), "```", "",
       "## (a) Types", "", "| Type | TEST groups | TRAIN groups |", "|---|---|---|"]
md += [f"| {t} | {a} | {b} |" for t, (a, b) in ta.items()]
md += ["", "Per-pair and exact-rule rows: split_check_k10.csv."]
(OUT / "split_check_k10.md").write_text("\n".join(md) + "\n")
print(json.dumps(summ, indent=1))
