"""Pre-generation audit of the proposed expanded held-out evaluation design (audit only; changes nothing).

Proposal under audit: in each of the 15 frozen TEST groups, use EVERY k-subset of the group's six exact rules for
k = 1..5 (6 + 15 + 20 + 15 + 6 = 62 subsets) with the existing COA ordering template per subset, on the group's
three tasks, 20 responses per prompt per generator.

What this script reads (metadata only):
  * data/manifest/prompts.parquet, prompt_rules.parquet, rule_pools.jsonl, tasks.parquet (frozen manifest; rule ids,
    types, rule texts, task keys, splits, chains, COA rows). It never reads prompt outcomes.
  * results/tables/desc_*.csv (committed TRAIN-only descriptive tables) for plausible violation rates and effect sizes.
It never reads generations, verifier labels, features or predictions, and it writes only to
results/audits/evaluation_expansion/. Kept outside src/ so the frozen source-tree hash is unchanged.

Run (from the repository root):
    PYTHONPATH=src python audits/evaluation_expansion_audit.py
    # options: --manifest-dir DIR  --out DIR  --mc-reps N  --seed S
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from itertools import combinations, permutations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from mife import design

ROOT = Path(__file__).resolve().parents[1]
FROZEN_PROMPTS_SHA256 = "f63c0aeaeec8872ec22df70f3feccbb4ecf464ee24368a001aace0cc5ccf1b14"  # results/prespec/FROZEN.json
N_RESP = 20
N_MODELS = 2
K_MAX = 5
PROMPT_HEADER = "\n\nYour response should follow the instructions below:\n"  # mife.source.PROMPT_HEADER
RULE_PREFIX = "- "  # mife.source.rule_line_prefix()


def _exclusive_groups() -> list[set[str]]:
    """Read mife.source.EXCLUSIVE_GROUPS without importing mife.source (which needs the pinned ManyIFEval checkout)."""
    tree = ast.parse((ROOT / "src" / "mife" / "source.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "EXCLUSIVE_GROUPS" for t in node.targets):
            return [set(s) for s in ast.literal_eval(node.value)]
    raise RuntimeError("EXCLUSIVE_GROUPS not found")


def _types() -> list[str]:
    tree = ast.parse((ROOT / "src" / "mife" / "source.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "TYPE_INFO" for t in node.targets):
            return sorted(ast.literal_eval(node.value))
    raise RuntimeError("TYPE_INFO not found")


TYPES = _types()
EXCL = _exclusive_groups()


def conflicts(a: str, b: str) -> bool:
    return a == b or any(a in g and b in g for g in EXCL)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ===================================================================================================== loading
def load(manifest_dir: Path):
    sha = sha256_file(manifest_dir / "prompts.parquet")
    assert sha == FROZEN_PROMPTS_SHA256, f"prompts.parquet does not match the frozen manifest ({sha})"
    P = pd.read_parquet(manifest_dir / "prompts.parquet",
                        columns=["prompt_id", "group_id", "split", "task_key", "task_family", "config_id", "subset_id",
                                 "chain_index", "k", "coa_row", "ordered_rule_ids", "n_chars"])
    R = pd.read_parquet(manifest_dir / "prompt_rules.parquet",
                        columns=["prompt_id", "position", "rule_id", "type_id", "rule_text", "char_start", "group_id", "split"])
    pools = {}
    for line in open(manifest_dir / "rule_pools.jsonl"):
        p = json.loads(line)
        pools[p["group_id"]] = p
    tasks = pd.read_parquet(manifest_dir / "tasks.parquet", columns=["task_key", "task_text", "family", "group_id", "split"])
    return P, R, pools, tasks, sha


# =========================================================================================== expanded manifest
def expanded_configs(pool: dict) -> list[dict]:
    """All k-subsets (k=1..5) x COA template. Existing subsets keep their chain-order base (so all 246 existing ordered
    configurations are reproduced exactly); new subsets use the group's chain-label order (A..F) as base."""
    order = pool["chain_order"]
    rank = {r: i for i, r in enumerate(order)}
    existing_base = {}
    for c_idx, chain in enumerate(design.chains(order), start=1):
        for k in range(1, K_MAX + 1):
            existing_base.setdefault(frozenset(chain[:k]), (tuple(chain[:k]), c_idx))
    out = []
    for k in range(1, K_MAX + 1):
        tmpl = design.coa_template(k)
        for sub in combinations(order, k):
            s = frozenset(sub)
            if s in existing_base:
                base, chain_idx = existing_base[s]
            else:
                base, chain_idx = tuple(sorted(sub, key=rank.get)), None
            for row, sigma in enumerate(tmpl):
                out.append({"k": k, "subset": s, "base": base, "chain_index": chain_idx, "coa_row": row,
                            "order": tuple(base[x] for x in sigma), "subset_status": "existing" if s in existing_base else "new"})
    return out


def naive_base_reuse(pool: dict, existing: set) -> dict:
    """How many existing configs would be reproduced if EVERY subset used the chain-label base (a tempting but wrong
    implementation)."""
    order = pool["chain_order"]
    n = Counter()
    for k in range(1, K_MAX + 1):
        for sub in combinations(order, k):
            for sigma in design.coa_template(k):
                if tuple(sub[x] for x in sigma) in existing:
                    n[k] += 1
    return dict(n)


# ================================================================================================== helpers
def _pct(a, b):
    return round(100.0 * a / b, 2) if b else float("nan")


def summarize_balance(counter: Counter, keys) -> dict:
    vals = np.array([counter.get(key, 0) for key in keys], dtype=float)
    return {"cells": len(vals), "min": int(vals.min()) if len(vals) else 0, "max": int(vals.max()) if len(vals) else 0,
            "mean": round(float(vals.mean()), 3) if len(vals) else 0,
            "cv": round(float(vals.std() / vals.mean()), 4) if len(vals) and vals.mean() > 0 else float("nan"),
            "n_zero": int((vals == 0).sum())}


# ================================================================================================== AUDIT 1
def factor_a(G: dict, rtype: dict, text_len: dict, task_len: dict, group_tasks: dict) -> tuple[pd.DataFrame, ...]:
    """Position coverage, companion balance, position-pair comparisons and char-offset nuisance."""
    cov_rows, pair_rows, nuis_rows, comp_rows = [], [], [], []
    for k in range(1, K_MAX + 1):
        pos_obs = Counter()           # (position) -> prompt-target observations (x tasks)
        per_cell = []                 # per (group, subset, target): orderings at each position
        comp_ok = comp_bad = 0
        dist_end = defaultdict(list)  # position -> chars from end of target rule to end of prompt (per config, target)
        start_off = defaultdict(list)
        for g, cfgs in G.items():
            nt = len(group_tasks[g])
            by_sub = defaultdict(list)
            for c in cfgs:
                if c["k"] == k:
                    by_sub[c["subset"]].append(c["order"])
            for s, orders in by_sub.items():
                for t in s:
                    cnt = Counter(o.index(t) + 1 for o in orders)
                    per_cell.append([cnt.get(p, 0) for p in range(1, k + 1)])
                    for p in range(1, k + 1):
                        pos_obs[p] += cnt.get(p, 0) * nt
                        rows = [o for o in orders if o[p - 1] == t]
                        cm = Counter((o[q], q) for o in rows for q in range(k) if q != p - 1)
                        cells = [(c2, q) for c2 in s if c2 != t for q in range(k) if q != p - 1]
                        if k >= 3:
                            if len(set(cm.get(x, 0) for x in cells)) == 1:
                                comp_ok += 1
                            else:
                                comp_bad += 1
                for o in orders:
                    lens = [len(RULE_PREFIX) + text_len[r] + 1 for r in o]
                    total = sum(lens) - 1
                    cum = 0
                    for p, r in enumerate(o, start=1):
                        start_off[p].append(cum)
                        cum += lens[p - 1]
                        dist_end[p].append(total - (cum - 1))
        cells = np.array(per_cell)
        n_cells = len(per_cell) * 3  # x 3 tasks per group
        for p in range(1, k + 1):
            cov_rows.append({"k": k, "position": p, "prompt_target_observations": pos_obs[p],
                             "orderings_per_task_subset_target_min": int(cells[:, p - 1].min()),
                             "orderings_per_task_subset_target_max": int(cells[:, p - 1].max()),
                             "task_subset_target_cells": n_cells,
                             "mean_chars_target_end_to_prompt_end": round(float(np.mean(dist_end[p])), 1),
                             "sd_chars_target_end_to_prompt_end": round(float(np.std(dist_end[p])), 1),
                             "mean_chars_rule_block_before_target": round(float(np.mean(start_off[p])), 1)})
        if k >= 2:
            for a, b in combinations(range(1, k + 1), 2):
                pair_rows.append({"k": k, "pos_a": a, "pos_b": b, "transition_size": b - a,
                                  "task_subset_target_comparisons": n_cells,
                                  "orderings_per_side": int(cells[:, a - 1].min()),
                                  "responses_per_side": int(cells[:, a - 1].min()) * N_RESP})
        comp_rows.append({"k": k, "target_position_cells_checked": comp_ok + comp_bad,
                          "cells_every_companion_at_every_other_position_equally": comp_ok,
                          "cells_unbalanced": comp_bad,
                          "note": "k<=2: no companion-position balance question" if k <= 2 else ""})
    return pd.DataFrame(cov_rows), pd.DataFrame(pair_rows), pd.DataFrame(comp_rows)


def strict_appends(xs: set, cfg_set: set, pool_rules: list):
    """Yield (X, Y, appended, present) for every ordered X in xs and every rule r not in X; present = X+(r,) in cfg_set."""
    for X in xs:
        if len(X) >= K_MAX:
            continue
        for r in pool_rules:
            if r not in X:
                Y = X + (r,)
                yield X, Y, r, (Y in cfg_set)


def factor_b1(G: dict, rtype: dict, group_tasks: dict, existing: dict) -> dict:
    summ, by_pos, by_type, by_group, by_task, bal_rows = [], [], [], [], [], []
    for design_name, src in (("expanded", G), ("current_246", existing)):
        for k in range(1, K_MAX):
            present = potential = prefix_present = n_k1 = 0
            tgt_pos = Counter(); tgt_type = Counter(); grp = Counter(); app_type = Counter()
            rule_pos = defaultdict(Counter)  # group -> (rule, pos) -> count
            app_rule = defaultdict(Counter)  # group -> (target, appended) -> count
            for g, cfgs in src.items():
                S = {c["order"] for c in cfgs}
                rules = sorted({r for c in cfgs for r in c["order"]})
                nt = len(group_tasks[g])
                for X, Y, r, ok in strict_appends({o for o in S if len(o) == k}, S, rules):
                    potential += 1
                    if not ok:
                        continue
                    present += 1
                    grp[g] += 1
                    app_type[rtype[r]] += nt
                    for p, t in enumerate(X, start=1):
                        tgt_pos[p] += nt
                        tgt_type[rtype[t]] += nt
                        rule_pos[g][(t, p)] += 1
                        app_rule[g][(t, r)] += 1
                ys = [o for o in S if len(o) == k + 1]
                n_k1 += len(ys)
                prefix_present += sum(1 for y in ys if y[:k] in S)
            ntask = sum(len(group_tasks[g]) for g in src)
            full_universe = math.perm(6, k) * (6 - k) * len(src)
            summ.append({"design": design_name, "transition": f"{k}->{k + 1}", "k": k,
                         "ordered_pairs_per_group": present / len(src),
                         "prompt_pairs_total": present * 3, "target_comparisons_total": present * 3 * k,
                         "potential_appends_from_present_k_configs": potential, "present": present,
                         "pct_of_potential_from_present_k_configs": _pct(present, potential),
                         "full_permutation_universe_pairs": full_universe,
                         "pct_of_full_permutation_universe": _pct(present, full_universe),
                         "k_plus_1_configs": n_k1, "k_plus_1_configs_with_prefix_present": prefix_present,
                         "pct_k_plus_1_with_prefix": _pct(prefix_present, n_k1),
                         "missing_vs_potential": potential - present})
            if design_name != "expanded":
                continue
            for p, n in sorted(tgt_pos.items()):
                by_pos.append({"transition": f"{k}->{k + 1}", "target_position": p, "target_comparisons": n})
            for t, n in sorted(tgt_type.items()):
                by_type.append({"transition": f"{k}->{k + 1}", "type_id": t, "target_comparisons": n,
                                "appended_type_comparisons(prompt_pairs)": app_type.get(t, 0)})
            for g, n in sorted(grp.items()):
                by_group.append({"transition": f"{k}->{k + 1}", "group_id": g, "ordered_pairs": n,
                                 "prompt_pairs": n * 3, "target_comparisons": n * 3 * k})
            # balance of (target rule x position) and (target x appended rule) within each group
            rp = Counter(); ar = Counter()
            for g in src:
                rules = sorted({r for c in src[g] for r in c["order"]})
                keys = [(t, p) for t in rules for p in range(1, k + 1)]
                b = summarize_balance(rule_pos[g], keys)
                keys2 = [(t, r) for t in rules for r in rules if t != r]
                b2 = summarize_balance(app_rule[g], keys2)
                bal_rows.append({"transition": f"{k}->{k + 1}", "group_id": g,
                                 "target_rule_x_position_min": b["min"], "target_rule_x_position_max": b["max"],
                                 "target_rule_x_position_cv": b["cv"],
                                 "target_x_appended_rule_min": b2["min"], "target_x_appended_rule_max": b2["max"],
                                 "target_x_appended_rule_cv": b2["cv"]})
    return {"summary": pd.DataFrame(summ), "by_position": pd.DataFrame(by_pos), "by_type": pd.DataFrame(by_type),
            "by_group": pd.DataFrame(by_group), "balance": pd.DataFrame(bal_rows)}


def factor_b2(G: dict, group_tasks: dict) -> pd.DataFrame:
    """Marginal k: per (target, position p, k) observations; companion identity balance."""
    rows = []
    for k in range(1, K_MAX + 1):
        obs_min, obs_max, comp_cv, n_cells = [], [], [], 0
        subs_full = True
        for g, cfgs in G.items():
            rules = sorted({r for c in cfgs for r in c["order"]})
            ks = [c for c in cfgs if c["k"] == k]
            subs_full &= len({c["subset"] for c in ks}) == math.comb(6, k)
            for t in rules:
                for p in range(1, k + 1):
                    cells = [c for c in ks if c["order"][p - 1] == t]
                    n_cells += 1
                    obs_min.append(len(cells)); obs_max.append(len(cells))
                    if k >= 2:
                        cc = Counter(r for c in cells for r in c["order"] if r != t)
                        v = np.array([cc.get(r, 0) for r in rules if r != t], float)
                        comp_cv.append(v.std() / v.mean())
        rows.append({"k": k, "all_C(6,k)_subsets_present_in_every_group": subs_full,
                     "subsets_per_group": math.comb(6, k), "subsets_containing_target": math.comb(5, k - 1),
                     "configs_with_target_at_fixed_position_per_task": int(np.min(obs_min)),
                     "configs_with_target_at_fixed_position_per_task_max": int(np.max(obs_max)),
                     "responses_per_task_target_position_cell": int(np.min(obs_min)) * N_RESP,
                     "target_position_cells_total(x3 tasks)": n_cells * 3,
                     "companion_presence_cv_within_cell(max)": round(float(np.max(comp_cv)), 4) if comp_cv else float("nan"),
                     "share_of_target_cells_with_each_companion": round((k - 1) / 5, 3) if k >= 2 else 0})
    return pd.DataFrame(rows)


def factor_c(G: dict, rtype: dict, group_tasks: dict, existing: dict) -> dict:
    summ, by_pos, typepair = [], [], []
    tp_groups = defaultdict(set); tp_pairs = Counter()
    for design_name, src in (("expanded", G), ("current_246", existing)):
        for k in range(2, K_MAX + 1):
            present = potential = 0
            pos_ct = Counter()
            for g, cfgs in src.items():
                S = {c["order"] for c in cfgs if c["k"] == k}
                rules = sorted({r for c in cfgs for r in c["order"]})
                for X in S:
                    for j in range(k):
                        for r in rules:
                            if r in X:
                                continue
                            Y = X[:j] + (r,) + X[j + 1:]
                            if X < Y:
                                potential += 1
                                if Y in S:
                                    present += 1
                                    if design_name == "expanded":
                                        for i in range(k):
                                            if i != j:
                                                pos_ct[(i + 1, j + 1)] += 3
                                        key = tuple(sorted((rtype[X[j]], rtype[r])))
                                        tp_groups[(k, key)].add(g); tp_pairs[(k, key)] += 1
            full = math.perm(6, k) * k * (6 - k) // 2 * len(src)
            summ.append({"design": design_name, "k": k, "ordered_pairs_per_group": present / len(src),
                         "prompt_pairs_total": present * 3, "target_comparisons_total": present * 3 * (k - 1),
                         "potential_from_present_configs": potential, "present": present,
                         "pct_of_potential_from_present_configs": _pct(present, potential),
                         "full_permutation_universe_pairs": full, "pct_of_full_permutation_universe": _pct(present, full)})
            if design_name == "expanded":
                for (tp, rp), n in sorted(pos_ct.items()):
                    by_pos.append({"k": k, "target_position": tp, "replaced_position": rp, "target_comparisons": n})
    for (k, key), gs in sorted(tp_groups.items()):
        typepair.append({"k": k, "swapped_type_a": key[0], "swapped_type_b": key[1], "n_groups": len(gs),
                         "ordered_pairs": tp_pairs[(k, key)], "prompt_pairs": tp_pairs[(k, key)] * 3})
    return {"summary": pd.DataFrame(summ), "by_position": pd.DataFrame(by_pos), "type_pairs": pd.DataFrame(typepair)}


def factor_c_marginal(G: dict) -> pd.DataFrame:
    rows = []
    for k in range(2, K_MAX + 1):
        pres, absn, valid, total = [], [], 0, 0
        for g, cfgs in G.items():
            rules = sorted({r for c in cfgs for r in c["order"]})
            ks = [c for c in cfgs if c["k"] == k]
            for t in rules:
                for p in range(1, k + 1):
                    cells = [c for c in ks if c["order"][p - 1] == t]
                    for cpn in rules:
                        if cpn == t:
                            continue
                        a = sum(1 for c in cells if cpn in c["subset"]); b = len(cells) - a
                        pres.append(a); absn.append(b); total += 1; valid += (a > 0 and b > 0)
        rows.append({"k": k, "task_target_position_companion_cells(x3 tasks)": total * 3,
                     "valid_cells(both present and absent)": valid * 3,
                     "configs_with_companion_present_per_cell": int(np.min(pres)),
                     "configs_with_companion_absent_per_cell": int(np.min(absn)),
                     "responses_present_side": int(np.min(pres)) * N_RESP, "responses_absent_side": int(np.min(absn)) * N_RESP,
                     "note": "absent side at k=5 is a single subset (the one omitting the companion)" if k == 5 else ""})
    return pd.DataFrame(rows)


def factor_d(G: dict, pools: dict, P_test: pd.DataFrame, tasks: pd.DataFrame, group_tasks: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for g in sorted(G):
        pool_tasks = sorted(pools[g]["task_keys"])
        man = P_test[P_test.group_id == g]
        same_cfgs = man.groupby("task_key")["config_id"].apply(frozenset)
        rows.append({"group_id": g, "tasks": "|".join(map(str, pool_tasks)), "pool_tasks_match_manifest": sorted(same_cfgs.index) == pool_tasks,
                     "current_configs_identical_across_tasks": len(set(same_cfgs)) == 1,
                     "families": "|".join(map(str, tasks.set_index("task_key").loc[pool_tasks, "family"])),
                     "n_distinct_families": tasks.set_index("task_key").loc[pool_tasks, "family"].nunique(),
                     "task_chars_min": int(tasks.set_index("task_key").loc[pool_tasks, "task_text"].str.len().min()),
                     "task_chars_max": int(tasks.set_index("task_key").loc[pool_tasks, "task_text"].str.len().max()),
                     "expanded_configs": len(G[g]), "task_pairs": 3,
                     "prompt_pairs": len(G[g]) * 3, "target_comparisons": sum(c["k"] for c in G[g]) * 3})
    D = pd.DataFrame(rows)
    summ = pd.DataFrame([{"groups": len(D), "tasks": int(D.tasks.str.count(r"\|").add(1).sum()),
                          "task_pairs_total": int(D.task_pairs.sum()), "prompt_pairs_total": int(D.prompt_pairs.sum()),
                          "target_comparisons_total": int(D.target_comparisons.sum()),
                          "all_pool_tasks_match": bool(D.pool_tasks_match_manifest.all()),
                          "all_current_configs_identical": bool(D.current_configs_identical_across_tasks.all()),
                          "groups_with_3_distinct_families": int((D.n_distinct_families == 3).sum())}])
    return D, summ


# ================================================================================================== AUDIT 2
def sample_types(rng: random.Random) -> list[str]:
    """Type-level replica of mife.rule_sampler.sample_pool (uniform proposal, D1 conflicts, completability lookahead).
    Parameter-feasibility rejections are ignored (the frozen manifest logs 0 type rejections)."""
    def completable(cur):
        def dfs(c, start):
            if len(c) == 6:
                return True
            for t in start:
                if t in c or any(conflicts(t, x) for x in c):
                    continue
                if dfs(c + [t], [u for u in start if u > t]):
                    return True
            return False
        return dfs(list(cur), TYPES)
    chosen, banned = [], set()
    while len(chosen) < 6:
        el = [t for t in TYPES if t not in chosen and t not in banned and not any(conflicts(t, c) for c in chosen)]
        t = rng.choice(el)
        if not completable(chosen + [t]):
            banned.add(t); continue
        chosen.append(t)
    return chosen


def audit2(pools: dict, G: dict, rtype: dict, group_tasks: dict, reps: int, seed: int) -> dict:
    test = sorted(g for g, p in pools.items() if p["split"] == "test")
    train = sorted(g for g, p in pools.items() if p["split"] == "train")
    tset = {g: [r["type_id"] for r in pools[g]["rules"]] for g in pools}
    # prompts containing each type / pair in the expanded design (x3 tasks)
    type_prompts = Counter(); pair_prompts = Counter(); pair_tasks = defaultdict(set)
    for g in test:
        for c in G[g]:
            ts = sorted(rtype[r] for r in c["order"])
            for t in ts:
                type_prompts[t] += 3
            for a, b in combinations(ts, 2):
                pair_prompts[(a, b)] += 3
                pair_tasks[(a, b)].update(group_tasks[g])
    rows = []
    for t in TYPES:
        inst = [r for g in test for r in pools[g]["rules"] if r["type_id"] == t]
        rows.append({"type_id": t, "n_test_groups": sum(t in tset[g] for g in test),
                     "n_train_groups": sum(t in tset[g] for g in train),
                     "test_exact_instances": len({r["exact_rule_hash"] for r in inst}),
                     "test_distinct_parameter_settings": len({json.dumps(r["kwargs"], sort_keys=True) for r in inst}),
                     "test_tasks": 3 * sum(t in tset[g] for g in test),
                     "expanded_test_prompts_containing": type_prompts.get(t, 0),
                     "test_groups": "|".join(g for g in test if t in tset[g])})
    TF = pd.DataFrame(rows)
    prow = []
    for a, b in combinations(TYPES, 2):
        gs = [g for g in test if a in tset[g] and b in tset[g]]
        inst = {(next(r["exact_rule_hash"] for r in pools[g]["rules"] if r["type_id"] == a),
                 next(r["exact_rule_hash"] for r in pools[g]["rules"] if r["type_id"] == b)) for g in gs}
        prow.append({"type_a": a, "type_b": b, "can_co_occur": not conflicts(a, b), "n_test_groups": len(gs),
                     "n_train_groups": sum(a in tset[g] and b in tset[g] for g in train),
                     "test_exact_instance_pairs": len(inst), "test_tasks": 3 * len(gs),
                     "expanded_test_prompts_with_both": pair_prompts.get((a, b), 0), "test_groups": "|".join(gs)})
    TP = pd.DataFrame(prow)
    # sampler validation: simulated vs observed type inclusion frequency over all 72 frozen pools
    rng = random.Random(seed)
    sims = [sample_types(rng) for _ in range(reps)]
    sim_freq = Counter(t for s in sims for t in s)
    obs_freq = Counter(t for g in pools for t in tset[g])
    VAL = pd.DataFrame([{"type_id": t, "observed_share_of_72_pools": round(obs_freq[t] / len(pools), 4),
                         "simulated_share": round(sim_freq[t] / reps, 4)} for t in TYPES])
    # projection: current 15 test pools + E simulated new pools
    feas_pairs = [(a, b) for a, b in combinations(TYPES, 2) if not conflicts(a, b)]
    proj = []
    cur_sets = [set(tset[g]) for g in test]
    prng = np.random.default_rng(seed + 1)
    for E in (0, 5, 10, 15, 20, 30, 45):
        mins, n_lt3, n_lt5, p_ge2, p_ge3, p_zero = [], [], [], [], [], []
        for r in range(500):
            extra = [set(sims[i]) for i in prng.integers(0, reps, E)]
            allp = cur_sets + extra
            tc = np.array([sum(t in s for s in allp) for t in TYPES])
            pc = np.array([sum(a in s and b in s for s in allp) for a, b in feas_pairs])
            mins.append(tc.min()); n_lt3.append((tc < 3).sum()); n_lt5.append((tc < 5).sum())
            p_ge2.append((pc >= 2).mean()); p_ge3.append((pc >= 3).mean()); p_zero.append((pc == 0).sum())
        proj.append({"extra_pools": E, "total_test_pools": 15 + E,
                     "min_groups_per_type_median": float(np.median(mins)),
                     "p_every_type_in_ge5_groups": round(float(np.mean(np.array(mins) >= 5)), 3),
                     "types_in_lt3_groups_mean": round(float(np.mean(n_lt3)), 2),
                     "types_in_lt5_groups_mean": round(float(np.mean(n_lt5)), 2),
                     "feasible_pairs_unobserved_mean": round(float(np.mean(p_zero)), 1),
                     "share_feasible_pairs_ge2_groups": round(float(np.mean(p_ge2)), 3),
                     "share_feasible_pairs_ge3_groups": round(float(np.mean(p_ge3)), 3)})
    return {"types": TF, "pairs": TP, "sampler_validation": VAL, "projection": pd.DataFrame(proj),
            "n_feasible_pairs": len(feas_pairs)}


# ================================================================================================== AUDIT 3
def single_prompt_table() -> pd.DataFrame:
    rows = []
    for p in (0.05, 0.10, 0.20, 0.50):
        for n in (20, 40, 60, 100):
            se = math.sqrt(p * (1 - p) / n)
            v = np.arange(n + 1); pmf = stats.binom.pmf(v, n, p)
            lo, hi = stats.binom.ppf(0.025, n, p) / n, stats.binom.ppf(0.975, n, p) / n
            rows.append({"true_p": p, "n_responses": n, "se_pp": round(100 * se, 2), "pm_1.96se_pp": round(196 * se, 1),
                         "central95_observed_rate_lo": lo, "central95_observed_rate_hi": hi,
                         "p_observe_zero_violations": round(float(pmf[0]), 4),
                         "resolution_pp(one response)": round(100 / n, 1)})
    return pd.DataFrame(rows)


def agresti_caffo_reliable(v1, n1, v2, n2):
    p1, p2 = (v1 + 1) / (n1 + 2), (v2 + 1) / (n2 + 2)
    se = np.sqrt(p1 * (1 - p1) / (n1 + 2) + p2 * (1 - p2) / (n2 + 2))
    d = p2 - p1
    return (d - 1.959963984540054 * se > 0) | (d + 1.959963984540054 * se < 0)


def matched_diff_table() -> pd.DataFrame:
    """Exact (binomial enumeration) properties of one matched pair, using the project's Agresti-Caffo 'reliable' rule
    (src/mife/probing/controlled.py)."""
    rows = []
    for p0 in (0.10, 0.18):
        for d in (0.025, 0.05, 0.10, 0.20):
            p1 = p0 + d
            for n in (20, 40, 60, 100, 200):
                v = np.arange(n + 1)
                a, b = stats.binom.pmf(v, n, p0), stats.binom.pmf(v, n, p1)
                J = np.outer(a, b)
                V1, V2 = np.meshgrid(v, v, indexing="ij")
                rel = agresti_caffo_reliable(V1, n, V2, n, )
                right = rel & (V2 > V1)
                sign_ok = (V2 > V1)
                se = math.sqrt((p0 * (1 - p0) + p1 * (1 - p1)) / n)
                rows.append({"baseline_p": p0, "true_diff_pp": 100 * d, "n_per_prompt": n,
                             "se_diff_pp": round(100 * se, 2), "snr(diff/se)": round(d / se, 2),
                             "p_observed_sign_correct(strict)": round(float((J * sign_ok).sum()), 3),
                             "p_observed_tie": round(float((J * (V1 == V2)).sum()), 3),
                             "p_reliable_and_correct(AgrestiCaffo)": round(float((J * right).sum()), 3),
                             "p_reliable_wrong_sign": round(float((J * (rel & (V2 < V1))).sum()), 4)})
    return pd.DataFrame(rows)


def contrast_weights(G: dict, group_tasks: dict, rtype: dict) -> dict:
    """For each factor contrast, per group: weights over (task, config, target) cells, normalised so the contrast is an
    average effect. Returns {name: list of (g, {prompt_key: {target: w}})}. Tasks are identical in design, so weights
    are computed for one task and the task dimension is applied analytically (mean over 3 tasks)."""
    out = defaultdict(list)
    for g, cfgs in G.items():
        rules = sorted({r for c in cfgs for r in c["order"]})
        S = {c["order"] for c in cfgs}
        for k in range(2, K_MAX + 1):
            # A: last minus first position, averaged over (subset, target)
            w = defaultdict(lambda: defaultdict(float))
            by_sub = defaultdict(list)
            for c in cfgs:
                if c["k"] == k:
                    by_sub[c["subset"]].append(c["order"])
            ncell = sum(len(s) for s in by_sub)
            for s, orders in by_sub.items():
                for t in s:
                    first = [o for o in orders if o[0] == t]; last = [o for o in orders if o[-1] == t]
                    for o in last:
                        w[o][t] += 1 / (len(last) * ncell)
                    for o in first:
                        w[o][t] -= 1 / (len(first) * ncell)
            out[f"A_k{k}_last_minus_first"].append((g, w))
            # A per rule (type-specific): rule r only; averaged over rules -> handled in variance step via subsetting
            for r in rules:
                wr = defaultdict(lambda: defaultdict(float))
                cells = [(s, orders) for s, orders in by_sub.items() if r in s]
                for s, orders in cells:
                    first = [o for o in orders if o[0] == r]; last = [o for o in orders if o[-1] == r]
                    for o in last:
                        wr[o][r] += 1 / (len(last) * len(cells))
                    for o in first:
                        wr[o][r] -= 1 / (len(first) * len(cells))
                out[f"A_k{k}_last_minus_first_single_rule"].append((g, wr))
        for k in range(1, K_MAX):
            pairs = [(X, X + (r,)) for X in S if len(X) == k for r in rules if r not in X and X + (r,) in S]
            n = len(pairs) * k
            w = defaultdict(lambda: defaultdict(float))
            for X, Y in pairs:
                for t in X:
                    w[Y][t] += 1 / n; w[X][t] -= 1 / n
            out[f"B1_{k}to{k + 1}_strict_append"].append((g, w))
            for r0 in rules:
                pr = [(X, Y) for X, Y in pairs if r0 in X]
                wr = defaultdict(lambda: defaultdict(float))
                for X, Y in pr:
                    wr[Y][r0] += 1 / len(pr); wr[X][r0] -= 1 / len(pr)
                out[f"B1_{k}to{k + 1}_strict_append_single_rule"].append((g, wr))
            # B2: marginal mean at k+1 minus k, target at fixed position p<=k, averaged over (target, p)
            w = defaultdict(lambda: defaultdict(float))
            cells = [(t, p) for t in rules for p in range(1, k + 1)]
            for t, p in cells:
                lo = [o for o in S if len(o) == k and o[p - 1] == t]
                hi = [o for o in S if len(o) == k + 1 and o[p - 1] == t]
                for o in hi:
                    w[o][t] += 1 / (len(hi) * len(cells))
                for o in lo:
                    w[o][t] -= 1 / (len(lo) * len(cells))
            out[f"B2_{k}to{k + 1}_marginal"].append((g, w))
        for k in range(2, K_MAX + 1):
            # C marginal, single companion c: mean(t at p, c present) - mean(t at p, c absent), averaged over (t, p)
            ks = [o for o in S if len(o) == k]
            for cpn in rules:
                w = defaultdict(lambda: defaultdict(float))
                cells = [(t, p) for t in rules if t != cpn for p in range(1, k + 1)]
                for t, p in cells:
                    at = [o for o in ks if o[p - 1] == t]
                    pr = [o for o in at if cpn in o]; ab = [o for o in at if cpn not in o]
                    if not pr or not ab:
                        continue
                    for o in pr:
                        w[o][t] += 1 / (len(pr) * len(cells))
                    for o in ab:
                        w[o][t] -= 1 / (len(ab) * len(cells))
                out[f"C_k{k}_marginal_single_companion"].append((g, w))
        # D: task b minus task a over all configs and targets (one task pair)
        w = {o: {t: 1.0 / sum(len(x) for x in S) for t in o} for o in S}
        out["D_task_pair_all_configs"].append((g, w))
    return out


def variance_terms(wlist):
    """Mean over groups of (sum_w2, sum over prompts of (sum_t w)^2) for ONE task's weights."""
    s2, sp = [], []
    for _, w in wlist:
        a = b = 0.0
        for o, tw in w.items():
            vals = list(tw.values())
            a += sum(x * x for x in vals); b += sum(vals) ** 2
        s2.append(a); sp.append(b)
    return float(np.mean(s2)), float(np.mean(sp))


def aggregate_precision(W: dict, TF: pd.DataFrame) -> pd.DataFrame:
    """SE of an across-group average effect, analytic, under explicit scenario assumptions (see explainer).
    Model per (prompt j, target t): observed rate = pi_jt + e_jt, Var(e)=pq/n, corr(e_jt, e_jt') = rho (same response);
    pi_jt contains an idiosyncratic deviation u_jt (SD sigma_u) that matching cannot remove; the group-level effect
    varies across groups with SD tau. Contrasts that compare two tasks have weights +w on task b and -w on task a.
    Within a group, factor A/B/C estimates average the same contrast over 3 tasks."""
    rows = []
    n_types = TF.set_index("type_id")["n_test_groups"]
    G_type_median = float(n_types[n_types > 0].median()); G_type_min = int(n_types[n_types > 0].min())
    for name, wl in W.items():
        s2, sp = variance_terms(wl)
        task_div = 3.0 if not name.startswith("D_") else 0.5  # D: (task b - task a) doubles the weight mass
        single = "single" in name
        for p in (0.10, 0.18):
            for rho in (0.0, 0.3):
                for sig_u in (0.0, 0.05, 0.10):
                    for n in (20, 40, 60, 100):
                        gen = p * (1 - p) / n * ((1 - rho) * s2 + rho * sp)
                        idio = sig_u ** 2 * s2
                        within = (gen + idio) / task_div
                        for tau in (0.0, 0.01, 0.02, 0.04):
                            for Glab, G in (("15 groups", 15), ("30 groups", 30), ("45 groups", 45),
                                            ("type: median groups", G_type_median), ("type: min groups", G_type_min)):
                                if Glab.startswith("type") and not single and not name.startswith("C_"):
                                    continue
                                if name.startswith("D_") and Glab.startswith("type"):
                                    continue
                                se = math.sqrt((tau ** 2 + within) / G)
                                rows.append({"contrast": name, "baseline_p": p, "rho_within_response": rho,
                                             "sigma_u_pp": 100 * sig_u, "tau_between_group_pp": 100 * tau,
                                             "n_per_prompt": n, "groups": Glab, "G": G,
                                             "within_group_se_pp": round(100 * math.sqrt(within), 3),
                                             "generation_noise_se_within_group_pp": round(100 * math.sqrt(gen / task_div), 3),
                                             "se_pp": round(100 * se, 3), "ci95_halfwidth_pp(t_{G-1})": round(100 * se * stats.t.ppf(0.975, max(G - 1, 1)), 3),
                                             "share_var_from_generation_noise": round(gen / task_div / (tau ** 2 + within), 3) if (tau ** 2 + within) > 0 else float("nan")})
    return pd.DataFrame(rows)


def mc_check(W: dict, reps: int, seed: int) -> pd.DataFrame:
    """Monte Carlo check of the analytic generation-noise SE for one contrast (rho = 0, sigma_u = 0, tau = 0)."""
    rng = np.random.default_rng(seed)
    rows = []
    for name in ("B1_3to4_strict_append", "A_k5_last_minus_first"):
        g, w = W[name][0]
        cells = [(o, t, x) for o, tw in w.items() for t, x in tw.items()]
        wv = np.array([x for _, _, x in cells])
        for p, n in ((0.18, 20), (0.10, 20)):
            est = (rng.binomial(n, p, size=(reps, len(cells))) / n) @ wv
            s2 = float((wv ** 2).sum())
            rows.append({"contrast": name, "group": g, "p": p, "n": n, "mc_sd_pp(one task)": round(100 * est.std(), 3),
                         "analytic_sd_pp(one task)": round(100 * math.sqrt(p * (1 - p) / n * s2), 3), "reps": reps})
    return pd.DataFrame(rows)


def bootstrap_coverage(reps: int, seed: int) -> pd.DataFrame:
    """Coverage of the paired group bootstrap (percentile) and t(G-1) intervals for a mean of G group-level effects."""
    rng = np.random.default_rng(seed)
    rows = []
    for G in (15, 30, 45):
        for dist in ("normal", "t3 (heavy tails)", "skewed (lognormal)"):
            cov_b = cov_t = 0; widths = []
            for _ in range(reps):
                if dist == "normal":
                    x = rng.normal(0, 1, G)
                elif dist.startswith("t3"):
                    x = rng.standard_t(3, G)
                else:
                    z = rng.lognormal(0, 1, G); x = z - math.exp(0.5)
                m = x.mean()
                bs = rng.choice(x, size=(1000, G), replace=True).mean(1)
                lo, hi = np.quantile(bs, [0.025, 0.975])
                cov_b += lo <= 0 <= hi
                se = x.std(ddof=1) / math.sqrt(G); q = stats.t.ppf(0.975, G - 1)
                cov_t += m - q * se <= 0 <= m + q * se
                widths.append((hi - lo) / (2 * q * se))
            rows.append({"groups": G, "group_effect_distribution": dist, "reps": reps,
                         "coverage_percentile_bootstrap": round(cov_b / reps, 3), "coverage_t_interval": round(cov_t / reps, 3),
                         "bootstrap_width_relative_to_t": round(float(np.mean(widths)), 3)})
    return pd.DataFrame(rows)


# ================================================================================================== AUDIT 5
def augmentation(G: dict, group_tasks: dict) -> tuple[dict, pd.DataFrame]:
    """Option B variants. B-prefix: add every missing 4-prefix of an expanded k=5 configuration (makes 4->5 strict
    appends available for every k=5 configuration). B-S4: full 24 orderings for every 4-subset."""
    variants = {}
    rows = []
    for name in ("expanded", "B_prefix", "B_S4"):
        Gv = {}
        for g, cfgs in G.items():
            S = [dict(c) for c in cfgs]
            have = {c["order"] for c in S}
            if name == "B_prefix":
                for c in cfgs:
                    if c["k"] == 5 and c["order"][:4] not in have:
                        have.add(c["order"][:4])
                        S.append({"k": 4, "subset": frozenset(c["order"][:4]), "order": c["order"][:4], "added": True})
            if name == "B_S4":
                for s in {c["subset"] for c in cfgs if c["k"] == 4}:
                    for o in permutations(sorted(s)):
                        if o not in have:
                            have.add(o); S.append({"k": 4, "subset": s, "order": o, "added": True})
            Gv[g] = S
        variants[name] = Gv
        added = sum(1 for g in Gv for c in Gv[g] if c.get("added"))
        # recompute strict append and replacement coverage, and k=4 position balance
        a34 = a45 = 0; c4 = c5 = 0; bal = []
        for g, S in Gv.items():
            st = {c["order"] for c in S}
            rules = sorted({r for c in S for r in c["order"]})
            a34 += sum(1 for X in st if len(X) == 3 for r in rules if r not in X and X + (r,) in st)
            a45 += sum(1 for X in st if len(X) == 4 for r in rules if r not in X and X + (r,) in st)
            for k, acc in ((4, "c4"), (5, "c5")):
                sk = {o for o in st if len(o) == k}
                n = sum(1 for X in sk for j in range(k) for r in rules if r not in X and X < X[:j] + (r,) + X[j + 1:]
                        and X[:j] + (r,) + X[j + 1:] in sk)
                if k == 4:
                    c4 += n
                else:
                    c5 += n
            for s in {c["subset"] for c in S if c["k"] == 4}:
                rows4 = [c["order"] for c in S if c["k"] == 4 and c["subset"] == s]
                cnt = Counter((o[p], p) for o in rows4 for p in range(4))
                v = np.array([cnt.get((t, p), 0) for t in s for p in range(4)], float)
                bal.append((v.min(), v.max(), len(rows4)))
        bal = np.array(bal)
        cvs = []
        for g, S in Gv.items():
            st = {c["order"] for c in S}
            rules = sorted({r for c in S for r in c["order"]})
            cnt = Counter((t, p) for X in st if len(X) == 4 for r in rules if r not in X and X + (r,) in st
                          for p, t in enumerate(X, start=1))
            v = np.array([cnt.get((t, p), 0) for t in rules for p in range(1, 5)], float)
            cvs.append(v.std() / v.mean())
        rows.append({"variant": name, "append_4to5_target_rule_x_position_cv": round(float(np.max(cvs)), 4), "added_k4_orderings_per_group": added / len(Gv),
                     "added_prompts_total": added * 3, "added_generations_both_llms_at_20": added * 3 * N_RESP * N_MODELS,
                     "strict_append_3to4_ordered_pairs_per_group": a34 / len(Gv),
                     "strict_append_4to5_ordered_pairs_per_group": a45 / len(Gv),
                     "strict_replacement_k4_pairs_per_group": c4 / len(Gv),
                     "strict_replacement_k5_pairs_per_group": c5 / len(Gv),
                     "k4_orderings_per_subset_min": int(bal[:, 2].min()), "k4_orderings_per_subset_max": int(bal[:, 2].max()),
                     "k4_rule_x_position_count_min": int(bal[:, 0].min()), "k4_rule_x_position_count_max": int(bal[:, 1].max()),
                     "k4_subsets_position_balanced": int(sum(1 for r in bal if r[0] == r[1]))})
    return variants, pd.DataFrame(rows)


def budget_table(n_cfg_exp: int, n_cfg_cur: int, aug: pd.DataFrame) -> pd.DataFrame:
    rows = []
    def add(option, desc, prompts_total, n, reusable_gen, notes=""):
        gens = prompts_total * n * N_MODELS
        rows.append({"option": option, "description": desc, "unique_test_prompts": prompts_total,
                     "responses_per_prompt": n, "total_generations_both_llms": gens,
                     "already_generated_reusable": reusable_gen, "new_generations_needed": gens - reusable_gen,
                     "notes": notes})
    cur_prompts, exp_prompts = n_cfg_cur * 3, n_cfg_exp * 3  # configs are summed over the 15 groups; x3 tasks
    cur_gen = cur_prompts * N_RESP * N_MODELS
    add("current", "frozen 246-config design (already generated, sealed)", cur_prompts, 20, cur_gen)
    for n in (20, 40, 60, 100):
        add("A" if n == 20 else "D", f"expanded 456-config design, {n} responses/prompt", exp_prompts, n, cur_gen,
            "reuse assumes the same backend and sampler (Betty, vLLM 0.25.0, BF16)")
    b = aug.set_index("variant")
    for v, label in (("B_prefix", "expanded + missing 4-prefixes of k=5 configs"), ("B_S4", "expanded + all 24 orderings of every 4-subset")):
        add("B", label, exp_prompts + int(b.loc[v, "added_prompts_total"]), 20, cur_gen)
    per_pool_exp, per_pool_cur = n_cfg_exp // 15 * 3, n_cfg_cur // 15 * 3
    for E in (5, 10, 15, 30):
        add("C", f"expanded design + {E} new rule pools on re-tripled TEST tasks (456 configs each)",
            exp_prompts + E * per_pool_exp, 20, cur_gen, "changes the held-out protocol: needs explicit approval")
        add("C-lite", f"expanded design + {E} new pools with the 246-config design",
            exp_prompts + E * per_pool_cur, 20, cur_gen, "changes the held-out protocol: needs explicit approval")
        add("C-only", f"current 246 design + {E} new pools with the 246-config design",
            cur_prompts + E * per_pool_cur, 20, cur_gen, "changes the held-out protocol: needs explicit approval")
    add("E", "expanded + 15 new pools (456 configs each) on re-tripled TEST tasks + missing 4-prefixes in all 30 pools",
        exp_prompts + 15 * per_pool_exp + 2 * int(b.loc["B_prefix", "added_prompts_total"]), 20, cur_gen,
        "changes the held-out protocol: needs explicit approval")
    return pd.DataFrame(rows)


# ===================================================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-dir", default=str(ROOT / "data" / "manifest"))
    ap.add_argument("--out", default=str(ROOT / "results" / "audits" / "evaluation_expansion"))
    ap.add_argument("--mc-reps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20261122)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    P, R, pools, tasks, sha = load(Path(a.manifest_dir))
    test_groups = sorted(g for g, p in pools.items() if p["split"] == "test")
    assert len(test_groups) == 15 and sorted(P[P.split == "test"].group_id.unique()) == test_groups
    rtype = {r["rule_id"]: r["type_id"] for p in pools.values() for r in p["rules"]}
    text_len = {r["rule_id"]: len(r["rule_text"]) for p in pools.values() for r in p["rules"]}
    group_tasks = {g: sorted(pools[g]["task_keys"]) for g in test_groups}
    P_test = P[P.split == "test"]

    # ---- current design (as frozen) and expanded design
    existing = {}
    for g in test_groups:
        cur = P_test[(P_test.group_id == g)].drop_duplicates("config_id")
        existing[g] = [{"k": int(r.k), "subset": frozenset(r.ordered_rule_ids), "order": tuple(r.ordered_rule_ids)}
                       for r in cur.itertuples()]
        ref = {tuple(c["ordered_rule_labels"]) for c in design.configurations(pools[g]["chain_order"])}
        assert {c["order"] for c in existing[g]} == ref, f"{g}: manifest configs differ from design.configurations"
    G = {g: expanded_configs(pools[g]) for g in test_groups}
    checks = []
    for g in test_groups:
        per_k = Counter(c["k"] for c in G[g]); subs = Counter(len(c["subset"]) for c in {c["subset"]: c for c in G[g]}.values())
        orders = [c["order"] for c in G[g]]
        cur_set = {c["order"] for c in existing[g]}
        reuse = sum(o in cur_set for o in orders)
        for k in range(1, 6):
            for s in {c["subset"] for c in G[g] if c["k"] == k}:
                rows = [design.coa_template(k)[c["coa_row"]] for c in G[g] if c["subset"] == s]
                design.check_coa(rows, k)
        checks.append({"group_id": g, **{f"configs_k{k}": per_k[k] for k in range(1, 6)},
                       **{f"subsets_k{k}": subs[k] for k in range(1, 6)}, "configs_total": len(orders),
                       "unique_orders": len(set(orders)), "existing_reused": reuse,
                       "naive_chain_label_base_would_reuse": sum(naive_base_reuse(pools[g], cur_set).values())})
    C = pd.DataFrame(checks)
    assert (C[[f"configs_k{k}" for k in range(1, 6)]].values == [6, 30, 120, 180, 120]).all()
    assert (C[[f"subsets_k{k}" for k in range(1, 6)]].values == [6, 15, 20, 15, 6]).all()
    assert (C.configs_total == 456).all() and (C.unique_orders == 456).all() and (C.existing_reused == 246).all()
    n_exp, n_cur = int(C.configs_total.sum()), 246 * 15
    assert n_exp * 3 == 20520 and n_exp * 3 * N_RESP == 410400 and n_exp * 3 * N_RESP * N_MODELS == 820800
    assert n_cur * 3 * N_RESP * N_MODELS == 442800 == len(P_test) * N_RESP * N_MODELS
    C.to_csv(out / "design_validation_by_group.csv", index=False)
    pd.DataFrame([{"k": k, "rule_subsets": math.comb(6, k), "coa_orderings_per_subset": len(design.coa_template(k)),
                   "ordered_configs_per_task": math.comb(6, k) * len(design.coa_template(k)),
                   "current_configs_per_task": 6 * len(design.coa_template(k)),
                   "new_configs_per_task": (math.comb(6, k) - 6) * len(design.coa_template(k)),
                   "prompt_rule_instances_per_task": math.comb(6, k) * len(design.coa_template(k)) * k}
                  for k in range(1, 6)]).to_csv(out / "design_counts_by_k.csv", index=False)
    exp_rows = [{"group_id": g, "k": c["k"], "subset_status": c["subset_status"], "chain_index": c["chain_index"],
                 "coa_row": c["coa_row"], "ordered_rule_ids": "|".join(c["order"]),
                 "ordered_type_ids": "|".join(rtype[r] for r in c["order"]),
                 "already_generated": c["order"] in {e["order"] for e in existing[g]}} for g in test_groups for c in G[g]]
    pd.DataFrame(exp_rows).to_csv(out / "expanded_manifest_configs.csv", index=False)

    # ---- Audit 1
    A_cov, A_pairs, A_comp = factor_a(G, rtype, text_len, {}, group_tasks)
    A_cov.to_csv(out / "A_position_coverage.csv", index=False)
    A_pairs.to_csv(out / "A_position_comparisons.csv", index=False)
    A_comp.to_csv(out / "A_companion_position_balance.csv", index=False)
    B1 = factor_b1(G, rtype, group_tasks, existing)
    for key, df in B1.items():
        df.to_csv(out / f"B1_strict_append_{key}.csv", index=False)
    B2 = factor_b2(G, group_tasks); B2.to_csv(out / "B2_marginal_k_coverage.csv", index=False)
    Cc = factor_c(G, rtype, group_tasks, existing)
    for key, df in Cc.items():
        df.to_csv(out / f"C_strict_replacement_{key}.csv", index=False)
    Cm = factor_c_marginal(G); Cm.to_csv(out / "C_marginal_composition_coverage.csv", index=False)
    D, Dsum = factor_d(G, pools, P_test, tasks, group_tasks)
    D.to_csv(out / "D_task_matching_by_group.csv", index=False); Dsum.to_csv(out / "D_task_matching_summary.csv", index=False)

    # ---- Audit 2
    A2 = audit2(pools, G, rtype, group_tasks, reps=a.mc_reps * 5, seed=a.seed)
    A2["types"].to_csv(out / "rule_type_frequencies.csv", index=False)
    A2["pairs"].to_csv(out / "rule_type_pair_frequencies.csv", index=False)
    A2["sampler_validation"].to_csv(out / "pool_sampler_validation.csv", index=False)
    A2["projection"].to_csv(out / "added_pool_coverage_projection.csv", index=False)

    # ---- Audit 3
    single_prompt_table().to_csv(out / "precision_single_prompt.csv", index=False)
    matched_diff_table().to_csv(out / "precision_matched_pair.csv", index=False)
    W = contrast_weights(G, group_tasks, rtype)
    aggregate_precision(W, A2["types"]).to_csv(out / "precision_aggregate_scenarios.csv", index=False)
    mc_check(W, a.mc_reps, a.seed).to_csv(out / "precision_mc_check.csv", index=False)
    bootstrap_coverage(2000, a.seed).to_csv(out / "precision_group_bootstrap_coverage.csv", index=False)

    # ---- Audit 5
    _, aug = augmentation(G, group_tasks)
    aug.to_csv(out / "option_B_ordering_augmentation.csv", index=False)
    budget_table(n_exp, n_cur, aug).to_csv(out / "generation_budget_scenarios.csv", index=False)

    # ---- TRAIN-only empirical anchors (committed tables)
    T = ROOT / "results" / "tables"
    pos = pd.read_csv(T / "desc_by_position.csv"); kk = pd.read_csv(T / "desc_by_k.csv")
    anchors = []
    for m in ("llama", "qwen"):
        km = kk[(kk.model == m) & (kk.k != "all")].copy(); km["k"] = km.k.astype(int); km = km.set_index("k").est
        for k in range(1, 5):
            anchors.append({"model": m, "quantity": f"marginal rate k={k + 1} minus k={k} (all targets)", "train_pp": round(100 * (km[k + 1] - km[k]), 2)})
        pm = pos[pos.model == m]
        for k in range(2, 6):
            s = pm[pm.k == k].set_index("position").est
            anchors.append({"model": m, "quantity": f"k={k}: last minus first position", "train_pp": round(100 * (s[k] - s[1]), 2)})
        anchors.append({"model": m, "quantity": "overall violation rate", "train_pp": round(100 * float(kk[(kk.model == m) & (kk.k == "all")].est.iloc[0]), 2)})
    pd.DataFrame(anchors).to_csv(out / "train_only_effect_anchors.csv", index=False)

    meta = {"audit": "evaluation_expansion", "manifest_prompts_sha256": sha, "matches_frozen_manifest": sha == FROZEN_PROMPTS_SHA256,
            "test_groups": test_groups, "seed": a.seed, "mc_reps": a.mc_reps,
            "inputs_read": ["data/manifest/prompts.parquet (metadata columns)", "data/manifest/prompt_rules.parquet (metadata columns)",
                            "data/manifest/rule_pools.jsonl", "data/manifest/tasks.parquet", "results/tables/desc_by_position.csv",
                            "results/tables/desc_by_k.csv"],
            "not_read": ["generations", "verification labels", "features", "predictions", "any TEST outcome"]}
    (out / "audit_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
