"""Build the immutable prompt manifest (tasks -> groups -> pools -> chains/COA -> prompts -> split)."""
from __future__ import annotations

import json
import random
import string
from collections import Counter
from itertools import combinations

import pandas as pd

from . import SCHEMA_VERSION, design, feasibility, pool_audit, rule_sampler, source as S, tasks as T
from .config import DATA_DIR, config_sha256, derive_seed, load_config
from .storage import sha256_text, write_json, write_jsonl, write_parquet

MANIFEST_DIR = DATA_DIR / "manifest"


def build(out_dir=MANIFEST_DIR) -> dict:
    cfg = load_config()
    d, seeds = cfg["design"], cfg["seeds"]
    repro = S.verify_source_reproduction()

    fams = T.load_families()
    all_task_keys = sorted(S.tasks())  # every ManyIFEval task
    assert len(all_task_keys) == d["n_tasks"] == d["n_groups"] * d["tasks_per_group"]
    groups = T.form_groups(all_task_keys, fams, seeds["task_grouping"], d["tasks_per_group"])
    assert len(groups) == d["n_groups"]

    pools, group_meta = {}, []
    for gi, task_keys in enumerate(groups, start=1):
        gid = f"g{gi:02d}"
        pool = rule_sampler.sample_pool(derive_seed(seeds["rule_sampling"], gid))
        rng = random.Random(derive_seed(seeds["chain_ordering"], gid))
        perm = list(range(6))
        rng.shuffle(perm)  # randomized chain ordering of the six sampled rules
        for i, r in enumerate(pool["rules"]):
            r["rule_id"] = f"{gid}-r{i + 1}"
            r["sampling_index"] = i + 1
        order = [pool["rules"][p]["rule_id"] for p in perm]
        for pos, rid in enumerate(order):
            next(r for r in pool["rules"] if r["rule_id"] == rid)["chain_label"] = string.ascii_uppercase[pos]
        pool.update(group_id=gid, task_keys=task_keys, chain_order=order)
        pools[gid] = pool
        group_meta.append({"group_id": gid, "task_keys": task_keys,
                           "families": [fams[t]["family"] for t in task_keys],
                           "types": [r["type_id"] for r in pool["rules"]]})

    # Task feasibility: each group's pool must be satisfiable for each of its three tasks (no relevance filter).
    reading = S.types_reading_task_text()
    task_checks = {}
    for gid, pool in pools.items():
        rules6 = [(r["type_id"], r["kwargs"]) for r in pool["rules"]]
        res = feasibility.check_tasks(rules6, pool["witness"], {k: S.tasks()[k]["task_text"] for k in pool["task_keys"]})
        bad = {k: v for k, v in res.items() if any(v)}
        assert not bad, f"{gid}: rule pool not satisfiable for task(s) {bad}"
        task_checks[gid] = sorted(res)
    task_feasibility = {
        "definition": "pool witness passes every official verifier with each task's text (no relevance filter)",
        "types_whose_verifier_reads_task_text": reading,
        "n_group_task_pairs_checked": sum(len(v) for v in task_checks.values()), "n_infeasible": 0}

    test_groups, split_info = T.split_groups(group_meta, d["n_test_groups"], seeds["split"])
    split_of = {g["group_id"]: ("test" if g["group_id"] in test_groups else "train") for g in group_meta}

    # COA templates are validated before any prompt is produced
    for k in range(1, d["k_max"] + 1):
        design.check_coa(design.coa_template(k), k)

    prompt_rows, rule_rows = [], []
    for gid, pool in pools.items():
        rmap = {r["rule_id"]: r for r in pool["rules"]}
        configs = design.configurations(pool["chain_order"], d["k_max"])
        _check_group_configs(configs, pool["chain_order"], d)
        for c in configs:
            subset_id = f"{gid}-c{c['chain_index']}-k{c['k']}"
            config_id = f"{subset_id}-o{c['coa_row']:02d}"
            texts = [rmap[r]["rule_text"] for r in c["ordered_rule_labels"]]
            for tkey in pool["task_keys"]:
                task_text = S.tasks()[tkey]["task_text"]
                prompt = S.serialize_prompt(task_text, texts)
                pid = f"{config_id}-t{tkey}"
                spans, cursor = [], len(task_text) + len(S.PROMPT_HEADER)
                for pos, (rid, txt) in enumerate(zip(c["ordered_rule_labels"], texts), start=1):
                    start = cursor + len(S.rule_line_prefix())
                    assert prompt[start:start + len(txt)] == txt
                    spans.append((start, start + len(txt)))
                    r = rmap[rid]
                    rule_rows.append({
                        "prompt_id": pid, "position": pos, "k": c["k"], "rule_id": rid,
                        "chain_label": r["chain_label"], "type_id": r["type_id"], "category": r["category"],
                        "code_prefix": r["code_prefix"], "kwargs_json": json.dumps(r["kwargs"], sort_keys=True),
                        "rule_text": txt, "char_start": start, "char_end": start + len(txt),
                        "exact_rule_hash": r["exact_rule_hash"], "group_id": gid, "split": split_of[gid],
                    })
                    cursor = start + len(txt) + 1  # newline
                prompt_rows.append({
                    "prompt_id": pid, "group_id": gid, "split": split_of[gid], "task_key": tkey,
                    "task_family": fams[tkey]["family"], "config_id": config_id, "subset_id": subset_id,
                    "subset_key": "|".join(sorted(c["subset_chain_order"])), "chain_index": c["chain_index"],
                    "k": c["k"], "coa_row": c["coa_row"], "coa_perm": c["coa_perm"],
                    "ordered_rule_ids": c["ordered_rule_labels"],
                    "ordered_type_ids": [rmap[r]["type_id"] for r in c["ordered_rule_labels"]],
                    "task_text": task_text, "task_char_span": [0, len(task_text)],
                    "prompt_text": prompt, "prompt_sha256": sha256_text(prompt), "n_chars": len(prompt),
                })

    prompts = pd.DataFrame(prompt_rows)
    rules = pd.DataFrame(rule_rows)
    audit = _audit(prompts, rules, pools, d, split_of)
    conformance = pool_audit.audit(pools)
    assert conformance["n_in_pool"] == conformance["n_exact_rules"], conformance["out_of_pool"]

    tasks_df = pd.DataFrame([{
        "task_key": k, "task_text": v["task_text"], "family": fams[k]["family"],
        "family_ambiguous": fams[k]["ambiguous"], "alt_family": fams[k]["alt_family"],
        "group_id": next((g["group_id"] for g in group_meta if k in g["task_keys"]), None),
    } for k, v in sorted(S.tasks().items())])
    tasks_df["split"] = tasks_df["group_id"].map(split_of)

    write_parquet(prompts, out_dir / "prompts.parquet")
    write_parquet(rules, out_dir / "prompt_rules.parquet")
    write_parquet(tasks_df, out_dir / "tasks.parquet")
    write_jsonl([{k: v for k, v in p.items()} | {"split": split_of[g]} for g, p in pools.items()],
                out_dir / "rule_pools.jsonl")
    summary = {
        "schema_version": SCHEMA_VERSION, "config_sha256": config_sha256(),
        "environment": S.environment_fingerprint(), "source_reproduction": repro,
        "family_counts": dict(Counter(v["family"] for v in fams.values())), "task_keys": all_task_keys,
        "groups": {g["group_id"]: g["task_keys"] for g in group_meta},
        "split": {"test_groups": sorted(test_groups), **{k: v for k, v in split_info.items() if k != "components"}},
        "split_components_multi": [c for c in split_info["components"] if len(c) > 1],
        "sampler_version": rule_sampler.SAMPLER_VERSION, "coa_version": design.COA_VERSION,
        "audit": audit,
        "task_feasibility": task_feasibility,
        "pool_conformance": {"n_exact_rules": conformance["n_exact_rules"], "n_in_pool": conformance["n_in_pool"],
                             "relation_split_all": conformance["relation_split_all"],
                             "d3_param_rejections": len(conformance["d3_rejected_draws"])},
        "manifest_sha256": sha256_text("".join(prompts["prompt_sha256"]) + "".join(prompts["prompt_id"])),
    }
    write_json(summary, out_dir / "manifest_summary.json")
    write_json(conformance, out_dir / "pool_audit.json")
    return summary


def _check_group_configs(configs, order, d):
    per_k = Counter(c["k"] for c in configs)
    assert dict(per_k) == {k: 6 * n for k, n in d["coa_orderings_per_k"].items()}, per_k
    assert len(configs) == 246
    keys = {tuple(c["ordered_rule_labels"]) for c in configs}
    assert len(keys) == 246, "ordered configurations must be unique"
    assert sum(c["k"] for c in configs) == 1026
    for k in range(1, 6):
        subsets = {frozenset(c["subset_chain_order"]) for c in configs if c["k"] == k}
        assert len(subsets) == 6, f"need 6 distinct subsets at k={k}"
        for r in order:  # each rule appears in exactly k subsets at level k
            assert sum(1 for s in subsets if r in s) == k
    # marginal and pairwise position balance within each exact subset
    for k in range(2, 6):
        for s in {frozenset(c["subset_chain_order"]) for c in configs if c["k"] == k}:
            rows = [c["ordered_rule_labels"] for c in configs if c["k"] == k and frozenset(c["subset_chain_order"]) == s]
            for r in s:
                for p in range(k):
                    assert sum(1 for row in rows if row[p] == r) == k - 1
            for a, b in combinations(range(k), 2):
                for p1, p2 in ((a, b), (b, a)):
                    pairs = Counter((row[p1], row[p2]) for row in rows)
                    assert len(pairs) == k * (k - 1) and set(pairs.values()) == {1}


def _audit(prompts, rules, pools, d, split_of) -> dict:
    n_groups = d["n_groups"]
    n_tr, n_te = d["n_train_groups"], d["n_test_groups"]
    assert n_tr + n_te == n_groups
    assert len(prompts) == n_groups * 3 * 246, len(prompts)
    assert len(rules) == n_groups * 3 * 1026, len(rules)
    assert prompts["prompt_id"].is_unique and prompts["prompt_sha256"].is_unique
    per_task = prompts.groupby("task_key").size()
    assert (per_task == 246).all() and len(per_task) == d["n_tasks"]
    rule_task = rules["prompt_id"].map(prompts.set_index("prompt_id")["task_key"])
    assert (rules.groupby(rule_task).size() == 1026).all()
    # same 246 ordered configurations across the group's three tasks
    for gid, grp in prompts.groupby("group_id"):
        sets = grp.groupby("task_key")["config_id"].apply(frozenset)
        assert len(set(sets)) == 1 and len(next(iter(sets))) == 246
    sp = prompts.groupby("split").agg(groups=("group_id", "nunique"), tasks=("task_key", "nunique"), prompts=("prompt_id", "size"))
    assert sp.loc["train"].tolist() == [n_tr, 3 * n_tr, 738 * n_tr], sp
    assert sp.loc["test"].tolist() == [n_te, 3 * n_te, 738 * n_te], sp
    assert prompts.groupby("task_key")["split"].nunique().max() == 1
    for members in T.RELATED_CLUSTERS.values():
        sides = set(prompts[prompts["task_key"].isin(members)]["split"])
        assert len(sides) <= 1, f"related cluster crosses split: {members}"
    # exposure audits
    pool_rules = [(p["group_id"], split_of[p["group_id"]], r) for p in pools.values() for r in p["rules"]]
    type_groups = Counter((s, r["type_id"]) for _, s, r in pool_rules)
    hashes_train = {r["exact_rule_hash"] for _, s, r in pool_rules if s == "train"}
    test_rules = [r for _, s, r in pool_rules if s == "test"]
    all_hashes = Counter(r["exact_rule_hash"] for _, _, r in pool_rules)
    return {
        "n_prompts": len(prompts), "n_prompt_rule_rows": len(rules),
        "split_counts": sp.reset_index().to_dict("records"),
        "type_group_counts": {f"{s}:{t}": c for (s, t), c in sorted(type_groups.items())},
        "category_slot_counts": dict(Counter(f"{s}:{r['category']}" for _, s, r in pool_rules)),
        "family_by_split": {f"{a}:{b}": int(n) for (a, b), n in
                            prompts.drop_duplicates("task_key").groupby(["split", "task_family"]).size().items()},
        "test_rule_instances_seen_in_train": sum(1 for r in test_rules if r["exact_rule_hash"] in hashes_train),
        "test_rule_instances": len(test_rules),
        "exact_rule_recurrences_any": sum(1 for c in all_hashes.values() if c > 1),
        "param_rejections": sum(1 for p in pools.values() for e in p["sampling_log"] if e["event"] == "params_rejected"),
        "type_rejections": sum(1 for p in pools.values() for e in p["sampling_log"] if e["event"].startswith("type_rejected")),
        "groups_with_3_distinct_families": int(sum(len(set(prompts[prompts.group_id == g].task_family)) == 3 for g in pools)),
        "n_chars_quantiles": prompts["n_chars"].quantile([0, 0.5, 0.9, 0.99, 1]).to_dict(),
    }

