"""Task-family loading, three-task grouping of all 216 tasks, and the group split."""
from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict

from . import source as S
from .config import ROOT

FAMILY_FILE = ROOT / "configs" / "task_families_v1.csv"
MIN_FAMILY_SIZE = 12  # sparse-family merge threshold (docs/task_family_rubric.md)

RELATED_CLUSTERS = {
    "japan": [1001, 1092],
    "layton": [3245, 3478],
    "us_president_persona": [1258, 2304, 3204, 2471],
    "internet_history": [122, 179],
    "sleep": [1072, 209],
    "bike_for_kids": [1580, 301],
    "google": [1551, 2985],
    "us_elections": [2441, 2097],
}


def load_families() -> dict[int, dict]:
    with open(FAMILY_FILE) as f:
        rows = {int(r["task_key"]): r for r in csv.DictReader(f)}
    keys = set(S.tasks())
    if set(rows) != keys:
        raise AssertionError(f"family file mismatch: missing {keys - set(rows)}, extra {set(rows) - keys}")
    counts = Counter(r["family"] for r in rows.values())
    small = [f for f, c in counts.items() if c < MIN_FAMILY_SIZE]
    if small:
        raise AssertionError(f"sparse families need a predeclared merge: {small}")
    return {k: {"family": r["family"], "ambiguous": r["ambiguous"] == "1",
                "alt_family": r["alt_family"] or None, "note": r["note"]} for k, r in rows.items()}


def form_groups(task_keys: list[int], families: dict[int, dict], seed: int, size: int = 3) -> list[list[int]]:
    """Greedy max-diversity grouping: each group draws from the `size` families with the most
    remaining tasks (random tie-break), so same-family groups occur only when unavoidable."""
    rng = random.Random(seed)
    pools = defaultdict(list)
    for k in task_keys:
        pools[families[k]["family"]].append(k)
    for f in pools:
        rng.shuffle(pools[f])
    groups = []
    while any(pools.values()):
        fams = [f for f in pools if pools[f]]
        rng.shuffle(fams)
        fams.sort(key=lambda f: -len(pools[f]))
        g = []
        i = 0
        while len(g) < size:
            f = fams[i % len(fams)] if len(fams) >= size else max(
                (x for x in pools if pools[x]), key=lambda x: len(pools[x]))
            g.append(pools[f].pop())
            i += 1
        groups.append(sorted(g))
    assert sorted(t for g in groups for t in g) == sorted(task_keys)
    return groups


def split_groups(group_meta: list[dict], n_test: int, seed: int, n_candidates: int = 20000) -> tuple[set[str], dict]:
    """Choose test groups by seeded random search minimizing a pre-generation balance score.

    group_meta items: {group_id, task_keys, families, types}. Related-task clusters are kept
    on one side by treating cluster-linked groups as indivisible components.
    """
    gid_of_task = {t: g["group_id"] for g in group_meta for t in g["task_keys"]}
    parent = {g["group_id"]: g["group_id"] for g in group_meta}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for members in RELATED_CLUSTERS.values():
        gids = [gid_of_task[t] for t in members if t in gid_of_task]
        for a in gids[1:]:
            parent[find(a)] = find(gids[0])
    comps = defaultdict(list)
    for g in group_meta:
        comps[find(g["group_id"])].append(g["group_id"])
    comp_list = sorted(comps.values(), key=lambda c: c[0])
    meta = {g["group_id"]: g for g in group_meta}

    fam_all = Counter(f for g in group_meta for f in g["families"])
    type_all = Counter(t for g in group_meta for t in g["types"])
    n_groups = len(group_meta)

    def score(test: list[str]):
        fam_t = Counter(f for gid in test for f in meta[gid]["families"])
        type_t = Counter(t for gid in test for t in meta[gid]["types"])
        type_tr = {t: type_all[t] - type_t[t] for t in type_all}
        if any(type_t[t] < 1 for t in S.TYPES) or any(type_tr.get(t, 0) < 5 for t in S.TYPES):
            return None
        f_dev = sum(abs(fam_t[f] / (3 * len(test)) - c / (3 * n_groups)) for f, c in fam_all.items())
        t_dev = sum(abs(type_t[t] / (6 * len(test)) - c / (6 * n_groups)) for t, c in type_all.items())
        return f_dev + t_dev

    rng = random.Random(seed)
    best = None
    for i in range(n_candidates):
        order = comp_list[:]
        rng.shuffle(order)
        test = []
        for c in order:
            if len(test) + len(c) <= n_test:
                test += c
            if len(test) == n_test:
                break
        if len(test) != n_test:
            continue
        s = score(test)
        if s is not None and (best is None or s < best[0] - 1e-12):
            best = (s, sorted(test), i)
    if best is None:
        raise RuntimeError("no admissible split found")
    return set(best[1]), {"score": best[0], "candidate_index": best[2], "n_candidates": n_candidates,
                          "components": comp_list}
