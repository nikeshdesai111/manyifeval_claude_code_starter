"""Split amendment S3 (2026-11-23): a ~70/30 group split with test-side coverage of compatible type pairs.

The 2026-11-14 split (`tasks.split_groups`, 15 test groups) is left untouched. This module proposes a replacement
split that adds one hard check: every compatible pair of instruction types (a pair the sampler allows in one pool)
must co-occur, within one rule pool, in at least `min_pair_test_groups` TEST groups. Incompatible pairs (the
mutually exclusive groups in `source.EXCLUSIVE_GROUPS`) never share a pool and are exempt.

Why the search differs from `tasks.split_groups`: random fills almost never reach pair coverage 2 (0 of 20,000
draws at 22 groups), so rejection sampling cannot find an admissible split. Each seeded restart is instead
improved by size-preserving unit swaps that minimise (hard shortfall, balance score). The balance score and
the related-task units are exactly those of `tasks.split_groups`.

Only pool metadata (task families, the six rule types per group) is used. No generations, labels or features.
"""
from __future__ import annotations

import random
from collections import defaultdict
from itertools import combinations

import numpy as np

from . import source as S
from .tasks import RELATED_CLUSTERS

AMENDMENT_ID = "split-s3-2026-11-23"


def compatible_type_pairs(types=None) -> list[tuple[str, str]]:
    types = sorted(types or S.TYPES)
    return [(a, b) for a, b in combinations(types, 2) if not S.conflicts(a, b)]


def related_units(group_meta: list[dict]) -> list[list[str]]:
    """Groups linked by RELATED_CLUSTERS, merged transitively (same rule as tasks.split_groups)."""
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
    return sorted(comps.values(), key=lambda c: c[0])


class _Problem:
    def __init__(self, group_meta, n_test, min_pair, min_type_test, min_type_train, types=None, keep_related_units=True,
                 min_pair_train=0, exception_priority="test"):
        self.types = sorted(types or S.TYPES)
        self.pairs = compatible_type_pairs(self.types)
        self.fams = sorted({f for g in group_meta for f in g["families"]})
        self.gids = [g["group_id"] for g in group_meta]
        self.units = related_units(group_meta) if keep_related_units else [[g] for g in sorted(self.gids)]
        self.n_test, self.n_groups = n_test, len(group_meta)
        self.min_pair, self.min_type_test, self.min_type_train = min_pair, min_type_test, min_type_train
        self.min_pair_train = min_pair_train
        ti = {t: i for i, t in enumerate(self.types)}
        pi = {p: i for i, p in enumerate(self.pairs)}
        fi = {f: i for i, f in enumerate(self.fams)}
        meta = {g["group_id"]: g for g in group_meta}
        nu = len(self.units)
        self.U_type = np.zeros((nu, len(self.types)))
        self.U_pair = np.zeros((nu, len(self.pairs)))
        self.U_fam = np.zeros((nu, len(self.fams)))
        self.size = np.array([len(u) for u in self.units])
        for k, u in enumerate(self.units):
            for gid in u:
                g = meta[gid]
                for t in g["types"]:
                    self.U_type[k, ti[t]] += 1
                for a, b in combinations(sorted(g["types"]), 2):
                    if (a, b) in pi:
                        self.U_pair[k, pi[(a, b)]] += 1
                for f in g["families"]:
                    self.U_fam[k, fi[f]] += 1
        self.type_all = self.U_type.sum(0)
        self.pair_all = self.U_pair.sum(0)
        self.fam_all = self.U_fam.sum(0)
        # per-pair requirements; a pair too rare for both minima satisfies the prioritised side first
        if exception_priority == "test":
            self.req_test = np.minimum(min_pair, self.pair_all)
            self.req_train = np.minimum(min_pair_train, self.pair_all - self.req_test)
        elif exception_priority == "train":
            self.req_train = np.minimum(min_pair_train, self.pair_all)
            self.req_test = np.minimum(min_pair, self.pair_all - self.req_train)
        else:
            raise ValueError(exception_priority)

    def infeasible_pairs(self):
        return [(self.pairs[i], int(c)) for i, c in enumerate(self.pair_all) if c < self.min_pair]

    def exceptions(self):
        return [{"pair": "|".join(self.pairs[i]), "all_groups": int(self.pair_all[i]),
                 "required_test": int(self.req_test[i]), "required_train": int(self.req_train[i])}
                for i in np.where((self.req_test < self.min_pair) | (self.req_train < self.min_pair_train))[0]]

    def evaluate(self, typ, pair, fam):
        """Arrays may carry a leading batch axis. Returns (shortfall, score)."""
        short = (np.clip(self.req_test - pair, 0, None).sum(-1)
                 + np.clip(self.req_train - (self.pair_all - pair), 0, None).sum(-1)
                 + np.clip(self.min_type_test - typ, 0, None).sum(-1)
                 + np.clip(self.min_type_train - (self.type_all - typ), 0, None).sum(-1))
        # identical to tasks.split_groups: L1 family-share gap + L1 type-share gap (test vs all 72 groups)
        score = (np.abs(fam / (3 * self.n_test) - self.fam_all / (3 * self.n_groups)).sum(-1)
                 + np.abs(typ / (6 * self.n_test) - self.type_all / (6 * self.n_groups)).sum(-1))
        return short, score

    def fill(self, rng):
        order = list(range(len(self.units)))
        rng.shuffle(order)
        sel = []
        for k in order:
            if sum(self.size[sel]) + self.size[k] <= self.n_test:
                sel.append(k)
            if sum(self.size[sel]) == self.n_test:
                return sel
        return None

    def climb(self, sel, max_steps):
        """Best-improvement size-preserving swaps on the key shortfall * BIG + score."""
        big = 1000.0
        in_test = np.zeros(len(self.units), bool)
        in_test[sel] = True
        for _ in range(max_steps):
            typ, pair, fam = (M[in_test].sum(0) for M in (self.U_type, self.U_pair, self.U_fam))
            s0, c0 = self.evaluate(typ, pair, fam)
            cur = s0 * big + c0
            out_idx, in_idx = np.where(in_test)[0], np.where(~in_test)[0]
            best = None
            for o in out_idx:
                cand = in_idx[self.size[in_idx] == self.size[o]]
                if not len(cand):
                    continue
                s, c = self.evaluate(typ - self.U_type[o] + self.U_type[cand],
                                     pair - self.U_pair[o] + self.U_pair[cand],
                                     fam - self.U_fam[o] + self.U_fam[cand])
                key = s * big + c
                j = int(np.argmin(key))
                if key[j] < cur - 1e-12 and (best is None or key[j] < best[0] - 1e-12):
                    best = (key[j], o, cand[j])
            if best is None:
                break
            in_test[best[1]], in_test[best[2]] = False, True
        typ, pair, fam = (M[in_test].sum(0) for M in (self.U_type, self.U_pair, self.U_fam))
        s, c = self.evaluate(typ, pair, fam)
        return in_test, int(s), float(c), pair


def split_groups_pair_coverage(group_meta: list[dict], n_test: int, seed: int, min_pair_test_groups: int = 2,
                               min_type_test_groups: int = 1, min_type_train_groups: int = 5,
                               n_restarts: int = 1000, max_steps: int = 200, types=None,
                               keep_related_units: bool = True, min_pair_train_groups: int = 0,
                               allow_pair_exceptions: bool = False,
                               exception_priority: str = "test") -> tuple[set[str], dict]:
    """Seeded multi-start local search for the best-balanced admissible split (see module docstring).

    Admissible: every compatible type pair in >= min_pair_test_groups TEST groups, every type in
    >= min_type_test_groups TEST groups and >= min_type_train_groups TRAIN groups, related units kept whole.
    Among admissible splits the lowest balance score wins; ties keep the earliest restart.
    min_pair_train_groups also requires every compatible pair in that many TRAIN groups. A pair that occurs in
    fewer than min_pair_test_groups + min_pair_train_groups groups overall cannot meet both; with
    allow_pair_exceptions=True the side named by exception_priority keeps its minimum and the other side gets the
    remaining groups (recorded under "pair_exceptions"), otherwise a ValueError is raised. The owner chose
    exception_priority="test" on 2026-11-23 (the rare pair goes 2 test / 0 train).
    keep_related_units=False treats every group as its own unit (RELATED_CLUSTERS ignored; variant for comparison).
    """
    P = _Problem(group_meta, n_test, min_pair_test_groups, min_type_test_groups, min_type_train_groups, types,
                 keep_related_units, min_pair_train_groups, exception_priority)
    exc = P.exceptions()
    if exc and not allow_pair_exceptions:
        raise ValueError(f"pair minima infeasible (test >= {min_pair_test_groups}, train >= {min_pair_train_groups}) "
                         f"for these compatible pairs; allow_pair_exceptions=True relaxes them: {exc}")
    bad = [] if allow_pair_exceptions else P.infeasible_pairs()
    if bad:
        raise ValueError(f"min_pair_test_groups={min_pair_test_groups} is infeasible: these compatible pairs occur in "
                         f"fewer groups across all {P.n_groups}: {bad}")
    rng = random.Random(seed)
    best, n_admissible, n_filled = None, 0, 0
    for r in range(n_restarts):
        sel = P.fill(rng)
        if sel is None:
            continue
        n_filled += 1
        in_test, short, score, pair = P.climb(sel, max_steps)
        if short:
            continue
        n_admissible += 1
        if best is None or score < best[0] - 1e-12:
            best = (score, in_test.copy(), r, pair.copy())
    if best is None:
        raise RuntimeError(f"no admissible split found in {n_restarts} restarts")
    score, in_test, r, pair = best
    test = sorted(g for k in np.where(in_test)[0] for g in P.units[k])
    typ = P.U_type[in_test].sum(0)
    info = {
        "amendment": AMENDMENT_ID, "score": score, "restart_index": r, "n_restarts": n_restarts,
        "n_restarts_filled": n_filled, "n_restarts_admissible": n_admissible, "seed": seed, "n_test": n_test,
        "min_pair_test_groups": min_pair_test_groups, "min_type_test_groups": min_type_test_groups,
        "min_type_train_groups": min_type_train_groups, "n_compatible_pairs": len(P.pairs),
        "keep_related_units": keep_related_units, "units": P.units,
        "min_pair_train_groups": min_pair_train_groups, "pair_exceptions": exc,
        "exception_priority": exception_priority,
        "pair_test_groups": {f"{a}|{b}": int(c) for (a, b), c in zip(P.pairs, pair)},
        "pair_all_groups": {f"{a}|{b}": int(c) for (a, b), c in zip(P.pairs, P.pair_all)},
        "type_test_groups": {t: int(c) for t, c in zip(P.types, typ)},
        "type_train_groups": {t: int(c) for t, c in zip(P.types, P.type_all - typ)},
        "min_pair_coverage": int(pair.min()),
        "n_pairs_at_least_3": int((pair >= 3).sum()),
        "pair_train_groups": {f"{a}|{b}": int(c) for (a, b), c in zip(P.pairs, P.pair_all - pair)},
        "n_pairs_absent_from_train": int(((P.pair_all - pair) == 0).sum()),
        "min_pair_train_coverage": int((P.pair_all - pair).min()),
        "related_units_split": [u for u in related_units(group_meta) if len(u) > 1
                                and 0 < sum(g in test for g in u) < len(u)],
    }
    return set(test), info


def group_meta_from_prompts(prompts) -> list[dict]:
    """group_meta (as manifest.build passes to tasks.split_groups) rebuilt from a prompts.parquet frame.

    Needs columns group_id, task_key, task_family, ordered_type_ids. A group's six pool types are the union of
    its prompts' ordered_type_ids (every pool rule appears in some k=5 prefix).
    """
    out = []
    for gid, d in prompts.groupby("group_id", sort=True):
        tasks = d.drop_duplicates("task_key").sort_values("task_key")
        types = sorted({t for ids in d["ordered_type_ids"] for t in ids})
        assert len(types) == 6, (gid, types)
        out.append({"group_id": gid, "task_keys": [int(t) for t in tasks["task_key"]],
                    "families": list(tasks["task_family"]), "types": types})
    return out


def pair_coverage(group_meta: list[dict], test: set[str], types=None) -> dict[str, int]:
    """Number of TEST groups whose pool contains both types, for every compatible pair."""
    pairs = compatible_type_pairs(types)
    out = {f"{a}|{b}": 0 for a, b in pairs}
    for g in group_meta:
        if g["group_id"] in test:
            for a, b in combinations(sorted(g["types"]), 2):
                if f"{a}|{b}" in out:
                    out[f"{a}|{b}"] += 1
    return out
