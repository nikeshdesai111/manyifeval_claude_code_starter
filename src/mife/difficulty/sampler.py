"""Pool sampler: 10 of the 18 rule types per seed, each with an easy/medium/hard level.

Constraints (plan "Sampling one pool", 2026-11-27):
  * 10 distinct types, least-used first (within the pool's split, then overall), seeded tie-break;
  * at most 2 types per category;
  * at least 3 easy, 3 medium, 3 hard; the 10th slot rotates E -> M -> H across accepted pools;
  * levels go to the types that have had that level least often within the split;
  * never a forbidden (type, level) pair (configs/difficulty/rules.yaml);
  * keywords: the two keyword rules get different keywords; TRAIN and TEST draw from disjoint word lists.
Redraw policy mirrors pilots/high_k/code/build.py extend_pool: on a failed final check (here the LLM compatibility
judge) the whole pool is redrawn, up to 50 attempts, and every rejected pool is logged with its E/M/H mix.
``search_design`` repeats the whole 40-pool draw over sampler seeds and keeps the first that meets the pair-coverage
target (every type pair in >= 2 TEST and >= 1 TRAIN pools).
"""
from __future__ import annotations

import hashlib
import itertools
import json
import random
from collections import Counter, defaultdict
from typing import Callable

import numpy as np
from scipy.optimize import linear_sum_assignment

from . import catalog as K

MAX_ATTEMPTS = 50
SAMPLER_VERSION = "difficulty-sampler-v0"


def derive_seed(*parts) -> int:
    h = hashlib.sha256(json.dumps([str(p) for p in parts]).encode()).digest()
    return int.from_bytes(h[:8], "big") & ((1 << 63) - 1)


def level_mix(rules: list[dict]) -> dict:
    c = Counter(r["designed_difficulty"] for r in rules)
    return {lvl: c.get(lvl, 0) for lvl in K.LEVELS}


class PoolSampler:
    def __init__(self, sampler_seed: int):
        self.seed = sampler_seed
        self.type_split = defaultdict(Counter)                    # split -> type -> pools
        self.type_all = Counter()
        self.level_split = defaultdict(lambda: defaultdict(Counter))  # split -> type -> level -> pools
        self.n_accepted = 0
        self.log: list[dict] = []

    # ------------------------------------------------------------------ one draw
    def _draw_types(self, split: str, rng: random.Random, jitter: float) -> list[str]:
        cat = K.catalog()
        ts = K.types()
        key = {t: (self.type_split[split][t] + rng.uniform(0, jitter), self.type_all[t] + rng.uniform(0, jitter),
                   rng.random()) for t in ts}
        chosen, per_cat = [], Counter()
        for t in sorted(ts, key=key.__getitem__):
            c = K.category_of(t)
            if per_cat[c] < cat["max_per_category"]:
                chosen.append(t)
                per_cat[c] += 1
            if len(chosen) == cat["pool_size"]:
                break
        return chosen

    def _assign_levels(self, split: str, types: list[str], rng: random.Random, jitter: float) -> list[str]:
        m = K.catalog()["min_per_level"]
        extra = K.LEVELS[self.n_accepted % 3]
        slots = [lvl for lvl in K.LEVELS for _ in range(m)] + [extra] * (len(types) - 3 * m)
        cost = np.array([[self.level_split[split][t][lvl] + rng.uniform(0, 0.5 + jitter) for lvl in slots]
                         for t in types])
        r, c = linear_sum_assignment(cost)
        out = [None] * len(types)
        for i, j in zip(r, c):
            out[i] = slots[j]
        return out

    def draw(self, split: str, rng: random.Random, attempt: int) -> list[tuple[str, str]]:
        jitter = 0.0 if attempt == 0 else 0.75 * attempt
        for _inner in range(200):
            types = self._draw_types(split, rng, jitter)
            levels = self._assign_levels(split, types, rng, jitter)
            pairs = list(zip(types, levels))
            if not any(K.forbidden(a, b) for a, b in itertools.combinations(pairs, 2)):
                return pairs
            self.log.append({"event": "forbidden_pair_redraw", "split": split, "pool": pairs})
            jitter += 0.5
        raise RuntimeError("could not avoid forbidden pairs")

    def instantiate(self, pairs: list[tuple[str, str]], split: str, rng: random.Random,
                    task_texts: list[str] | None = None) -> list[dict]:
        # keywords that already occur in the group's tasks would make keyword rules trivially easier: never draw them
        from .text import words as _words
        taken: set[str] = {w for t in (task_texts or []) for w in _words(t)}
        rules = []
        for t, lvl in pairs:
            kw = K.instantiate(t, lvl, rng, split, taken)
            if "keyword" in kw:
                taken.add(kw["keyword"])
            rules.append(K.rule_record(t, lvl, kw))
        return rules

    # ------------------------------------------------------------------ accepted pool with judge
    def sample_pool(self, seed_id: str, split: str, judge: Callable[[list[dict], int], dict] | None = None,
                    task_texts: list[str] | None = None) -> dict:
        """Draw a pool for one seed. ``judge(rules, attempt)`` returns {"accepted": bool, "reason": str, ...}."""
        rng = random.Random(derive_seed(self.seed, "pool", seed_id))
        for attempt in range(MAX_ATTEMPTS):
            pairs = self.draw(split, rng, attempt)
            rules = self.instantiate(pairs, split, rng, task_texts)
            verdict = judge(rules, attempt) if judge else {"accepted": True, "reason": "no judge"}
            if verdict["accepted"]:
                break
            self.log.append({"event": "pool_rejected", "seed_id": seed_id, "split": split, "attempt": attempt,
                             "types": [r["type_id"] for r in rules], "mix": level_mix(rules),
                             "levels": {r["type_id"]: r["designed_difficulty"] for r in rules},
                             "reason": verdict.get("reason"), "conflicting_rules": verdict.get("conflicting_rules")})
        else:
            raise RuntimeError(f"{seed_id}: no compatible pool in {MAX_ATTEMPTS} attempts")
        for r in rules:
            self.type_split[split][r["type_id"]] += 1
            self.type_all[r["type_id"]] += 1
            self.level_split[split][r["type_id"]][r["designed_difficulty"]] += 1
        self.n_accepted += 1
        self.log.append({"event": "pool_accepted", "seed_id": seed_id, "split": split, "attempt": attempt,
                         "mix": level_mix(rules)})
        return {"seed_id": seed_id, "split": split, "sampler_seed": self.seed, "attempts": attempt + 1,
                "rules": rules, "level_mix": level_mix(rules), "sampler_version": SAMPLER_VERSION,
                "catalog_sha256": K.catalog_sha256(), "judge_verdict": verdict}


# ---------------------------------------------------------------------- design-level search

def pair_coverage(pools: list[dict]) -> dict:
    cov = defaultdict(Counter)
    for p in pools:
        ts = sorted(r["type_id"] for r in p["rules"])
        for a, b in itertools.combinations(ts, 2):
            cov[(a, b)][p["split"]] += 1
    pairs = list(itertools.combinations(sorted(K.types()), 2))
    bad = [pr for pr in pairs if cov[pr]["test"] < 2 or cov[pr]["train"] < 1]
    return {"n_pairs": len(pairs), "n_failing": len(bad), "failing": bad[:20]}


def sample_design(seeds: list[tuple[str, str]], sampler_seed: int, judge=None) -> tuple[list[dict], PoolSampler]:
    s = PoolSampler(sampler_seed)
    return [s.sample_pool(sid, split, judge) for sid, split in seeds], s


def search_design(seeds: list[tuple[str, str]], base_seed: int, n_tries: int = 400, judge=None) -> dict:
    """First sampler seed (base_seed + i) whose 40 pools meet the pair-coverage target."""
    for i in range(n_tries):
        pools, s = sample_design(seeds, base_seed + i, judge)
        cov = pair_coverage(pools)
        if cov["n_failing"] == 0:
            return {"sampler_seed": base_seed + i, "tries": i + 1, "pools": pools, "coverage": cov, "log": s.log}
    raise RuntimeError(f"no design met pair coverage in {n_tries} tries")
