"""Per-group sampling of six jointly compatible exact ManyIFEval rule instances (D1-D3).

Procedure (frozen before generation, logged per group):
1. Type proposal: repeatedly propose a type uniformly at random from the eligible set
   (not yet chosen, not D1-conflicting with chosen types). Lookahead: accept only if the
   chosen set can still be completed to six pairwise-compatible types (exhaustive DFS).
2. Instantiation: draw the accepted type's parameters from IFEval's code-default pool (D2): the
   official instruction class's build_description() called with no arguments, exactly as the
   pinned ManyIFEval/IFEval code does, seeded from the group's RNG. Then check feasibility of the
   partial pool (D3). On a proven contradiction (e.g. "less than 1 sentence"), resample this
   type's parameters (up to MAX_PARAM_TRIES), then reject the type and backtrack. D3 resampling
   is the only filter applied to code-default draws; rejection rates per type are in the manifest audit.
3. Final check: the complete six-rule pool must have a witness (implies every subset is feasible).

Parameter policy (docs/audit_findings.md M2): parameters come ONLY from the released code's default
generators. They are never drawn from the empirical distribution of the published ManyIFEval parameter values,
and no distribution is invented for values outside the code's ranges. This reproduces the released code's
procedure; it does not claim to reproduce the original authors' dataset-generation distribution. Every
proposal, rejection (with proof) and resampling decision is written to the group's sampling_log.
"""
from __future__ import annotations

import hashlib
import json
import random

from . import feasibility, source as S

N_RULES = 6
MAX_PARAM_TRIES = 25
SAMPLER_VERSION = "rule_sampler/ifeval_code_defaults"


def _field_order(type_id: str) -> list[str]:
    """Canonical field order = the order used by the released ManyIFEval data for this type."""
    return list(next(iter(S.tasks().values()))["source_rules"][type_id].keys())


def sample_params(type_id: str, rng: random.Random) -> dict:
    """One draw from IFEval's code-default parameter pool for `type_id`.

    Instantiates the OFFICIAL instruction class from the pinned instructions_registry and calls
    build_description() with no kwargs, so the official code alone defines the ranges and
    distributions. The official code draws from Python's global `random`; it is seeded from `rng`
    and its previous state is restored afterwards, so callers' global random state is untouched.
    """
    state = random.getstate()
    try:
        random.seed(rng.getrandbits(64))
        inst = S.instruction_class(type_id)(type_id)
        inst.build_description()
        args = inst.get_instruction_args() or {}
    finally:
        random.setstate(state)
    args = {k: v for k, v in args.items() if v is not None}
    order = _field_order(type_id)
    assert set(args) == set(order), (type_id, args, order)
    return {f: args[f] for f in order}


def _completable(chosen: list[str], banned: set[str]) -> bool:
    def dfs(cur, start_types):
        if len(cur) == N_RULES:
            return True
        for t in start_types:
            if t in cur or t in banned or any(S.conflicts(t, c) for c in cur):
                continue
            if dfs(cur + [t], [u for u in start_types if u > t]):
                return True
        return False
    return dfs(list(chosen), S.TYPES)


def rule_hash(type_id: str, kwargs: dict) -> str:
    return hashlib.sha256(json.dumps([type_id, kwargs], sort_keys=True).encode()).hexdigest()[:16]


def sample_pool(seed: int) -> dict:
    rng = random.Random(seed)
    log: list[dict] = []
    chosen: list[tuple[str, dict]] = []
    rejected_types: set[str] = set()
    while len(chosen) < N_RULES:
        types = [t for t, _ in chosen]
        eligible = [t for t in S.TYPES if t not in types and t not in rejected_types
                    and not any(S.conflicts(t, c) for c in types)]
        if not eligible:
            raise RuntimeError("sampler dead end (should be impossible under D1)")
        t = rng.choice(eligible)
        if not _completable(types + [t], rejected_types):
            log.append({"event": "type_rejected_lookahead", "type": t})
            rejected_types.add(t)
            continue
        accepted = False
        for attempt in range(MAX_PARAM_TRIES):
            kw = sample_params(t, rng)
            res = feasibility.check_pool(chosen + [(t, kw)], seed=rng.randrange(1 << 30))
            if res["feasible"]:
                chosen.append((t, kw))
                log.append({"event": "accepted", "type": t, "kwargs": kw, "param_attempt": attempt})
                accepted = True
                break
            log.append({"event": "params_rejected", "type": t, "kwargs": kw, "proofs": res["proofs"]})
        if not accepted:
            log.append({"event": "type_rejected_params_exhausted", "type": t})
            rejected_types.add(t)
    final = feasibility.check_pool(chosen, seed=rng.randrange(1 << 30))
    assert final["feasible"], final
    return {
        "rules": [{"type_id": t, "kwargs": kw, "rule_text": S.render_rule_text(t, kw),
                   "category": S.category_of(t), "code_prefix": S.code_prefix_of(t),
                   "exact_rule_hash": rule_hash(t, kw)} for t, kw in chosen],
        "sampling_log": log,
        "witness": final["witness"],
        "seed": seed,
        "sampler_version": SAMPLER_VERSION,
    }
