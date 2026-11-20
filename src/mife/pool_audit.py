"""Conformance audit: every exact rule in the manifest lies in IFEval's code-default pool.

The bounds are read from the pinned official constants (instructions.py), not restated here, so the audit
checks the manifest against the same source of truth the sampler calls. (tests/test_rule_sampler_pool.py
spells the pool out literally as an independent check of these constants.)
"""
from __future__ import annotations

from collections import Counter

from . import source as S
from instruction_following_eval import instructions as I, instructions_util as U

REL = set(I._COMPARISON_RELATION)


def pool_spec() -> dict:
    """{type_id: {field: ("int", lo, hi) | ("rel",) | ("words2",) | ("word",) | ("letter",)}}; {} = fixed text."""
    return {
        "keywords:existence": {"keywords": ("words2",)},
        "keywords:forbidden_words": {"forbidden_words": ("words2",)},
        "keywords:frequency": {"keyword": ("word",), "frequency": ("int", 1, I._KEYWORD_FREQUENCY), "relation": ("rel",)},
        "keywords:letter_frequency": {"letter": ("letter",), "let_frequency": ("int", 1, I._LETTER_FREQUENCY),
                                      "let_relation": ("rel",)},
        "length_constraints:number_words": {"num_words": ("int", I._NUM_WORDS_LOWER_LIMIT, I._NUM_WORDS_UPPER_LIMIT),
                                            "relation": ("rel",)},
        "length_constraints:number_sentences": {"num_sentences": ("int", 1, I._MAX_NUM_SENTENCES), "relation": ("rel",)},
        "length_constraints:number_paragraphs": {"num_paragraphs": ("int", 1, I._NUM_PARAGRAPHS)},
        "detectable_content:number_placeholders": {"num_placeholders": ("int", 1, I._NUM_PLACEHOLDERS)},
        "detectable_format:number_bullet_lists": {"num_bullets": ("int", 1, I._NUM_BULLETS)},
        "change_case:capital_word_frequency": {"capital_frequency": ("int", 1, I._ALL_CAPITAL_WORD_FREQUENCY),
                                               "capital_relation": ("rel",)},
        **{t: {} for t in ("detectable_format:title", "change_case:english_capital", "change_case:english_lowercase",
                           "startend:quotation", "punctuation:no_comma")},
    }


def in_pool(type_id: str, kwargs: dict) -> bool:
    spec = pool_spec()[type_id]
    if set(kwargs) != set(spec):
        return False
    wl = set(U.WORD_LIST)
    for f, kind in spec.items():
        v = kwargs[f]
        ok = {"int": lambda: isinstance(v, int) and kind[1] <= v <= kind[-1],
              "rel": lambda: v in REL,
              "words2": lambda: isinstance(v, list) and len(v) == 2 and len(set(v)) == 2 and set(v) <= wl,
              "word": lambda: v in wl,
              "letter": lambda: isinstance(v, str) and len(v) == 1 and "a" <= v <= "z"}[kind[0]]()
        if not ok:
            return False
    return True


def audit(pools: dict) -> dict:
    rules = [r for p in pools.values() for r in p["rules"]]
    bad = [(r["rule_id"], r["type_id"], r["kwargs"]) for r in rules if not in_pool(r["type_id"], r["kwargs"])]
    per_type = {}
    for t in S.TYPES:
        rs = [r["kwargs"] for r in rules if r["type_id"] == t]
        h = {}
        for f, kind in pool_spec()[t].items():
            if kind[0] == "words2":
                h[f] = {"n_distinct_words": len({w for kw in rs for w in kw[f]}), "list_lengths": dict(Counter(len(kw[f]) for kw in rs))}
            elif kind[0] == "word":
                h[f] = {"n_distinct_words": len({kw[f] for kw in rs})}
            else:
                h[f] = dict(sorted(Counter(kw[f] for kw in rs).items()))
        # D3 resampling: draws rejected as provably infeasible, per type
        acc = sum(1 for p in pools.values() for e in p["sampling_log"] if e["event"] == "accepted" and e["type"] == t)
        rej = sum(1 for p in pools.values() for e in p["sampling_log"] if e["event"] == "params_rejected" and e["type"] == t)
        per_type[t] = {"n_rules": len(rs), "histograms": h, "d3_param_draws": acc + rej, "d3_param_rejections": rej,
                       "d3_rejection_rate": rej / (acc + rej) if acc + rej else None}
    rel_fields = [(t, f) for t, sp in pool_spec().items() for f, k in sp.items() if k[0] == "rel"]
    relation_split = {f"{t}.{f}": dict(Counter(r["kwargs"][f] for r in rules if r["type_id"] == t)) for t, f in rel_fields}
    rejected = [e for p in pools.values() for e in p["sampling_log"] if e["event"] == "params_rejected"]
    return {"n_exact_rules": len(rules), "n_in_pool": len(rules) - len(bad), "out_of_pool": bad,
            "relation_split": relation_split,
            "relation_split_all": dict(Counter(v for d in relation_split.values() for v, c in d.items() for _ in range(c))),
            "per_type": per_type, "d3_rejected_draws": [{"type": e["type"], "kwargs": e["kwargs"]} for e in rejected]}
