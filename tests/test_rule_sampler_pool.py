"""Rule parameters are draws from IFEval's code-default pool (official build_description() with no kwargs).

The expected pool below is written out independently from the pinned instructions.py constants, so these tests
fail if the sampler ever draws from anything else. Random draws here are test fixtures, not research data.
"""
import random
from collections import Counter

import pytest

from mife import rule_sampler as RS, source as S
from instruction_following_eval import instructions_util as U

WL = set(U.WORD_LIST)
REL = {"less than", "at least"}
LOWER = set("abcdefghijklmnopqrstuvwxyz")
INT_RANGES = {  # (type, field): inclusive range in the official code
    ("keywords:frequency", "frequency"): (1, 3),
    ("keywords:letter_frequency", "let_frequency"): (1, 10),
    ("length_constraints:number_words", "num_words"): (100, 500),
    ("length_constraints:number_sentences", "num_sentences"): (1, 20),
    ("length_constraints:number_paragraphs", "num_paragraphs"): (1, 5),
    ("detectable_content:number_placeholders", "num_placeholders"): (1, 4),
    ("detectable_format:number_bullet_lists", "num_bullets"): (1, 5),
    ("change_case:capital_word_frequency", "capital_frequency"): (1, 20),
}
FIELDS = {
    "keywords:existence": ["keywords"],
    "keywords:frequency": ["keyword", "frequency", "relation"],
    "keywords:forbidden_words": ["forbidden_words"],
    "keywords:letter_frequency": ["letter", "let_frequency", "let_relation"],
    "length_constraints:number_words": ["num_words", "relation"],
    "length_constraints:number_sentences": ["num_sentences", "relation"],
    "length_constraints:number_paragraphs": ["num_paragraphs"],
    "detectable_content:number_placeholders": ["num_placeholders"],
    "detectable_format:number_bullet_lists": ["num_bullets"],
    "change_case:capital_word_frequency": ["capital_frequency", "capital_relation"],
}
FIXED = {"detectable_format:title", "change_case:english_capital", "change_case:english_lowercase",
         "startend:quotation", "punctuation:no_comma"}


def in_pool(t: str, kw: dict) -> bool:
    if t in FIXED:
        return kw == {}
    if sorted(kw) != sorted(FIELDS[t]):
        return False
    for f, v in kw.items():
        if (t, f) in INT_RANGES:
            lo, hi = INT_RANGES[(t, f)]
            if not (isinstance(v, int) and lo <= v <= hi):
                return False
        elif f in ("relation", "let_relation", "capital_relation"):
            if v not in REL:
                return False
        elif f in ("keywords", "forbidden_words"):
            if not (isinstance(v, list) and len(v) == 2 and len(set(v)) == 2 and set(v) <= WL):
                return False
        elif f == "keyword":
            if v not in WL:
                return False
        elif f == "letter":
            if not (isinstance(v, str) and len(v) == 1 and v in LOWER):
                return False
        else:
            return False
    return True


def test_fifteen_types_covered():
    assert set(FIELDS) | FIXED == set(S.TYPES)


@pytest.mark.parametrize("t", S.TYPES)
def test_every_draw_in_pool_and_renders(t):
    rng = random.Random(1234)
    for _ in range(500):
        kw = RS.sample_params(t, rng)
        assert in_pool(t, kw), (t, kw)
        S.render_rule_text(t, kw)  # official template accepts it


def test_every_pool_rule_over_many_seeds_in_pool():
    for seed in range(40):
        for r in RS.sample_pool(seed)["rules"]:
            assert in_pool(r["type_id"], r["kwargs"]), r


def test_global_random_state_untouched():
    random.seed(99)
    before = random.getstate()
    for t in S.TYPES:
        RS.sample_params(t, random.Random(5))
    assert random.getstate() == before
    RS.sample_pool(3)
    assert random.getstate() == before


def test_same_seed_same_pool_and_params():
    a, b = RS.sample_pool(17), RS.sample_pool(17)
    assert [(r["type_id"], r["kwargs"]) for r in a["rules"]] == [(r["type_id"], r["kwargs"]) for r in b["rules"]]
    assert a["sampler_version"] == "rule_sampler/ifeval_code_defaults"
    x = [RS.sample_params("length_constraints:number_words", random.Random(8)) for _ in range(3)]
    assert x[0] == x[1] == x[2]


def _chi2(counts: Counter, values: list) -> tuple[float, int]:
    n = sum(counts.values())
    e = n / len(values)
    return sum((counts.get(v, 0) - e) ** 2 / e for v in values), len(values) - 1


N_DRAWS = 20000


@pytest.mark.parametrize("t", sorted(FIELDS))
def test_raw_draw_distributions(t):
    """Before D3: relations ~50/50, integer thresholds ~uniform over the full range, letters ~uniform a-z."""
    rng = random.Random(20261111)
    draws = [RS.sample_params(t, rng) for _ in range(N_DRAWS)]
    for f in FIELDS[t]:
        vals = Counter(d[f] if not isinstance(d[f], list) else len(d[f]) for d in draws)
        if f.endswith("relation"):
            share = vals["less than"] / N_DRAWS
            assert abs(share - 0.5) < 0.02, (t, f, share)
        elif (t, f) in INT_RANGES:
            lo, hi = INT_RANGES[(t, f)]
            support = list(range(lo, hi + 1))
            assert set(vals) == set(support), (t, f, sorted(set(support) - set(vals)))
            x2, df = _chi2(vals, support)
            assert x2 < df + 6 * (2 * df) ** 0.5, (t, f, x2, df)   # loose: ~6 sd above the mean
        elif f == "letter":
            x2, df = _chi2(vals, sorted(LOWER))
            assert set(vals) == LOWER and x2 < df + 6 * (2 * df) ** 0.5
        elif f in ("keywords", "forbidden_words"):
            assert set(vals) == {2}
            words = Counter(w for d in draws for w in d[f])
            assert len(words) > 0.9 * len(WL)                   # draws spread across the whole WORD_LIST
