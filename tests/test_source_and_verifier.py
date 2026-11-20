"""Source integration and verifier fidelity. Fixtures here are hand-written test strings, not research data."""
import random
import sys
import types

import pytest

from mife import source as S, verifier as V


def test_reproduces_all_released_prompts():
    rep = S.verify_source_reproduction()
    assert rep == {"prompts_reproduced": 2160, "tasks": 216, "distinct_type_lists": 18}


def test_fifteen_types_and_six_categories():
    assert len(S.TYPES) == 15
    assert {S.category_of(t) for t in S.TYPES} == {"keywords", "length_counts", "formatting", "letter_case",
                                                   "beginning_ending", "punctuation"}
    released = {t for task in S.tasks().values() for t in task["source_rules"]}
    assert released == set(S.TYPES)


def test_d1_rule_matches_released_lists_and_is_maximal():
    for task in S.tasks().values():
        tl = task["source_type_list"]
        assert len(tl) == 10
        for t in S.TYPES:
            if t not in tl:  # every excluded type conflicts with an included one => lists are maximal
                assert any(S.conflicts(t, u) for u in tl)


def _official_wrapper():
    sys.modules.setdefault("matplotlib", types.ModuleType("matplotlib"))
    sys.modules.setdefault("matplotlib.pyplot", types.ModuleType("matplotlib.pyplot"))
    sys.modules["matplotlib"].pyplot = sys.modules["matplotlib.pyplot"]
    import utils  # the pinned ManyIFEval utils.py (on sys.path via mife.source)
    return utils.test_instruction_following


@pytest.mark.parametrize("seed", range(20))
def test_verifier_matches_official_wrapper(seed):
    official = _official_wrapper()
    rng = random.Random(seed)
    task = rng.choice(list(S.tasks().values()))
    rules = list(task["source_rules"].items())
    rng.shuffle(rules)
    rules = rules[:5]
    for text in ["", "hello world.", '"<<TITLE>>\n* ONE\n* TWO\nTHE DAY IS GOOD."', "a, b. c! [x] [y] *** next"]:
        _, ref = official(text, [t for t, _ in rules], [kw for _, kw in rules], task["task_text"])
        ours = V.verify(text, rules, task["task_text"])
        assert ours == [0 if f else 1 for f in ref]


def test_langdetect_deterministic():
    text = "THE CAT SAT ON THE MAT AND THE DOG RAN AWAY."
    rules = [("change_case:english_capital", {})]
    assert len({tuple(V.verify(text, rules, "")) for _ in range(10)}) == 1
