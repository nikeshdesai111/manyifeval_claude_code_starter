import pytest

from mife_ifb import inventory, plain_meaning as pm, source, witness

TASK = "Write a short story about a lighthouse keeper who finds a message in a bottle on the beach and decides to answer it."


@pytest.mark.parametrize("t", inventory.candidates())
def test_every_single_type_gets_a_witness(t):
    kw = source.sample_default_kwargs(t, rng_seed=4242)
    r = witness.build_witness({t: kw}, task_text=TASK, seed=1)
    assert r.ok, r.reasons.most_common(3)
    kw.pop("reference_text", None)
    assert pm.witness_ok(t, kw, r.text, TASK) == (True, "")


def test_deterministic_for_a_seed():
    kw = source.sample_default_kwargs("count:keywords_multiple", rng_seed=5)
    a = witness.build_witness({"count:keywords_multiple": kw}, seed=3)
    b = witness.build_witness({"count:keywords_multiple": kw}, seed=3)
    assert a.text == b.text and a.ok


@pytest.mark.parametrize("pair", [
    ("words:alphabet", "format:title_case"), ("ratio:sentence_type", "count:punctuation"),
    ("format:list", "format:line_indent"), ("format:quotes", "format:quote_unquote"),
    ("sentence:keyword", "words:keywords_specific_position"), ("ratio:overlap", "format:list"),
])
def test_hand_picked_pairs(pair):
    rules = {t: source.sample_default_kwargs(t, rng_seed=99 + i) for i, t in enumerate(pair)}
    r = witness.build_witness(rules, task_text=TASK, seed=2)
    assert r.ok, r.reasons.most_common(3)
    assert witness.verify(rules, r.text, TASK) == []


def test_contradictory_pair_returns_no_witness():
    rules = {"ratio:sentence_type": {}, "ratio:sentence_balance": {}}
    r = witness.build_witness(rules, seed=0, attempts=4)
    assert not r.ok and r.text is None


def test_overlap_needs_task_text():
    with pytest.raises(ValueError):
        witness.build_witness({"ratio:overlap": {"percentage": 30}})


def test_verify_reports_failures():
    bad = witness.verify({"format:no_whitespace": {}}, "has some spaces here", None)
    assert bad and bad[0][0] == "format:no_whitespace"


def test_lexicon_closed_classes_match_official_verifiers():
    from mife_ifb import lexicon as lx
    # every listed name, pronoun and conjunction is counted by the official verifier
    names = " ".join(lx.PERSON_NAMES)
    assert source.official_check("count:person_names", {"N": 50}, names)
    assert source.official_check("count:pronouns", {"N": len(lx.PRONOUNS)}, " ".join(lx.PRONOUNS))
    assert source.official_check("count:conjunctions", {"small_n": 7}, " ".join(lx.CONJUNCTIONS) + " end")
    assert source.official_check("words:palindrome", {}, " ".join(lx.PALINDROMES))


@pytest.mark.parametrize("other", ["words:last_first", "ratio:sentence_type", "sentence:increment"])
def test_thesis_terminal_placement(other):
    # inside </i> under words:last_first, after it for sentence-count rules (fix 17)
    rules = {"format:thesis": {}, other: source.sample_default_kwargs(other, rng_seed=7)}
    r = witness.build_witness(rules, seed=0)
    assert r.ok, r.reasons.most_common(3)
    import re
    assert bool(re.search(r"[.?!]</i>", r.text)) == (other == "words:last_first")


LAYOUTS = [t for t in witness.LAYOUT_TYPES]
_SEPS = ["-", "...", "!?!?", "SEPARATOR"]


def _layout_sets():
    import itertools
    out = []
    for k in (2, 3, 4):
        for combo in itertools.combinations(LAYOUTS, k):
            if {"format:no_bullets_bullets", "format:sub-bullets"} <= set(combo):
                continue  # proven conflict
            out.append(combo)
    return out


@pytest.mark.parametrize("combo", _layout_sets(), ids=lambda c: "+".join(t.split(":")[1] for t in c))
def test_combined_layouts_get_a_witness(combo):
    """Two, three or four layouts in one text, every list separator: the joint witness passes
    every official verifier and every plain-meaning check."""
    seps = _SEPS if "format:list" in combo else [None]
    for sep in seps:
        rules = {t: ({"sep": sep} if t == "format:list" else {}) for t in combo}
        r = witness.build_witness(rules, seed=0)
        assert r.ok, (sep, r.reasons.most_common(3))
        for t, kw in rules.items():
            assert pm.witness_ok(t, kw, r.text, None) == (True, ""), (t, sep)


def test_no_bullets_with_sub_bullets_stays_infeasible():
    r = witness.build_witness({"format:no_bullets_bullets": {}, "format:sub-bullets": {}}, seed=0)
    assert not r.ok


def test_copy_model_matches_rendered_tokens():
    """The copy route matches trigrams of rendered tokens, with sentence-initial capitals."""
    c = witness._Copy.of("Write a short story about a keeper. The keeper finds a bottle.")
    assert ("a", "short", "story") in c.tri and ("keeper", ".", "The") in c.tri
    assert "short" in c.nxt[("Write", "a")]


@pytest.mark.parametrize("pct", [10, 30, 50])
def test_overlap_copy_route_hits_the_window(pct):
    rules = {"ratio:overlap": {"percentage": pct}, "format:quotes": {}}
    text = witness._copy_route(rules, TASK, pct, 3, False, lambda t: not witness.verify(rules, t, TASK))
    assert text is not None
    assert abs(witness._overlap_value(text, TASK) - pct) <= 2
    assert not witness.verify(rules, text, TASK)
