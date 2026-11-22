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
