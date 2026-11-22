import hashlib
import importlib.util
import json
from pathlib import Path

from mife_ifb import inventory, plain_meaning as pm, source

IFB = Path(__file__).resolve().parents[1]
RECORD = json.loads((IFB / "freeze" / "plain_meaning_frozen.json").read_text())


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_checks_are_frozen():
    assert _sha(IFB / RECORD["module"]) == RECORD["module_sha256"], \
        "plain_meaning.py changed after the stage-1 freeze; changes need Nicky's approval and a new freeze record"
    assert _sha(IFB / RECORD["doc"]) == RECORD["doc_sha256"]
    assert RECORD["types"] == inventory.candidates()


def test_doc_matches_code():
    spec = importlib.util.spec_from_file_location("freeze", IFB / "scripts" / "freeze_plain_meaning.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m.render_doc() == (IFB / RECORD["doc"]).read_text(encoding="utf-8")


def test_one_check_per_candidate():
    assert set(pm.CHECKS) == set(inventory.candidates())
    assert all(c.rule for c in pm.CHECKS.values())


def test_examples_pass_and_loopholes_are_rejected():
    for t, c in pm.CHECKS.items():
        e = c.ok
        assert source.official_check(t, e.kwargs, e.response, e.task_text), (t, "ok example fails official")
        assert pm.witness_ok(t, e.kwargs, e.response, e.task_text) == (True, ""), t
        if c.loophole is not None:
            e = c.loophole
            assert source.official_check(t, e.kwargs, e.response, e.task_text), (t, "loophole not accepted by official")
            ok, why = pm.check(t, e.response, e.kwargs, e.task_text)
            assert not ok and why, t


def test_g1():
    assert pm.g1("")[0] is False
    assert pm.g1("one two")[0] is False
    assert pm.g1("one,two,three")[0] is True
    assert pm.witness_ok("format:no_whitespace", {}, "a,b")[0] is False


def test_quotes_ignore_apostrophes_and_need_alternation():
    ok = pm.check("format:quotes", "It's \"she said 'he yelled \"don't go\" twice' once\" ok")
    assert ok[0], ok
    assert not pm.check("format:quotes", "He said \"one 'two' three\" only")[0]  # two levels


def test_parentheses_need_real_matching():
    assert pm.check("format:parentheses", "Look (a [b {c (d [deep words])}])")[0]
    assert not pm.check("format:parentheses", "Look (a [b {c (d [deep)])}] words")[0]


def test_plain_checks_never_relax_official():
    # witness_ok must be False whenever the official verifier fails
    assert pm.witness_ok("ratio:sentence_type", {}, "Is it? Yes it is.")[0] is False
