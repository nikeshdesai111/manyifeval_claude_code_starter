"""Offline tests of the task-side code: keyword screen, judge schemas, code acceptance (no API calls)."""
import jsonschema
import pytest

from mife.difficulty import tasks as TK


def _verdict(mech, level, d=(), i=(), fail=None):
    checks = {c: {"pass": c != fail, "reason": "n/a" if c != fail else "bad"} for c in TK.CHECK_NAMES}
    return {"task_id": f"x-{TK.MECH_SHORT[mech]}-{TK.LEVEL_SHORT[level]}", "mechanism": mech, "level": level,
            "distractors_found": list(d), "inferences_found": list(i), "checks": checks, "length_note": ""}


def _task(mech, level, text="My bill rose. Why?", hits=()):
    return {"task_id": f"x-{TK.MECH_SHORT[mech]}-{TK.LEVEL_SHORT[level]}", "mechanism": mech, "level": level,
            "task_text": text, "screen_hits": list(hits)}


def test_screen_flags_form_words_only():
    assert TK.screen("Explain it in three sentences, at least.") == ["at least", "sentences"]
    assert TK.screen("Keep it under 200 words") == ["keep it under", "words"]
    assert TK.screen("My rent rose by 40 dollars in March. Why did my balance drop?") == []


def test_judge_schema_accepts_well_formed_and_rejects_missing_fields():
    good = {"seed_id": "x", "core_objective": "c", "tasks": [_verdict("distractors", "easy")]}
    jsonschema.validate(good, TK.TASK_JUDGE_SCHEMA)
    bad = {"seed_id": "x", "core_objective": "c", "tasks": [{**_verdict("distractors", "easy")}]}
    del bad["tasks"][0]["checks"]["count_correct"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, TK.TASK_JUDGE_SCHEMA)


def test_accept_counts_lists_itself():
    ok, _ = TK.accept_task(_task("distractors", "medium"), _verdict("distractors", "medium", d=["a", "b"]), None)
    assert ok
    ok, why = TK.accept_task(_task("distractors", "medium"), _verdict("distractors", "medium", d=["a"]), None)
    assert not ok and "designed 2" in why[0]
    ok, why = TK.accept_task(_task("reasoning", "hard"), _verdict("reasoning", "hard", i=["1", "2", "3"], d=["z"]), None)
    assert not ok and "must be 0" in why[0]
    ok, why = TK.accept_task(_task("reasoning", "easy"), _verdict("reasoning", "easy", i=["1"], fail="count_correct"), None)
    assert not ok and why[0].startswith("count_correct")


def test_accept_hard_must_contain_medium_distractors_and_screen():
    t = _task("distractors", "hard", text="A. B. C. D. Why?")
    ok, _ = TK.accept_task(t, _verdict("distractors", "hard", d=list("abcd")), ["A.", "B."])
    assert ok
    ok, why = TK.accept_task(t, _verdict("distractors", "hard", d=list("abcd")), ["A.", "E."])
    assert not ok and "missing medium" in why[0]
    t2 = _task("reasoning", "easy", hits=["words"])
    ok, why = TK.accept_task(t2, _verdict("reasoning", "easy", i=["1"]), None)
    assert not ok and "keyword screen" in why[0]


def test_compat_schema():
    v = {"pass": False, "conflicting_rules": ["R1", "R2"], "reason": "x"}
    jsonschema.validate({"pool_id": "p", "rules_jointly_satisfiable": v,
                         "tasks": [{"task_id": "t", "reasonably_possible": v}]}, TK.COMPAT_SCHEMA)
