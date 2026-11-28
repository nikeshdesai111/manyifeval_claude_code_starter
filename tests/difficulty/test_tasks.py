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


def test_rewrite_feedback_keeps_accepted_and_pins_medium_distractors():
    prev = [{"task_id": f"x-dis-{TK.LEVEL_SHORT[l]}", "level": l, "mechanism": "distractors", "task_text": f"T {l}",
             "distractors": ["D1.", "D2."] if l == "medium" else []} for l in TK.LEVELS]
    res = {"x-dis-hard": {"why": ["judge found 5 distractors (designed 4)"]}}
    fb = TK.rewrite_feedback(prev, ["hard"], res)
    assert "Rewrite only the hard version" in fb and "keep easy exactly:\nT easy" in fb
    assert "keep medium exactly:\nT medium" in fb and "D1. | D2." in fb and "designed 4" in fb


def test_compat_prompt_contains_definitions_and_every_checker_spec():
    import random
    from mife.difficulty import catalog as K, checkers as C, text as T
    rng = random.Random(3)
    rules = [K.rule_record(t, "hard", K.instantiate(t, "hard", rng, "train", set())) for t in K.types()[:10]]
    rules += [K.rule_record(t, "easy", K.instantiate(t, "easy", rng, "train", set())) for t in K.types()[10:]]
    tasks = [{"task_id": "x-dis-easy", "task_text": "Why did my bill rise?"}]
    p = TK.compat_prompt("pool1", rules, tasks)
    assert T.DEFINITIONS in p
    for key in ("SECTION header lines are not sentences", "digit runs outside SECTION markers and hashtags",
                "emoji may follow a sentence's end punctuation", "ignore case", "Paragraphs are blocks",
                "Words are runs of letters"):
        assert key in p
    for i, r in enumerate(rules):
        assert f"R{i + 1}: {r['rule_text']}" in p
        assert C.spec(r["type_id"], r["kwargs"]) in p
    assert "{" not in p.split("The 10 rules")[0].replace("{ }", "")  # no unfilled template fields
