from collections import Counter

import pytest

from mife import design, feasibility as F, rule_sampler as RS, source as S, tasks as T
from mife.config import DATA_DIR


@pytest.mark.parametrize("k,n", [(1, 1), (2, 2), (3, 6), (4, 12), (5, 20)])
def test_coa_templates(k, n):
    rows = design.coa_template(k)
    assert len(rows) == n
    design.check_coa(rows, k)


def test_coa_rejects_bad_array():
    rows = design.coa_template(4)
    with pytest.raises(AssertionError):
        design.check_coa(rows[:-1] + [rows[0]], 4)


def test_configurations_counts_and_balance():
    order = list("ABCDEF")
    cfg = design.configurations(order)
    assert len(cfg) == 246 and sum(c["k"] for c in cfg) == 1026
    assert len({tuple(c["ordered_rule_labels"]) for c in cfg}) == 246
    for k in range(1, 6):
        subsets = {frozenset(c["subset_chain_order"]) for c in cfg if c["k"] == k}
        assert len(subsets) == 6
        assert all(sum(r in s for s in subsets) == k for r in order)
    per_rule = Counter(r for c in cfg for r in c["ordered_rule_labels"])
    assert set(per_rule.values()) == {171}


def test_feasibility_proofs():
    assert F.proofs([("length_constraints:number_sentences", {"num_sentences": 1, "relation": "less than"})])
    clash = [("keywords:forbidden_words", {"forbidden_words": ["active", "ring"]}),
             ("keywords:existence", {"keywords": ["active", "owner"]})]  # released task 1381 rules
    assert any(p.startswith("P_KW_FORBIDDEN") for p in F.proofs(clash))
    letter = [("keywords:letter_frequency", {"letter": "e", "let_frequency": 2, "let_relation": "less than"}),
              ("keywords:existence", {"keywords": ["tree", "owner"]})]
    assert any(p.startswith("P_LETTER_LESS") for p in F.proofs(letter))
    assert not F.check_pool(clash)["feasible"]


def test_witness_passes_official_verifiers():
    for seed in range(200):
        pool = RS.sample_pool(seed)
        rules = [(r["type_id"], r["kwargs"]) for r in pool["rules"]]
        from mife import verifier
        assert verifier.verify(pool["witness"], rules, "") == [0] * 6


def test_sampler_deterministic_and_valid():
    a, b = RS.sample_pool(123), RS.sample_pool(123)
    assert a["rules"] == b["rules"]
    types = [r["type_id"] for r in a["rules"]]
    assert len(set(types)) == 6
    assert not any(S.conflicts(x, y) for i, x in enumerate(types) for y in types[i + 1:])
    for r in a["rules"]:
        assert r["rule_text"] == S.render_rule_text(r["type_id"], r["kwargs"])


def test_no_verifier_reads_the_task_text():
    """Satisfiability of a rule set is task-independent for the 15 ManyIFEval types."""
    assert S.types_reading_task_text() == []


def test_unusual_but_possible_task_rule_pairs_are_kept():
    """Feasibility-only policy: a rule that is merely unusual for a task stays (no relevance filter)."""
    from mife import feasibility
    cases = {
        1857: [("length_constraints:number_sentences", {"num_sentences": 17, "relation": "at least"})],  # catchy tweet
        1551: [("length_constraints:number_words", {"num_words": 170, "relation": "less than"})],  # two cover letters
        1928: [("punctuation:no_comma", {})],                                                       # quotes a comma sentence
        3703: [("detectable_format:title", {}), ("startend:quotation", {})],                         # finish the sentence
    }
    for key, pool in cases.items():
        res = feasibility.check_pool(pool, seed=0)
        assert res["feasible"], (key, res)
        got = feasibility.check_tasks(pool, res["witness"], {key: S.tasks()[key]["task_text"]})
        assert got == {key: [0] * len(pool)}


@pytest.mark.skipif(not (DATA_DIR / "manifest" / "rule_pools.jsonl").exists(), reason="manifest not built")
def test_every_group_pool_is_feasible_for_each_of_its_tasks():
    import json

    from mife import feasibility
    pools = [json.loads(line) for line in open(DATA_DIR / "manifest" / "rule_pools.jsonl")]
    assert len(pools) == 72
    for p in pools:
        rules = [(r["type_id"], r["kwargs"]) for r in p["rules"]]
        res = feasibility.check_tasks(rules, p["witness"], {k: S.tasks()[k]["task_text"] for k in p["task_keys"]})
        assert len(res) == 3 and all(not any(v) for v in res.values()), (p["group_id"], res)
