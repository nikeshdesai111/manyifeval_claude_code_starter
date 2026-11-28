"""Pool sampler and catalog tests (synthetic 24 TRAIN / 16 TEST seed split; no real seeds or judge calls)."""
import itertools
import random
import re
from collections import Counter

import pytest

from mife.difficulty import catalog as K
from mife.difficulty import sampler as S
from mife.difficulty import text as T

SEEDS = [(f"sc{i + 1:02d}", "train" if i < 24 else "test") for i in range(40)]


@pytest.fixture(scope="module")
def design():
    return S.search_design(SEEDS, base_seed=2026112700, n_tries=400)


def test_design_meets_every_pool_constraint(design):
    for p in design["pools"]:
        ts = [r["type_id"] for r in p["rules"]]
        assert len(ts) == 10 and len(set(ts)) == 10
        assert max(Counter(K.category_of(t) for t in ts).values()) <= 2
        mix = p["level_mix"]
        assert min(mix.values()) >= 3 and sum(mix.values()) == 10
        pairs = [(r["type_id"], r["designed_difficulty"]) for r in p["rules"]]
        assert not any(K.forbidden(a, b) for a, b in itertools.combinations(pairs, 2))
        kws = [r["kwargs"]["keyword"] for r in p["rules"] if "keyword" in r["kwargs"]]
        assert len(kws) == len(set(kws))


def test_design_type_balance_and_pair_coverage(design):
    pools = design["pools"]
    for split, lo, hi in (("train", 13, 14), ("test", 8, 9)):
        c = Counter(r["type_id"] for p in pools if p["split"] == split for r in p["rules"])
        assert set(c) == set(K.types())
        assert lo <= min(c.values()) and max(c.values()) <= hi, (split, c)
    assert design["coverage"]["n_failing"] == 0
    total = Counter(r["designed_difficulty"] for p in pools for r in p["rules"])
    assert sorted(total.values()) == [133, 133, 134]  # 400 slots: tenth slot rotates


def test_train_and_test_word_lists_disjoint(design):
    for p in design["pools"]:
        wl = K.wordlists()
        other = "train" if p["split"] == "test" else "test"
        for r in p["rules"]:
            if "keyword" in r["kwargs"]:
                assert r["kwargs"]["keyword"] in wl["keywords"][p["split"]]
            if "phrase" in r["kwargs"]:
                assert r["kwargs"]["phrase"] not in sum(wl["closing_phrases"][other].values(), [])


def test_judge_rejection_redraws_whole_pool_and_logs_mix():
    s = S.PoolSampler(7)
    calls = []

    def judge(rules, attempt):
        calls.append([r["type_id"] for r in rules])
        if attempt < 2:
            return {"accepted": False, "reason": "synthetic rejection", "conflicting_rules": [rules[0]["type_id"]]}
        return {"accepted": True, "reason": "ok"}

    p = s.sample_pool("pc01", "train", judge)
    assert p["attempts"] == 3 and len(calls) == 3
    rej = [e for e in s.log if e["event"] == "pool_rejected"]
    assert len(rej) == 2 and all(sum(e["mix"].values()) == 10 for e in rej)
    assert calls[0] != calls[1] or [r for r in rej[0]["levels"].items()] != [r for r in rej[1]["levels"].items()]


def test_wordlist_constraints():
    wl = K.wordlists()
    for split in ("train", "test"):
        for w in wl["keywords"][split]:
            assert re.fullmatch(r"[a-z]{3,8}", w) and w != "section"
        for n, phrases in wl["closing_phrases"][split].items():
            for ph in phrases:
                ws = T.words(ph)
                assert len(ws) == int(n), ph
                assert all(T.letters_in(w) <= 8 for w in ws), ph
                assert all(a[0] != b[0] for a, b in zip(ws, ws[1:])), ph
                assert len(set(ws)) == len(ws), ph
                assert not re.search(r"\d", ph) and "section" not in ws
    assert not set(wl["keywords"]["train"]) & set(wl["keywords"]["test"])
    tr = sum(wl["closing_phrases"]["train"].values(), [])
    te = sum(wl["closing_phrases"]["test"].values(), [])
    norm = lambda s: " ".join(T.words(s))
    for a in tr:
        for b in te:
            assert norm(a) not in norm(b) and norm(b) not in norm(a), (a, b)
    # phrases contain neither split's keywords
    allkw = set(wl["keywords"]["train"]) | set(wl["keywords"]["test"])
    assert not any(set(T.words(p)) & allkw for p in tr + te)


def test_forbidden_pairs_are_the_approved_two():
    fp = K.catalog()["forbidden_pairs"]
    assert len(fp) == 2
    assert K.forbidden(("longest_sentence", "hard"), ("keyword_position", "hard"))
    assert K.forbidden(("closing_phrase", "hard"), ("longest_sentence", "hard"))
    assert not K.forbidden(("longest_sentence", "medium"), ("keyword_position", "hard"))


def test_catalog_shape():
    cat = K.catalog()
    assert len(cat["types"]) == 18
    assert Counter(v["category"] for v in cat["types"].values()) == {c: 3 for c in cat["categories"]}


def test_rule_texts_render_and_mention_their_parameter():
    rng = random.Random(0)
    for t in K.types():
        for lvl in K.LEVELS:
            kw = K.instantiate(t, lvl, rng, "train", set())
            txt = K.rule_text(t, kw)
            assert txt and txt.endswith(".")
            if "n" in kw and t not in ("keyword_sentence", "keyword_position"):
                assert str(kw["n"]) in txt


def test_keywords_never_drawn_from_the_groups_own_tasks():
    wl = K.wordlists()["keywords"]["train"]
    tasks = ["My " + " and ".join(wl[:-2]) + " plans."]          # every keyword but the last two appears in a task
    for seed in range(20):
        s = S.PoolSampler(seed)
        p = s.sample_pool(f"x{seed}", "train", task_texts=tasks)
        kws = [r["kwargs"]["keyword"] for r in p["rules"] if "keyword" in r["kwargs"]]
        assert set(kws) <= set(wl[-2:])
