"""Round-2 reserve types: checkers on constructed pass / fail / edge cases, and the round-2 draft catalog."""
import itertools
import random
from collections import Counter

import pytest

from mife.difficulty import catalog as K
from mife.difficulty import checkers as C


def test_person_names():
    assert C.person_names("Emma met Liam and Noah.", n=3).passed
    assert not C.person_names("Emma met Emma.", n=2).passed
    assert not C.person_names("emma and liam", n=1).passed          # capitalised, as IFBench
    assert not C.person_names("Emmanuel and Leopold", n=1).passed   # whole names only
    assert not C.person_names("Mary-Grace came", n=1).passed        # part of a hyphenated name is not the name


def test_stopword_share():
    assert C.stopword_share("Budget planning requires careful tracking.", pct=10).passed
    assert not C.stopword_share("It is the one that is in the box.", pct=50).passed
    assert C.stopword_share("It is the cat.", pct=75).passed and not C.stopword_share("It is the cat.", pct=74).passed
    assert not C.stopword_share("", pct=50).passed
    assert C.stopword_share("Don't panic, team.", pct=34).passed    # don't is one stop word of 3


def test_sentence_increment():
    assert C.sentence_increment("Go. Go now. Go now please.", n=3).passed
    assert not C.sentence_increment("Go now. Go. Go now please.", n=3).passed
    assert not C.sentence_increment("Go. Go now.", n=3).passed            # too few sentences
    assert not C.sentence_increment("Go now. Go too.", n=2).passed        # equal is not longer
    assert C.sentence_increment("Hi. Pack well. Then move out. Now rest.", n=3).passed  # only the first n count


def test_postscript():
    ok = "Main text here.\n\nP.S. " + " ".join(["word"] * 12)
    assert C.postscript(ok, n=10).passed and not C.postscript(ok, n=13).passed
    assert C.postscript("Text.\n\n**P.S.** call the movers early today please now", n=5).passed
    assert not C.postscript("Text.\n\nP.S. call early.\n\nMore text after.", n=1).passed   # must be the last paragraph
    assert not C.postscript("Text. P.S. inline is not a paragraph", n=1).passed
    assert C.postscript("Text.\n\nP. S. spaced marker works fine", n=4).passed


def test_nested_quotes():
    assert C.nested_quotes('He said "go".', n=1).passed
    assert C.nested_quotes('She said "he told me \'wait\' twice".', n=2).passed
    assert C.nested_quotes('A "b \'c "d" e\' f" g', n=3).passed
    assert not C.nested_quotes('A "b \'c "d" e\' f" g', n=4).passed
    assert not C.nested_quotes("It's my parents' car, don't worry.", n=1).passed   # apostrophes are not quotes
    assert C.nested_quotes("“Mom said ‘relax’”", n=2).passed        # curly quotes normalised
    assert not C.nested_quotes('An "unclosed quote', n=1).passed


def test_divider_lines():
    two = "Para one.\n\n---\n\nPara two.\n\nPara three."
    assert C.divider_lines(two, scope="first_gap").passed
    assert not C.divider_lines(two, scope="every_paragraph").passed
    allp = "One.\n---\nTwo.\n\n***\n\nThree."
    assert C.divider_lines(allp, scope="every_paragraph").passed
    assert C.divider_lines(allp, scope="every_sentence").passed    # each block is one sentence, every gap divided
    sent = "First sentence.\n---\nSecond one.\n---\nThird."
    assert C.divider_lines(sent, scope="every_sentence").passed
    assert not C.divider_lines("Two sentences. In one block.\n---\nThird.", scope="every_sentence").passed
    assert not C.divider_lines("No dividers.\n\nAt all.", scope="first_gap").passed
    assert not C.divider_lines("Only one paragraph.", scope="every_paragraph").passed
    assert not C.divider_lines("Text\n- - -\nMore", scope="first_gap").passed   # spaced dashes are not a divider


@pytest.fixture
def round2():
    K.use_catalog(K.CFG / "rules_round2.yaml")
    yield K.catalog()
    K.use_catalog(K.CFG / "rules.yaml")


def test_round2_catalog_shape_texts_and_checkers(round2):
    assert len(round2["types"]) == 18
    rng = random.Random(0)
    for t in K.types():
        assert t in C.CHECKERS
        for lvl in K.LEVELS:
            kw = K.instantiate(t, lvl, rng, "train", set())
            txt = K.rule_text(t, kw)
            assert txt.endswith(".") and C.spec(t, kw)
            assert isinstance(C.check(t, "Some text. More text?", kw).passed, bool)


def test_round2_sampler_meets_constraints_and_pair_coverage(round2):
    from mife.difficulty import sampler as S
    seeds = [(f"sc{i + 1:02d}", "train" if i < 24 else "test") for i in range(40)]
    d = S.search_design(seeds, base_seed=2026112800, n_tries=400)
    for p in d["pools"]:
        ts = [r["type_id"] for r in p["rules"]]
        assert len(set(ts)) == 10 and max(Counter(K.category_of(t) for t in ts).values()) <= 2
        assert min(p["level_mix"].values()) >= 3
    assert d["coverage"]["n_failing"] == 0


def test_bold_first_word():
    assert C.bold_first_word("**Pack** early. Then rest.\n\n__Call__ movers. Done.", scope="paragraph_first").passed
    assert not C.bold_first_word("**Pack** early. Then rest.\n\nCall movers.", scope="paragraph_first").passed
    assert not C.bold_first_word("**Pack** early. Then rest.", scope="first_two_paragraphs").passed
    assert C.bold_first_word("**Pack** early. **Then** rest.\n\n**Call** now.\n\nplain third.",
                             scope="first_two_paragraphs").passed
    assert C.bold_first_word("- **Pack** boxes\n- **Label** them", scope="every_sentence").passed      # list items
    assert C.bold_first_word("## **Plan**\n😀 **First** step. \"**Second** step.\"", scope="every_sentence").passed
    assert C.bold_first_word("**The whole sentence is bold.** **Next** one.", scope="every_sentence").passed
    assert not C.bold_first_word("The **plan** works. **Next** one.", scope="every_sentence").passed  # not the first word
    assert not C.bold_first_word("*Pack* early.", scope="paragraph_first").passed                     # italics, not bold
    assert not C.bold_first_word("**12** boxes moved.", scope="paragraph_first").passed               # digits are not a word
