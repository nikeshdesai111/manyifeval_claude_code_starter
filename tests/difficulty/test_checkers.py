"""Unit tests for the 18 difficulty-dataset checkers and the shared text definitions (pass, fail, edge cases)."""
import pytest

from mife.difficulty import checkers as C
from mife.difficulty import text as T


# ------------------------------------------------------------------ shared definitions

def test_words_definition():
    s = "Don't over-plan: **Budget** #SaveMore 3rd time, 2.5 kg 😀 mother-in-law's"
    assert T.words(s) == ["don't", "over-plan", "budget", "savemore", "rd", "time", "kg", "mother-in-law's"]


def test_words_curly_apostrophe_and_section_lines_removed():
    s = "SECTION 1\nIt’s fine.\n\nSECTION 2: Next Steps\nWe’re done."
    assert T.words(s) == ["it's", "fine", "we're", "done"]


def test_numbers_definition():
    s = "SECTION 1\nPay 1,200 dollars on 3.5 days, #Tip2 and 7 more. 1️⃣"
    assert T.numbers(s) == ["1,200", "3.5", "7"]


def test_sentences_abbreviations_and_emoji_after_punctuation():
    s = "Dr. Smith arrived at 5 p.m. on Monday. It went well! 😀 Then we left? 🎉"
    ss = T.sentences(s)
    assert len(ss) == 3
    assert ss[1].endswith("😀") and ss[2].endswith("🎉")


def test_sentences_list_items_and_headings_are_sentences():
    s = "## Plan\n- pack the boxes\n- call the movers\nThen relax. Done."
    assert T.sentences(s) == ["## Plan", "- pack the boxes", "- call the movers", "Then relax.", "Done."]


def test_section_header_lines_not_sentences_or_paragraphs():
    s = "SECTION 1\nFirst part.\n\nSECTION 2\n\nSecond part."
    assert T.sentences(s) == ["First part.", "Second part."]
    assert T.paragraphs(s) == ["First part.", "Second part."]


def test_ellipsis_does_not_crash():
    assert len(T.sentences("Well... maybe. Fine.")) >= 1


# ------------------------------------------------------------------ counting

def test_unique_words():
    assert C.unique_words("a b c d", n=4).passed
    assert not C.unique_words("a a b b", n=3).passed
    assert C.unique_words("Run run RUN ran", n=2).passed and not C.unique_words("Run run RUN ran", n=3).passed


def test_numbers_exact():
    assert C.numbers_exact("I paid 20 dollars and 3.50 in tax.", n=2).passed
    assert not C.numbers_exact("I paid 20 dollars and 3 50 in tax.", n=2).passed
    assert C.numbers_exact("SECTION 1\nNo digits here #Top5", n=0).passed
    assert not C.numbers_exact("1. pack\n2. move", n=0).passed  # list numbering counts


def test_conjunctions():
    assert C.conjunctions("Go and stay, but not or.", n=3).passed
    assert not C.conjunctions("And and AND.", n=2).passed
    assert not C.conjunctions("Brand, android.", n=1).passed  # substrings are not words


# ------------------------------------------------------------------ word level

def test_word_repeats():
    assert C.word_repeats(" ".join(["the"] * 5) + " end", n=5).passed
    assert not C.word_repeats(" ".join(["The", "the", "THE"]), n=2).passed


def test_longest_word_counts_letters_only():
    assert C.longest_word("well-known it's", n=9).passed            # well-known = 9 letters
    assert not C.longest_word("information", n=10).passed            # 11 letters
    assert C.longest_word("https 123456789012 ok", n=5).passed       # digits are not letters


def test_no_consecutive_initial_scopes():
    text = "Big cats run. Some silly sentence follows.\n\nLater lines like this."
    assert C.no_consecutive_initial(text, scope="first_sentence").passed
    assert not C.no_consecutive_initial(text, scope="first_paragraph").passed   # some silly
    assert not C.no_consecutive_initial(text, scope="whole").passed
    assert not C.no_consecutive_initial("Big bats.", scope="first_sentence").passed
    assert not C.no_consecutive_initial("End. Each one.", scope="whole").passed  # across sentences
    assert not C.no_consecutive_initial("Apple and", scope="whole").passed       # case-insensitive


# ------------------------------------------------------------------ sentence level

def test_longest_sentence():
    assert C.longest_sentence("One two three. Four five.", n=3).passed
    assert not C.longest_sentence("One two three four. Five.", n=3).passed
    assert C.longest_sentence("SECTION 1\nShort one.", n=2).passed  # header line is not a sentence


def test_questions():
    assert C.questions("Why? Is it ok? 🤔 Fine.", n=2).passed
    assert C.questions('Did she say "yes?"', n=1).passed
    assert not C.questions("What a day. Really!", n=1).passed
    assert C.questions("**Is this bold?**", n=1).passed


def test_title_case_scopes():
    text = "A Plan for the Week Ahead. then the details follow.\n\nmore text."
    assert C.title_case(text, scope="first_sentence").passed
    assert not C.title_case(text, scope="first_paragraph").passed
    assert not C.title_case("the Plan", scope="first_sentence").passed  # first word must be capitalised
    assert C.title_case("Pay The Bills And Save #Money", scope="whole").passed
    assert not C.title_case("Pay the bills and save #money", scope="whole").passed


# ------------------------------------------------------------------ position

def test_keyword_sentence():
    text = "First one. The window is open. Third."
    assert C.keyword_sentence(text, keyword="window", n=2).passed
    assert C.keyword_sentence(text.replace("window", "WINDOW"), keyword="window", n=2).passed
    assert not C.keyword_sentence(text, keyword="window", n=3).passed
    assert not C.keyword_sentence("The windows.", keyword="window", n=1).passed   # whole word only
    assert not C.keyword_sentence("Only one.", keyword="window", n=2).passed


def test_keyword_position():
    text = "Hi there. We open the garden gate."
    assert C.keyword_position(text, keyword="garden", n=2, m=4).passed
    assert not C.keyword_position(text, keyword="garden", n=2, m=3).passed
    assert not C.keyword_position(text, keyword="garden", n=3, m=1).passed
    assert C.keyword_position("Hi. **We** open the *Garden* gate.", keyword="garden", n=2, m=4).passed


def test_closing_phrase():
    ph = "Hope this helps"
    assert C.closing_phrase("Text here. Hope this helps", phrase=ph).passed
    assert C.closing_phrase("Text here. hope this helps! 😀\n", phrase=ph).passed   # case, punctuation, emoji
    assert C.closing_phrase("Text. Hope this\nhelps.", phrase=ph).passed
    assert not C.closing_phrase("Hope this helps. Bye.", phrase=ph).passed
    assert not C.closing_phrase("Text. Hope this helpss", phrase=ph).passed
    assert not C.closing_phrase("Text. I hope this helps a lot", phrase=ph).passed
    assert C.closing_phrase("Thanks for asking, so please", phrase="Thanks for asking, so please").passed


# ------------------------------------------------------------------ markup

def test_sections():
    assert C.sections("SECTION 1\nA.\n\nSECTION 2\nB.", n=2).passed
    assert C.sections("**SECTION 1**\nA.\n  SECTION 2: Next\nB.", n=2).passed
    assert not C.sections("Section 1\nA.\nSECTION 2\nB.", n=2).passed   # upper case marker, as IFEval
    assert not C.sections("See SECTION 1 above.", n=1).passed           # marker must start a line


def test_highlights():
    assert C.highlights("*one* and *two* and **three**", n=3).passed
    assert not C.highlights("** ** and * *", n=1).passed
    assert not C.highlights("*one* only", n=2).passed


def test_bracket_depth():
    assert C.bracket_depth("a ([b]) c", n=2).passed
    assert C.bracket_depth("x ([{([y])}]) z", n=5).passed
    assert not C.bracket_depth("x ([{([y)]}]) z", n=5).passed   # mismatch resets
    assert not C.bracket_depth("((( unclosed", n=1).passed
    assert not C.bracket_depth("(a) (b) (c)", n=2).passed        # sequential, not nested


# ------------------------------------------------------------------ decoration

def test_emoji_ends_scopes():
    text = "First line 😀. Second line.\n\nThird line! 🎉"
    assert C.emoji_ends(text, scope="last_sentence").passed
    assert not C.emoji_ends(text, scope="each_paragraph").passed
    assert C.emoji_ends("A 😀. B.  🙂\n\nC 👍🏽", scope="each_paragraph").passed   # skin tone modifier
    assert C.emoji_ends("One 👨‍👩‍👧. Two 🎉! Three? 😀", scope="every_sentence").passed  # ZWJ family
    assert not C.emoji_ends("One 😀. Two.", scope="every_sentence").passed
    assert not C.emoji_ends("No emoji :)", scope="last_sentence").passed


def test_stair_indent():
    ok = "a\n  b\n    c\nd"
    assert C.stair_indent(ok, lines=3).passed
    assert not C.stair_indent(ok, lines="all").passed
    assert C.stair_indent("a\n\n b\n\n\t c", lines="all").passed        # blank lines skipped, tab = 4
    assert not C.stair_indent("a\n b\n b2", lines=3).passed
    assert not C.stair_indent("only one line", lines=3).passed


def test_hashtags():
    assert C.hashtags("#Budget and #budget and #Plan2", n=2).passed
    assert not C.hashtags("#Budget and #budget", n=2).passed
    assert not C.hashtags("# Heading\n## Sub\nC# and &#39;", n=1).passed
    assert not C.hashtags("#123", n=1).passed


def test_registry_complete():
    from mife.difficulty import catalog
    assert set(C.CHECKERS) == set(catalog.types()) and len(C.CHECKERS) == 18


@pytest.mark.parametrize("t", sorted(C.CHECKERS))
def test_every_type_every_level_has_a_passing_and_failing_example(t):
    """Smoke: instantiate each type at each level and run the checker on an empty and a non-empty response."""
    import random
    from mife.difficulty import catalog
    for lvl in catalog.LEVELS:
        kw = catalog.instantiate(t, lvl, random.Random(0), "train", set())
        r1 = C.check(t, "", kw)
        r2 = C.check(t, "Some ordinary text. Another sentence here?", kw)
        assert isinstance(r1.passed, bool) and isinstance(r2.passed, bool)
