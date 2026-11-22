"""Plain-meaning checks, one per candidate type (frozen in stage 1, before any audit).

Purpose: a *witness* (a constructed response that proves a set of rules can be satisfied
together) must satisfy each rule in its plain meaning, not only through a verifier loophole.
Example: ``ratio:sentence_type`` ("2:1 declarative to interrogative") is passed by the official
verifier when a response has zero sentences of either kind; the plain-meaning check needs at
least one question and two statements.

Scope (decided 2026-11-22): these checks apply to WITNESSES ONLY (pairwise audit, six-rule
pool check, per-task witnesses). Labels for generated responses always come from the official
IFBench verifiers alone. The checks add requirements on top of the official verifier; they
never relax it.

Every check = G1 (applies to all types) + the type's own condition. Each type has a passing
example, and, where the check adds a condition, a "loophole" example that the official verifier
accepts but the check rejects. Both are tested.

Changing this file changes its hash; ``tests/test_plain_meaning.py`` refuses any change unless
the freeze record is regenerated with Nicky's approval.
"""
from __future__ import annotations

import re
import string
from dataclasses import dataclass, field
from typing import Callable

from . import source

_LETTER_RUN = re.compile(r"[^\W\d_]+")
_PUNCT = "".join(string.punctuation)


def _sentences(text: str) -> list[str]:
    return source._bootstrap()[2].split_into_sentences(text)


def _has_letter(s: str) -> bool:
    return bool(_LETTER_RUN.search(s))


def _words(s: str) -> list[str]:
    """Whitespace words with surrounding punctuation stripped, empty ones dropped."""
    return [w for w in (t.strip(_PUNCT) for t in s.split()) if w]


Result = tuple[bool, str]


def g1(response: str) -> Result:
    """G1, for every type: non-blank, with at least three runs of letters (so the response says something)."""
    if not response or not response.strip():
        return False, "G1: blank response"
    n = len(_LETTER_RUN.findall(response))
    return (n >= 3, "" if n >= 3 else f"G1: only {n} letter runs (need 3)")


def _ok(cond: bool, why: str) -> Result:
    return (True, "") if cond else (False, why)


# --------------------------------------------------------------------------------------------
# Type-specific conditions. Signature: (response, kwargs, task_text) -> (ok, reason)
# --------------------------------------------------------------------------------------------

def _word_count_range(r, kw, t):
    toks = re.findall(r"\w+", r)
    share = sum(_has_letter(x) for x in toks) / max(1, len(toks))
    return _ok(share >= 0.9, f"only {share:.0%} of counted words contain a letter (need 90%)")


def _unique_word_count(r, kw, t):
    u = {w.lower() for w in _words(r) if _has_letter(w)}
    return _ok(len(u) >= kw["N"], f"{len(u)} distinct real words (need {kw['N']})")


def _sentence_type(r, kw, t):
    s = _sentences(r)
    q = sum(x.endswith("?") for x in s)
    d = sum(x.endswith(".") for x in s)
    return _ok(q >= 1 and d >= 2, f"{d} statements and {q} questions (need at least 2 and 1)")


def _sentence_balance(r, kw, t):
    s = _sentences(r)
    c = [sum(x.endswith(e) for x in s) for e in ".?!"]
    return _ok(min(c) >= 1, f"sentence counts .?! = {c} (need at least one of each)")


def _conjunctions(r, kw, t):
    u = {w.lower() for w in _words(r)} & {"and", "but", "for", "nor", "or", "so", "yet"}
    return _ok(len(u) >= kw["small_n"], f"{len(u)} different conjunctions after case/punctuation folding (need {kw['small_n']})")


def _overlap(r, kw, t):
    import nltk
    g = set(nltk.ngrams(nltk.word_tokenize(r), 3))
    return _ok(len(g) >= 5, f"{len(g)} distinct trigrams (need 5, so the percentage is not set by one or two trigrams)")


def _numbers(r, kw, t):
    plain = re.findall(r"\d+(?:[.,]\d+)*", r)
    return _ok(len(plain) == kw["N"], f"{len(plain)} numbers as a reader counts them (need exactly {kw['N']})")


def _alphabet(r, kw, t):
    w = r.translate(str.maketrans("", "", string.punctuation)).split()
    if not all(x.isalpha() for x in w):
        return False, "a word contains a non-letter"
    single = sum(len(x) == 1 for x in w) / max(1, len(w))
    return _ok(single <= 0.2, f"{single:.0%} single-letter words (at most 20%)")


def _vowel(r, kw, t):
    n = len(_words(r))
    return _ok(n >= 5, f"{n} words (need 5 for a paragraph)")


def _palindrome(r, kw, t):
    w = r.translate(str.maketrans("", "", string.punctuation)).lower().split()
    p = {x for x in w if len(x) >= 5 and x == x[::-1] and x.isalpha() and len(set(x)) >= 2}
    return _ok(len(p) >= 10, f"{len(p)} distinct real palindromes (need 10)")


def _at_least_two_sentences(r, kw, t):
    n = len(_sentences(r))
    return _ok(n >= 2, f"{n} sentence (need 2 so the cross-sentence rule is exercised)")


def _parentheses(r, kw, t):
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: list[tuple[str, int]] = []
    for i, ch in enumerate(r):
        if ch in "([{":
            stack.append((ch, i))
        elif ch in pairs:
            if not stack or stack[-1][0] != pairs[ch]:
                stack = []
                continue
            _, start = stack.pop()
            if len(stack) + 1 >= 5 and _has_letter(r[start + 1:i]):
                return True, ""
    return False, "no correctly matched group at depth 5 that contains words"


def _is_apostrophe(r, i):
    return 0 < i < len(r) - 1 and r[i - 1].isalpha() and r[i + 1].isalpha()


def _quotes(r, kw, t):
    stack: list[tuple[str, int]] = []
    deep_ok = False
    for i, ch in enumerate(r):
        if ch not in "\"'" or (ch == "'" and _is_apostrophe(r, i)):
            continue
        if stack and stack[-1][0] == ch:
            _, start = stack.pop()
            depth = len(stack) + 1
            chars = [c for c, _ in stack] + [ch]
            alternating = all(a != b for a, b in zip(chars, chars[1:]))
            if depth >= 3 and alternating and _has_letter(r[start + 1:i]):
                deep_ok = True
            if depth == 1 and deep_ok:
                return True, ""
            if depth == 1:
                deep_ok = False
        else:
            stack.append((ch, i))
    return False, "no closed, alternating \"/' nesting three deep with words inside"


def _letters_only_words(r, kw, t):
    w = r.translate(str.maketrans("", "", string.punctuation)).split()
    return _ok(all(_has_letter(x) for x in w), "a counted 'word' has no letters (a number)")


def _newline(r, kw, t):
    n = len([x for x in r.split("\n") if x.strip()])
    return _ok(n >= 2, f"{n} line (need at least two words, one per line)")


def _sentence_words(r, kw, t):
    s = _sentences(r)
    return _ok(all(len(_words(x)) >= 2 for x in s), "a sentence has fewer than two words")


def _words_japanese(r, kw, t):
    n = len(r.split())
    return _ok(n >= kw["N"], f"{n} words, so no word sits at position {kw['N']} (the rule is never exercised)")


def _first_n_sentences(key):
    def f(r, kw, t):
        s = _sentences(r)[: int(kw[key])]
        return _ok(all(len(_words(x)) >= 2 for x in s), f"one of the first {kw[key]} sentences has fewer than two words")
    return f


def _paragraph_last_first(r, kw, t):
    p = [x for x in r.split("\n") if x.strip()]
    return _ok(all(len(_words(x)) >= 3 for x in p), "a paragraph has fewer than three words")


def _line_indent(r, kw, t):
    n = len([x for x in r.split("\n") if x.strip()])
    return _ok(n >= 3, f"{n} lines (need three steps of stairs)")


def _quote_unquote(r, kw, t):
    v = r.replace("“", '"').replace("”", '"')
    parts = v.split('"')
    if len(parts) < 3 or len(parts) % 2 == 0:
        return False, "no complete quoted phrase"
    quoted = parts[1::2]
    after = parts[2::2]
    ok = all(_has_letter(q) for q in quoted) and all(_has_letter(a) for a in after)
    return _ok(ok, "a quoted phrase is empty or is not followed by an unquoted explanation")


def _list(r, kw, t):
    sep = kw["sep"]
    items = [ln.lstrip()[len(sep):] for ln in r.splitlines() if ln.lstrip().startswith(sep)]
    return _ok(len(items) >= 2 and all(_has_letter(x) for x in items), "fewer than two list items with text")


def _thesis(r, kw, t):
    s = r.lstrip()
    for o, c in (("<i>", "</i>"), ("<em>", "</em>")):
        if s.startswith(o) and c in s:
            inner, rest = s[len(o):].split(c, 1)
            ok = len(_words(inner)) >= 3 and _has_letter(rest)
            return _ok(ok, "the italic thesis has fewer than three words or nothing follows it")
    return False, "the response does not begin with an italic (HTML) thesis"


def _sub_bullets(r, kw, t):
    lines = [x.strip() for x in r.split("\n") if x.strip()]
    if sum(ln.startswith("*") for ln in lines) < 1:
        return False, "no '*' bullet"
    if any("*" in ln[1:] for ln in lines) or any("*" in ln for ln in lines if not ln.startswith("*")):
        return False, "'*' is used outside bullet markers"
    cur_bullet, cur_has_sub = None, True
    for ln in lines:
        if ln.startswith("*"):
            if not cur_has_sub or not _has_letter(ln[1:]):
                return False, "a bullet has no text or no '-' sub-bullet"
            cur_bullet, cur_has_sub = ln, False
        elif ln.startswith("-") and cur_bullet is not None:
            if not _has_letter(ln[1:]):
                return False, "an empty sub-bullet"
            cur_has_sub = True
    return _ok(cur_has_sub, "the last bullet has no '-' sub-bullet")


def _no_bullets_bullets(r, kw, t):
    b = [x.strip()[1:] for x in r.split("\n") if x.strip().startswith("*")]
    return _ok(all(_has_letter(x) for x in b), "a bullet has no text")


def _words_position(r, kw, t):
    n = len(source.instructions_module()._word_tokens_without_punctuation(r))
    return _ok(n >= 4, f"{n} words, so the 2nd and 2nd-to-last word are the same word (need 4)")


_LABELS = ("My Answer:", "My Conclusion:", "Future Outlook:")


def _output_template(r, kw, t):
    pos = [r.find(x) for x in _LABELS]
    if min(pos) < 0 or pos != sorted(pos):
        return False, "template labels missing or out of order"
    ends = pos[1:] + [len(r)]
    ok = all(_has_letter(r[p + len(lab):e]) for p, lab, e in zip(pos, _LABELS, ends))
    return _ok(ok, "a template slot is left empty")


def _none(r, kw, t):
    return True, ""


# --------------------------------------------------------------------------------------------
# The registry. ``rule`` is the written definition (what the docs show and what Nicky reviews).
# --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Example:
    response: str
    kwargs: dict = field(default_factory=dict)
    task_text: str | None = None


@dataclass(frozen=True)
class Check:
    rule: str
    fn: Callable
    ok: Example
    loophole: Example | None = None


G1_ONLY = "G1 only: the official verifier already captures the plain meaning once the response says something."

_KW5 = {"keyword1": "cat", "keyword2": "dog", "keyword3": "sun", "keyword4": "sea", "keyword5": "sky"}
_TASK = "Write a short story about a lighthouse keeper who finds a message in a bottle on the beach."

CHECKS: dict[str, Check] = {
    "count:word_count_range": Check(
        "At least 90% of the counted words (\\w+ tokens) contain a letter, so the range is not met by padding with numbers.",
        _word_count_range,
        Example("The river runs past the old mill.", {"min_words": 5, "max_words": 8}),
        Example("Yes no maybe 1 2 3 4 5", {"min_words": 5, "max_words": 8})),
    "count:unique_word_count": Check(
        "At least N distinct words that contain a letter (case and surrounding punctuation folded); numbers and bare punctuation do not count.",
        _unique_word_count,
        Example("Cats chase small grey mice daily.", {"N": 5}),
        Example("Red blue green 1 2", {"N": 5})),
    "ratio:stop_words": Check(G1_ONLY, _none, Example("Bright rivers carve deep canyons slowly.", {"percentage": 40})),
    "ratio:sentence_type": Check(
        "At least one question and at least two statements (the 2:1 ratio is not met by having neither).",
        _sentence_type,
        Example("Rain fell all night. The roads flooded. Will it stop soon?"),
        Example("Wow, what a storm!")),
    "ratio:sentence_balance": Check(
        "At least one statement, one question and one exclamation (balance is not met by having none).",
        _sentence_balance,
        Example("The sun rose. Is it warm? What a day!"),
        Example("Look at that storm cloud")),
    "count:conjunctions": Check(
        "At least small_n different coordinating conjunctions after folding case and punctuation ('And', 'and,' and 'AND' are one).",
        _conjunctions,
        Example("I ran and hid, but I was seen, so I left.", {"small_n": 3}),
        Example("And and, AND we went home.", {"small_n": 3})),
    "count:person_names": Check(G1_ONLY, _none, Example("Emma met Liam at the market.", {"N": 2})),
    "ratio:overlap": Check(
        "At least 5 distinct word trigrams in the response, so the overlap percentage is not set by one or two trigrams. The reference text is the task's own text.",
        _overlap,
        Example("Write a short story about a lighthouse keeper who sang loudly to the gulls at dawn and dusk.", {"percentage": 41}, _TASK),
        Example("Write a short note", {"percentage": 50}, _TASK)),
    "count:numbers": Check(
        "The number of numbers a reader counts (digit runs, with a decimal point or thousands comma inside a number) also equals N; '2-3' is two numbers, not one.",
        _numbers,
        Example("We bought 3 apples and 12 pears.", {"N": 2}),
        Example("The score was 2-3 today.", {"N": 1})),
    "words:alphabet": Check(
        "Every word is made of letters only, and at most 20% of words are single letters (the loop is not met by writing 'A b c d').",
        _alphabet,
        Example("Apples bloom carefully during each fall."),
        Example("A b c d e")),
    "words:vowel": Check(
        "At least 5 words, so it is a paragraph.",
        _vowel,
        Example("Tom sat on a red mat today."),
        Example("Hi yes no")),
    "words:consonants": Check(G1_ONLY, _none, Example("Strong brown crabs crawl.")),
    "sentence:alliteration_increment": Check(
        "At least 2 sentences, so the increase is exercised.",
        _at_least_two_sentences,
        Example("Big bears roam. Silly snakes slide slowly."),
        Example("Cats like milk.")),
    "words:palindrome": Check(
        "At least 10 DISTINCT palindromes of 5+ letters, each made of letters only and with at least two different letters ('aaaaa' and repeats do not count).",
        _palindrome,
        Example("level radar civic kayak rotor refer madam stats tenet racecar"),
        Example(" ".join(["level"] * 10))),
    "count:punctuation": Check(G1_ONLY, _none, Example("Wait?! Yes, really: go; now. Hi! Why?")),
    "format:parentheses": Check(
        "A correctly matched group at nesting depth 5 that contains words.",
        _parentheses,
        Example("Note (one [two {three (four [five])}])"),
        Example("Look at this ((((()))))")),
    "format:quotes": Check(
        "Quotes nested at least 3 deep, alternating \" and ' at each level, closed back to the outside, with words in the innermost quote. Apostrophes inside words are not quotes.",
        _quotes,
        Example("He said \"she told me 'they shouted \"run\" loudly' today\" again."),
        Example("He said \"I heard 'a \"\"' here\" ok")),
    "words:prime_lengths": Check(
        "Every word contains a letter (numbers are not words).",
        _letters_only_words,
        Example("We are all at the sea"),
        Example("Hi 12 we 123 go")),
    "format:newline": Check(
        "At least two words, one per line.",
        _newline,
        Example("Rivers\nflow\nslowly")),
    "format:emoji": Check(G1_ONLY, _none, Example("I love the sun \U0001F31E. Rain is fun too \U0001F600.")),
    "ratio:sentence_words": Check(
        "Each of the three sentences has at least two words.",
        _sentence_words,
        Example("Cats nap. Dogs run. Owls fly."),
        Example("Cats. Dogs. Owls.")),
    "count:words_japanese": Check(
        "At least N words, so at least one word position is required to be Japanese.",
        _words_japanese,
        Example("I like 寿司 and also 猫 very", {"N": 3}),
        Example("Hello there my friend", {"N": 5})),
    "words:start_verb": Check(G1_ONLY, _none, Example("Write the letter today.")),
    "words:repeats": Check(G1_ONLY, _none, Example("The cat sat on a mat.", {"small_n": 2})),
    "sentence:keyword": Check(
        "Sentences 1 to N each have at least two words (the position is not reached by one-word sentences).",
        _first_n_sentences("N"),
        Example("I walked out. The river was calm.", {"word": "river", "N": 2}),
        Example("Hi. River runs fast.", {"word": "river", "N": 2})),
    "count:pronouns": Check(G1_ONLY, _none, Example("I told her that we would go.", {"N": 3})),
    "words:odd_even_syllables": Check(
        "Every word contains a letter (numbers are not words).",
        _letters_only_words,
        Example("cat apple dog table"),
        Example("cat 12 dog 34 sun")),
    "words:last_first": Check(
        "At least 2 sentences, so the chaining is exercised.",
        _at_least_two_sentences,
        Example("I saw the dog. Dog barked at me."),
        Example("The cat slept all day.")),
    "words:paragraph_last_first": Check(
        "Every paragraph has at least three words (a one-word paragraph starts and ends with itself).",
        _paragraph_last_first,
        Example("rain falls on the hills of rain\nsun warms every sun"),
        Example("Hello\nWorld\nAgain")),
    "sentence:increment": Check(
        "At least 2 sentences, so the increment is exercised.",
        _at_least_two_sentences,
        Example("Birds sing. Birds sing very loud.", {"small_n": 2}),
        Example("Birds sing all morning.", {"small_n": 2})),
    "words:no_consecutive": Check(G1_ONLY, _none, Example("The big cat ran home.")),
    "format:line_indent": Check(
        "At least three non-empty lines, so there are stairs.",
        _line_indent,
        Example("Step one\n  Step two\n    Step three"),
        Example("Just one line here")),
    "format:quote_unquote": Check(
        "At least one double-quoted phrase; every quoted phrase has words and is followed by unquoted text with words.",
        _quote_unquote,
        Example("He said \"go home\" which meant leave."),
        Example("There are no quotes here at all.")),
    "format:list": Check(
        "At least two items that start with the separator and have text after it.",
        _list,
        Example("- apples\n- pears\n- plums", {"sep": "-"}),
        Example("My two items are here\n-\n-", {"sep": "-"})),
    "format:thesis": Check(
        "The response begins with an HTML-italic thesis (<i> or <em>) of at least three words, followed by non-italic text. The response is treated as one section unless it has explicit section headers.",
        _thesis,
        Example("<i>Cities need more trees.</i> Trees cool streets."),
        Example("Some intro text <i>x</i> more")),
    "format:sub-bullets": Check(
        "At least one '*' bullet line; every bullet has text and is followed by at least one '-' sub-bullet line with text; '*' appears only as bullet markers.",
        _sub_bullets,
        Example("* Fruit\n- apple\n* Veg\n- leek"),
        Example("Just text, no bullets at all.")),
    "format:no_bullets_bullets": Check(
        "Every '*' bullet has text.",
        _no_bullets_bullets,
        Example("I like fruit. I also like veg.\n* apples\n* leeks"),
        Example("I like fruit. I like veg.\n*\n*")),
    "count:keywords_multiple": Check(
        G1_ONLY, _none,
        Example("cat dog dog sun sun sun sea sea sea sea sea sky sky sky sky sky sky sky", _KW5)),
    "words:keywords_specific_position": Check(
        "Sentences 1 to n each have at least two words.",
        _first_n_sentences("n"),
        Example("We walked far. The calm river flowed.", {"keyword": "river", "n": 2, "m": 3}),
        Example("Hi. A calm river flows.", {"keyword": "river", "n": 2, "m": 3})),
    "words:words_position": Check(
        "At least four words, so the second and second-to-last words are different positions.",
        _words_position,
        Example("The river is wide, river banks", {"keyword": "river"}),
        Example("A river flows", {"keyword": "river"})),
    "format:title_case": Check(G1_ONLY, _none, Example("The Quick Brown Fox Jumps")),
    "format:output_template": Check(
        "The three labels appear in order and each slot contains words.",
        _output_template,
        Example("My Answer: yes\nMy Conclusion: it works\nFuture Outlook: bright"),
        Example("My Answer: My Conclusion: Future Outlook:")),
    "format:no_whitespace": Check(G1_ONLY, _none, Example("Rivers,flow,slowly.")),
}


def check(type_id: str, response: str, kwargs: dict | None = None, task_text: str | None = None) -> Result:
    """Plain-meaning check for one rule: G1, then the type's own condition."""
    ok, why = g1(response)
    if not ok:
        return ok, why
    return CHECKS[type_id].fn(response, kwargs or {}, task_text)


def witness_ok(type_id: str, kwargs: dict, response: str, task_text: str | None = None) -> Result:
    """Official strict verifier AND the plain-meaning check (what a witness must pass)."""
    if not source.official_check(type_id, kwargs, response, task_text):
        return False, "official verifier fails"
    return check(type_id, response, kwargs, task_text)
