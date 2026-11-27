"""18 deterministic checkers for the difficulty-controlled dataset (rule_levels_v0).

Each checker takes the response text and the rule's parameters and returns a ``Result`` (passed + a short detail).
All of them use the shared definitions in ``mife.difficulty.text``. The checking logic for published types follows
IFBench (allenai/IFBench @ 488918665d53f867fa908a293769ec6e71e74aec, Apache-2.0) and IFEval
(google-research instruction_following_eval), adapted to the pinned definitions and to our scope/length parameters:

* unique_words           IFBench count:unique_word_count   (>= n distinct words)
* numbers_exact          IFBench count:numbers             (== n numbers; decimals and SECTION/hashtag digits handled)
* conjunctions           IFBench count:conjunctions        (>= n different of and/but/for/nor/or/so/yet)
* word_repeats           IFBench words:repeats             (no word more than n times)
* longest_word           new                               (every word <= n letters)
* no_consecutive_initial IFBench words:no_consecutive      (scope: first sentence / first paragraph / whole)
* longest_sentence       new                               (every sentence <= n words)
* questions              new                               (>= n sentences ending in '?')
* title_case             IFBench format:title_case         (scope; pinned minor-word list)
* keyword_sentence       IFBench sentence:keyword          (keyword in sentence n)
* keyword_position       IFBench words:keywords_specific_position (keyword is word m of sentence n)
* closing_phrase         IFEval startend:end_checker       (ends with the exact phrase; trailing emoji allowed)
* sections               IFEval detectable_format:multiple_sections (>= n SECTION markers)
* highlights             IFEval detectable_format:number_highlighted_sections (>= n *spans*)
* bracket_depth          IFBench format:parentheses        (a properly closed nesting of depth >= n)
* emoji_ends             IFBench format:emoji              (scope: last sentence / each paragraph / every sentence)
* stair_indent           IFBench format:line_indent        (first 3 / first 6 / all non-blank lines strictly deeper)
* hashtags               new                               (>= n different #hashtags)
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable

from . import text as T

CHECKER_VERSION = "difficulty-checkers-v0"
CONJUNCTIONS = ("and", "but", "for", "nor", "or", "so", "yet")  # IFBench's list
# Words that may stay lower case in title case (pinned; articles, coordinating conjunctions, short prepositions).
TITLE_MINOR = frozenset("a an the and but or nor for so yet as at by in of off on per to up via vs with from into onto "
                        "over than".split())


@dataclass(frozen=True)
class Result:
    passed: bool
    detail: str = ""


# ----------------------------------------------------------------------------------------------- scopes

def _scope_text(text: str, scope: str) -> str:
    if scope == "whole":
        return T.remove_section_lines(T.normalize(text))
    if scope in ("first_paragraph", "first_two_paragraphs"):
        ps = T.paragraphs(text)
        return "\n\n".join(ps[: 1 if scope == "first_paragraph" else 2])
    if scope in ("first_sentence", "first_two_sentences"):
        ss = T.sentences(text)
        return " ".join(ss[: 1 if scope == "first_sentence" else 2])
    raise ValueError(f"unknown scope {scope!r}")


# ----------------------------------------------------------------------------------------------- counting

def unique_words(text: str, n: int) -> Result:
    k = len(set(T.words(text)))
    return Result(k >= n, f"{k} unique words (need >= {n})")


def numbers_exact(text: str, n: int) -> Result:
    nums = T.numbers(text)
    return Result(len(nums) == n, f"{len(nums)} numbers {nums[:12]} (need exactly {n})")


def conjunctions(text: str, n: int) -> Result:
    found = sorted(set(T.words(text)) & set(CONJUNCTIONS))
    return Result(len(found) >= n, f"{len(found)} different conjunctions {found} (need >= {n})")


# ----------------------------------------------------------------------------------------------- word level

def word_repeats(text: str, n: int) -> Result:
    c = Counter(T.words(text))
    worst = c.most_common(1)
    if not worst:
        return Result(True, "no words")
    w, k = worst[0]
    return Result(k <= n, f"most frequent word {w!r} x{k} (max {n})")


def longest_word(text: str, n: int) -> Result:
    ws = T.words(text)
    if not ws:
        return Result(True, "no words")
    w = max(ws, key=T.letters_in)
    return Result(T.letters_in(w) <= n, f"longest word {w!r} has {T.letters_in(w)} letters (max {n})")


def no_consecutive_initial(text: str, scope: str) -> Result:
    ws = T.words(_scope_text(text, scope))
    for a, b in zip(ws, ws[1:]):
        if a[0] == b[0]:
            return Result(False, f"{a!r} {b!r} share a first letter (scope {scope})")
    if not ws:
        return Result(False, f"no words in scope {scope}")
    return Result(True, f"{len(ws)} words checked (scope {scope})")


# ----------------------------------------------------------------------------------------------- sentence level

def longest_sentence(text: str, n: int) -> Result:
    ss = T.sentences(text)
    if not ss:
        return Result(True, "no sentences")
    lens = [len(T.words(s)) for s in ss]
    i = max(range(len(lens)), key=lens.__getitem__)
    return Result(lens[i] <= n, f"longest sentence #{i + 1} has {lens[i]} words (max {n})")


def questions(text: str, n: int) -> Result:
    k = sum(T.sentence_core_end(s).endswith("?") for s in T.sentences(text))
    return Result(k >= n, f"{k} questions (need >= {n})")


def title_case(text: str, scope: str) -> Result:
    ws = T.words(_scope_text(text, scope), keep_case=True)
    if not ws:
        return Result(False, f"no words in scope {scope}")
    for i, w in enumerate(ws):
        if w.lower() in TITLE_MINOR and i > 0:
            continue
        if not w[0].isupper():
            return Result(False, f"word {i + 1} {w!r} is not capitalised (scope {scope})")
    return Result(True, f"{len(ws)} words title case (scope {scope})")


# ----------------------------------------------------------------------------------------------- position

def keyword_sentence(text: str, keyword: str, n: int) -> Result:
    ss = T.sentences(text)
    if len(ss) < n:
        return Result(False, f"only {len(ss)} sentences (keyword must be in sentence {n})")
    ok = keyword.lower() in T.words(ss[n - 1])
    return Result(ok, f"sentence {n}: {ss[n - 1][:80]!r}")


def keyword_position(text: str, keyword: str, n: int, m: int) -> Result:
    ss = T.sentences(text)
    if len(ss) < n:
        return Result(False, f"only {len(ss)} sentences (need sentence {n})")
    ws = T.words(ss[n - 1])
    if len(ws) < m:
        return Result(False, f"sentence {n} has only {len(ws)} words (need word {m})")
    return Result(ws[m - 1] == keyword.lower(), f"word {m} of sentence {n} is {ws[m - 1]!r} (need {keyword.lower()!r})")


def _phrase_norm(s: str) -> str:
    s = T.sentence_core_end(T.normalize(s))
    s = s.rstrip(".!?;:,…").rstrip()
    s = T.sentence_core_end(s)
    return re.sub(r"\s+", " ", s).lower()


def closing_phrase(text: str, phrase: str) -> Result:
    resp, ph = _phrase_norm(text), _phrase_norm(phrase)
    ok = resp.endswith(ph) and (len(resp) == len(ph) or not resp[-len(ph) - 1].isalnum())
    return Result(ok, f"ends with {resp[-len(ph) - 10:]!r}")


# ----------------------------------------------------------------------------------------------- markup

def sections(text: str, n: int) -> Result:
    k = len(T.SECTION_LINE_RE.findall(T.normalize(text)))
    return Result(k >= n, f"{k} SECTION markers (need >= {n})")


def highlights(text: str, n: int) -> Result:
    t = T.normalize(text)
    single = [h for h in re.findall(r"\*[^\n\*]*\*", t) if h.strip("*").strip()]
    double = [h for h in re.findall(r"\*\*[^\n\*]*\*\*", t) if h.removeprefix("**").removesuffix("**").strip()]
    k = len(single) + len(double)
    return Result(k >= n, f"{k} highlighted spans (need >= {n})")


_OPEN, _CLOSE = "([{", ")]}"


def max_closed_depth(text: str) -> int:
    stack: list[str] = []
    best = 0
    for ch in T.normalize(text):
        if ch in _OPEN:
            stack.append(ch)
        elif ch in _CLOSE:
            if stack and _OPEN.index(stack[-1]) == _CLOSE.index(ch):
                best = max(best, len(stack))
                stack.pop()
            else:
                stack = []  # mismatch resets, as IFBench
    return best


def bracket_depth(text: str, n: int) -> Result:
    d = max_closed_depth(text)
    return Result(d >= n, f"max closed bracket depth {d} (need >= {n})")


# ----------------------------------------------------------------------------------------------- decoration

def _ends_with_emoji(sentence: str) -> bool:
    s = sentence.rstrip()
    while s and s[-1] in ".!?;:,\"')]}*_" and not T.is_emoji_end(s):
        s = s[:-1].rstrip()
    return T.is_emoji_end(s)


def emoji_ends(text: str, scope: str) -> Result:
    ps = T.paragraphs(text)
    if not ps:
        return Result(False, "empty response")
    if scope == "last_sentence":
        targets = [T.sentences(text)[-1:]]
    elif scope == "first_and_last_sentence":
        ss = T.sentences(text)
        targets = [ss[:1] + ss[-1:]]
    elif scope == "each_paragraph":
        targets = [T.sentences_of_paragraph(p)[-1:] for p in ps]
    elif scope == "every_sentence":
        targets = [T.sentences(text)]
    else:
        raise ValueError(scope)
    sents = [s for grp in targets for s in grp]
    if not sents:
        return Result(False, "no sentences")
    for s in sents:
        if not _ends_with_emoji(s):
            return Result(False, f"no emoji at end of {s[-60:]!r} (scope {scope})")
    return Result(True, f"{len(sents)} sentence ends checked (scope {scope})")


def stair_indent(text: str, lines) -> Result:
    ls = T.lines_nonblank(text)
    need = len(ls) if lines == "all" else int(lines)
    if len(ls) < max(need, 2):
        return Result(False, f"only {len(ls)} non-blank lines (need {need})")
    widths = [T.indent_width(x) for x in ls[:need]]
    for i in range(1, len(widths)):
        if widths[i] <= widths[i - 1]:
            return Result(False, f"line {i + 1} indent {widths[i]} <= line {i} indent {widths[i - 1]}")
    return Result(True, f"indents {widths[:8]}{'...' if len(widths) > 8 else ''}")


def hashtags(text: str, n: int) -> Result:
    tags = sorted(set(T.hashtags(text)))
    return Result(len(tags) >= n, f"{len(tags)} different hashtags {tags[:8]} (need >= {n})")


# ----------------------------------------------------------------------------------------------- registry

CHECKERS: dict[str, Callable[..., Result]] = {
    "unique_words": unique_words, "numbers_exact": numbers_exact, "conjunctions": conjunctions,
    "word_repeats": word_repeats, "longest_word": longest_word, "no_consecutive_initial": no_consecutive_initial,
    "longest_sentence": longest_sentence, "questions": questions, "title_case": title_case,
    "keyword_sentence": keyword_sentence, "keyword_position": keyword_position, "closing_phrase": closing_phrase,
    "sections": sections, "highlights": highlights, "bracket_depth": bracket_depth,
    "emoji_ends": emoji_ends, "stair_indent": stair_indent, "hashtags": hashtags,
}


def check(type_id: str, text: str, kwargs: dict) -> Result:
    """Run one rule. ``kwargs`` are the instantiated parameters (see catalog.instantiate)."""
    return CHECKERS[type_id](text, **kwargs)
