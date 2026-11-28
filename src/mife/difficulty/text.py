"""Shared, pinned text definitions used by every difficulty-dataset checker.

These follow the plan's "fixed definitions" (rule_levels_v0, validated 2026-11-27):

* Words: runs of letters with internal apostrophes or hyphens, lowercased. Markdown symbols, emoji, digits and SECTION
  header lines are removed first. Hashtag text counts as a word ("#budget" -> "budget").
* Sentences: one pinned splitter (nltk punkt_tab, English, nltk version recorded in ``SPLITTER_ID``). SECTION header
  lines are not sentences. Markdown headings and list items are always their own sentence(s). An emoji may follow
  end punctuation: emoji that punkt attaches to the start of the next sentence are moved back to the sentence they end.
* Paragraphs: blocks separated by blank lines (SECTION header lines removed; empty blocks dropped).
* Numbers: digit runs outside SECTION markers and hashtags; a decimal or thousands-separated number counts once.
* Keyword and phrase matching ignores case. Curly quotes/apostrophes are normalised to straight ones.

Everything here is deterministic and depends only on the response text, ``nltk`` (punkt_tab data) and ``emoji``.
"""
from __future__ import annotations

import functools
import re
import unicodedata

import emoji as _emoji
import nltk

# The pinned definitions, stated word for word in the compatibility judge's prompt (tests check they stay in sync).
DEFINITIONS = """\
- A SECTION header line is a line that starts with the word SECTION in capitals followed by a number (optionally after markdown symbols such as # or **), for example "SECTION 2: Costs". The whole line is a header line.
- Words are runs of letters with internal apostrophes or hyphens, lowercased. Before words are taken, SECTION header lines, markdown symbols, emoji and digits are removed. Hashtag text counts as a word (#budget gives the word budget). A word's length is its number of letters.
- Sentences come from one pinned sentence splitter (nltk punkt, English). SECTION header lines are not sentences. Markdown headings and list items are always sentences of their own. An emoji may follow a sentence's end punctuation and belongs to that sentence; emoji are never words.
- Paragraphs are blocks of text separated by blank lines; SECTION header lines are not part of any paragraph.
- Numbers are digit runs outside SECTION markers and hashtags; a decimal or a number with thousands separators counts once (2.5 and 1,200 are one number each). List numbering counts. Emoji keycaps are not numbers.
- Keyword and closing-phrase matches ignore case. A keyword must appear as a whole word. The closing phrase is compared after removing trailing emoji, whitespace, closing quotes or brackets and end punctuation.
- Indentation is the number of leading spaces on a line (a tab counts as 4); blank lines are skipped. Other rules ignore leading whitespace.
- No rule limits the total length of the answer."""

EMOJI_VERSION = _emoji.__version__
NLTK_VERSION = nltk.__version__
SPLITTER_ID = f"nltk-{NLTK_VERSION}-punkt_tab-english"

# A SECTION marker starts a line: optional markdown decoration, the word SECTION (upper case, as IFEval), a number.
# The whole line is a header line: it is not a sentence, not part of a paragraph, and its words are not words.
SECTION_LINE_RE = re.compile(r"^[ \t]*(?:[#>*_]+[ \t]*)?SECTION[ \t]+\d+\b.*$", re.M)
SECTION_MARKER_RE = re.compile(r"SECTION[ \t]+\d+")
HASHTAG_RE = re.compile(r"(?<![\w#&])#([^\W\d_][\w'\-]*)")
WORD_RE = re.compile(r"[^\W\d_]+(?:['\-][^\W\d_]+)*")
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
LIST_ITEM_RE = re.compile(r"^[ \t]*(?:[-*+•]|\d+[.)])[ \t]+")
HEADING_RE = re.compile(r"^[ \t]*#{1,6}[ \t]+")
BLANK_SPLIT_RE = re.compile(r"\n[ \t]*\n")

_QUOTES = str.maketrans({"‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
                         "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"',
                         "‐": "-", "‑": "-"})


def normalize(text: str) -> str:
    """NFC, straight quotes/apostrophes, ASCII hyphens, \\r\\n -> \\n."""
    return unicodedata.normalize("NFC", text).translate(_QUOTES).replace("\r\n", "\n").replace("\r", "\n")


def remove_section_lines(text: str) -> str:
    return SECTION_LINE_RE.sub("", text)


def remove_emoji(text: str) -> str:
    return _emoji.replace_emoji(text, replace=" ")


def is_emoji_end(s: str) -> bool:
    """True if ``s`` ends with an emoji (any sequence the pinned ``emoji`` package knows, incl. ZWJ and skin tones)."""
    s = s.rstrip()
    if not s:
        return False
    found = _emoji.emoji_list(s)
    return bool(found) and found[-1]["match_end"] == len(s)


def strip_trailing_emoji(s: str) -> str:
    """Remove trailing emoji (and the whitespace around them), repeatedly."""
    s = s.rstrip()
    while s:
        found = _emoji.emoji_list(s)
        if found and found[-1]["match_end"] == len(s):
            s = s[: found[-1]["match_start"]].rstrip()
            # variation selectors / zero-width joiners left behind
            s = s.rstrip("️‍").rstrip()
        else:
            break
    return s


def _clean_for_words(text: str) -> str:
    t = normalize(text)
    t = remove_section_lines(t)
    t = HASHTAG_RE.sub(lambda m: " " + m.group(1) + " ", t)
    t = remove_emoji(t)
    t = re.sub(r"\d", " ", t)
    t = re.sub(r"[*_`#>|~\[\]()]", " ", t)  # markdown symbols (brackets are punctuation for words anyway)
    return t


def words(text: str, keep_case: bool = False) -> list[str]:
    ws = WORD_RE.findall(_clean_for_words(text))
    return ws if keep_case else [w.lower() for w in ws]


def letters_in(word: str) -> int:
    return sum(ch.isalpha() for ch in word)


def numbers(text: str) -> list[str]:
    t = normalize(text)
    t = SECTION_MARKER_RE.sub(" ", t)
    t = HASHTAG_RE.sub(" ", t)
    t = _emoji.replace_emoji(t, replace=" ")  # keycap emoji such as 1️⃣ are emoji, not digits
    return NUMBER_RE.findall(t)


def hashtags(text: str) -> list[str]:
    return [m.lower() for m in HASHTAG_RE.findall(normalize(text))]


@functools.lru_cache(maxsize=1)
def _punkt():
    from nltk.tokenize.punkt import PunktTokenizer
    return PunktTokenizer("english")


def paragraphs(text: str) -> list[str]:
    t = remove_section_lines(normalize(text))
    return [p.strip("\n") for p in BLANK_SPLIT_RE.split(t) if p.strip()]


def _segments(paragraph: str) -> list[str]:
    """Lines of a paragraph -> splitter segments: headings and list items stand alone, other lines are joined."""
    segs, cur = [], []
    for line in paragraph.split("\n"):
        if not line.strip():
            continue
        if HEADING_RE.match(line) or LIST_ITEM_RE.match(line):
            if cur:
                segs.append(" ".join(cur))
                cur = []
            segs.append(line.strip())
        else:
            cur.append(line.strip())
    if cur:
        segs.append(" ".join(cur))
    return segs


def _reattach_emoji(sents: list[str]) -> list[str]:
    """punkt puts an emoji that follows end punctuation at the start of the next sentence; move it back."""
    out: list[str] = []
    for s in sents:
        s = s.strip()
        if out:
            lead = _emoji.emoji_list(s)
            moved = ""
            while lead and lead[0]["match_start"] == 0:
                moved += " " + s[: lead[0]["match_end"]]
                s = s[lead[0]["match_end"]:].lstrip().lstrip("️‍").lstrip()
                lead = _emoji.emoji_list(s)
            if moved:
                out[-1] = out[-1] + moved
        if s and (WORD_RE.search(_clean_for_words(s)) or NUMBER_RE.search(s) or not out):
            out.append(s)
        elif s and out:  # punctuation-only leftovers stay with the sentence before
            out[-1] = out[-1] + " " + s
    return [s for s in out if s.strip()]


def sentences_of_paragraph(paragraph: str) -> list[str]:
    raw: list[str] = []
    for seg in _segments(paragraph):
        raw.extend(_punkt().tokenize(seg))
    return _reattach_emoji(raw)


def sentences(text: str) -> list[str]:
    out: list[str] = []
    for p in paragraphs(text):
        out.extend(sentences_of_paragraph(p))
    return out


_END_STRIP = " \t\n\"')]}*_`"


def sentence_core_end(s: str) -> str:
    """A sentence with trailing emoji, closing quotes/brackets and markdown removed (for '?' and phrase checks)."""
    s = strip_trailing_emoji(s)
    prev = None
    while prev != s:
        prev = s
        s = strip_trailing_emoji(s.rstrip(_END_STRIP))
    return s


def lines_nonblank(text: str) -> list[str]:
    return [ln for ln in normalize(text).split("\n") if ln.strip()]


def indent_width(line: str) -> int:
    """Leading indentation in spaces, a tab counts as 4 (tab stops are not used: each tab adds 4)."""
    n = 0
    for ch in line:
        if ch == " ":
            n += 1
        elif ch == "\t":
            n += 4
        elif ch in " 　":
            n += 1
        else:
            break
    return n
