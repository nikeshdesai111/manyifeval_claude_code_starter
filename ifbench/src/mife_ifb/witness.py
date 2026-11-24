"""Programmatic witness builder (stage 2).

A *witness* for a set of rules is one constructed response that passes every rule's official
IFBench verifier AND its frozen plain-meaning check. Finding one proves the set is satisfiable;
failing to find one proves nothing (decided 2026-11-22: such cases are investigated, then either
proven conflicts or "unsupported by the builder").

No language model is used (decision D3). The builder works in five steps, all seeded:

1. **Plan.** Compile the rules into a plan: number of sentences and their lengths, sentence-end
   punctuation, pinned words (a keyword at sentence n / word m, a verb first, ...), required
   tokens (keywords with exact counts, names, pronouns, conjunctions, palindromes, numbers),
   word filters (vowel set, prime lengths, consonant clusters), sequence rules (alphabet order,
   no shared first letter, syllable parity, alliteration runs), and a layout (prose, list,
   bullets, template, thesis, one word per line, stairs, no whitespace).
2. **Search.** Fill the word slots by depth-first search with backtracking over a filtered
   vocabulary (IFBench's own WORD_LIST plus a small documented supplement), placing required
   tokens and respecting pins, filters, sequence rules and repeat caps.
3. **Render.** Turn the word stream into text: casing, emoji, internal punctuation, nested
   brackets or quotes, layout and indentation.
4. **Repair.** Local search for length objectives (three sentences of equal character length)
   and, for ``ratio:overlap``, choose a span of the task text to append so that the trigram
   overlap lands in the target window.
5. **Verify.** Every rule through the official verifier and the plain-meaning check. Only a
   text that passes all of them is returned.

Each call tries up to ``attempts`` (default 24, as ManyIFEval) independently seeded plans.
"""
from __future__ import annotations

import math
import random
from functools import lru_cache
import sys
from collections import Counter
from dataclasses import dataclass, field

from . import lexicon as lx
from . import plain_meaning as pm
from . import source

sys.setrecursionlimit(10000)

ALPHA = "abcdefghijklmnopqrstuvwxyz"
LAYOUT_TYPES = ["format:output_template", "format:thesis", "format:no_bullets_bullets", "format:sub-bullets",
                "format:list"]
TEMPLATE = ["My Answer:", "My Conclusion:", "Future Outlook:"]
DFS_BUDGET = 40000
ACCEPT_CAP = 40        # in-window overlap texts fully verified per attempt before giving up
COPY_ROUNDS = 6        # copy-mode searches per attempt (two plans, three quota corrections each)
COPY_SPANS = 500       # task spans tried when the copy route falls back to inserting one


class Infeasible(Exception):
    """The plan cannot be made for this attempt (reason in the message)."""


@dataclass
class Plan:
    n: int
    lengths: list
    terms: list
    pins: dict = field(default_factory=dict)          # (sentence, index) -> word; index may be negative
    req: list = field(default_factory=list)            # per sentence: Counter of required tokens
    greq: Counter = field(default_factory=Counter)     # required anywhere (used with sequence rules)
    forbid: set = field(default_factory=set)
    vowels: frozenset | None = None
    prime: bool = False
    cluster: bool = False
    alpha: bool = False
    noconsec: bool = False
    parity: bool = False
    allit: list | None = None
    last_first: bool = False
    para_lf: bool = False
    max_rep: int | None = None
    title: bool = False
    layout: str = "prose"
    sep: str | None = None
    lines: list = field(default_factory=list)          # list of lists of sentence indices
    indent: bool = False
    newline: bool = False
    nows: bool = False
    emoji: bool = False
    punct: bool = False
    parens: bool = False
    quotes: bool = False
    qu: bool = False
    stop_fill: bool = True
    equal_chars: bool = False
    minimal: bool = False
    emoji_attach: bool = False
    tight: bool = False
    interro: bool = False
    layouts: tuple = ()                                # every layout rule present, in LAYOUT_TYPES order
    kinds: list = field(default_factory=list)          # per line, when two or more layouts combine
    nb_first: int = 1                                  # index of the first '*' line of no_bullets_bullets
    types: frozenset = frozenset()                     # the rule types of the set

    @property
    def composite(self) -> bool:
        return len(self.layouts) >= 2


@dataclass
class Result:
    ok: bool
    text: str | None = None
    attempt: int | None = None
    reasons: Counter = field(default_factory=Counter)


# ---------------------------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------------------------

def _static_ok(plan: Plan, w: str, last: bool = False) -> bool:
    f = lx.feat(w)
    if plan.vowels is not None and not f.vowels <= plan.vowels:
        return False
    extra = 1 if (last and plan.emoji_attach) else 0   # an emoji attached to a sentence's last word adds one character
    if plan.prime and (f.length + extra not in lx.PRIMES or not any(c.isalpha() for c in w)):
        return False
    if plan.cluster and not f.cluster:
        return False
    return True


def _need_letters(types) -> bool:
    return bool({"words:alphabet", "words:prime_lengths", "words:odd_even_syllables", "words:consonants"} & set(types))


def _layout_of(types) -> str:
    for t in LAYOUT_TYPES:
        if t in types:
            return t.split(":")[1]
    return "prose"


def make_plan(rules: dict, rng: random.Random, task_text: str | None, copy_len: tuple | None = None) -> Plan:
    """``copy_len`` (overlap copy route only): the default sentence-length range, and fewer sentences,
    so that copied task-text runs are long relative to the sentence boundaries they cannot cross."""
    T = rules
    layout = _layout_of(T)
    plan = Plan(n=0, lengths=[], terms=[], layout=layout)
    plan.layouts = tuple(t.split(":")[1] for t in LAYOUT_TYPES if t in T)
    plan.types = frozenset(T)
    if plan.composite and {"no_bullets_bullets", "sub-bullets"} <= set(plan.layouts):
        raise Infeasible("no_bullets_bullets and sub-bullets cannot share a text (proven conflict)")
    plan.sep = T.get("format:list", {}).get("sep")
    plan.prime = "words:prime_lengths" in T
    plan.cluster = "words:consonants" in T
    plan.alpha = "words:alphabet" in T
    plan.noconsec = "words:no_consecutive" in T
    plan.parity = "words:odd_even_syllables" in T
    plan.last_first = "words:last_first" in T
    plan.para_lf = "words:paragraph_last_first" in T
    plan.title = "format:title_case" in T
    plan.indent = "format:line_indent" in T
    plan.newline = "format:newline" in T
    plan.nows = "format:no_whitespace" in T
    plan.emoji = "format:emoji" in T
    plan.punct = "count:punctuation" in T
    plan.parens = "format:parentheses" in T
    plan.quotes = "format:quotes" in T
    plan.qu = "format:quote_unquote" in T
    plan.equal_chars = "ratio:sentence_words" in T
    shape = bool({"words:alphabet", "words:prime_lengths", "words:consonants", "words:odd_even_syllables",
                  "words:no_consecutive"} & set(T))
    # with word-shape rules, emoji and list/bullet markers are attached to a word instead of standing alone,
    # because the shape verifiers read every whitespace-separated token as a word
    plan.emoji_attach = plan.emoji and shape
    plan.tight = shape or plan.newline
    plan.interro = plan.punct and plan.equal_chars
    plan.max_rep = T["words:repeats"]["small_n"] if "words:repeats" in T else None
    stop_cap = T["ratio:stop_words"]["percentage"] if "ratio:stop_words" in T else None
    plan.stop_fill = stop_cap is None
    overlap = T.get("ratio:overlap")
    plan.minimal = bool(overlap and overlap["percentage"] > 40) and copy_len is None

    # ---- required tokens (global), chosen to pass the static filters ----
    required: list[str] = []
    exact_kw = []
    if "count:keywords_multiple" in T:
        kw = T["count:keywords_multiple"]
        for i, c in zip(range(1, 6), [1, 2, 3, 5, 7]):
            exact_kw.append(kw[f"keyword{i}"].lower())
            required += [kw[f"keyword{i}"].lower()] * c
        plan.forbid |= set(exact_kw)

    def pick(pool, k, what, distinct=True):
        ok = [w for w in pool if _static_ok_pre(T, w)]
        rng.shuffle(ok)
        if distinct:
            if len(ok) < k:
                raise Infeasible(f"only {len(ok)} {what} pass the word filters (need {k})")
            return ok[:k]
        if not ok:
            raise Infeasible(f"no {what} passes the word filters")
        cap = plan.max_rep or 10 ** 9
        out = []
        for w in ok * (k // max(1, len(ok)) + 2):
            if Counter(out)[w] < cap and len(out) < k:
                out.append(w)
        if len(out) < k:
            raise Infeasible(f"repeat cap leaves too few {what}")
        return out

    if "count:person_names" in T:
        required += [w.lower() for w in pick(lx.PERSON_NAMES, T["count:person_names"]["N"], "names")]
    if "count:conjunctions" in T:
        required += pick(lx.CONJUNCTIONS, T["count:conjunctions"]["small_n"], "conjunctions")
    if "count:pronouns" in T:
        pr = [p for p in lx.PRONOUNS if p != "i" or not plan.prime]
        required += pick(pr, T["count:pronouns"]["N"], "pronouns", distinct=False)
    if "words:palindrome" in T:
        required += pick(lx.PALINDROMES, 10, "palindromes")
    if "count:numbers" in T:
        if "words:alphabet" in T:
            raise Infeasible("a number cannot start with a letter (alphabet rule)")
        if _need_letters(T):
            # numbers written as ordinals ("3rd", "21st"): a digit run plus letters, so they are words
            def ordinal(x):
                suf = "th" if 10 <= x % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(x % 10, "th")
                return f"{x}{suf}"
            pool = [ordinal(x) for x in range(2, 99)]
            pool = [w for w in pool if _static_ok_pre(T, w)]
            if len(pool) < T["count:numbers"]["N"]:
                raise Infeasible("too few ordinal numbers pass the word filters")
            nums = rng.sample(pool, T["count:numbers"]["N"])
        else:
            nums = [str(x) for x in rng.sample(range(2, 99), T["count:numbers"]["N"])]
        required += nums
    kw_sentence = None
    if "sentence:keyword" in T:
        kw_sentence = (T["sentence:keyword"]["N"] - 1, T["sentence:keyword"]["word"].lower())

    # ---- vowel set ----
    if "words:vowel" in T:
        need = set()
        for w in required + ([kw_sentence[1]] if kw_sentence else []):
            need |= set(lx.feat(w).vowels)
        for key in ("words:keywords_specific_position", "words:words_position"):
            if key in T:
                need |= set(lx.feat(T[key]["keyword"].lower()).vowels)
        if "output_template" in plan.layouts:
            need |= set("aeiou") & set("my answer conclusion future outlook")
        if len(need) > 3:
            raise Infeasible("required words use more than three vowels")
        rest = [v for v in "aeoiu" if v not in need]
        while len(need) < 3:
            v = rng.choices(rest, weights=[5, 4, 3, 3, 1][:len(rest)])[0]
            need.add(v)
            rest.remove(v)
        plan.vowels = frozenset(need)
    for w in required:
        if not _static_ok(plan, w):
            raise Infeasible(f"required word '{w}' fails the word filters")

    # ---- number of sentences ----
    lo_n = 2 if plan.minimal else rng.randint(1, 3) if copy_len else rng.randint(3, 5)
    fixed_n = None
    if kw_sentence:
        lo_n = max(lo_n, kw_sentence[0] + 1)
    if "words:keywords_specific_position" in T:
        lo_n = max(lo_n, T["words:keywords_specific_position"]["n"])
    mult = 1
    if "ratio:sentence_type" in T or "ratio:sentence_balance" in T:
        mult = 3
        lo_n = max(lo_n, 3)
    if "ratio:sentence_balance" in T and plan.punct:
        lo_n = max(lo_n, 6)
    if plan.equal_chars:
        fixed_n = 3
    if "sentence:alliteration_increment" in T:
        lo_n = max(lo_n, 2)
    if plan.last_first or "sentence:increment" in T:
        lo_n = max(lo_n, 2)
    if plan.composite:
        lo_n = max(lo_n, len(_composite_kinds(plan.layouts, 0)))
    else:
        lo_n = max(lo_n, {"list": 2, "sub-bullets": 2, "no_bullets_bullets": 4, "output_template": 3, "thesis": 2}.get(layout, 1))
    if plan.indent or plan.punct:
        lo_n = max(lo_n, 3)
    if layout == "sub-bullets" and not plan.composite and lo_n % 2:
        lo_n += 1
    n = fixed_n or lo_n
    if fixed_n and fixed_n < lo_n:
        raise Infeasible("exactly three sentences, but another rule needs more")
    while n % mult:
        n += 1
    if fixed_n and n != fixed_n:
        raise Infeasible("three sentences cannot satisfy the sentence-type ratio")
    if layout == "sub-bullets" and not plan.composite and n % 2:
        n += 1

    # ---- sentence lengths ----
    inc = T.get("sentence:increment", {}).get("small_n")
    base_lo, base_hi = (3, 4) if plan.minimal else copy_len or (5, 9)
    if plan.equal_chars:
        base_lo, base_hi = 3, 5
    if plan.parens or plan.quotes:
        base_lo = max(base_lo, 7)
    if plan.qu:
        base_lo = max(base_lo, 4)
    if plan.punct:
        base_lo = max(base_lo, 4)
    base_hi = max(base_hi, base_lo)

    def lengths_for(n, l0):
        if inc:
            return [l0 + i * inc for i in range(n)]
        return [rng.randint(max(base_lo, l0), max(base_hi, l0)) for _ in range(n)]

    l0 = rng.randint(base_lo, base_hi) if not inc else rng.randint(3, 5)
    lengths = lengths_for(n, l0)
    # minimum lengths
    mins = [2] * n
    if "words:keywords_specific_position" in T:
        k = T["words:keywords_specific_position"]
        mins[k["n"] - 1] = max(mins[k["n"] - 1], k["m"] + 1)
    if "sentence:alliteration_increment" in T:
        plan.allit = [0] + [i + 1 for i in range(1, n)]
        mins = [max(a, b + 1) for a, b in zip(mins, plan.allit)]
    if plan.para_lf:
        mins = [max(m, 3) for m in mins]
    if "words:words_position" in T:
        mins[-1] = max(mins[-1], 3)
        if "words:keywords_specific_position" in T:
            k = T["words:keywords_specific_position"]
            if k["n"] - 1 == n - 1:  # keep the keyword pin clear of the second-to-last word
                mins[-1] = max(mins[-1], k["m"] + 2)
    if plan.parens or plan.quotes:
        mins[0] = max(mins[0], 7)
    if plan.punct:
        mins[0] = max(mins[0], 6)

    def fix_mins(lengths):
        if inc:
            need = max(m - i * inc for i, m in enumerate(mins))
            return [max(lengths[0], need) + i * inc for i in range(n)]
        return [max(a, b) for a, b in zip(lengths, mins)]

    lengths = fix_mins(lengths)

    # ---- demands on the total ----
    stop_req = sum(1 for w in required if w in lx.stopwords())
    stop_req += 1 if (kw_sentence and kw_sentence[1] in lx.stopwords()) else 0
    stop_req += 2 if "output_template" in plan.layouts else 0
    need_total = len(required) + n
    if "count:unique_word_count" in T:
        need_total = max(need_total, T["count:unique_word_count"]["N"] + 3 + len(required) - len(set(required)))
    if stop_cap is not None and stop_req:
        need_total = max(need_total, math.ceil(stop_req * 100 / max(stop_cap, 0.5)) + 2)
    if plan.alpha and required:
        same = Counter(w[0] for w in required).most_common(1)[0][1]
        need_total = max(need_total, 26 * (same + 1))
    wc = T.get("count:word_count_range")
    extra = sum({"output_template": 5, "thesis": 2}.get(x, 0) for x in plan.layouts) + (n if plan.sep == "SEPARATOR" else 0)

    def grow(lengths, total):
        if inc:
            while sum(lengths) < total:
                lengths = [x + 1 for x in lengths]
            return lengths
        while sum(lengths) < total:
            lengths[rng.randrange(n)] += 1
        return lengths

    if fixed_n or plan.equal_chars:
        if sum(lengths) < need_total:
            lengths = grow(lengths, need_total)
    else:
        lengths = grow(lengths, need_total)
    if wc:
        lo, hi = wc["min_words"] - extra, wc["max_words"] - extra
        target = rng.randint(max(lo, sum(lengths)), max(hi, sum(lengths)))
        if sum(lengths) > hi:
            raise Infeasible("other rules need more words than the word-count range allows")
        if inc:
            # adjust n and l0 to hit [lo, hi] exactly
            best = None
            for nn in range(n, n + 40, mult):
                for ll in range(max(2, mins[0]), 60):
                    tot = nn * ll + inc * nn * (nn - 1) // 2
                    if lo <= tot <= hi and (not fixed_n or nn == fixed_n):
                        best = (nn, ll)
                        break
                if best:
                    break
            if not best:
                raise Infeasible("no sentence count and start length fit the word-count range with the increment")
            n, l0 = best
            lengths = [l0 + i * inc for i in range(n)]
            mins = (mins + [2] * n)[:n]
            if plan.allit:
                plan.allit = [0] + [i + 1 for i in range(1, n)]
        else:
            if not fixed_n:
                single_sb = layout == "sub-bullets" and not plan.composite
                step = mult * (2 if single_sb and mult % 2 else 1) if single_sb else mult
                if single_sb and step % 2:
                    step *= 2
                while target / n > 18:
                    for _ in range(step):
                        n += 1
                        lengths.append(rng.randint(base_lo, base_hi))
                        mins.append(2)
                        if plan.allit is not None:
                            plan.allit.append(len(plan.allit) + 1)
                            mins[-1] = plan.allit[-1] + 1
                lengths = [max(a, b) for a, b in zip(lengths, mins)]
            lengths = grow(lengths, target)
            while sum(lengths) > target:
                i = rng.randrange(n)
                if lengths[i] > mins[i]:
                    lengths[i] -= 1
                elif all(a <= b for a, b in zip(lengths, mins)):
                    raise Infeasible("minimum sentence lengths exceed the word-count range")
    if plan.alpha and plan.para_lf:
        # first word == last word of a paragraph while letters advance by one per word: 26k + 1 words
        lengths = [27] * n
    plan.n, plan.lengths = n, lengths

    # ---- sentence ends ----
    if "ratio:sentence_balance" in T:
        terms = []
        for _ in range(n // 3):
            t = [".", "?", "!"]
            rng.shuffle(t)
            terms += t
    elif "ratio:sentence_type" in T:
        terms = []
        for _ in range(n // 3):
            t = [".", ".", "?"]
            rng.shuffle(t)
            terms += t
    else:
        terms = ["."] * n
    if plan.interro:
        terms = [".", "?", "!"]
        rng.shuffle(terms)
    elif plan.punct:
        if "?" not in terms:
            terms[1] = "?" if "ratio:sentence_type" not in T else terms[1]
        if "?" not in terms:
            raise Infeasible("no question mark slot for the punctuation rule")
        bang = [i for i, t in enumerate(terms) if t == "!"]
        if "ratio:sentence_balance" in T:
            if len(bang) < 2:
                raise Infeasible("need two exclamations for '?!' plus '!'")
            terms[[b for b in bang if b != n - 1][0]] = "?!"  # nltk splits a final '?!' into '?' + '!'
        else:
            cand = [i for i, t in enumerate(terms) if t == "." and terms.count(".") > (2 if "ratio:sentence_type" in T else 1)]
            if len(cand) < 2 and "ratio:sentence_type" not in T:
                raise Infeasible("no room for '!' and '?!' sentences")
            if "ratio:sentence_type" in T:
                # '!' sentences do not count for the ratio: add them as extra sentences at the end
                plan.n += 2
                plan.lengths += [rng.randint(base_lo, base_hi), rng.randint(base_lo, base_hi)]
                terms += ["?!", "!"]  # a text-final '?!' is split by nltk into '?' and '!'
                if plan.allit is not None:
                    plan.allit += [len(plan.allit) + 1, len(plan.allit) + 2]
                    plan.lengths[-2:] = [max(x, a + 1) for x, a in zip(plan.lengths[-2:], plan.allit[-2:])]
            else:
                a, b = (cand[0], cand[1]) if cand[1] != n - 1 else (cand[1], cand[0])
                terms[a] = "!"
                terms[b] = "?!"  # never text-final: nltk splits a final '?!' into '?' and '!'
    plan.terms = terms
    n = plan.n

    # ---- line grouping ----
    if plan.composite:
        plan.kinds = _composite_kinds(plan.layouts, n)
        # one sentence per line; the sentences beyond the structure's minimum join the last line
        # before the closing bullets. Indices stay in reading order (the search fills them in order).
        last = len(plan.kinds) - 1 - (2 if "no_bullets_bullets" in plan.layouts else 0)
        extra = n - len(plan.kinds)
        plan.lines, k = [], 0
        for li in range(len(plan.kinds)):
            size = 1 + (extra if li == last else 0)
            plan.lines.append(list(range(k, k + size)))
            k += size
        assert k == n
        if "no_bullets_bullets" in plan.layouts:
            plan.nb_first = plan.kinds.index("nbullet")
    elif layout == "no_bullets_bullets":
        plan.lines = [[0, 1]] + [[i] for i in range(2, n)]
    elif layout == "output_template":
        plan.lines = [[0], [1], list(range(2, n))]
    elif layout == "thesis":
        plan.lines = [[0] + list(range(1, n))] if not (plan.indent or plan.para_lf) else [[0, 1]] + [[i] for i in range(2, n)]
    elif layout in ("list", "sub-bullets") or plan.indent or (plan.para_lf and "words:vowel" not in T):
        plan.lines = [[i] for i in range(n)]
    else:
        plan.lines = [list(range(n))]

    # ---- pins ----
    plan.req = [Counter() for _ in range(n)]
    pins = {}
    if "words:keywords_specific_position" in T:
        k = T["words:keywords_specific_position"]
        lead = _lead_words(plan, k["n"] - 1)
        idx = k["m"] - 1 - lead
        if idx < 0:
            raise Infeasible("layout words occupy the keyword position")
        pins[(k["n"] - 1, idx)] = k["keyword"].lower()
    if "words:words_position" in T:
        kwd = T["words:words_position"]["keyword"].lower()
        lead = _lead_words(plan, 0)
        if lead > 1:
            raise Infeasible("layout words occupy the second word")
        for key in ((0, 1 - lead), (n - 1, plan.lengths[n - 1] - 2)):
            if key in pins and pins[key] != kwd:
                raise Infeasible("two position rules pin different words to one slot")
        pins[(0, 1 - lead)] = kwd
        if (n - 1, plan.lengths[n - 1] - 2) not in pins:
            pins[(n - 1, -2)] = kwd
    if "words:start_verb" in T:
        if layout in ("output_template", "thesis") or plan.sep or (plan.kinds and plan.kinds[0] != "text"):
            raise Infeasible("layout puts a non-verb first")
        verbs = [v for v in lx.VERBS if _static_ok(plan, v)]
        if not verbs:
            raise Infeasible("no verb passes the word filters")
        pins[(0, 0)] = rng.choice(verbs)
        if plan.lengths[0] > 2 and (0, 1) not in pins and rng.random() < 0.7:
            dets = [d for d in ("the", "a", "this") if _static_ok(plan, d) and (stop_cap is None)]
            if dets:
                pins[(0, 1)] = rng.choice(dets)
    plan.pins = pins

    # ---- distribute required tokens ----
    if kw_sentence:
        s, w = kw_sentence
        if not _static_ok(plan, w):
            raise Infeasible("the sentence keyword fails the word filters")
        plan.req[s][w] += 1
    if plan.max_rep is not None:
        over = [w for w, c in Counter(required).items() if c > plan.max_rep]
        if over:
            raise Infeasible(f"'{over[0]}' is required {Counter(required)[over[0]]} times but words may repeat at most {plan.max_rep} times")
    if plan.alpha or plan.parity or plan.noconsec or plan.allit is not None:
        plan.greq = Counter(required)
        if sum(plan.lengths) - len(pins) < len(required) + sum(sum(c.values()) for c in plan.req):
            raise Infeasible("not enough word slots for the required tokens")
        required = []
    cap = [plan.lengths[i] - sum(1 for (s, j) in pins if s == i) - sum(plan.req[i].values()) for i in range(n)]
    order = required[:]
    rng.shuffle(order)
    for w in order:
        opts = [i for i in range(n) if cap[i] > 0]
        if not opts:
            raise Infeasible("not enough word slots for the required tokens")
        i = rng.choice(opts)
        plan.req[i][w] += 1
        cap[i] -= 1
    if exact_kw:
        for (s, j), w in pins.items():
            if w in exact_kw:
                raise Infeasible("a pinned keyword is also an exact-count keyword")
    return plan


def _static_ok_pre(T, w):
    """Filter used while choosing required tokens, before the vowel set is fixed."""
    w = w.lower()
    f = lx.feat(w)
    if "words:prime_lengths" in T and f.length not in lx.PRIMES:
        return False
    if "words:consonants" in T and not f.cluster:
        return False
    if "words:vowel" in T and len(f.vowels) > 3:
        return False
    return True


def _composite_kinds(layouts, n: int) -> list:
    """Line kinds for two or more layouts, in reading order (one sentence per line before extras).

    thesis line first (the thesis must open the response), then the template's first label, the
    list items and the bullet/sub-bullet pair inside the first slot, the other two labels, any plain
    sentence line no_bullets_bullets still needs (two sentences before its bullets), and the two
    closing '*' bullets last (after the first bullet every line must be a bullet).
    """
    L = set(layouts)
    kinds = []
    if "thesis" in L:
        kinds.append("thesis")
    if "output_template" in L:
        kinds.append("label0")
    if "list" in L:
        kinds += ["item", "item"]
    if "sub-bullets" in L:
        kinds += ["bullet", "sub"]
    if "output_template" in L:
        kinds += ["label1", "label2"]
    if "no_bullets_bullets" in L:
        kinds += ["text"] * max(0, 2 - len(kinds))
        kinds += ["nbullet", "nbullet"]
    return kinds


def _lead_words(plan: Plan, s: int) -> int:
    """Alphanumeric layout words that precede sentence s on its line (they count as words)."""
    if plan.kinds:
        for li, grp in enumerate(plan.lines):
            if grp and grp[0] == s:
                kind = plan.kinds[li]
                if kind.startswith("label"):
                    return 2
                if kind == "thesis":
                    return 1
                if kind == "item" and plan.sep == "SEPARATOR":
                    return 1
                return 0
        return 0
    for li, grp in enumerate(plan.lines or [[s]]):
        if grp and grp[0] == s:
            if plan.layout == "output_template" and li < 3:
                return 2
            if plan.layout == "thesis" and li == 0:
                return 1
            if plan.sep == "SEPARATOR":
                return 1
            return 0
    return 0


# ---------------------------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------------------------

class _Search:
    def __init__(self, plan: Plan, rng: random.Random, copy: "_Copy | None" = None, quota: int | None = None):
        self.p, self.rng = plan, rng
        self.copy, self.quota, self.matched, self.used = copy, quota, 0, Counter()
        vocab = [w for w in lx.vocabulary() if w not in plan.forbid and _static_ok(plan, w)]
        if len(vocab) < 40 and plan.stop_fill:
            vocab += [w for w in lx.stop_vocabulary() if w not in plan.forbid and _static_ok(plan, w)]
        if not vocab:
            raise Infeasible("the word filters leave no filler words")
        self.vocab = vocab
        self.vocab_last = vocab
        if plan.emoji_attach and plan.prime:
            self.vocab_last = [w for w in lx.vocabulary() if w not in plan.forbid and _static_ok(plan, w, last=True)]
        self.by_first_last = {}
        for w in self.vocab_last:
            self.by_first_last.setdefault(w[0], []).append(w)
        self.by_first = {}
        for w in vocab:
            self.by_first.setdefault(w[0], []).append(w)
        self.counts = Counter()
        self.nodes = 0
        self.stream = [[None] * L for L in plan.lengths]
        self.req = [Counter(c) for c in plan.req]
        self.greq = Counter(plan.greq)
        self.free = [sum(1 for j in range(L) if (s, j) not in plan.pins and (s, j - L) not in plan.pins)
                     for s, L in enumerate(plan.lengths)]
        if copy is not None:
            # task words the search may copy: any lower-case alphabetic word except the classes that
            # other rules count exactly (stop words under a stop-word cap, pronouns, conjunctions,
            # exact keywords); the word filters still apply through seq_ok
            bad = set(plan.forbid)
            if not plan.stop_fill:
                bad |= lx.stopwords()
            if "count:pronouns" in plan.types:
                bad |= set(lx.PRONOUNS)
            if "count:conjunctions" in plan.types:
                bad |= set(lx.CONJUNCTIONS)
            self.copy_ok = {t.lower() for t in copy.starts if t.lower() not in bad}

    def _out(self, i, w):
        p = self.p
        if w == "i":
            return "I"
        return w.capitalize() if p.title or (i == 0 and not p.newline) else w

    def _before(self, s, i):
        """The two rendered tokens before slot (s, i); a sentence's terminal mark is one token."""
        def tok(s2, i2):
            if i2 >= 0:
                return self._out(i2, self.stream[s2][i2])
            return self.p.terms[s2] if s2 >= 0 else None
        if i >= 2:
            return tok(s, i - 2), tok(s, i - 1)
        if s == 0:
            return (None, tok(s, 0)) if i == 1 else (None, None)
        last = self.p.lengths[s - 1] - 1
        return (self.p.terms[s - 1], tok(s, 0)) if i == 1 else (tok(s - 1, last), self.p.terms[s - 1])

    def _grams(self, s, i, w):
        """Task trigrams completed by placing w at (s, i), including the one closed by the terminal."""
        b1, b2 = self._before(s, i)
        o = self._out(i, w)
        out = [(b1, b2, o)]
        if i == self.p.lengths[s] - 1:
            out.append((b2, o, self.p.terms[s]))
        return [g for g in out if g in self.copy.tri]

    def _hits(self, s, i, w):
        """New (not yet used) task trigrams w would add. The overlap verifier counts distinct trigrams,
        so a repeated one changes nothing."""
        return sum(1 for g in self._grams(s, i, w) if not self.used[g])

    def _copy_order(self, s, i, cands):
        """Overlap control: below the quota, words that continue the task text come first (those that
        complete a task trigram before those that only follow the previous token), so runs of the task
        are copied; at the quota, words that would complete a task trigram go last."""
        c = self.copy
        if self.matched < self.quota:
            b1, b2 = self._before(s, i)
            toks = list(c.nxt.get((b1, b2), ())) + list(c.nxt1.get(b2, ()))
            if not toks:
                toks = self.rng.sample(c.starts, min(6, len(c.starts)))
            seen, nxt = set(), []
            for t in toks:
                w = t.lower()
                if w in seen or not t.isalpha() or self._out(i, w) != t or w not in self.copy_ok:
                    continue
                seen.add(w)
                nxt.append(w)
            nxt.sort(key=lambda w: -self._hits(s, i, w))
            return nxt + cands
        return sorted(cands, key=lambda w: self._hits(s, i, w) > 0)

    def prev(self, s, i):
        if i > 0:
            return self.stream[s][i - 1]
        if s > 0:
            return self.stream[s - 1][-1]
        return None

    def pin(self, s, i):
        p = self.p
        L = p.lengths[s]
        w = p.pins.get((s, i)) or p.pins.get((s, i - L))
        if w:
            return w
        if p.last_first and i == 0 and s > 0:
            return self.stream[s - 1][-1]
        if p.para_lf and i == L - 1:
            for grp in p.lines:
                if grp[-1] == s:
                    return self.stream[grp[0]][0] if grp[0] != s or i > 0 else None
        return None

    def seq_ok(self, s, i, w):
        p = self.p
        if p.max_rep is not None and self.counts[w] >= p.max_rep:
            return False
        return self.rel_ok(s, i, w, self.prev(s, i))

    def rel_ok(self, s, i, w, prev):
        p = self.p
        if not _static_ok(p, w, last=(i == p.lengths[s] - 1)):
            return False
        if prev is None:
            return True
        f, pf = lx.feat(w), lx.feat(prev)
        if p.alpha and (not f.first.isalpha() or not pf.first.isalpha()
                        or f.first != ALPHA[(ALPHA.index(pf.first) + 1) % 26]):
            return False
        if p.noconsec and f.first == pf.first:
            return False
        if p.parity and f.parity == pf.parity:
            return False
        if p.allit is not None and i > 0:
            run = p.allit[s]
            same = f.first == pf.first
            if i < run and not same:
                return False
            if i >= run and same:
                return False
        return True

    def next_pos(self, s, i):
        if i + 1 < self.p.lengths[s]:
            return s, i + 1
        if s + 1 < self.p.n:
            return s + 1, 0
        return None

    def filler(self, s, i):
        p = self.p
        prev = self.prev(s, i)
        last = i == p.lengths[s] - 1
        pool = self.vocab_last if last else self.vocab
        byf = self.by_first_last if last else self.by_first
        if prev is not None:
            pf = lx.feat(prev).first
            if p.alpha and pf.isalpha():
                pool = byf.get(ALPHA[(ALPHA.index(pf) + 1) % 26], [])
            elif p.allit is not None and i > 0 and i < p.allit[s]:
                pool = byf.get(pf, [])
        k = min(len(pool), 30)
        cand = self.rng.sample(pool, k) if k else []
        cand.sort(key=lambda w: self.counts[w] > 0)
        return cand

    def free_left(self, s, i):
        return sum(1 for j in range(i, self.p.lengths[s]) if self.pin(s, j) is None or j == i)

    def run(self) -> bool:
        return self._dfs(0, 0)

    def _dfs(self, s, i) -> bool:
        p = self.p
        if s == p.n:
            return sum(self.greq.values()) == 0
        if i == p.lengths[s]:
            if sum(self.req[s].values()):
                return False
            return self._dfs(s + 1, 0)
        self.nodes += 1
        if self.nodes > DFS_BUDGET:
            raise Infeasible("search budget exhausted")
        pinned = self.pin(s, i)
        if pinned:
            cands = [pinned]
        else:
            local = [w for w, c in self.req[s].items() if c > 0]
            glob = [w for w, c in self.greq.items() if c > 0]
            slots = sum(1 for j in range(i, p.lengths[s]) if self.pin(s, j) is None)
            later = sum(self.free[k] - sum(self.req[k].values()) for k in range(s + 1, p.n))
            n_local, n_glob = sum(self.req[s].values()), sum(self.greq.values())
            self.rng.shuffle(local)
            self.rng.shuffle(glob)
            prev = self.prev(s, i)
            if p.alpha and prev is not None and lx.feat(prev).first.isalpha():
                want = ALPHA[(ALPHA.index(lx.feat(prev).first) + 1) % 26]
                glob.sort(key=lambda w: w[0] != want)
                local.sort(key=lambda w: w[0] != want)
            if n_local >= slots:
                cands = local
            elif n_local + n_glob >= slots + later:
                cands = local + glob
            else:
                pr = (n_local / slots) + (n_glob / max(1, slots + later)) + 0.2
                lead = [w for w in glob if p.alpha and prev is not None and w[0] == want] if p.alpha and prev is not None and lx.feat(prev).first.isalpha() else []
                if lead:
                    cands = lead + local + self.filler(s, i) + glob
                elif (local or glob) and self.rng.random() < pr:
                    cands = local + glob + self.filler(s, i)
                else:
                    cands = self.filler(s, i) + local + glob
                if self.copy is not None:
                    cands = self._copy_order(s, i, cands)
        tried = set()
        for w in cands:
            if w in tried:
                continue
            tried.add(w)
            if not self.seq_ok(s, i, w):
                continue
            src = None
            if not pinned:
                if self.req[s][w] > 0:
                    src = self.req[s]
                elif self.greq[w] > 0:
                    src = self.greq
            if src is None and not pinned and w in p.forbid:
                continue
            grams = self._grams(s, i, w) if self.copy is not None else []
            for g in grams:
                self.used[g] += 1
                self.matched += self.used[g] == 1
            self.stream[s][i] = w
            self.counts[w] += 1
            if src is not None:
                src[w] -= 1
            if self._dfs(s, i + 1):
                return True
            self.counts[w] -= 1
            for g in grams:
                self.matched -= self.used[g] == 1
                self.used[g] -= 1
            if src is not None:
                src[w] += 1
            self.stream[s][i] = None
        return False


# ---------------------------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------------------------

_NAMES = {n.lower(): n for n in lx.PERSON_NAMES}


def _case(plan: Plan, words: list[str], first_in_sentence: bool = True) -> list[str]:
    out = []
    for j, w in enumerate(words):
        if w in _NAMES:
            w = _NAMES[w]
        elif w == "i":
            w = "I"
        elif plan.title or (j == 0 and first_in_sentence and not plan.newline):
            w = w[:1].upper() + w[1:]
        out.append(w)
    return out


def render(plan: Plan, stream: list[list[str]], rng: random.Random) -> str:
    return finalize(plan, compose(plan, stream, rng))


def compose(plan: Plan, stream: list[list[str]], rng: random.Random) -> list[str]:
    """Lines of text before the whole-text layouts (one word per line, stairs, no whitespace)."""
    sents, bodies = [], []
    emo = lx.EMOJI[:]
    rng.shuffle(emo)
    wrap_s = 0
    for s, words in enumerate(stream):
        toks = _case(plan, words)
        if plan.punct and s == 0 and len(toks) >= 4:
            k = 1 if len(toks) >= 5 else 0   # keep the first word bare (first-word rules read it)
            toks[k] += ","
            toks[k + 1] += ";"
            toks[k + 2] += ":"
            if plan.interro and len(toks) >= k + 5:
                toks[k + 3] += "\u203d"      # the interrobang character, inside the sentence
        if plan.parens and s == wrap_s and len(toks) >= 7:
            toks[1] = "(" + toks[1]
            toks[2] = "[" + toks[2]
            toks[3] = "{" + toks[3]
            toks[4] = "(" + toks[4]
            toks[5] = "[" + toks[5] + "])}])"
        if plan.quotes and s == wrap_s and len(toks) >= 7:
            toks[1] = '"' + toks[1]
            toks[2] = "'" + toks[2]
            toks[3] = '"' + toks[3] + '"'
            toks[4] = toks[4] + "'"
            toks[5] = toks[5] + '"'
        if plan.qu and not plan.quotes and len(toks) >= 4 and s == 0:
            toks[1] = '"' + toks[1]
            toks[2] = toks[2] + '"'
        body = " ".join(toks)
        if plan.emoji:
            body += ("" if plan.emoji_attach else " ") + emo[s % len(emo)]
        bodies.append(body)
        sents.append(body + plan.terms[s])
    lines = []
    sp = "" if plan.tight else " "
    for li, grp in enumerate(plan.lines):
        text = " ".join(sents[i] for i in grp)
        if plan.kinds:
            kind = plan.kinds[li]
            if kind == "thesis":
                rest = "" if len(grp) == 1 else " " + " ".join(sents[i] for i in grp[1:])
                text = ("<i>" + sents[grp[0]] + "</i>" + rest if plan.last_first
                        else "<i>" + bodies[grp[0]] + "</i>" + plan.terms[grp[0]] + rest)
            elif kind.startswith("label"):
                text = TEMPLATE[int(kind[-1])] + " " + text
            elif kind == "item":
                text = plan.sep + sp + text
            elif kind in ("bullet", "nbullet"):
                text = "*" + sp + text
            elif kind == "sub":
                text = "-" + sp + text
            lines.append(text)
            continue
        if plan.layout == "output_template":
            text = TEMPLATE[li] + " " + text
        elif plan.layout == "thesis" and li == 0:
            rest = "" if len(grp) == 1 else " " + " ".join(sents[i] for i in grp[1:])
            if plan.last_first:
                # terminal inside the tag: words:last_first strips punctuation from the sentence end, and
                # "word</i>." would leave "word</i", which no longer matches the next sentence's first word
                text = "<i>" + sents[grp[0]] + "</i>" + rest
            else:
                # terminal after the closing tag, so NLTK ends the thesis sentence there (sentence-count rules)
                text = "<i>" + bodies[grp[0]] + "</i>" + plan.terms[grp[0]] + rest
        elif plan.layout == "list":
            text = plan.sep + ("" if plan.tight else " ") + text
        elif plan.layout == "sub-bullets":
            text = ("*" if li % 2 == 0 else "-") + ("" if plan.tight else " ") + text
        elif plan.layout == "no_bullets_bullets" and li > 0:
            text = "*" + ("" if plan.tight else " ") + text
        lines.append(text)
    return lines


def finalize(plan: Plan, lines: list[str]) -> str:
    if plan.newline:
        toks = []
        for li, ln in enumerate(lines):
            for j, t in enumerate(ln.split()):
                # after the first bullet every line must be a bullet line, so each word carries the marker
                if "no_bullets_bullets" in plan.layouts and li >= plan.nb_first and j > 0:
                    t = "*" + t
                toks.append(t)
        lines = toks
    if plan.indent:
        lines = [" " * (2 * k) + ln for k, ln in enumerate(lines)]
    text = "\n".join(lines)
    if plan.nows:
        text = ",".join(text.split())
    return text


# ---------------------------------------------------------------------------------------------
# Repair and verification
# ---------------------------------------------------------------------------------------------

def _equalize(plan: Plan, srch: _Search, rng: random.Random, budget: int = 4000) -> bool:
    """Swap filler words until the three sentences have the same character length."""
    sents = source._bootstrap()[2].split_into_sentences

    def lens():
        txt = render(plan, srch.stream, random.Random(0))
        ss = sents(txt)
        return [len(x.strip()) for x in ss] if len(ss) == 3 else None

    for _ in range(budget):
        L = lens()
        if L is None:
            return False
        if len(set(L)) == 1:
            return True
        target = max(L)
        s = rng.choice([k for k, v in enumerate(L) if v < target])
        i = rng.randrange(plan.lengths[s])
        if srch.pin(s, i) is not None or plan.req[s] and srch.stream[s][i] in plan.req[s]:
            continue
        old = srch.stream[s][i]
        need = target - L[s]
        cands = [w for w in srch.filler(s, i) if 0 < len(w) - len(old) <= need and srch.counts[w] == 0]
        if not cands:
            # shrink the longest instead
            s = L.index(target)
            i = rng.randrange(plan.lengths[s])
            if srch.pin(s, i) is not None:
                continue
            old = srch.stream[s][i]
            cands = [w for w in srch.filler(s, i) if 0 < len(old) - len(w) <= target - min(L) and srch.counts[w] == 0]
            if not cands:
                continue
        w = None
        nxt = srch.next_pos(s, i)
        for c in cands:
            if not srch.rel_ok(s, i, c, srch.prev(s, i)):
                continue
            if nxt and not srch.rel_ok(nxt[0], nxt[1], srch.stream[nxt[0]][nxt[1]], c):
                continue
            w = c
            break
        if w is None:
            continue
        srch.counts[old] -= 1
        srch.stream[s][i] = w
        srch.counts[w] += 1
    return False


@lru_cache(maxsize=64)
def _ref_trigrams(ref: str) -> frozenset:
    import nltk

    return frozenset(nltk.ngrams(nltk.word_tokenize(ref), 3))


def _overlap_value(text: str, ref: str) -> float:
    import nltk

    g = set(nltk.ngrams(nltk.word_tokenize(text), 3))
    return 100.0 * len(g & _ref_trigrams(ref)) / len(g) if g else 0.0


@dataclass
class _Copy:
    """The task text as the word search sees it: its token trigrams, indexed by their first two
    tokens, and the tokens that can open a run. The search works on lower-case words; ``out`` gives
    the form a word takes in the rendered text (capitalised at a sentence start or under title case),
    and only that form is matched, because the overlap verifier compares tokens exactly."""
    tri: set
    nxt: dict      # (t1, t2) -> sorted third tokens
    nxt1: dict     # t1 -> sorted second tokens
    starts: list

    @classmethod
    def of(cls, ref: str) -> "_Copy":
        import nltk

        toks = nltk.word_tokenize(ref)
        tri = set(nltk.ngrams(toks, 3))
        nxt, nxt1 = {}, {}
        for a, b, c in tri:
            nxt.setdefault((a, b), set()).add(c)
        for a, b in nltk.ngrams(toks, 2):
            nxt1.setdefault(a, set()).add(b)
        starts = sorted({t for t in toks if t.isalpha()})
        return cls(tri, {k: sorted(v) for k, v in nxt.items()}, {k: sorted(v) for k, v in nxt1.items()}, starts)


def _fit_overlap(plan: Plan, lines: list[str], ref: str, pct: int, rng: random.Random,
                 alone_ok: bool = False, accept=None, max_spans: int = 3000) -> str | None:
    """Insert a span of the task text (as its own sentence) so the trigram overlap is within pct +/- 2.

    The span may go at the start, the end, or after the first line; whole-text layouts are applied
    after insertion, so one-word-per-line, stairs and no-whitespace texts stay well-formed.
    """
    base = finalize(plan, lines)
    if abs(_overlap_value(base, ref) - pct) <= 2:
        return base
    import nltk

    tested = 0
    R = _ref_trigrams(ref)
    gb = set(nltk.ngrams(nltk.word_tokenize(base), 3))
    words = ref.split()
    spans = [(a, b) for a in range(len(words)) for b in range(a + 3, len(words) + 1)]
    rng.shuffle(spans)
    for a, b in spans[:max_spans]:
        raw = " ".join(words[a:b])
        span = raw.rstrip(".?!,;:") + "."
        # cheap screen (2026-11-24): the overlap of base + span, from the two trigram sets with two
        # new trigrams for the junction, must be within 6 points of the target before any candidate
        # text is rendered and measured exactly; spans far off cannot land in the window
        st = set(nltk.ngrams(nltk.word_tokenize(span), 3))
        u = gb | st
        est = 100.0 * len(u & R) / (len(u) + 2)
        alone = 100.0 * len(st & R) / len(st) if st else 0.0
        if abs(est - pct) > 6 and not (alone_ok and abs(alone - pct) <= 6):
            continue
        cands = [lines[:-1] + [lines[-1] + " " + span], [span + " " + lines[0]] + lines[1:]]
        if len(lines) > 1:
            cands.append([lines[0] + " " + span] + lines[1:])
        else:
            import re as _re
            m = _re.search(r"[.?!] ", lines[0])
            if m:
                cands.append([lines[0][:m.end()] + span + " " + lines[0][m.end():]])
        texts = [finalize(plan, c) for c in cands]
        if alone_ok:
            texts += [raw, span]
        for cand in texts:
            if abs(_overlap_value(cand, ref) - pct) <= 2 and pm.check("ratio:overlap", cand, {}, ref)[0]:
                if accept is None:
                    return cand
                # keep looking past a text in the window that breaks another rule (up to a cap)
                if accept(cand):
                    return cand
                tested += 1
                if tested >= ACCEPT_CAP:
                    return None
    return None


def _copy_route(rules, ref, pct, seed, alone_ok, accept) -> str | None:
    """Overlap by copying: the word search prefers task-text continuations until a quota of copied
    trigrams is reached, then avoids completing new task trigrams. The quota starts at pct% of the
    planned trigrams and is corrected from the measured overlap of the rendered text. If the text is
    still below the window, a task span is inserted as in the first route.

    Sentence lengths are sized from the task: only trigrams made of words (no punctuation, digits or
    mid-sentence capitals) can be copied, so a text much longer than M / (pct/100) words, where M is
    the number of such trigrams, would need new trigrams beyond the target share. Low targets use
    ordinary lengths."""
    import nltk

    rng = random.Random(seed ^ 0x5EED)
    copy = _Copy.of(ref)
    usable = sum(1 for g in copy.tri if all(t.isalpha() or t in ".?!" for t in g))
    if not usable:
        return None
    if pct <= 40:
        span = (4, 9)
    else:
        per = max(4, round(usable / (pct / 100) / 2))   # about two sentences of copied text
        span = (max(3, per - 3), per + 3)
    plan = None
    for r in range(COPY_ROUNDS):
        if r % 3 == 0:   # a fresh plan every third round
            plan = make_plan(rules, rng, ref, copy_len=span)
            quota = round(pct / 100 * (sum(plan.lengths) + plan.n))
        srch = _Search(plan, rng, copy=copy, quota=quota)
        try:
            if not srch.run():
                continue
        except Infeasible:
            continue
        if plan.equal_chars and not _equalize(plan, srch, rng):
            continue
        lines = compose(plan, srch.stream, rng)
        text = finalize(plan, lines)
        got = _overlap_value(text, ref)
        if abs(got - pct) <= 2 and pm.check("ratio:overlap", text, {}, ref)[0] and accept(text):
            return text
        if got < pct - 2 and r % 3 == 2:   # last round of a plan: try a span insertion (fewer spans, for time)
            t = _fit_overlap(plan, lines, ref, pct, rng, alone_ok=alone_ok, accept=accept, max_spans=COPY_SPANS)
            if t is not None:
                return t
        n_tri = max(1, len(set(nltk.ngrams(nltk.word_tokenize(text), 3))))
        step = round((pct - got) / 100 * n_tri) or (1 if got < pct else -1)
        quota = max(0, quota + step)
    return None


def verify(rules: dict, text: str, task_text: str | None) -> list[tuple[str, str]]:
    bad = []
    for t, kw in rules.items():
        ok, why = pm.witness_ok(t, kw, text, task_text)
        if not ok:
            bad.append((t, why))
    return bad


def build_witness(rules, task_text: str | None = None, seed: int = 0, attempts: int = 24) -> Result:
    """Try up to ``attempts`` seeded constructions; return the first text that passes everything."""
    rules = {t: dict(kw or {}) for t, kw in (rules.items() if isinstance(rules, dict) else rules)}
    for t in rules:
        rules[t].pop("reference_text", None)
    if "ratio:overlap" in rules and not task_text:
        raise ValueError("ratio:overlap needs the task text")
    res = Result(ok=False)
    overlap = "ratio:overlap" in rules
    alone_ok = set(rules) <= {"ratio:overlap", "count:word_count_range", "words:repeats", "ratio:stop_words"}
    accept = (lambda t: not verify(rules, t, task_text)) if overlap else None
    for a in range(attempts):
        rng = random.Random(seed * 1000003 + a)
        try:
            plan = make_plan(rules, rng, task_text)
            srch = _Search(plan, rng)
            if not srch.run():
                raise Infeasible("search found no word sequence")
            if plan.equal_chars and not _equalize(plan, srch, rng):
                raise Infeasible("could not equalize sentence lengths")
            lines = compose(plan, srch.stream, rng)
            text = finalize(plan, lines)
            if overlap:
                pct = rules["ratio:overlap"]["percentage"]
                text = _fit_overlap(plan, lines, task_text, pct, rng, alone_ok=alone_ok, accept=accept)
                if text is None:
                    # second route: copy runs of the task text inside the word search itself
                    text = _copy_route(rules, task_text, pct, seed * 1000003 + a, alone_ok, accept)
                if text is None:
                    raise Infeasible("no task-text span puts the overlap in the window")
            bad = verify(rules, text, task_text)
            if not bad:
                res.ok, res.text, res.attempt = True, text, a
                return res
            for t, why in bad:
                res.reasons[f"verify {t}: {why}"] += 1
        except Infeasible as e:
            res.reasons[str(e)] += 1
    return res
