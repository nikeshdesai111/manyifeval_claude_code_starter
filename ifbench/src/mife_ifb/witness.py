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


@dataclass
class Result:
    ok: bool
    text: str | None = None
    attempt: int | None = None
    reasons: Counter = field(default_factory=Counter)


# ---------------------------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------------------------

def _static_ok(plan: Plan, w: str) -> bool:
    f = lx.feat(w)
    if plan.vowels is not None and not f.vowels <= plan.vowels:
        return False
    if plan.prime and (f.length not in lx.PRIMES or not w.isalpha()):
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


def make_plan(rules: dict, rng: random.Random, task_text: str | None) -> Plan:
    T = rules
    layout = _layout_of(T)
    plan = Plan(n=0, lengths=[], terms=[], layout=layout)
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
    plan.max_rep = T["words:repeats"]["small_n"] if "words:repeats" in T else None
    stop_cap = T["ratio:stop_words"]["percentage"] if "ratio:stop_words" in T else None
    plan.stop_fill = stop_cap is None
    overlap = T.get("ratio:overlap")
    plan.minimal = bool(overlap and overlap["percentage"] > 40)

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
        if _need_letters(T):
            raise Infeasible("numbers are not words for a word-shape rule")
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
        if plan.layout == "output_template":
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
    lo_n = 2 if plan.minimal else rng.randint(3, 5)
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
    lo_n = max(lo_n, {"list": 2, "sub-bullets": 2, "no_bullets_bullets": 4, "output_template": 3, "thesis": 2}.get(layout, 1))
    if plan.indent or plan.punct:
        lo_n = max(lo_n, 3)
    if layout == "sub-bullets" and lo_n % 2:
        lo_n += 1
    n = fixed_n or lo_n
    if fixed_n and fixed_n < lo_n:
        raise Infeasible("exactly three sentences, but another rule needs more")
    while n % mult:
        n += 1
    if fixed_n and n != fixed_n:
        raise Infeasible("three sentences cannot satisfy the sentence-type ratio")
    if layout == "sub-bullets" and n % 2:
        n += 1

    # ---- sentence lengths ----
    inc = T.get("sentence:increment", {}).get("small_n")
    base_lo, base_hi = (3, 4) if plan.minimal else (5, 9)
    if plan.equal_chars:
        base_lo, base_hi = 3, 5
    if plan.parens or plan.quotes:
        base_lo = max(base_lo, 7)
    if plan.qu:
        base_lo = max(base_lo, 4)
    if plan.punct:
        base_lo = max(base_lo, 4)

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
    if plan.parens or plan.quotes:
        mins[0] = max(mins[0], 7)

    def fix_mins(lengths):
        if inc:
            need = max(m - i * inc for i, m in enumerate(mins))
            return [max(lengths[0], need) + i * inc for i in range(n)]
        return [max(a, b) for a, b in zip(lengths, mins)]

    lengths = fix_mins(lengths)

    # ---- demands on the total ----
    stop_req = sum(1 for w in required if w in lx.stopwords())
    stop_req += 1 if (kw_sentence and kw_sentence[1] in lx.stopwords()) else 0
    stop_req += 2 if layout == "output_template" else 0
    need_total = len(required) + n
    if "count:unique_word_count" in T:
        need_total = max(need_total, T["count:unique_word_count"]["N"] + 3)
    if stop_cap is not None and stop_req:
        need_total = max(need_total, math.ceil(stop_req * 100 / max(stop_cap, 0.5)) + 2)
    if plan.alpha and required:
        same = Counter(w[0] for w in required).most_common(1)[0][1]
        need_total = max(need_total, 26 * (same + 1))
    wc = T.get("count:word_count_range")
    extra = {"output_template": 5, "thesis": 2}.get(layout, 0) + (n if plan.sep == "SEPARATOR" else 0)

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
                while target / n > 18:
                    n += mult
                    lengths.append(rng.randint(base_lo, base_hi))
                    mins.append(2)
                    if plan.allit is not None:
                        plan.allit.append(len(plan.allit) + 1)
                        mins[-1] = plan.allit[-1] + 1
                    if layout == "sub-bullets" and n % 2:
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
    if plan.punct:
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
    if layout == "no_bullets_bullets":
        terms[0] = terms[1] = "." if terms[0] in ".!?" else terms[0]
    plan.terms = terms
    n = plan.n

    # ---- line grouping ----
    if layout == "no_bullets_bullets":
        plan.lines = [[0, 1]] + [[i] for i in range(2, n)]
    elif layout == "output_template":
        plan.lines = [[0], [1], list(range(2, n))]
    elif layout == "thesis":
        plan.lines = [[0] + list(range(1, n))] if not (plan.indent or plan.para_lf) else [[0, 1]] + [[i] for i in range(2, n)]
    elif layout in ("list", "sub-bullets") or plan.indent or plan.para_lf:
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
        pins[(0, 1 - lead)] = kwd
        pins[(n - 1, -2)] = kwd
    if "words:start_verb" in T:
        if layout in ("output_template", "thesis") or plan.sep:
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


def _lead_words(plan: Plan, s: int) -> int:
    """Alphanumeric layout words that precede sentence s on its line (they count as words)."""
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
    def __init__(self, plan: Plan, rng: random.Random):
        self.p, self.rng = plan, rng
        vocab = [w for w in lx.vocabulary() if w not in plan.forbid and _static_ok(plan, w)]
        if len(vocab) < 40 and plan.stop_fill:
            vocab += [w for w in lx.stop_vocabulary() if w not in plan.forbid and _static_ok(plan, w)]
        if not vocab:
            raise Infeasible("the word filters leave no filler words")
        self.vocab = vocab
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
        if not _static_ok(p, w):
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
        pool = self.vocab
        if prev is not None:
            pf = lx.feat(prev).first
            if p.alpha and pf.isalpha():
                pool = self.by_first.get(ALPHA[(ALPHA.index(pf) + 1) % 26], [])
            elif p.allit is not None and i > 0 and i < p.allit[s]:
                pool = self.by_first.get(pf, [])
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
            self.stream[s][i] = w
            self.counts[w] += 1
            if src is not None:
                src[w] -= 1
            if self._dfs(s, i + 1):
                return True
            self.counts[w] -= 1
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
    sents = []
    emo = lx.EMOJI[:]
    rng.shuffle(emo)
    wrap_s = 0
    for s, words in enumerate(stream):
        toks = _case(plan, words)
        if plan.punct and s == 0 and len(toks) >= 4:
            toks[0] += ","
            toks[1] += ";"
            toks[2] += ":"
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
            body += " " + emo[s % len(emo)]
        sents.append(body + plan.terms[s])
    lines = []
    for li, grp in enumerate(plan.lines):
        text = " ".join(sents[i] for i in grp)
        if plan.layout == "output_template":
            text = TEMPLATE[li] + " " + text
        elif plan.layout == "thesis" and li == 0:
            first = sents[grp[0]]
            text = "<i>" + first + "</i>" + ("" if len(grp) == 1 else " " + " ".join(sents[i] for i in grp[1:]))
        elif plan.layout == "list":
            text = plan.sep + " " + text
        elif plan.layout == "sub-bullets":
            text = ("* " if li % 2 == 0 else "- ") + text
        elif plan.layout == "no_bullets_bullets" and li > 0:
            text = "* " + text
        lines.append(text)
    if plan.newline:
        toks = [t for ln in lines for t in ln.split()]
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


def _overlap_value(text: str, ref: str) -> float:
    import nltk

    g = set(nltk.ngrams(nltk.word_tokenize(text), 3))
    r = set(nltk.ngrams(nltk.word_tokenize(ref), 3))
    return 100.0 * len(g & r) / len(g) if g else 0.0


def _fit_overlap(text: str, ref: str, pct: int, rng: random.Random, alone_ok: bool = False) -> str | None:
    """Append (or prepend) a span of the task text so the trigram overlap is within pct +/- 2."""
    if abs(_overlap_value(text, ref) - pct) <= 2:
        return text
    words = ref.split()
    spans = [(a, b) for a in range(len(words)) for b in range(a + 3, len(words) + 1)]
    rng.shuffle(spans)
    for a, b in spans[:3000]:
        raw = " ".join(words[a:b])
        span = raw.rstrip(".?!,;:")
        cands = [text + " " + span + ".", span + ". " + text]
        if alone_ok:
            cands += [raw, span + "."]
        for cand in cands:
            if abs(_overlap_value(cand, ref) - pct) <= 2 and pm.check("ratio:overlap", cand, {}, ref)[0]:
                return cand
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
    for a in range(attempts):
        rng = random.Random(seed * 1000003 + a)
        try:
            plan = make_plan(rules, rng, task_text)
            srch = _Search(plan, rng)
            if not srch.run():
                raise Infeasible("search found no word sequence")
            if plan.equal_chars and not _equalize(plan, srch, rng):
                raise Infeasible("could not equalize sentence lengths")
            text = render(plan, srch.stream, rng)
            if "ratio:overlap" in rules:
                text = _fit_overlap(text, task_text, rules["ratio:overlap"]["percentage"], rng,
                                    alone_ok=set(rules) <= {"ratio:overlap", "count:word_count_range", "words:repeats", "ratio:stop_words"})
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
