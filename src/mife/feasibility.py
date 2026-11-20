"""D3 joint-feasibility check: reject only provable contradictions.

A rule pool is ACCEPTED only if a constructive witness response passes every official verifier.
A pool is REJECTED only if an explicit proof rule shows it is unsatisfiable. If neither happens,
UnresolvedFeasibility is raised (a pipeline bug to fix, never a silent rejection).

Witness semantics follow the plain meaning of the instruction: required keywords appear as
standalone words (no substring loopholes such as satisfying "active" via "inactive").

Task feasibility: a group's rule pool must be satisfiable for each of its three tasks. That is the only
task-level requirement; there is no relevance or appropriateness filter, so a rule that is merely unusual for a
task (e.g. "at least 17 sentences" for a tweet) is kept. check_tasks() confirms it by running the official
verifiers on the pool's witness with each task's text. Because the witness satisfies all six rules, it also
satisfies every subset, i.e. every prompt built from the pool.
"""
from __future__ import annotations

import random
import re

from . import verifier

# Helper vocabulary for witnesses only (NOT a benchmark artifact).
_FILLER = (
    "the a an and to of in on at for with by from as is was are were be been it this that "
    "we you they he she our your their my his her its will can may must should would could "
    "day time year way work plan team world life city home school water food book story road "
    "trip park garden music light hand house room door table window family friend people child "
    "good great small big new old long short simple clear quick slow warm cold bright quiet "
    "make take give find keep help show bring hold turn start begin learn read write walk run "
    "talk think know see look feel like love want need try use call ask move live stay play "
    "here there now then today soon often always also very just only still well again together "
    "morning evening night river hill forest field street market shop train bus car path view"
).split()
# extra natural carriers for rare letters (used only for "at least N <letter>" padding)
_CARRIERS = ("question quality quarter queen quite equal square request size zone zero amazing "
             "box next six extra job enjoy major jump joy fix wax lazy puzzle quiz jacket").split()


class UnresolvedFeasibility(RuntimeError):
    pass


def _get(pool, type_id):
    for t, kw in pool:
        if t == type_id:
            return kw
    return None


def proofs(pool: list[tuple[str, dict]]) -> list[str]:
    """Explicit unsatisfiability proofs. Empty list => no contradiction proven."""
    out = []
    ex = _get(pool, "keywords:existence")
    fr = _get(pool, "keywords:frequency")
    fb = _get(pool, "keywords:forbidden_words")
    lf = _get(pool, "keywords:letter_frequency")
    ns = _get(pool, "length_constraints:number_sentences")
    nw = _get(pool, "length_constraints:number_words")
    cw = _get(pool, "change_case:capital_word_frequency")

    if ns and ns["relation"] == "less than" and ns["num_sentences"] <= 1:
        out.append("P_SENT0: 'less than 1 sentences' requires an empty response, which fails every rule")

    required = list(ex["keywords"]) if ex else []
    f_need = fr["frequency"] if fr and fr["relation"] == "at least" else 0
    if fr and f_need > 0:
        required.append(fr["keyword"])
    if fb:
        forb = {w.lower() for w in fb["forbidden_words"]}
        clash = sorted({w for w in required if w.lower() in forb})
        if clash:
            out.append(f"P_KW_FORBIDDEN: required keyword(s) {clash} are also forbidden")

    if fr and fr["relation"] == "less than" and ex:
        forced = sum(len(re.findall(fr["keyword"], w, flags=re.IGNORECASE)) for w in ex["keywords"])
        if forced >= fr["frequency"]:
            out.append(f"P_FREQ_LESS: existence keywords force >= {forced} occurrences of "
                       f"'{fr['keyword']}' but it must appear less than {fr['frequency']} times")

    if lf and lf["let_relation"] == "less than":
        letter = lf["letter"]
        forced = sum(w.lower().count(letter) for w in (ex["keywords"] if ex else []))
        if fr and f_need:
            forced += fr["keyword"].lower().count(letter) * f_need
        if forced >= lf["let_frequency"]:
            out.append(f"P_LETTER_LESS: required keywords force >= {forced} '{letter}' "
                       f"but fewer than {lf['let_frequency']} are allowed")

    if nw and nw["relation"] == "less than":
        n_tokens = (len(ex["keywords"]) if ex else 0) + f_need
        cap_need = cw["capital_frequency"] if cw and cw["capital_relation"] == "at least" else 0
        forced = max(n_tokens, cap_need)
        if forced >= nw["num_words"]:
            out.append(f"P_WORDS_LESS: >= {forced} required word tokens but fewer than {nw['num_words']} allowed")
    return out


def _build_witness(pool, rng: random.Random, compact: bool, per: int = 6) -> str:
    ex = _get(pool, "keywords:existence")
    fr = _get(pool, "keywords:frequency")
    fb = _get(pool, "keywords:forbidden_words")
    lf = _get(pool, "keywords:letter_frequency")
    ns = _get(pool, "length_constraints:number_sentences")
    nw = _get(pool, "length_constraints:number_words")
    npar = _get(pool, "length_constraints:number_paragraphs")
    ph = _get(pool, "detectable_content:number_placeholders")
    bl = _get(pool, "detectable_format:number_bullet_lists")
    title = _get(pool, "detectable_format:title") is not None
    cw = _get(pool, "change_case:capital_word_frequency")
    upper = _get(pool, "change_case:english_capital") is not None
    case_rule = upper or _get(pool, "change_case:english_lowercase") is not None
    quote = _get(pool, "startend:quotation") is not None

    banned_letter = lf["letter"] if lf and lf["let_relation"] == "less than" else None
    forb = {w.lower() for w in fb["forbidden_words"]} if fb else set()
    avoid_sub = fr["keyword"].lower() if fr and fr["relation"] == "less than" else None
    req_kw = fr["keyword"].lower() if fr else None

    def ok(w):
        return (w not in forb and (banned_letter is None or banned_letter not in w)
                and (avoid_sub is None or avoid_sub not in w) and (req_kw is None or req_kw not in w))

    vocab = [w for w in _FILLER if ok(w)]
    if len(vocab) < 5:
        raise UnresolvedFeasibility("filler vocabulary exhausted")
    rng.shuffle(vocab)
    it = iter(vocab * 400)

    def filler(n):
        return [next(it) for _ in range(n)]

    content = list(ex["keywords"]) if ex else []
    if fr and fr["relation"] == "at least":
        content += [fr["keyword"]] * fr["frequency"]
    if cw and cw["capital_relation"] == "at least":
        content += [w.upper() for w in filler(cw["capital_frequency"])]
    if ph:
        content += [f"[{w}]" for w in filler(ph["num_placeholders"])] if not compact else ["[ ]"] * ph["num_placeholders"]
    rng.shuffle(content)

    # sentence plan
    if ns and ns["relation"] == "at least":
        n_sent = ns["num_sentences"]
    elif ns:  # less than (>=2)
        n_sent = 1
    else:
        n_sent = 1 if compact else 3
    per = 1 if compact else per
    sentences = [filler(per) for _ in range(n_sent)]
    for i, w in enumerate(content):
        sentences[i % n_sent].append(w)

    def text_of(sents):
        return [" ".join(s) + "." for s in sents]

    # word-count padding
    def count_all(sents, extra):
        return sum(len(re.findall(r"\w+", s)) for s in text_of(sents)) + extra

    extra_words = (1 if title else 0) + (bl["num_bullets"] if bl else 0) * (0 if compact else 2)
    if nw and nw["relation"] == "at least":
        while count_all(sentences, extra_words) < nw["num_words"]:
            sentences[rng.randrange(n_sent)].extend(filler(8))

    # letter "at least" padding
    if lf and lf["let_relation"] == "at least":
        letter = lf["letter"]
        have = " ".join(text_of(sentences)).lower().count(letter)
        need = lf["let_frequency"] - have
        if need > 0:
            carriers = [w for w in _FILLER + _CARRIERS if letter in w and ok(w)]
            if carriers and not (nw and nw["relation"] == "less than"):
                n_added = 0
                while need > 0:
                    w = rng.choice(carriers)
                    sentences[rng.randrange(n_sent)].append(w)
                    need -= w.count(letter)
                    n_added += 1
                if case_rule:  # keep the text recognisably English for langdetect
                    for _ in range(3 * n_added):
                        sentences[rng.randrange(n_sent)].append(next(it))
            else:
                sentences[0].append(letter * (need + 1))

    paragraphs_text = text_of(sentences)
    lines = []
    if title:
        lines.append("<<" + " ".join(filler(1 if compact else 2)) + ">>")
    n_par = npar["num_paragraphs"] if npar else 1
    # distribute sentences across paragraphs (each paragraph non-empty)
    if n_par > len(paragraphs_text):
        paragraphs_text += [" ".join(filler(2)) + "." for _ in range(n_par - len(paragraphs_text))]
    chunks = [[] for _ in range(n_par)]
    for i, s in enumerate(paragraphs_text):
        chunks[i % n_par].append(s)
    bullets = ["* " + " ".join(filler(1 if compact else 2)) for _ in range(bl["num_bullets"])] if bl else []
    blocks = []
    one_sentence = ns is not None and ns["relation"] == "less than"
    for i, c in enumerate(chunks):
        block = " ".join(c)
        if i == n_par - 1 and bullets:
            # with a sentence cap, unpunctuated bullet lines go first so punkt sees one sentence
            block = ("\n".join(bullets) + "\n" + block) if one_sentence else (block + "\n" + "\n".join(bullets))
        blocks.append(block)
    if quote and not title and bullets and one_sentence and n_par == 1:
        lines.append(" ".join(filler(2)))  # keep the opening quote off the first bullet line
    body = "\n***\n".join(blocks) if npar else "\n\n".join(blocks)
    text = "\n".join(lines + [body])
    if upper:
        text = text.upper()
    if quote:
        text = '"' + text + '"'
    return text


def check_tasks(pool: list[tuple[str, dict]], witness: str, task_texts: dict) -> dict:
    """Official verifiers on the pool's witness with each task's text: {task_key: [violations per rule]}.

    A task is feasible for the pool when its list is all zeros.
    """
    return {k: verifier.verify(witness, pool, task_text=t) for k, t in task_texts.items()}


def check_pool(pool: list[tuple[str, dict]], seed: int = 0, attempts: int = 24) -> dict:
    """Return {'feasible': bool, 'proofs': [...], 'witness': str|None}."""
    p = proofs(pool)
    if p:
        return {"feasible": False, "proofs": p, "witness": None}
    rng = random.Random(seed)
    last = None
    for a in range(attempts):
        try:
            w = _build_witness(pool, rng, compact=(a % 4 == 1), per=[6, 1, 15, 30][a % 4])
        except UnresolvedFeasibility as e:
            last = str(e)
            continue
        v = verifier.verify(w, pool, task_text="")
        if not any(v):
            return {"feasible": True, "proofs": [], "witness": w}
        last = f"violations {v}"
    raise UnresolvedFeasibility(f"no witness and no proof for pool {pool}: {last}")
