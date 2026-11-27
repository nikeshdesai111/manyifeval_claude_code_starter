"""Rule catalog (configs/difficulty/rules.yaml): instantiation and rule text for the 18 types."""
from __future__ import annotations

import functools
import hashlib
import json
import random
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
CFG = ROOT / "configs" / "difficulty"
LEVELS = ("easy", "medium", "hard")


@functools.lru_cache(maxsize=1)
def catalog() -> dict:
    return yaml.safe_load((CFG / "rules.yaml").read_text())


@functools.lru_cache(maxsize=1)
def wordlists() -> dict:
    return yaml.safe_load((CFG / "wordlists.yaml").read_text())


def catalog_sha256() -> str:
    h = hashlib.sha256()
    for f in ("rules.yaml", "wordlists.yaml"):
        h.update((CFG / f).read_bytes())
    return h.hexdigest()


def types() -> list[str]:
    return list(catalog()["types"])


def category_of(t: str) -> str:
    return catalog()["types"][t]["category"]


def forbidden(a: tuple[str, str], b: tuple[str, str]) -> bool:
    """(type, level) pairs that may never share a pool."""
    for x, y, _why in catalog()["forbidden_pairs"]:
        p, q = (x["type"], x["level"]), (y["type"], y["level"])
        if {a, b} == {p, q}:
            return True
    return False


def _word_split(split: str) -> str:
    """Pilot (calibration/confirmation) seeds draw from the TRAIN lists; TEST from TEST."""
    return "test" if split == "test" else "train"


def instantiate(t: str, level: str, rng: random.Random, split: str, taken_keywords: set[str]) -> dict:
    """Checker kwargs for one rule. Keywords are drawn so the two keyword rules of a pool differ."""
    spec = catalog()["types"][t]
    v = spec["levels"][level]
    if t == "keyword_sentence":
        kw = rng.choice(sorted(set(wordlists()["keywords"][_word_split(split)]) - taken_keywords))
        return {"keyword": kw, "n": int(v)}
    if t == "keyword_position":
        kw = rng.choice(sorted(set(wordlists()["keywords"][_word_split(split)]) - taken_keywords))
        return {"keyword": kw, "n": int(v[0]), "m": int(v[1])}
    if t == "closing_phrase":
        return {"phrase": rng.choice(wordlists()["closing_phrases"][_word_split(split)][int(v)])}
    if spec["param"] == "scope":
        return {"scope": v}
    if spec["param"] == "lines":
        return {"lines": v}
    return {"n": int(v)}


def _s(n: int) -> str:
    return "" if n == 1 else "s"


def _ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


_SCOPE = {"first_sentence": "In the first sentence of your response", "first_two_sentences": "In the first two sentences",
          "first_paragraph": "In the first paragraph of your response",
          "first_two_paragraphs": "In the first two paragraphs", "whole": "In your entire response"}


def rule_text(t: str, kw: dict) -> str:
    """The instruction shown to the model. Definitions are stated where the checker depends on them."""
    n = kw.get("n")
    if t == "unique_words":
        return f"Use at least {n} different words in your response."
    if t == "numbers_exact":
        return (f"Include exactly {n} numbers written as digits in your response (every separate run of digits "
                f"counts, including list numbering; a decimal such as 2.5 counts as one number).")
    if t == "conjunctions":
        return (f"Use at least {n} different coordinating conjunctions in your response, chosen from: "
                f"and, but, or, so, yet, for, nor.")
    if t == "word_repeats":
        return f"Do not use any word more than {n} times in your response."
    if t == "longest_word":
        return f"Do not use any word longer than {n} letters in your response."
    if t == "no_consecutive_initial":
        return f"{_SCOPE[kw['scope']]}, no two consecutive words may start with the same letter."
    if t == "longest_sentence":
        return f"Every sentence in your response must have at most {n} words."
    if t == "questions":
        return (f"Include at least {n} question{_s(n)} (sentence{_s(n)} ending with a question mark) in your response.")
    if t == "title_case":
        what = {"first_sentence": "Write the first sentence of your response",
                "first_two_sentences": "Write the first two sentences of your response",
                "first_paragraph": "Write the first paragraph of your response",
                "first_two_paragraphs": "Write the first two paragraphs of your response",
                "whole": "Write your entire response"}[kw["scope"]]
        return (f"{what} in title case: capitalise the first letter of every word except short words "
                f"such as a, an, the, and, or, of, in, to.")
    if t == "keyword_sentence":
        return f"Include the word \"{kw['keyword']}\" in the {_ordinal(n)} sentence of your response."
    if t == "keyword_position":
        return (f"Make the word \"{kw['keyword']}\" the {_ordinal(kw['m'])} word of the {_ordinal(n)} sentence "
                f"of your response.")
    if t == "closing_phrase":
        return f"End your response with the exact phrase \"{kw['phrase']}\". No other words should follow it."
    if t == "sections":
        return (f"Divide your response into at least {n} sections. Start each section with a line that says "
                f"SECTION X, such as: SECTION 1.")
    if t == "highlights":
        return f"Highlight at least {n} part{_s(n)} of your response with markdown, i.e. *highlighted part*."
    if t == "bracket_depth":
        return (f"Include brackets nested at least {n} levels deep somewhere in your response, using "
                f"( ), [ ] or {{ }}, for example " + _bracket_example(n) + ".")
    if t == "emoji_ends":
        return {"last_sentence": "End the last sentence of your response with an emoji.",
                "first_and_last_sentence": "End the first and the last sentence of your response with an emoji.",
                "each_paragraph": "End every paragraph of your response with an emoji.",
                "every_sentence": "End every sentence of your response with an emoji."}[kw["scope"]]
    if t == "stair_indent":
        if kw["lines"] == "all":
            return ("Indent every line of your response with more leading spaces than the line before it, "
                    "so the lines form a staircase.")
        return (f"Make the first {kw['lines']} lines of your response a staircase: each of these lines must start "
                f"with more leading spaces than the line before it.")
    if t == "hashtags":
        return (f"Include at least {n} " + ("hashtag" if n == 1 else "different hashtags") + " (such as #example) in your response.")
    raise KeyError(t)


def _bracket_example(n: int) -> str:
    o = "".join("([{"[i % 3] for i in range(n))
    c = "".join(")]}"[i % 3] for i in reversed(range(n)))
    return f"{o}a{c}"


def rule_record(t: str, level: str, kw: dict) -> dict:
    return {"type_id": t, "category": category_of(t), "designed_difficulty": level, "kwargs": kw,
            "rule_text": rule_text(t, kw),
            "rule_instance_id": hashlib.sha256(json.dumps([t, kw], sort_keys=True).encode()).hexdigest()[:12],
            "checker_id": f"{t}@difficulty-checkers-v0", "catalog_version": catalog()["version"]}
