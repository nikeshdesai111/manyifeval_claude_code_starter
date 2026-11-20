"""Official ManyIFEval rule verification with per-rule binary outcomes.

Semantics replicate `utils.test_instruction_following` from the pinned release exactly
(rule is followed iff response.strip() is non-empty AND check_following(response)), with
langdetect seeded (DetectorFactory.seed = 0) for deterministic case-rule checks.
"""
from __future__ import annotations

import re

from . import source as S

VERIFIER_VERSION = "manyifeval@5c1ba29+nltk-vendored+langdetect-seed0/v1"


def follows(response: str, type_id: str, kwargs: dict, task_text: str) -> bool:
    inst = S.official_checker(type_id, kwargs, task_text)
    return bool(response.strip()) and bool(inst.check_following(response))


def verify(response: str, rules: list[tuple[str, dict]], task_text: str) -> list[int]:
    """Return a violation vector (1 = violated, 0 = followed), one entry per rule, in order."""
    return [0 if follows(response, t, kw, task_text) else 1 for t, kw in rules]


def diagnostics(response: str) -> dict:
    """Cheap measured quantities used by the verifiers; saved for audits, not used as labels."""
    from instruction_following_eval import instructions_util as U
    import langdetect

    try:
        lang = langdetect.detect(response) if response.strip() else None
    except langdetect.LangDetectException:
        lang = "undetectable"
    paragraphs = re.split(r"\s?\*\*\*\s?", response)
    return {
        "n_chars": len(response),
        "n_words": U.count_words(response),
        "n_sentences": U.count_sentences(response) if response.strip() else 0,
        "n_paragraph_chunks": len(paragraphs),
        "n_bullets": len(re.findall(r"^\s*\*[^\*].*$", response, flags=re.MULTILINE))
        + len(re.findall(r"^\s*-.*$", response, flags=re.MULTILINE)),
        "n_placeholders": len(re.findall(r"\[.*?\]", response)),
        "has_title": bool(re.search(r"<<[^\n]+>>", response)),
        "quoted": len(response.strip()) > 1 and response.strip()[0] == '"' and response.strip()[-1] == '"',
        "n_commas": response.count(","),
        "is_upper": response.isupper(),
        "is_lower": response.islower(),
        "langdetect": lang,
        "empty": not response.strip(),
    }
