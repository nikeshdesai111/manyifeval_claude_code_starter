"""Deterministic, rule-based implementation of the task-family rubric.

Codifies `docs/task_family_rubric.md` (v1) as transparent keyword/pattern rules over the
task text, so the family labelling can be re-run by script. It does NOT replace the
hand labels in `configs/task_families_v1.csv`, which remain the labels the pipeline uses
(`mife.tasks.load_families`). Its purpose is reproducibility and an agreement audit.

Method
------
Each family has a list of named cue rules (regular expressions over the lower-cased task
text). The rubric's tie-breaking precedence is applied literally: the first family, in the
order TRANSFORM > CREATIVE > PERSUASIVE > PRACTICAL > ANALYTICAL > INFO, with any matching
cue wins; INFO is the default when nothing matches ("otherwise it is a factual
explanation"). Tone words (funny, weird, angry, casual ...) carry no cue, so tone never
changes the family. Correspondence (letters and emails, including to friends) is a
PRACTICAL cue by the rubric's policy, so it is PRACTICAL unless an earlier family fires
(e.g. an explicitly persuasive email is PERSUASIVE).

The cues were written from the rubric's family definitions and precedence list only. They
were frozen (committed) before being compared with the hand labels and were not adjusted
to agree with them; see `docs/task_family_rules_agreement.md`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

RULES_VERSION = "family_rules_v1"
PRECEDENCE = ["TRANSFORM", "CREATIVE", "PERSUASIVE", "PRACTICAL", "ANALYTICAL", "INFO"]

# Verbs that introduce the deliverable the task asks for.
_MAKE = r"(?:write|compose|create|draft|make|generate|give me|come up with|re-create|hallucinate|produce|i want|i need)"
# A quoted span, or an explicit pointer to supplied text.
_SUPPLIED = r"(?:\"|“|'|the following|the text below|this\b|it\b)"

CUES: dict[str, list[tuple[str, str]]] = {
    # Rubric rule 1: the task supplies a text to transform (expand, elaborate, complete,
    # rephrase, ask about a given sentence), or the task is underspecified / other.
    "TRANSFORM": [
        ("elaborate_on_supplied", rf"\belaborate on {_SUPPLIED}"),
        ("expand_supplied", rf"\bexpand(?:ing)? {_SUPPLIED}"),
        ("complete_supplied", rf"\b(?:complete|finish) (?:the following|the sentence|this)\b"),
        ("text_below_starting_point", r"\btext below\b"),
        ("rephrase_supplied", r"\b(?:rephrase|reword|paraphrase|rewrite)\b|\bways to say the following\b"),
        ("question_about_given_sentence", r"\bgiven the sentence\b"),
        ("brainstorm_name", r"\bbrainstorm (?:a )?names?\b"),
        ("q_and_a_unspecified", r"^give me q ?(?:and|&) ?as?\.?$"),
        ("fragment_no_task", r"^(?:in|with|for|as|like) the style of\b"),
    ],
    # Rubric rule 2: the deliverable is in a creative form (verse, narrative, dialogue);
    # form beats topic. "Invented scenario" covers explicit role-play ("pretend you are").
    "CREATIVE": [
        ("make_creative_form", rf"\b{_MAKE}\b[^.?!]{{0,60}}?\b(?:story|stories|poem|poems|song|songs|lyrics|dialogue|conversation|fiction|fairy tale|rap|riddle|plot|limerick|haiku|verse|screenplay|script)\b"),
        ("in_creative_form", r"\b(?:in the form of|as) (?:a|an) (?:\w+ ){0,2}(?:poem|riddle|song|story|rap|dialogue)\b"),
        ("make_it_creative_form", r"\bmake it (?:a|an) (?:\w+ ){0,2}(?:poem|song|story|rap)\b"),
        ("rap_style", r"\bin a rap style\b|\bas a rap\b"),
        ("role_play", r"\bpretend (?:that )?you(?:'re| are)\b"),
    ],
    # Rubric rule 3: primary purpose is to sell or persuade.
    "PERSUASIVE": [
        ("advertising", r"\b(?:ad|ads|advertisement|advertising|ad copy|copy for an ad)\b"),
        ("pitch", r"\bpitch(?: deck)?\b"),
        ("product_description", r"\bproduct description\b"),
        ("persuade", r"\bpersuasive\b|\bpersuade\b|\bconvince (?!you\b)"),
        ("x_is_the_best", r"\bis the best\b|\bthe best (?:city|place|country) (?:in|to)\b"),
        ("marketing_post", r"\b(?:post|tweet)\b[^.?!]{0,40}\bfor (?:a|an|the) (?:\w+ ){0,3}(?:startup|company|business|brand)\b|\bcompany\b[^.?!]{0,60}\b(?:catchy )?tweet\b"),
    ],
    # Rubric rule 4: functional document, plan, instructions, advice or correspondence.
    "PRACTICAL": [
        ("functional_document", r"\b(?:resume|cv|cover letters?|itinerary|template|rubric|proposal|planning doc|job description|checklist|schedule|agenda)\b"),
        ("plan_trip", r"\bplan (?:a|an|my) (?:\d+[- ]\w+ )?(?:\w+ ){0,2}(?:trip|vacation|journey)\b"),
        ("correspondence", r"\b(?:letter|email|e-mail|memo|note) (?:to|for)\b|\bwrite (?:me |them )?(?:a|an) (?:\w+ ){0,2}(?:letter|email|e-mail)\b|\bhelp me write a letter\b"),
        ("how_to_steps", r"\bhow to\b|\bwhat are the steps\b|\bsteps to\b|\bthe best way to\b|\bhow (?:can|do|should) (?:i|you|we)\b"),
        ("advice_recommendation", r"\brecommend(?:ation|ations)?\b|\badvice\b|\bwhich \w+ should i\b|\bshould i\b"),
    ],
    # Rubric rule 5: comparison, evaluation or judgement.
    "ANALYTICAL": [
        ("comparison", r"\bdifferences? between\b|\bpros and cons\b|\b(?:vs\.?|versus)\b|\bcompare\b|\bcomparison\b|\bbetter way\b|\bis different from\b"),
        ("critique_review", r"\bcritique\b|\breview\b|\bevaluat(?:e|ion)\b|\bassess\b"),
        ("opinion_judgement", r"\bdo you think\b|\bwhat are your thoughts\b|\byour thoughts on\b|\bwhat do you think\b|\bis (?:this|that|it) a good\b|\bis (?:\w+ ){1,4}a good (?:idea|actor|actress|way|explanation)\b|\bmean to you\b"),
    ],
}

# When the deliverable is a critique/review of supplied material, words naming that material
# (an ad, a pitch) describe the object, not the deliverable's purpose, so they are not
# persuasion cues (rubric rule 3 asks for the deliverable's *primary purpose*).
_CRITIQUE_OF_SUPPLIED = re.compile(r"^(?:please )?(?:critique|review|evaluate|assess)\b")

_COMPILED = {fam: [(name, re.compile(rx)) for name, rx in rules] for fam, rules in CUES.items()}


@dataclass(frozen=True)
class Assignment:
    family: str
    rule: str            # the cue that decided the family ("default" for INFO fallback)
    matched: str         # matched text span ("" for default)
    all_families: tuple  # every family with at least one matching cue, in precedence order


def normalize(text: str) -> str:
    t = text.strip().lower()
    t = t.replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", t)


def classify(task_text: str) -> Assignment:
    t = normalize(task_text)
    hits = []
    for fam in PRECEDENCE[:-1]:
        if fam == "PERSUASIVE" and _CRITIQUE_OF_SUPPLIED.search(t):
            continue
        for name, rx in _COMPILED[fam]:
            m = rx.search(t)
            if m:
                hits.append((fam, name, m.group(0)))
                break
    if hits:
        fam, name, span = hits[0]
        return Assignment(fam, name, span, tuple(h[0] for h in hits))
    return Assignment("INFO", "default", "", ())
