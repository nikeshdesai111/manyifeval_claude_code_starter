"""The candidate instruction-type inventory (Nicky's decision, 2026-11-22).

IFBench has 58 out-of-distribution types. 15 are excluded before any audit, for reasons that
make them unusable in a task-independent, reusable rule pool. The remaining 43 are
*candidates*: the final inventory is chosen only after the stage-3 pairwise witness audit,
with no preset cap.
"""
from __future__ import annotations

from . import source

_TASK = "it is a complete task of its own (fixed question and fixed answer), so it cannot be applied to another task"
_REPEAT = "it asks the model to repeat or edit the prompt text, so its verifier reads the prompt and the rule depends on the task"

EXCLUDED: dict[str, str] = {
    "custom:multiples": _TASK,
    "custom:mcq_count_length": _TASK,
    "custom:reverse_newline": _TASK,
    "custom:word_reverse": _TASK,
    "custom:character_reverse": _TASK,
    "custom:sentence_alphabet": _TASK,
    "custom:european_capitals_sort": _TASK,
    "custom:csv_city": _TASK,
    "custom:csv_special_character": _TASK,
    "custom:csv_quotes": _TASK,
    "custom:date_format_list": _TASK,
    "repeat:repeat_change": _REPEAT,
    "repeat:repeat_simple": _REPEAT,
    "repeat:repeat_span": _REPEAT,
    "format:options": "the answer must be exactly one fixed option and nothing else, so it conflicts with every other "
                      "rule and with almost every task",
}


def candidates() -> list[str]:
    """The 43 candidate types, in IFBench registry order."""
    return [t for t in source.ood_types() if t not in EXCLUDED]


def parameterized() -> list[str]:
    return [t for t in candidates() if source.is_parameterized(t)]


def fixed() -> list[str]:
    return [t for t in candidates() if not source.is_parameterized(t)]


def table() -> list[dict]:
    rows = []
    for t in source.ood_types():
        inst = source.instruction(t)
        keys = [k for k in (inst.get_instruction_args_keys() or []) if k != "reference_text"]
        rows.append({
            "type": t,
            "category": source.category_of(t),
            "status": "excluded" if t in EXCLUDED else "candidate",
            "reason": EXCLUDED.get(t, ""),
            "parameters": keys,
            "reads_task_text": t in source.types_reading_task_text(),
        })
    return rows
