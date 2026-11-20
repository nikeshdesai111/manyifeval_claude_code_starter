"""Source-of-truth integration with the pinned ManyIFEval release.

Everything benchmark-defining (rule templates, verifiers, task texts, parameter values) is
read from the pinned checkout; nothing here re-implements benchmark semantics.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from functools import lru_cache

from .config import load_config, nltk_dir, source_dir

# --- make the official package importable with the vendored NLTK data -----------------
_SRC = str(source_dir())
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import nltk  # noqa: E402

nltk.data.path[:] = [str(nltk_dir())]  # use ONLY vendored tokenizer data

import langdetect  # noqa: E402
from langdetect import DetectorFactory  # noqa: E402

DetectorFactory.seed = 0  # official case checkers call langdetect, which is stochastic otherwise

from instruction_following_eval import instructions_registry as _registry  # noqa: E402
from instruction_following_eval import instructions_util as _util  # noqa: E402

PROMPT_HEADER = "\n\nYour response should follow the instructions below:\n"

# Plan v3.4 Section 3.1 categories mapped to code type IDs.
# NOTE: the plan's "character count" type is the code's number_words (WORD count); see audit M-minor.
TYPE_INFO = {
    "keywords:existence": ("keywords", "Include keywords"),
    "keywords:frequency": ("keywords", "Keyword occurrence count"),
    "keywords:forbidden_words": ("keywords", "Forbidden keywords"),
    "keywords:letter_frequency": ("length_counts", "Specific letter count"),
    "length_constraints:number_words": ("length_counts", "Word count"),
    "length_constraints:number_sentences": ("length_counts", "Sentence count"),
    "length_constraints:number_paragraphs": ("length_counts", "Paragraph count"),
    "detectable_content:number_placeholders": ("formatting", "Placeholder"),
    "detectable_format:number_bullet_lists": ("formatting", "Bullet points"),
    "detectable_format:title": ("formatting", "Title"),
    "change_case:english_capital": ("letter_case", "Uppercase"),
    "change_case:english_lowercase": ("letter_case", "Lowercase"),
    "change_case:capital_word_frequency": ("letter_case", "Uppercase word count"),
    "startend:quotation": ("beginning_ending", "Quotation marks"),
    "punctuation:no_comma": ("punctuation", "Comma prohibition"),
}
TYPES = sorted(TYPE_INFO)

# D1: ManyIFEval-implied mutually exclusive groups (reproduces all 18 released type lists).
EXCLUSIVE_GROUPS = [
    {"keywords:existence", "keywords:frequency"},
    {"change_case:capital_word_frequency", "change_case:english_capital", "change_case:english_lowercase"},
    {"length_constraints:number_words", "length_constraints:number_sentences", "length_constraints:number_paragraphs"},
]

# Types whose threshold is compared via a relation field ("less than" / "at least").
RELATION_FIELDS = {
    "keywords:frequency": ("relation", ["frequency"]),
    "keywords:letter_frequency": ("let_relation", ["let_frequency"]),
    "length_constraints:number_words": ("relation", ["num_words"]),
    "length_constraints:number_sentences": ("relation", ["num_sentences"]),
    "change_case:capital_word_frequency": ("capital_relation", ["capital_frequency"]),
}
WORD_LIST_FIELDS = {"keywords:existence": "keywords", "keywords:forbidden_words": "forbidden_words"}


def conflicts(a: str, b: str) -> bool:
    if a == b:
        return True
    return any(a in g and b in g for g in EXCLUSIVE_GROUPS)


def category_of(type_id: str) -> str:
    return TYPE_INFO[type_id][0]


def code_prefix_of(type_id: str) -> str:
    return type_id.split(":")[0]


@lru_cache(maxsize=None)
def source_rows() -> list[dict]:
    cfg = load_config()["source"]
    path = source_dir() / cfg["data_file"]
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != cfg["data_sha256"]:
        raise RuntimeError(f"ManyIFEval data hash mismatch: {digest}")
    return [json.loads(l) for l in raw.decode().splitlines() if l.strip()]


def source_commit() -> str:
    return subprocess.check_output(["git", "-C", str(source_dir()), "rev-parse", "HEAD"], text=True).strip()


@lru_cache(maxsize=None)
def tasks() -> dict[int, dict]:
    """216 source tasks keyed by ManyIFEval `key`."""
    out: dict[int, dict] = {}
    for r in source_rows():
        k = r["key"]
        if k not in out:
            out[k] = {
                "task_key": k,
                "task_prompt_raw": r["task_prompt"],
                "task_text": r["task_prompt"].strip(),
                "source_rules": r["rules"],
                "source_type_list": r["not_conflict_rules_list"],
            }
    return out


def render_rule_text(type_id: str, kwargs: dict) -> str:
    """Official ManyIFEval rule wording: IFEval build_description with newlines flattened."""
    inst = _registry.INSTRUCTION_DICT[type_id](type_id)
    return inst.build_description(**kwargs).replace("\n", " ")


def serialize_prompt(task_text: str, rule_texts: list[str]) -> str:
    return task_text + PROMPT_HEADER + "\n".join("- " + t for t in rule_texts)


def rule_line_prefix() -> str:
    return "- "


def verify_source_reproduction() -> dict:
    """Assert our serializer reproduces every released ManyIFEval prompt byte-for-byte."""
    n = 0
    for r in source_rows():
        texts = [render_rule_text(t, kw) for t, kw in zip(r["instruction_id_list"], r["kwargs"])]
        got = serialize_prompt(r["task_prompt"].strip(), texts)
        if got != r["prompt"]:
            raise AssertionError(f"serialization mismatch for task {r['key']} k={len(texts)}")
        n += 1
    type_lists = {tuple(sorted(t["source_type_list"])) for t in tasks().values()}
    for tl in type_lists:
        for i, a in enumerate(tl):
            for b in tl[i + 1:]:
                if conflicts(a, b):
                    raise AssertionError(f"released type list violates D1 rule: {a} + {b}")
    return {"prompts_reproduced": n, "tasks": len(tasks()), "distinct_type_lists": len(type_lists)}


def instruction_class(type_id: str):
    """The official IFEval/ManyIFEval instruction class for a type id (pinned registry)."""
    return _registry.INSTRUCTION_DICT[type_id]


def count_sentences(text: str) -> int:
    return _util.count_sentences(text)


def official_checker(type_id: str, kwargs: dict, task_text: str):
    """Build an official instruction object exactly as utils.test_instruction_following does."""
    inst = _registry.INSTRUCTION_DICT[type_id](type_id)
    inst.build_description(**kwargs)
    args = inst.get_instruction_args()
    if args and "prompt" in args:
        inst.build_description(prompt=task_text)
    return inst


def types_reading_task_text() -> list[str]:
    """Instruction types whose official verifier reads the task text (args key "prompt").

    Empty for the 15 ManyIFEval types: every verifier judges the response alone, so whether a rule set can be
    satisfied does not depend on which task it is attached to.
    """
    return sorted(t for t in TYPES if "prompt" in (instruction_class(t)(t).get_instruction_args_keys() or []))


def environment_fingerprint() -> dict:
    tok_files = sorted((nltk_dir() / "tokenizers").rglob("english*")) + sorted(
        (nltk_dir() / "tokenizers" / "punkt_tab" / "english").glob("*"))
    return {
        "source_commit": source_commit(),
        "nltk_version": nltk.__version__,
        "langdetect_seed": DetectorFactory.seed,
        "langdetect_version": getattr(langdetect, "__version__", "unknown"),
        "nltk_data_sha256": {str(p.relative_to(nltk_dir())): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in tok_files if p.is_file()},
    }
