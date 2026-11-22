"""Source-of-truth adapter for IFBench (allenai/IFBench at a pinned commit).

Everything that decides a label goes through the released IFBench code: the instruction
classes, their ``build_description()`` defaults, their wording and their ``check_following()``
verifiers. This module only

* pins the checkout and the NLTK data (no network, no other NLTK search path),
* exposes the official classes under stable names,
* renders and serializes prompts in the ManyIFEval prompt format, and
* runs the official strict and loose checks exactly as ``evaluation_lib`` does.

It never imports anything from the ManyIFEval package ``mife``.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

from .config import IFB_ROOT, load_config, nltk_data_dir, source_dir, test_file

# Same prompt layout as ManyIFEval (src/mife/source.py), so the probing code and the rule-span
# alignment carry over unchanged. Only the rule wording comes from IFBench.
PROMPT_HEADER = "\n\nYour response should follow the instructions below:\n"
RULE_PREFIX = "- "

CLASSIC_PREFIXES = (
    "keywords:", "language:", "length_constraints:", "detectable_content:", "detectable_format:",
    "combination:", "startend:", "change_case:", "punctuation:",
)


class NetworkDisabled(RuntimeError):
    pass


def _no_download(*args, **kwargs):  # pragma: no cover - only hit when vendored data is missing
    raise NetworkDisabled(
        f"nltk.download{args!r} was called: the vendored NLTK data is missing or unreadable. "
        "Run ifbench/scripts/fetch_source.sh; downloads are disabled by design.")


@lru_cache(maxsize=1)
def _bootstrap():
    """Import the pinned IFBench package with the vendored NLTK data and no network."""
    vendored = str(nltk_data_dir())
    os.environ["NLTK_DATA"] = vendored
    import nltk

    nltk.data.path[:] = [vendored]
    nltk.download = _no_download
    src = str(source_dir())
    if src not in sys.path:
        sys.path.insert(0, src)
    import ifbench  # noqa: F401
    from ifbench import instructions, instructions_registry, instructions_util

    pkg = Path(ifbench.__file__).resolve()
    if not pkg.is_relative_to(source_dir().resolve()):
        raise ImportError(f"ifbench imported from {pkg}, not from the pinned checkout {source_dir()}")
    # instructions.py prepends its own <pkg>/.nltk_data; keep the vendored directory as the only path.
    nltk.data.path[:] = [vendored]
    return instructions_registry, instructions, instructions_util


def registry() -> dict:
    return _bootstrap()[0].INSTRUCTION_DICT


def instructions_module():
    return _bootstrap()[1]


def ood_types() -> list[str]:
    """The 58 IFBench out-of-distribution types (registry order)."""
    return [k for k in registry() if not k.startswith(CLASSIC_PREFIXES)]


def classic_types() -> list[str]:
    return [k for k in registry() if k.startswith(CLASSIC_PREFIXES)]


def instruction(type_id: str):
    return registry()[type_id](type_id)


def category_of(type_id: str) -> str:
    return type_id.split(":", 1)[0]


# ---------------------------------------------------------------------------------------------
# D2: parameters from the released default generators
# ---------------------------------------------------------------------------------------------

# count:words_japanese: the released build_description() crashes when called with no arguments
# (it tests ``N % 10`` on the argument instead of the drawn value; instructions.py line 966).
# The shim draws N exactly as the released default would (random.randint(1, _NUM_WORD_CYCLE),
# one randint call on the module RNG) and passes it in, so the distribution and the RNG stream
# are unchanged. Wording still comes from the released code.
DEFAULT_SHIMS = {"count:words_japanese"}


def sample_default_kwargs(type_id: str, rng_seed: int | None = None) -> dict:
    """Draw one parameter setting from the released default generator for ``type_id``.

    IFBench generators use the module-level ``random``; if ``rng_seed`` is given it is seeded
    first (as ManyIFEval seeds per group). ``reference_text`` for ratio:overlap is filled by
    the caller (it is the task's text), never sampled.
    """
    if rng_seed is not None:
        random.seed(rng_seed)
    inst = instruction(type_id)
    if type_id == "count:words_japanese":
        n = random.randint(1, instructions_module()._NUM_WORD_CYCLE)
        inst.build_description(N=n)
    else:
        inst.build_description()
    args = inst.get_instruction_args() or {}
    return {k: v for k, v in args.items() if v is not None}


def is_parameterized(type_id: str) -> bool:
    """True if the released class exposes build_description() arguments (other than reference text)."""
    keys = instruction(type_id).get_instruction_args_keys() or []
    return bool([k for k in keys if k != "reference_text"])


def types_reading_task_text() -> list[str]:
    """Candidate types whose verifier reads the task text (only ratio:overlap, via reference_text)."""
    return sorted(t for t in ood_types() if "reference_text" in (instruction(t).get_instruction_args_keys() or []))


# ---------------------------------------------------------------------------------------------
# Rendering and serialization
# ---------------------------------------------------------------------------------------------

def _built(type_id: str, kwargs: dict, task_text: str | None = None):
    inst = instruction(type_id)
    kw = {k: v for k, v in (kwargs or {}).items() if v is not None}
    if type_id in types_reading_task_text():
        if task_text is None:
            raise ValueError(f"{type_id} needs the task text as its reference text")
        kw["reference_text"] = task_text
    inst.build_description(**kw)
    return inst


def render_rule_text(type_id: str, kwargs: dict) -> str:
    """The official instruction wording, byte-for-byte (decision 5: wording is never edited)."""
    kw = {k: v for k, v in (kwargs or {}).items() if v is not None and k != "reference_text"}
    if type_id in types_reading_task_text():
        kw["reference_text"] = ""
    text = instruction(type_id).build_description(**kw)
    if "\n" in text:
        raise ValueError(f"{type_id} renders a multi-line instruction; the bullet layout needs a decision")
    return text


def serialize_prompt(task_text: str, rule_texts: list[str]) -> str:
    """ManyIFEval layout: task, header, one '- ' line per rule."""
    return task_text + PROMPT_HEADER + "\n".join(RULE_PREFIX + t for t in rule_texts)


# ---------------------------------------------------------------------------------------------
# Official verification (labels come from here only)
# ---------------------------------------------------------------------------------------------

def official_check(type_id: str, kwargs: dict, response: str, task_text: str | None = None) -> bool:
    """Strict check, as evaluation_lib.test_instruction_following_strict: empty/blank fails."""
    inst = _built(type_id, kwargs, task_text)
    return bool(response and response.strip() and inst.check_following(response))


def loose_variants(response: str) -> list[str]:
    """The eight response variants of evaluation_lib.test_instruction_following_loose."""
    r = response.split("\n")
    first = "\n".join(r[1:]).strip()
    last = "\n".join(r[:-1]).strip()
    both = "\n".join(r[1:-1]).strip()
    return [response, response.replace("*", ""), first, last, both,
            first.replace("*", ""), last.replace("*", ""), both.replace("*", "")]


def official_check_loose(type_id: str, kwargs: dict, response: str, task_text: str | None = None) -> bool:
    inst = _built(type_id, kwargs, task_text)
    return any(v.strip() and inst.check_following(v) for v in loose_variants(response))


# ---------------------------------------------------------------------------------------------
# Pinning and provenance
# ---------------------------------------------------------------------------------------------

def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def source_commit() -> str:
    return subprocess.check_output(["git", "-C", str(source_dir()), "rev-parse", "HEAD"], text=True).strip()


def nltk_tree_hash(rel: str) -> str:
    """Hash of an unpacked NLTK package directory (paths + bytes), recorded for provenance."""
    root = nltk_data_dir() / rel
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(root)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def verify_pinned() -> dict:
    """Fail loudly unless the checkout, the test file and the NLTK zips match the config."""
    cfg = load_config()
    out = {}
    head = source_commit()
    if head != cfg["source"]["commit"]:
        raise AssertionError(f"IFBench checkout at {head}, config pins {cfg['source']['commit']}")
    out["commit"] = head
    status = subprocess.check_output(["git", "-C", str(source_dir()), "status", "--porcelain",
                                      "--untracked-files=no"], text=True).strip()
    if status:
        raise AssertionError(f"IFBench checkout has local modifications:\n{status}")
    sha = _sha256(test_file())
    if sha != cfg["source"]["test_file_sha256"]:
        raise AssertionError(f"IFBench test file sha256 {sha} != pinned {cfg['source']['test_file_sha256']}")
    out["test_file_sha256"] = sha
    zips = IFB_ROOT / "third_party" / "nltk_zips"
    out["nltk"] = {}
    for rel, want in cfg["nltk"]["packages"].items():
        z = zips / (rel.split("/")[-1] + ".zip")
        got = _sha256(z)
        if got != want:
            raise AssertionError(f"NLTK zip {z.name}: sha256 {got} != pinned {want}")
        if not (nltk_data_dir() / rel).is_dir():
            raise AssertionError(f"NLTK package {rel} is not unpacked under {nltk_data_dir()}")
        out["nltk"][rel] = {"zip_sha256": got, "tree_sha256": nltk_tree_hash(rel)}
    return out


def environment_fingerprint() -> dict:
    from importlib.metadata import PackageNotFoundError, version

    pk = {}
    for name in load_config()["packages"]:
        try:
            pk[name] = version(name)
        except PackageNotFoundError:
            pk[name] = None
    return {"python": sys.version.split()[0], "platform": platform.platform(), "packages": pk,
            "ifbench_commit": source_commit()}


def check_package_pins() -> dict:
    fp = environment_fingerprint()["packages"]
    want = load_config()["packages"]
    bad = {k: (fp[k], v) for k, v in want.items() if fp[k] != v}
    if bad:
        raise AssertionError(f"verifier dependency versions differ from the pins: {bad}")
    return fp


# ---------------------------------------------------------------------------------------------
# Released data (read for the drift audit and, in stage 4, for base tasks)
# ---------------------------------------------------------------------------------------------

def source_rows() -> list[dict]:
    with open(test_file(), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _clean_kwargs(kw: dict) -> dict:
    return {k: (int(v) if isinstance(v, float) and v.is_integer() else v) for k, v in kw.items() if v is not None}


def released_prompt_drift() -> dict:
    """How many released prompts contain every one of their instruction texts verbatim, as the
    pinned code renders them. Rows that do not are template drift between data and code."""
    rows = source_rows()
    verbatim, drift = 0, []
    for r in rows:
        ok = True
        for iid, kw in zip(r["instruction_id_list"], r["kwargs"]):
            try:
                d = instruction(iid).build_description(**_clean_kwargs(kw))
            except Exception as e:  # released kwargs the code cannot rebuild
                d = None
                err = repr(e)
            if not d or d not in r["prompt"]:
                ok = False
                drift.append({"key": r["key"], "type": iid, "rendered": (d or f"<error {err}>")[:160]})
        verbatim += ok
    n_constraints = {}
    for r in rows:
        n_constraints[len(r["instruction_id_list"])] = n_constraints.get(len(r["instruction_id_list"]), 0) + 1
    by_type = {}
    for d in drift:
        by_type[d["type"]] = by_type.get(d["type"], 0) + 1
    return {"rows": len(rows), "rows_verbatim": verbatim, "constraints_per_row": n_constraints,
            "drift_by_type": dict(sorted(by_type.items(), key=lambda x: -x[1])), "drift_examples": drift[:20]}
