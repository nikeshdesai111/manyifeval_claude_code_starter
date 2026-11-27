"""Stage 3: task writer (Prompts 1 and 2) and judges (task judge, pool compatibility judge) via OpenRouter.

Writer: anthropic/claude-sonnet-5.5 (provider anthropic); judge: openai/gpt-5.5 (provider openai). Both use
structured outputs (JSON schema); replies are validated again in code (jsonschema) and retried up to 3 times.
Neither endpoint accepts a temperature parameter (reasoning models; checked on OpenRouter 2026-11-27), so the plan's
writer 0.7 / judge 0 settings cannot be applied; provider defaults are used and recorded.

Per seed: Prompt 1 -> scenario; Prompt 2 x 2 mechanisms -> 6 tasks; keyword screen; task judge (all 6 together);
a failing mechanism is rewritten with the judge's reasons (<= 3 rewrites); then a pool is sampled and checked by the
compatibility judge (an invalid fail - no rules named or no reason - is re-judged; a valid fail redraws the pool).

Usage (test batch only): PYTHONPATH=src:.pydeps python -m mife.difficulty.tasks --seeds pc01 pc02
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import jsonschema
import pandas as pd
import yaml

from . import catalog as K
from . import openrouter as OR
from . import sampler as S

CFG = K.CFG
PROMPTS = CFG / "prompts"
SEEDS_CFG = yaml.safe_load((CFG / "seeds.yaml").read_text())
OUT = K.ROOT / "data" / "difficulty" / "tasks_test_batch"
WRITER = {"model": "anthropic/claude-sonnet-5.5", "provider": "anthropic"}
JUDGE = {"model": "openai/gpt-5.5", "provider": "openai", "reasoning": {"effort": "medium"}}
MAX_REWRITES = 3
MAX_JUDGE_RETRIES = 3
LEVELS = ("easy", "medium", "hard")
DESIGN = {"distractors": {"easy": 0, "medium": 2, "hard": 4}, "reasoning": {"easy": 1, "medium": 2, "hard": 3}}
MECH_SHORT = {"distractors": "dis", "reasoning": "rea"}
LEVEL_SHORT = {"easy": "easy", "medium": "med", "hard": "hard"}

# Keyword screen (plan "Keeping tasks free of rules"): any hit rejects the task.
SCREEN_RE = re.compile(r"\b(words?|sentences?|paragraphs?|bullets?|lists?|headings?|bold|format(?:ted|ting)?|tone|"
                       r"emojis?|hashtags?|quotes?|quotation|p\.s\.|begin|end with|include the word|at least|at most|"
                       r"no more than|in \w+ (?:points|steps|lines)|keep it (?:short|brief|under)|summary)\b", re.I)


def screen(task_text: str) -> list[str]:
    return sorted({m.group(0).lower() for m in SCREEN_RE.finditer(task_text)})


# ------------------------------------------------------------------------------------------- schemas

def _obj(props: dict) -> dict:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


STR, BOOL = {"type": "string"}, {"type": "boolean"}
STRS = {"type": "array", "items": STR}
SCENARIO_SCHEMA = _obj({"scenario_id": STR, "domain": STR, "subtopic": STR, "task_family": STR,
                        "scenario_description": STR, "core_task_objective": STR, "potential_source_facts": STRS,
                        "correct_answer": STR, "supports_both_mechanisms": BOOL, "source_reference": STR})
VARIANT = _obj({"task_id": STR, "task_text": STR, "designed_difficulty": {"type": "string", "enum": list(LEVELS)},
                "relevant_facts": STRS, "irrelevant_facts": STRS, "required_reasoning_steps": {"type": "integer"},
                "reasoning_dependencies": STRS, "expected_answer_or_content": STR, "task_completion_rubric": STR,
                "difficulty_justification": STR})
VARIANTS_SCHEMA = _obj({"scenario_id": STR, "domain": STR, "task_family": STR, "core_objective": STR,
                        "evolution_mechanism": {"type": "string", "enum": ["distractors", "reasoning"]},
                        "easy": VARIANT, "medium": VARIANT, "hard": VARIANT, "change_easy_to_medium": STR,
                        "change_medium_to_hard": STR, "unsuitable_for_mechanism": BOOL, "unsuitable_reason": STR})
CHECK = _obj({"pass": BOOL, "reason": STR})
CHECK_NAMES = ["level_matches_definition", "count_correct", "same_concept_and_answer", "only_intended_mechanism",
               "distractors_do_not_change_answer", "steps_dependent_and_not_skippable", "no_rules_or_formatting"]
TASK_JUDGE_SCHEMA = _obj({"seed_id": STR, "core_objective": STR, "tasks": {"type": "array", "items": _obj({
    "task_id": STR, "mechanism": {"type": "string", "enum": list(DESIGN)}, "level": {"type": "string", "enum": list(LEVELS)},
    "distractors_found": STRS, "inferences_found": STRS, "checks": _obj({c: CHECK for c in CHECK_NAMES}),
    "length_note": STR})}})
VERDICT = _obj({"pass": BOOL, "conflicting_rules": STRS, "reason": STR})
COMPAT_SCHEMA = _obj({"pool_id": STR, "rules_jointly_satisfiable": VERDICT, "tasks": {"type": "array", "items": _obj({
    "task_id": STR, "reasonably_possible": VERDICT})}})


def _call_json(role: dict, prompt: str, schema: dict, name: str, tag: str, variant: int = 0) -> tuple[dict, dict]:
    """One structured call, validated; parse/validation failures are retried with a new variant."""
    errors = []
    for k in range(MAX_JUDGE_RETRIES):
        r = OR.chat(role["model"], [{"role": "user", "content": prompt}], provider=role.get("provider"),
                    response_format=OR.json_schema_format(name, schema), reasoning=role.get("reasoning"),
                    max_tokens=16000, variant=variant * 10 + k, tag=tag)
        try:
            d = json.loads(r["content"])
            jsonschema.validate(d, schema)
            return d, {"cache_key": r["cache_key"], "cost": r["cost"], "cached": r["cached"], "usage": r["usage"],
                       "provider": r["provider"], "model": r["model"], "parse_retries": k, "errors": errors}
        except (json.JSONDecodeError, jsonschema.ValidationError) as e:
            errors.append(f"{type(e).__name__}: {str(e)[:200]}")
    raise RuntimeError(f"{tag}: no valid JSON after {MAX_JUDGE_RETRIES} tries: {errors}")


# ------------------------------------------------------------------------------------------- writer

def write_scenario(seed: dict) -> tuple[dict, dict]:
    d, t = seed["domain"], seed["task_type"]
    prompt = (PROMPTS / "p1_scenario.md").read_text().format(
        seed_id="{seed_id}", domain=d, domain_desc=SEEDS_CFG["domains"][d], task_type=t, task_type_desc=SEEDS_CFG["task_types"][t],
        cell_example=SEEDS_CFG["cell_examples"][d][t], raw_request=seed["text"])
    prompt = prompt.replace("{seed_id}", seed["seed_id"])
    d, meta = _call_json(WRITER, prompt, SCENARIO_SCHEMA, "seed_scenario", f"p1:{seed['seed_id']}")
    d["scenario"], d["core_objective"] = d["scenario_description"], d["core_task_objective"]
    return d, meta


def write_variants(seed: dict, scen: dict, mechanism: str, attempt: int, feedback: str) -> tuple[list[dict], dict]:
    mech = yaml.safe_load((PROMPTS / "mechanisms.yaml").read_text())
    fb = f"\nA reviewer rejected your previous versions. Fix these problems:\n{feedback}\n" if feedback else ""
    prompt = (PROMPTS / "p2_variants.md").read_text().format(
        seed_id=seed["seed_id"], domain=seed["domain"], task_type=seed["task_type"], scenario=scen["scenario"],
        core_objective=scen["core_objective"], correct_answer=scen["correct_answer"], mechanism=mechanism,
        mechanism_definition=mech[mechanism], feedback=fb)
    d, meta = _call_json(WRITER, prompt, VARIANTS_SCHEMA, "task_variants", f"p2:{seed['seed_id']}:{mechanism}",
                         variant=attempt)
    tasks = []
    for lvl in LEVELS:
        x = dict(d[lvl])
        x["writer_task_id"] = x.pop("task_id")
        x["level"] = lvl
        x["distractors"] = x["irrelevant_facts"] if mechanism == "distractors" else []
        x["inferences"] = x["reasoning_dependencies"] if mechanism == "reasoning" else []
        x["change_easy_to_medium"], x["change_medium_to_hard"] = d["change_easy_to_medium"], d["change_medium_to_hard"]
        x["unsuitable_for_mechanism"], x["unsuitable_reason"] = d["unsuitable_for_mechanism"], d["unsuitable_reason"]
        tasks.append(x)
    for x in tasks:
        x["task_id"] = f"{seed['seed_id']}-{MECH_SHORT[mechanism]}-{LEVEL_SHORT[x['level']]}"
        x["mechanism"] = mechanism
        x["writer_attempt"] = attempt
        x["screen_hits"] = screen(x["task_text"])
        x["n_words"] = len(x["task_text"].split())
    return tasks, meta


# ------------------------------------------------------------------------------------------- judges

def judge_tasks(seed: dict, scen: dict, tasks: list[dict], variant: int) -> tuple[dict, dict]:
    mech = yaml.safe_load((PROMPTS / "mechanisms.yaml").read_text())
    tt = "\n\n".join(f"[{x['task_id']}] mechanism={x['mechanism']} level={x['level']}\n{x['task_text']}" for x in tasks)
    prompt = (PROMPTS / "p4_task_judge.md").read_text().format(
        seed_id=seed["seed_id"], domain=seed["domain"], task_type=seed["task_type"],
        core_objective=scen["core_objective"], correct_answer=scen["correct_answer"],
        mechanisms="\n".join(mech.values()), tasks=tt)
    return _call_json(JUDGE, prompt, TASK_JUDGE_SCHEMA, "task_judge", f"p4:{seed['seed_id']}", variant=variant)


def accept_task(task: dict, v: dict, medium_distractors: list[str] | None) -> tuple[bool, list[str]]:
    """Code acceptance (plan "Validating easy, medium and hard")."""
    why = []
    for c in CHECK_NAMES:
        if not v["checks"][c]["pass"]:
            why.append(f"{c}: {v['checks'][c]['reason']}")
    want = DESIGN[task["mechanism"]][task["level"]]
    mine, other = (("distractors_found", "inferences_found") if task["mechanism"] == "distractors"
                   else ("inferences_found", "distractors_found"))
    if len(v[mine]) != want:
        why.append(f"judge found {len(v[mine])} {mine.split('_')[0]} (designed {want})")
    if v[other]:
        why.append(f"judge found {len(v[other])} {other.split('_')[0]} (must be 0)")
    if task["mechanism"] == "distractors" and task["level"] == "hard" and medium_distractors is not None:
        missing = [m for m in medium_distractors if m.strip() not in task["task_text"]]
        if missing:
            why.append(f"hard version is missing medium distractors: {missing}")
    if task["screen_hits"]:
        why.append(f"keyword screen: {task['screen_hits']}")
    return not why, why


def compat_judge(pool_id: str, rules: list[dict], tasks: list[dict], attempt: int, log: list) -> dict:
    rid = {f"R{i + 1}": r for i, r in enumerate(rules)}
    rr = "\n".join(f"{k}: {r['rule_text']}" for k, r in rid.items())
    tt = "\n\n".join(f"[{x['task_id']}]\n{x['task_text']}" for x in tasks)
    prompt = (PROMPTS / "p5_compat_judge.md").read_text().format(pool_id=pool_id, rules=rr, tasks=tt)
    for k in range(MAX_JUDGE_RETRIES):
        d, meta = _call_json(JUDGE, prompt, COMPAT_SCHEMA, "pool_compat", f"p5:{pool_id}:a{attempt}",
                             variant=attempt * 10 + k)
        verdicts = [d["rules_jointly_satisfiable"]] + [t["reasonably_possible"] for t in d["tasks"]]
        ids_ok = {t["task_id"] for t in d["tasks"]} == {x["task_id"] for x in tasks}
        invalid = [v for v in verdicts if not v["pass"] and (not v["conflicting_rules"] or not v["reason"].strip()
                                                              or not set(v["conflicting_rules"]) <= set(rid))]
        log.append({"pool_id": pool_id, "attempt": attempt, "judge_try": k, "verdict": d, "meta": meta,
                    "rules": [{"id": i, "type_id": r["type_id"], "level": r["designed_difficulty"],
                               "rule_text": r["rule_text"]} for i, r in rid.items()],
                    "invalid_fails": len(invalid), "task_ids_ok": ids_ok})
        if invalid or not ids_ok:
            continue
        fails = [v for v in verdicts if not v["pass"]]
        named = sorted({rid[x]["type_id"] for v in fails for x in v["conflicting_rules"]})
        return {"accepted": not fails, "reason": "; ".join(v["reason"] for v in fails) or "all pass",
                "conflicting_rules": named, "verdict": d}
    return {"accepted": False, "reason": "judge gave no valid verdict (unnamed rules or no reason)",
            "conflicting_rules": [], "invalid": True}


# ------------------------------------------------------------------------------------------- per seed

def run_seed(seed: dict, split: str, sampler_seed: int) -> dict:
    out = OUT / seed["seed_id"]
    out.mkdir(parents=True, exist_ok=True)
    scen, m1 = write_scenario(seed)
    (out / "scenario.json").write_text(json.dumps({"seed": seed, "scenario": scen, "meta": m1}, indent=1, default=str))
    tasks = {m: None for m in DESIGN}
    feedback = {m: "" for m in DESIGN}
    history = []
    accepted = {m: False for m in DESIGN}
    for rnd in range(MAX_REWRITES + 1):
        for m in DESIGN:
            if not accepted[m]:
                tasks[m], meta = write_variants(seed, scen, m, rnd, feedback[m])
                history.append({"round": rnd, "mechanism": m, "writer_meta": meta, "tasks": tasks[m]})
        six = tasks["distractors"] + tasks["reasoning"]
        verdict, jm = judge_tasks(seed, scen, six, variant=rnd)
        by_id = {v["task_id"]: v for v in verdict["tasks"]}
        res = {}
        med = next(x for x in tasks["distractors"] if x["level"] == "medium")["distractors"]
        for x in six:
            v = by_id.get(x["task_id"])
            ok, why = (False, ["judge returned no verdict for this task"]) if v is None else accept_task(x, v, med)
            res[x["task_id"]] = {"accepted": ok, "why": why, "verdict": v}
        history.append({"round": rnd, "judge": verdict, "judge_meta": jm, "acceptance": res})
        for m in DESIGN:
            fails = {tid: r["why"] for tid, r in res.items() if tid.split("-")[1] == MECH_SHORT[m] and not r["accepted"]}
            accepted[m] = not fails
            feedback[m] = "\n".join(f"- {tid}: " + "; ".join(w) for tid, w in fails.items())
        if all(accepted.values()):
            break
    (out / "tasks_history.json").write_text(json.dumps(history, indent=1, default=str))
    six = tasks["distractors"] + tasks["reasoning"]
    final = [{**x, "accepted": res[x["task_id"]]["accepted"], "acceptance_reasons": res[x["task_id"]]["why"],
              "judge_verdict": res[x["task_id"]]["verdict"]} for x in six]
    (out / "tasks.json").write_text(json.dumps(final, indent=1))
    seed_ok = all(accepted.values())
    # pool + compatibility judge (only for a seed whose 6 tasks were accepted; else use the last tasks to test it)
    compat_log: list = []
    sampler = S.PoolSampler(sampler_seed)
    pool = sampler.sample_pool(seed["seed_id"], split,
                               judge=lambda rules, a: compat_judge(seed["seed_id"], rules, six, a, compat_log))
    (out / "pool.json").write_text(json.dumps(pool, indent=1, default=str))
    (out / "compat_log.json").write_text(json.dumps({"judge_calls": compat_log, "sampler_log": sampler.log},
                                                    indent=1, default=str))
    return {"seed_id": seed["seed_id"], "domain": seed["domain"], "task_type": seed["task_type"],
            "seed_accepted": seed_ok, "rounds": rnd + 1, "accepted_tasks": sum(x["accepted"] for x in final),
            "pool_attempts": pool["attempts"], "pool_mix": pool["level_mix"],
            "pool_rejections": [e for e in sampler.log if e["event"] == "pool_rejected"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", required=True)
    ap.add_argument("--sampler-seed", type=int, default=2026112701)
    a = ap.parse_args()
    seeds = pd.read_parquet(K.ROOT / "data" / "difficulty" / "seeds.parquet").set_index("seed_id")
    summary = []
    for sid in a.seeds:
        s = seeds.loc[sid].to_dict()
        s["seed_id"] = sid
        split = "test" if s.get("split") == "test" else "train"
        summary.append(run_seed(s, split, a.sampler_seed))
        print(json.dumps(summary[-1], indent=1, default=str), flush=True)
    cost = OR.total_cost()
    (OUT / "summary.json").write_text(json.dumps({"seeds": summary, "cost": cost, "writer": WRITER, "judge": JUDGE},
                                                 indent=1, default=str))
    print(json.dumps(cost, indent=1))


if __name__ == "__main__":
    main()
