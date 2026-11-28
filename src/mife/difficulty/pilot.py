"""Calibration pilot (plan "Calibrating easy, medium and hard"), round r: every type at its 3 current levels, alone in
the prompt (k = 1), on all 48 tasks of the 8 calibration seeds, 5 responses per prompt per model.
54 rule instances x 48 tasks = 2,592 prompts; x 5 = 12,960 responses per model.

Steps (nothing here makes paid calls; task writing is `mife.difficulty.tasks --tasks-only`):
  build      prompts (K10 format: task + "Your response should follow the instructions below:" + "- rule"),
             per-model tokenization with the K10 revisions and chat templates (refuses truncation at 4,096),
             per-sample seeds, worklists in K10's schema (worklist_<m>.parquet, tokenized_<m>.parquet), summary.
  generate   Mac fallback: MLX, BF16 weights, K10 decoding (temperature 0.6, top-p 0.9, max 2,048), resumable.
             (Betty route: K10's generate.py reading this round's inputs; see the report.)
  verify     run the checkers (Gemma responses cut at <end_of_turn>, as K10's verify.py).
  analyze    pooled and per-model violation rates per type x level, 90% bootstrap over the 48 tasks, criteria.

Usage: PYTHONPATH=src:.pydeps python -m mife.difficulty.pilot {build,generate,verify,analyze} --round 1 [...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import catalog as K
from . import checkers as C
from .capability import MODELS
from .sampler import derive_seed

PILOT = K.ROOT / "data" / "difficulty" / "pilot"
TASKS_DIR = K.ROOT / "data" / "difficulty" / "tasks_pilot"
PROMPT_HEADER = "\n\nYour response should follow the instructions below:\n"   # K10 / ManyIFEval format
N_SAMPLES = 5
MAX_MODEL_LEN = 4096
MAX_NEW = 2048
DECODING = {"temperature": 0.6, "top_p": 0.9, "max_new_tokens": MAX_NEW}
EOT = "<end_of_turn>"


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def rule_instances(round_: int) -> list[dict]:
    rng = random.Random(derive_seed("pilot-rules", round_))
    out = []
    for t in K.types():
        for lvl in K.LEVELS:
            out.append(K.rule_record(t, lvl, K.instantiate(t, lvl, rng, "train", set())))
    return out


def calibration_seeds() -> list[str]:
    S = pd.read_parquet(K.ROOT / "data" / "difficulty" / "seeds.parquet")
    return sorted(S[S.role == "calibration"].seed_id)


def load_tasks(seed_ids: list[str]) -> list[dict]:
    tasks = []
    for sid in seed_ids:
        f = TASKS_DIR / sid / "tasks.json"
        ts = json.loads(f.read_text())
        bad = [t["task_id"] for t in ts if not t["accepted"]]
        if bad:
            raise SystemExit(f"{sid}: tasks not accepted {bad}; replace the seed before building the pilot")
        tasks += ts
    return tasks


def build(round_: int, seed_ids: list[str]) -> None:
    from transformers import AutoTokenizer
    out = PILOT / f"round{round_}" / "inputs"
    out.mkdir(parents=True, exist_ok=True)
    tasks, rules = load_tasks(seed_ids), rule_instances(round_)
    rows = []
    for t in tasks:
        for r in rules:
            text = t["task_text"] + PROMPT_HEADER + "- " + r["rule_text"]
            rows.append({"prompt_id": f"pl{round_}-{t['task_id']}-{r['type_id']}-{r['designed_difficulty']}",
                         "task_id": t["task_id"], "seed_id": t["task_id"].split("-")[0], "mechanism": t["mechanism"],
                         "task_level": t["level"], "type_id": r["type_id"], "level": r["designed_difficulty"],
                         "kwargs_json": json.dumps(r["kwargs"], sort_keys=True), "rule_text": r["rule_text"],
                         "rule_instance_id": r["rule_instance_id"], "prompt_text": text, "prompt_sha256": sha(text)})
    P = pd.DataFrame(rows)
    P.to_parquet(out / "prompts.parquet")
    summ = {"version": f"difficulty-pilot-r{round_}-v0", "run_name": f"difficulty_pilot_r{round_}",
            "max_model_len": MAX_MODEL_LEN, "n_chunks": 1, "prompts_sha256": hashlib.sha256(
                (out / "prompts.parquet").read_bytes()).hexdigest(),
            "round": round_, "n_prompts": len(P), "n_tasks": len(tasks), "n_rules": len(rules),
            "catalog_sha256": K.catalog_sha256(), "checker_version": C.CHECKER_VERSION, "decoding": DECODING,
            "n_samples": N_SAMPLES, "models": {}}
    for key, (hf, rev) in MODELS.items():
        tok = AutoTokenizer.from_pretrained(hf, revision=rev)
        ids = []
        for p in P.prompt_text:   # same rendering as K10 build.encode_with
            rendered = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
            assert rendered.count(p) == 1
            ids.append(list(tok(rendered, add_special_tokens=False)["input_ids"]))
        n = [len(x) for x in ids]
        if max(n) + MAX_NEW > MAX_MODEL_LEN:
            raise SystemExit(f"{key}: prompt of {max(n)} tokens + {MAX_NEW} new exceeds {MAX_MODEL_LEN}")
        T = pd.DataFrame({"prompt_id": P.prompt_id, "input_ids": ids, "n_input_tokens": n,
                          "rendered_sha256": [sha(json.dumps(x)) for x in ids]})
        T.to_parquet(out / f"tokenized_{key}.parquet")
        W = pd.DataFrame([{"prompt_id": p, "sample_index": s, "seed": derive_seed("pilot", round_, key, p, s) % (1 << 31),
                           "chunk": 0} for p in P.prompt_id for s in range(N_SAMPLES)])
        W.to_parquet(out / f"worklist_{key}.parquet")
        extra = []
        if key.startswith("gemma"):   # Nicky 2026-11-27: stop Gemma at <end_of_turn> in the pilot (K10 did not)
            eot = tok.convert_tokens_to_ids(EOT)
            assert eot == 107, eot
            extra = [int(eot)]
        summ["models"][key] = {"hf_id": hf, "revision": rev, "chat_template_kwargs": {}, "n_samples": N_SAMPLES,
                               "extra_stop_token_ids": extra, "extra_stop_tokens": [EOT] if extra else [],
                               "decoding": {"temperature": 0.6, "top_p": 0.9, "top_k": 0, "min_p": 0.0,
                                            "max_new_tokens": MAX_NEW},   # = K10 models.yaml
                               "n_responses": len(W), "max_input_tokens": max(n),
                               "mean_input_tokens": round(float(np.mean(n)), 1),
                               "worklist_sha256": hashlib.sha256((out / f"worklist_{key}.parquet").read_bytes()).hexdigest(),
                               "tokenized_sha256": hashlib.sha256((out / f"tokenized_{key}.parquet").read_bytes()).hexdigest()}
    (out / "build_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


def generate_mlx(round_: int, model_key: str, batch: int, limit: int | None) -> None:
    """Mac fallback. Resumable: one JSONL per model; finished (prompt, sample) pairs are skipped."""
    import mlx.core as mx
    from mlx_lm import batch_generate, load
    from mlx_lm import generate as mgen
    from mlx_lm.sample_utils import make_sampler
    inp = PILOT / f"round{round_}" / "inputs"
    W = pd.read_parquet(inp / f"worklist_{model_key}.parquet")
    T = pd.read_parquet(inp / f"tokenized_{model_key}.parquet").set_index("prompt_id")
    out = PILOT / f"round{round_}" / "generations" / f"{model_key}_mlx.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    have = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            have.add((r["prompt_id"], r["sample_index"]))
    todo = [r for r in W.itertuples(index=False) if (r.prompt_id, r.sample_index) not in have][:limit]
    hf, rev = MODELS[model_key]
    path = next((Path.home() / ".cache/huggingface/hub" / f"models--{hf.replace('/', '--')}" / "snapshots").glob(rev))
    model, tok = load(str(path))
    t0, n_tok = time.time(), 0
    with out.open("a") as fh:
        for b in range(0, len(todo), batch):
            chunk = todo[b:b + batch]
            sampler = make_sampler(temp=DECODING["temperature"], top_p=DECODING["top_p"])
            prompts = [[int(x) for x in T.loc[r.prompt_id, "input_ids"]] for r in chunk]
            if model_key.startswith("gemma"):
                # mlx-lm 0.32 batch_generate fails on Gemma 2 attention (broadcast error): one at a time, own seed
                texts = []
                for r, ids in zip(chunk, prompts):
                    mx.random.seed(int(r.seed))
                    texts.append(mgen(model, tok, ids, max_tokens=MAX_NEW, sampler=sampler))
            else:
                mx.random.seed(int(chunk[0].seed))   # one RNG stream per batch; per-sample seeds are recorded
                texts = batch_generate(model, tok, prompts, max_tokens=MAX_NEW, sampler=sampler).texts
            for r, text in zip(chunk, texts):
                n_tok += len(tok.encode(text))
                fh.write(json.dumps({"prompt_id": r.prompt_id, "sample_index": int(r.sample_index), "seed": int(r.seed),
                                     "model_key": model_key, "backend": "mlx", "response_text": text}) + "\n")
            fh.flush()
            el = time.time() - t0
            print(f"{model_key} {b + len(chunk)}/{len(todo)} {el:.0f}s {(b + len(chunk)) / el:.2f} resp/s "
                  f"{n_tok / el:.0f} tok/s", flush=True)


def verify(round_: int) -> pd.DataFrame:
    base = PILOT / f"round{round_}"
    P = pd.read_parquet(base / "inputs" / "prompts.parquet").set_index("prompt_id")
    rows = []
    for f in sorted((base / "generations").glob("*.jsonl")):
        for line in f.read_text().splitlines():
            g = json.loads(line)
            p = P.loc[g["prompt_id"]]
            text = g["response_text"].split(EOT, 1)[0]
            r = C.check(p.type_id, text, json.loads(p.kwargs_json))
            rows.append({"prompt_id": g["prompt_id"], "model_key": g["model_key"], "sample_index": g["sample_index"],
                         "task_id": p.task_id, "type_id": p.type_id, "level": p.level, "passed": r.passed,
                         "detail": r.detail, "eot_cut": EOT in g["response_text"]})
    V = pd.DataFrame(rows)
    V.to_parquet(base / "verification.parquet")
    return V


def analyze(round_: int, n_boot: int = 2000) -> pd.DataFrame:
    """Criteria frozen in the plan: order E<M<H pooled and in >= 2 of 3 models; each adjacent gap >= 10 pp with its
    90% bootstrap interval (over the 48 tasks) excluding 0; easy >= 3% violations, hard <= 90%."""
    base = PILOT / f"round{round_}"
    V = pd.read_parquet(base / "verification.parquet")
    V["viol"] = ~V.passed
    rng = np.random.default_rng(derive_seed("pilot-boot", round_) % (1 << 32))
    out = []
    for t, g in V.groupby("type_id"):
        tasks = sorted(g.task_id.unique())
        per = g.groupby(["task_id", "level"]).viol.mean().unstack()   # pooled over models and samples
        rate = {lvl: float(g[g.level == lvl].viol.mean()) for lvl in K.LEVELS}
        by_model = {m: {lvl: float(gm[gm.level == lvl].viol.mean()) for lvl in K.LEVELS} for m, gm in g.groupby("model_key")}
        boots = []
        for _ in range(n_boot):
            s = per.loc[rng.choice(tasks, len(tasks))].mean()
            boots.append((s["medium"] - s["easy"], s["hard"] - s["medium"]))
        b = np.array(boots)
        lo, hi = np.percentile(b, 5, axis=0), np.percentile(b, 95, axis=0)
        gap = (rate["medium"] - rate["easy"], rate["hard"] - rate["medium"])
        order_models = sum(v["easy"] < v["medium"] < v["hard"] for v in by_model.values())
        crit = {"order_pooled": rate["easy"] < rate["medium"] < rate["hard"], "order_models": order_models >= 2,
                "gap_em": gap[0] >= 0.10 and lo[0] > 0, "gap_mh": gap[1] >= 0.10 and lo[1] > 0,
                "easy_floor": rate["easy"] >= 0.03, "hard_ceiling": rate["hard"] <= 0.90}
        out.append({"type_id": t, **{f"viol_{k}": round(v, 4) for k, v in rate.items()},
                    "gap_em": round(gap[0], 4), "gap_em_ci90": [round(lo[0], 4), round(hi[0], 4)],
                    "gap_mh": round(gap[1], 4), "gap_mh_ci90": [round(lo[1], 4), round(hi[1], 4)],
                    "models_ordered": order_models, "by_model": by_model, **crit, "passes": all(crit.values())})
    A = pd.DataFrame(out)
    A.to_json(base / "calibration_table.json", orient="records", indent=1)
    return A


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["build", "generate", "verify", "analyze"])
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--seeds", nargs="+", default=None, help="default: the current calibration seeds")
    ap.add_argument("--model", choices=list(MODELS))
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    if a.step == "build":
        build(a.round, a.seeds or calibration_seeds())
    elif a.step == "generate":
        generate_mlx(a.round, a.model, a.batch, a.limit)
    elif a.step == "verify":
        print(verify(a.round).groupby(["model_key", "type_id", "level"]).passed.mean())
    else:
        print(analyze(a.round).to_string())


if __name__ == "__main__":
    main()
