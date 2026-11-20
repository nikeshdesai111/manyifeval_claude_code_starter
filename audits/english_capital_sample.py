"""100 english_capital responses that the official checker failed only at the langdetect step (TRAIN only).

Draws a random sample (seed 0) of 50 per generator from results/audits/english_capital_langdetect_only.parquet,
re-reads each response byte-for-byte from the generation shards of run `main`, and writes
results/audits/english_capital_langdetect_only_100.html (readable) and .csv (same rows).
Run: PYTHONPATH=src .venv/bin/python audits/english_capital_sample.py
"""
from __future__ import annotations

import html
import json

import pandas as pd

from mife.config import DATA_DIR
from mife.probing import data
from mife.probing.data import RESULTS

A = RESULTS / "audits"
N_PER_MODEL, SEED = 50, 0


def exact_texts(model: str, prompt_ids: list[str], response_ids: set[str]) -> pd.DataFrame:
    gdir = DATA_DIR / "generations" / model / data.RUN
    g = pd.concat([data._read_filtered(p, prompt_ids, ["response_id", "prompt_id", "sample_index", "response_text",
                                                        "n_output_tokens", "finish_reason"])
                   for p in sorted(gdir.glob("shard_*.parquet"))], ignore_index=True)
    return g[g.response_id.isin(response_ids)]


def build() -> dict:
    flagged = pd.read_parquet(A / "english_capital_langdetect_only.parquet").drop(columns=["response_text"])
    S = pd.concat([flagged[flagged.model == m].sample(N_PER_MODEL, random_state=SEED) for m in sorted(set(flagged.model))],
                  ignore_index=True)
    texts = pd.concat([exact_texts(m, sorted(set(d.prompt_id)), set(d.response_id)) for m, d in S.groupby("model")],
                      ignore_index=True)
    S = S.merge(texts.drop(columns=["prompt_id"]), on="response_id", how="left", validate="one_to_one")
    assert S.response_text.notna().all() and len(S) == 2 * N_PER_MODEL
    P = data.prompts("train").set_index("prompt_id")
    R = data.prompt_rules("train")
    S["prompt_text"] = S.prompt_id.map(P.prompt_text)
    R = R[R.prompt_id.isin(set(S.prompt_id))].sort_values(["prompt_id", "position"])
    rules = {pid: [f"{r.position}. {r.type_id} — {r.rule_text}" for r in d.itertuples()] for pid, d in R.groupby("prompt_id")}
    S["rules"] = S.prompt_id.map(rules)
    S = S.sort_values(["model", "response_id"]).reset_index(drop=True)
    S.insert(0, "n", range(1, len(S) + 1))
    cols = ["n", "model", "response_id", "prompt_id", "task_family", "k", "position", "diag_langdetect", "langdetect_lower",
            "top_langs", "assessment", "loop", "hit_max_tokens", "diag_n_words", "prompt_text", "response_text"]
    S.assign(rules=S.rules.map(" | ".join))[cols + ["rules"]].to_csv(A / "english_capital_langdetect_only_100.csv", index=False)
    (A / "english_capital_langdetect_only_100.html").write_text(_html(S, len(flagged), flagged.model.value_counts().to_dict()))
    return {"rows": len(S)}


def _html(S: pd.DataFrame, n_total: int, per_model: dict) -> str:
    e = html.escape
    cards = []
    for r in S.itertuples():
        rules = "".join(f"<li>{e(x)}</li>" for x in r.rules)
        tags = [f"langdetect: <b>{e(str(r.diag_langdetect))}</b>", f"lower-cased: <b>{e(str(r.langdetect_lower))}</b>",
                f"probabilities: {e(str(r.top_langs))}", f"{int(r.diag_n_words)} words"]
        if r.loop:
            tags.append("<span class='warn'>repetition loop</span>")
        if r.hit_max_tokens:
            tags.append("<span class='warn'>hit max tokens</span>")
        cards.append(f"""
<article class="card" data-model="{e(r.model)}">
  <header><span class="num">#{r.n}</span><span class="model {e(r.model)}">{e(r.model)}</span>
    <code>{e(r.response_id)}</code><span class="meta">{e(str(r.task_family))} · k={r.k} · english_capital at position {r.position}</span></header>
  <div class="tags">{" · ".join(tags)}</div>
  <details><summary>Prompt and rules</summary><ol class="rules">{rules}</ol><pre class="prompt">{e(r.prompt_text)}</pre></details>
  <pre class="resp">{e(r.response_text)}</pre>
</article>""")
    pm = ", ".join(f"{k} {v:,}" for k, v in sorted(per_model.items()))
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Langdetect-flagged responses</title>
<style>
:root{{--bg:#fafaf9;--fg:#1c1917;--muted:#57534e;--card:#fff;--line:#e7e5e4;--pre:#f5f5f4;--q:#7c3aed;--l:#0369a1;--warn:#b45309}}
@media (prefers-color-scheme:dark){{:root{{--bg:#1c1917;--fg:#f5f5f4;--muted:#a8a29e;--card:#292524;--line:#44403c;--pre:#231f1d;--q:#c4b5fd;--l:#7dd3fc;--warn:#fbbf24}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,system-ui,sans-serif}}
main{{max-width:980px;margin:0 auto;padding:24px 16px 64px}}
h1{{font-size:22px;margin:0 0 6px}} p.lead{{color:var(--muted);margin:0 0 16px}}
.bar{{position:sticky;top:0;background:var(--bg);padding:10px 0;border-bottom:1px solid var(--line);margin-bottom:16px;display:flex;gap:8px;flex-wrap:wrap}}
.bar button{{border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:6px;padding:4px 12px;cursor:pointer}}
.bar button.on{{border-color:var(--fg);font-weight:600}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px 16px;margin:0 0 14px}}
.card header{{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}} .num{{font-weight:700}}
.model{{font-weight:600;text-transform:uppercase;font-size:12px;letter-spacing:.04em}} .model.qwen{{color:var(--q)}} .model.llama{{color:var(--l)}}
code{{font-size:12px;color:var(--muted)}} .meta,.tags{{color:var(--muted);font-size:13px}} .tags{{margin:6px 0}} .warn{{color:var(--warn);font-weight:600}}
details{{margin:6px 0}} summary{{cursor:pointer;color:var(--muted);font-size:13px}} ol.rules{{font-size:13px;margin:6px 0;padding-left:20px}}
pre{{white-space:pre-wrap;word-break:break-word;background:var(--pre);border-radius:6px;padding:10px 12px;margin:6px 0 0;font:13px/1.45 ui-monospace,Menlo,monospace;max-height:420px;overflow:auto}}
pre.prompt{{max-height:260px}}
</style></head><body><main>
<h1>english_capital responses flagged only by langdetect</h1>
<p class="lead">TRAIN only. Each response is all uppercase (it passed <code>isupper()</code>) but the official checker failed it because
<code>langdetect.detect()</code> did not return "en". Population: {n_total:,} responses ({pm}). Shown: a random sample of
{N_PER_MODEL} per generator (seed {SEED}), each text re-read verbatim from the generation shards. Long responses scroll inside their box.</p>
<div class="bar"><button class="on" data-f="all">All 100</button><button data-f="qwen">Qwen</button><button data-f="llama">Llama</button></div>
{"".join(cards)}
</main><script>
document.querySelectorAll('.bar button').forEach(b=>b.onclick=()=>{{
 document.querySelectorAll('.bar button').forEach(x=>x.classList.toggle('on',x===b));
 document.querySelectorAll('.card').forEach(c=>c.style.display=(b.dataset.f==='all'||c.dataset.model===b.dataset.f)?'':'none');}});
</script></body></html>"""


if __name__ == "__main__":
    print(json.dumps(build()))
