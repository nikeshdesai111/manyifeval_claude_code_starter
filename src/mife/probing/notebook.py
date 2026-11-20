"""Generate and execute notebooks/results_explorer.ipynb (reads only results/ and data/; no LLM, no training).

Sections whose inputs do not exist yet (for example TEST results before unblinding) print 'pending'. The prompt
drill-down only offers TEST prompts after unblinding."""
from __future__ import annotations

import nbformat as nbf

from ..config import ROOT

NB = ROOT / "notebooks" / "results_explorer.ipynb"

CELLS = [
    ("md", "# ManyIFEval probing results explorer\n\nReads only `results/` and `data/`; loads no LLM and trains nothing. "
           "CV (training-only) and TEST results are labelled separately everywhere. Regenerate with "
           "`python -c \"from mife.probing import notebook; notebook.build()\"`."),
    ("md", "## 1. Setup and integrity check"),
    ("code", """import json, hashlib, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
ROOT = Path.cwd().resolve().parent if Path.cwd().name == 'notebooks' else Path.cwd().resolve()
sys.path.insert(0, str(ROOT / 'src'))
from mife.probing import data, configs as C
R, T, D = ROOT / 'results', ROOT / 'results' / 'tables', ROOT / 'data'
pd.set_option('display.width', 200); pd.set_option('display.max_columns', 30)
def tab(name):
    p = T / f'{name}.csv'
    return pd.read_csv(p) if p.exists() else None
def pending(what):
    print(f'pending: {what}')
man = json.loads((R / 'run_manifest.json').read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
paths = {'manifest_prompts': D/'manifest/prompts.parquet', 'manifest_prompt_rules': D/'manifest/prompt_rules.parquet',
         **{f'{m}_labels_prompt_rule': D/f'verification/{m}/main/labels_prompt_rule.parquet' for m in data.MODELS},
         **{f'{m}_labels_prompt': D/f'verification/{m}/main/labels_prompt.parquet' for m in data.MODELS},
         **{f'{m}_features_index': D/f'features/{m}/main/index.parquet' for m in data.MODELS}}
check = pd.DataFrame([{'file': k, 'matches_run_manifest': sha(p) == man['data_sha256'][k]} for k, p in paths.items()])
display(check)
assert check.matches_run_manifest.all(), 'data files differ from results/run_manifest.json'
print('test set unblinded:', data.test_unblinded())"""),
    ("md", "## 2. Dataset descriptives (TRAIN only)\n\nUse the filters to slice the type × k × position rates."),
    ("code", """tk = tab('desc_type_by_k_long'); bp = tab('desc_by_position'); fam = tab('desc_by_family')
display(tab('desc_by_k'))
try:
    import ipywidgets as W
    def show(model, type_id, k):
        d = tk[(tk.model == model)]
        if type_id != 'all': d = d[d.type_id == type_id]
        if k != 'all': d = d[d.k == int(k)]
        display(d[['type_id', 'k', 'est', 'lo', 'hi', 'n_groups']].round(4))
        if k != 'all':
            display(bp[(bp.model == model) & (bp.k == int(k))][['position', 'est', 'lo', 'hi']].round(4))
    W.interact(show, model=['llama', 'qwen'], type_id=['all'] + sorted(tk.type_id.unique()), k=['all', '1', '2', '3', '4', '5'])
except ImportError:
    display(tk.head(30))
display(fam)"""),
    ("code", """for f in ['desc_by_k', 'desc_by_type', 'desc_type_by_k']:
    p = R / 'figures' / f'{f}.png'
    if p.exists():
        from IPython.display import Image; display(Image(str(p)))
for t in ['desc_label_distribution', 'desc_truncation', 'desc_english_capital', 'desc_loops']:
    print(t); display(tab(t))"""),
    ("md", "## 3. Cross-validation (training-only): grid, selected layer and pooling, fold-to-fold variation"),
    ("code", """g = tab('cv_grid_all'); sel = tab('cv_selected')
display(tab('cv_layer_selection')); display(tab('cv_pooling_selection'))
if g is not None:
    for (m, o), d in g.groupby(['model', 'objective']):
        pv = d.pivot_table(index='key', columns='setting', values='cv_loss')
        fig, ax = plt.subplots(figsize=(8, 0.28 * len(pv) + 1.5))
        im = ax.imshow(pv.values, aspect='auto', cmap='viridis_r')
        ax.set_yticks(range(len(pv))); ax.set_yticklabels(pv.index, fontsize=7)
        ax.set_xticks(range(len(pv.columns))); ax.set_xticklabels(pv.columns, rotation=45, ha='right', fontsize=8)
        ax.set_title(f'CV loss (TRAIN folds) — {m}, objective {o}'); fig.colorbar(im, ax=ax); plt.show()
if sel is not None:
    display(sel[['model', 'objective', 'family', 'key', 'setting', 'epoch', 'cv_loss', 'fold_sd', 'minus_ref_structure']])
display(tab('cv_boundary_hits')); display(tab('cv_b2_grid'))"""),
    ("md", "## 4. Evaluation 1 leaderboard (TEST)"),
    ("code", """for o in (1, 2):
    t = tab(f'eval1_obj{o}')
    if t is None: pending(f'TEST evaluation, objective {o}'); continue
    for m, d in t.groupby('model'):
        print(m, 'objective', o); display(d.sort_values(d.columns[3]).reset_index(drop=True))
for name in ['eval1_obj1_by_type', 'eval1_obj1_by_k', 'eval1_obj2_by_k']:
    t = tab(name); display(t) if t is not None else pending(name)
for p in sorted((R / 'figures').glob('reliability_*.png')):
    from IPython.display import Image; display(Image(str(p)))"""),
    ("md", "## 5. Hypothesis contrasts (TEST): forest plots"),
    ("code", """c = tab('contrasts')
if c is None: pending('contrasts')
else:
    for (m, o), d in c.groupby(['model', 'objective']):
        d = d.reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(7, 0.25 * len(d) + 1))
        ax.errorbar(d.est, range(len(d)), xerr=[d.est - d.lo, d.hi - d.est], fmt='o', capsize=2)
        ax.axvline(0, color='grey', lw=1); ax.set_yticks(range(len(d)))
        ax.set_yticklabels([f'{a} → {b}' for a, b in zip(d.a, d.b)], fontsize=7); ax.invert_yaxis()
        ax.set_title(f'{m}, objective {o}: metric(b) − metric(a) (negative favours b)'); plt.show()"""),
    ("md", "## 6. Evaluation 2 (TEST): predicted vs observed differences"),
    ("code", """e = tab('eval2_summary')
if e is None: pending('Evaluation 2')
else:
    display(e)
    for p in sorted((R / 'figures').glob('eval2_*.png')):
        from IPython.display import Image; display(Image(str(p)))"""),
    ("md", "## 7. Error analysis\n\nLargest residuals for a chosen configuration and where configurations disagree most. "
           "Uses TEST predictions after unblinding, otherwise TRAINING out-of-fold CV predictions (labelled)."),
    ("code", """def load_preds(model, obj):
    tdir = R / 'predictions' / model / f'obj{obj}' / 'test'
    if data.test_unblinded() and tdir.exists() and any(tdir.glob('*.parquet')):
        return 'TEST', pd.read_parquet(next(tdir.glob('*.parquet')))
    s = tab('cv_selected'); cols = {}
    if s is None: return None, None
    s = s[(s.model == model) & (s.objective == obj)]
    base = None
    for r in s.itertuples():
        p = R / 'predictions' / model / f'obj{obj}' / 'cv' / f'{r.key}.parquet'
        if not p.exists(): continue
        d = pd.read_parquet(p); key = [c for c in ('prompt_id', 'position', 'rule_id', 'type_id', 'fold') if c in d]
        base = d[key] if base is None else base
        cols[r.config] = d[r.setting].values
    if base is None:
        pending(f'CV predictions for {model} objective {obj}'); return None, None
    return 'CV (TRAIN, out-of-fold)', base.assign(**cols)
src, P1 = load_preds('llama', 1)
if P1 is not None:
    lab = data.labels_prompt_rule('llama', 'train')[['prompt_id', 'position', 'p_hat']]
    P1 = P1.merge(lab, on=['prompt_id', 'position'])
    cfg = [c for c in C.CONFIGS + C.REFERENCES if c in P1]
    focus = 'B1+H1+A1' if 'B1+H1+A1' in P1 else cfg[0]
    P1['residual'] = P1['p_hat'] - P1[focus]
    print(src, '— largest |residual| for', focus); display(P1.reindex(P1.residual.abs().sort_values(ascending=False).index).head(15))
    P1['disagreement'] = P1[cfg].std(axis=1)
    print('largest disagreement across configurations'); display(P1.sort_values('disagreement', ascending=False).head(15)[['prompt_id', 'type_id', 'p_hat', 'disagreement'] + cfg[:6]])"""),
    ("md", "## 8. Prompt drill-down\n\nPick a `prompt_id` (TRAIN prompts; TEST prompts only after unblinding)."),
    ("code", """split = 'all' if data.test_unblinded() else 'train'
Pm = data.prompts(split); PRm = data.prompt_rules(split)
def drill(prompt_id):
    p = Pm.set_index('prompt_id').loc[prompt_id]
    print(f'{prompt_id}  split={p.split}  task_family={p.task_family}  k={p.k}\\n'); print(p.prompt_text, '\\n')
    rules = PRm[PRm.prompt_id == prompt_id][['position', 'rule_id', 'type_id', 'kwargs_json', 'rule_text']]
    for m in data.MODELS:
        L = data.labels_prompt_rule(m, p.split)
        rules = rules.merge(L[L.prompt_id == prompt_id][['position', 'p_hat']].rename(columns={'p_hat': f'p_hat_{m}'}), on='position')
        q = data.labels_prompt(m, p.split); print(m, 'q_hat =', float(q[q.prompt_id == prompt_id].q_hat.iloc[0]))
    display(rules)
    for m in data.MODELS:
        s, Pp = load_preds(m, 1)
        if Pp is not None and prompt_id in set(Pp.prompt_id):
            print(m, s, 'predictions'); display(Pp[Pp.prompt_id == prompt_id].drop(columns=[c for c in ('fold',) if c in Pp]).round(3))
    g = data.generations('llama', p.split, columns=['sample_index', 'response_text']).query('prompt_id == @prompt_id').sort_values('sample_index').head(3)
    V = data.verification('llama', 'verification_long.parquet', p.split, columns=['response_id', 'position', 'violated'])
    for r in g.itertuples():
        v = V[V.response_id == r.response_id].sort_values('position')
        print(f'--- llama sample {r.sample_index}: violated by position', dict(zip(v.position, v.violated)))
        print(r.response_text[:1200])
example = Pm.prompt_id.iloc[1234]
try:
    import ipywidgets as W
    W.interact(drill, prompt_id=W.Text(value=example))
except ImportError:
    drill(example)
drill(example)"""),
    ("md", "## 9. Llama vs Qwen"),
    ("code", """display(tab('desc_cross_model_type')); display(tab('desc_cross_model_corr'))
s = tab('cv_selected')
if s is not None:
    w = s.pivot_table(index=['objective', 'config'], columns='model', values='cv_loss')
    display(w)
e = tab('eval1_obj1')
if e is not None:
    display(e.pivot_table(index='config', columns='model', values=e.columns[3]))
else:
    pending('TEST comparison')"""),
]


def build(execute: bool = True, timeout: int = 1200) -> str:
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src) for kind, src in CELLS]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    NB.parent.mkdir(exist_ok=True)
    if execute:
        from nbclient import NotebookClient
        NotebookClient(nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": str(NB.parent)}}).execute()
    nbf.write(nb, NB)
    return str(NB)
