"""K10 versions of the paper's design exhibits (Tables 1, 6, 7, 27 and Figure 6), from design metadata and the K10
training-CV selections only. Counting TEST prompts and pool contents reads manifest metadata, never labels, responses
or features. Writes results/k10_paper/paper/k10_design_*.tex and figures_k10/pair_coverage_k10.pdf.
Run: PYTHONPATH=src .venv/bin/python -m mife.k10paper.ev.design_k10
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ... import splits  # noqa: E402
from ..paths import INP, OUT  # noqa: E402

PAPER = OUT / "paper"
FIG = PAPER / "figures_k10"
GL = {"llama": "Llama", "gemma2b": "Gemma-2B"}
TYPES = [  # report/neurips_draft/make_assets.py order and names
    ("keywords:existence", "Include keywords"), ("keywords:frequency", "Keyword count"),
    ("keywords:forbidden_words", "Forbidden words"), ("keywords:letter_frequency", "Letter count"),
    ("length_constraints:number_words", "Word count"), ("length_constraints:number_sentences", "Sentence count"),
    ("length_constraints:number_paragraphs", "Paragraph count"), ("detectable_content:number_placeholders", "Placeholders"),
    ("detectable_format:number_bullet_lists", "Bullet points"), ("detectable_format:title", "Title"),
    ("change_case:english_capital", "All uppercase"), ("change_case:english_lowercase", "All lowercase"),
    ("change_case:capital_word_frequency", "Uppercase-word count"), ("startend:quotation", "Quotation marks"),
    ("punctuation:no_comma", "No commas")]
MUTEX = [{"keywords:existence", "keywords:frequency"},
         {"change_case:capital_word_frequency", "change_case:english_capital", "change_case:english_lowercase"},
         {"length_constraints:number_words", "length_constraints:number_sentences", "length_constraints:number_paragraphs"}]


def _meta():
    P = pd.read_parquet(INP / "prompts.parquet", columns=["prompt_id", "group_id", "split", "task_key", "k", "subset_id",
                                                           "is_original"])
    assert set(P[P.split == "test"].group_id) == splits.test_groups()
    n_rules = P.k  # one prompt-rule row per rule in the prompt (metadata)
    return P.assign(rows=n_rules)


def counts_table() -> str:
    P = _meta()
    tr, te = P[P.split == "train"], P[P.split == "test"]
    ttr, tte = tr.task_key.nunique(), te.task_key.nunique()
    L = [r"\begin{tabular}{lrrrr}", r"\toprule",
         r" & Train (50 groups) & Test (22 groups, sealed) & Train per task & Test per task \\", r"\midrule",
         f"Tasks & {ttr} & {tte} & -- & -- \\\\",
         f"Prompts (ordered configurations) & {len(tr):,} & {len(te):,} & {len(tr) // ttr:,} & {len(te) // tte:,} \\\\",
         f"Prompt--rule rows & {tr.rows.sum():,} & {te.rows.sum():,} & {tr.rows.sum() // ttr:,} & {te.rows.sum() // tte:,} \\\\",
         f"Verified responses & {5 * len(tr):,} & {5 * len(te):,} & {5 * len(tr) // ttr:,} & {5 * len(te) // tte:,} \\\\",
         r"\bottomrule", r"\end{tabular}"]
    return wrap_table("K10 data per generator under the adopted split ($k=1..10$, 5 responses per prompt). The test set "
                      "contains every rule subset of its 22 pools (1,023 per pool; extra subsets with $2k$ Williams orders) "
                      "and stays sealed: every K10 result in this document uses the training groups only.",
                      "tab:counts-k10", "\n".join(L), size=r"\small")


def perk_table() -> str:
    P = _meta()
    tr = P[(P.split == "train")]
    te = P[(P.split == "test")]
    t0, s0 = tr.task_key.iloc[0], te.task_key.iloc[0]
    a = tr[tr.task_key == t0]
    b = te[te.task_key == s0]
    L = [r"\begin{tabular}{crrrrrr}", r"\toprule",
         r"$k$ & Subsets & Orders/subset & Prompts & Prompt--rule rows & Expanded subsets & Expanded prompts \\", r"\midrule"]
    tot = [0, 0, 0, 0, 0]
    for k in range(1, 11):
        x, y = a[a.k == k], b[b.k == k]
        ns, ne = x.subset_id.nunique(), y.subset_id.nunique()
        L.append(f"{k} & {ns} & {len(x) // ns} & {len(x):,} & {x.rows.sum():,} & {ne:,} & {len(y):,} \\\\")
        tot = [tot[0] + ns, tot[1] + len(x), tot[2] + x.rows.sum(), tot[3] + ne, tot[4] + len(y)]
    L += [r"\midrule", f"Total & {tot[0]} & & {tot[1]:,} & {tot[2]:,} & {tot[3]:,} & {tot[4]:,} \\\\", r"\bottomrule",
          r"\end{tabular}"]
    return wrap_table("K10 counts per task by $k$ in the training design (10 chains plus the full pool) and in the test "
                      "expansion (every subset; chain subsets keep their COA orders, added subsets get $2k$ Williams orders). "
                      "Orders: exact COAs except near-COAs at $k=6$ and $k=10$ (PRESPEC deviation).",
                      "tab:perk-k10", "\n".join(L), size=r"\footnotesize")


def pools():
    out = {}
    with open(INP / "pools.jsonl") as f:
        for line in f:
            d = json.loads(line)
            out[d["group_id"]] = {r["type_id"] for r in d["rules"]}
    return out


def pair_figure() -> str:
    pl = pools()
    test = splits.test_groups()
    idx = {t: i for i, (t, _) in enumerate(TYPES)}
    n = len(TYPES)
    M = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a, b = TYPES[i][0], TYPES[j][0]
            gs = [g for g, s in pl.items() if a in s and b in s]
            M[i, j] = sum(g in test for g in gs) if j > i else sum(g not in test for g in gs)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, color="#f0f0f0"))
                continue
            a, b = TYPES[i][0], TYPES[j][0]
            if any(a in g and b in g for g in MUTEX):
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, color="#d9d9d9", hatch="////", fill=False))
                continue
            v = M[i, j]
            cmap, vmax = (plt.get_cmap("Blues"), 22) if j > i else (plt.get_cmap("Oranges"), 50)
            ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, color=cmap(0.15 + 0.75 * min(v, vmax) / vmax)))
            ax.text(j, i, f"{int(v)}", ha="center", va="center", fontsize=6.5, color="white" if v / vmax > 0.55 else "black")
    ax.set_xlim(-.5, n - .5)
    ax.set_ylim(n - .5, -.5)
    names = [s for _, s in TYPES]
    ax.set_xticks(range(n), names, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(n), names, fontsize=7)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("K10 pools. Upper triangle: TEST groups containing the pair (22 total)\n"
                 "Lower triangle: TRAIN groups containing the pair (50 total)", fontsize=8)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "pair_coverage_k10.pdf")
    plt.close(fig)
    return (r"\begin{figure}[H]\centering\includegraphics[width=0.8\linewidth]{figures_k10/pair_coverage_k10.pdf}"
            r"\caption{K10 pair coverage of the adopted split: the number of test groups (upper triangle) and training "
            r"groups (lower triangle) whose 10-rule pool contains both types. Hatched cells are type pairs that ManyIFEval "
            r"never combines; with 10 of the 10 compatible types per pool, every compatible pair that the pool's exclusive-set "
            r"choices allow is present. Derived from pool metadata only.}\label{fig:pairs-k10}\end{figure}")


def types_table() -> str:
    pl = pools()
    test = splits.test_groups()
    L = [r"\begin{tabular}{lcccc}", r"\toprule", r"Rule type & Train groups (orig.) & Test groups (orig.) & Train groups (K10) & Test groups (K10) \\",
         r"\midrule"]
    orig = json.load(open(splits.ROOT / "results/split_s3/proposed_split.json")) if hasattr(splits, "ROOT") else None
    for t, name in TYPES:
        k_tr = sum(t in s and g not in test for g, s in pl.items())
        k_te = sum(t in s and g in test for g, s in pl.items())
        o_tr = orig["type_train_groups"][t] if orig else "--"
        o_te = orig["type_test_groups"][t] if orig else "--"
        L.append(f"{name} & {o_tr} & {o_te} & {k_tr} & {k_te} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return wrap_table("K10 rule types: the 15 types and their parameter generators are unchanged (Table~\\ref{tab:types}); "
                      "each K10 pool keeps its 6 frozen rules and adds 4 (the missing universal types plus one member of each "
                      "exclusive set it does not use), so every pool holds 10 types. Groups whose pool contains each type, "
                      "original 6-rule pools vs K10 10-rule pools.", "tab:types-k10", "\n".join(L), size=r"\footnotesize")


def cv_table() -> str:
    from ..cv import RESULTS_V2
    lab = {"REF_structure": "Structure only (S)", "B1": r"B1 (S\,+\,P)", "B2": r"B2 (S\,+\,P\,+\,R)", "H1": r"H1\,+\,S",
           "H2": r"H2\,+\,S", "RCAP": r"A2 (S\,+\,A\,+\,R)", "B1+H1": r"B1\,+\,H", "B2+H1": r"B2\,+\,H",
           "B2+A1": r"B2\,+\,A", "B2+H1+A1": r"B2\,+\,H\,+\,A", "XH1": r"Other generator's H1\,+\,S"}
    order = ["REF_structure", "B1", "B2", "H1", "H2", "RCAP", "B1+H1", "B2+H1", "B2+A1", "B2+H1+A1", "XH1"]
    gens = [g for g in ("llama", "gemma2b") if (RESULTS_V2 / "cv" / g / "obj1" / "selection.json").exists()]
    if not gens:
        raise FileNotFoundError("no K10 selection yet")
    sel = {}
    for g in gens:
        d = RESULTS_V2 / "cv" / g / "obj1"
        s = json.loads((d / "selection.json").read_text())["selected"]
        if (d / "selection_extra.json").exists():
            e = json.loads((d / "selection_extra.json").read_text())
            s["B2+H1+A1"] = {"cv_loss": e["cv_loss"], "n_features": None}
        if (d / "selection_xh1.json").exists():
            s["XH1"] = json.loads((d / "selection_xh1.json").read_text())
        sel[g] = s
    L = [r"\begin{tabular}{l" + "rr" * len(gens) + "}", r"\toprule",
         " & " + " & ".join(r"\multicolumn{2}{c}{" + GL[g] + "}" for g in gens) + r" \\",
         "Predictor & " + " & ".join(["log loss & dim."] * len(gens)) + r" \\", r"\midrule"]
    for i, p in enumerate(order):
        if not any(p in sel[g] for g in gens):
            continue
        if p in ("B1", "B1+H1"):
            L.append(r"\midrule")
        cells = []
        for g in gens:
            x = sel[g].get(p)
            cells += [f"{x['cv_loss']:.4f}" if x else "--", f"{x['n_features']}" if x and x.get("n_features") else "--"]
        L.append(lab[p] + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    rows = "; ".join(f"{GL[g]} row {json.loads((RESULTS_V2 / 'cv' / g / 'obj1' / 'selection.json').read_text())['hidden_row']}"
                     for g in gens)
    return wrap_table("K10 training cross-validation log loss per completion at the selected setting (5 grouped folds over "
                      f"the 50 training groups) and the number of input features. Selected hidden rows: {rows}. From "
                      r"\texttt{results/k10\_paper/v2/cv/}.", "tab:cv-k10", "\n".join(L), size=r"\footnotesize")


def wrap_table(cap, label, body, size=r"\small"):
    return "\n".join([r"\begin{table}[H]", r"\centering" + size, r"\caption{" + cap + "}", r"\label{" + label + "}", body,
                      r"\end{table}", ""])


def run():
    PAPER.mkdir(parents=True, exist_ok=True)
    (PAPER / "k10_design_counts.tex").write_text(counts_table())
    (PAPER / "k10_design_perk.tex").write_text(perk_table())
    (PAPER / "k10_design_pairs.tex").write_text(pair_figure())
    (PAPER / "k10_design_types.tex").write_text(types_table())
    try:
        (PAPER / "k10_design_cv.tex").write_text(cv_table())
    except FileNotFoundError as e:
        (PAPER / "k10_design_cv.tex").unlink(missing_ok=True)
        print("cv table not ready:", e)
    print("ok", PAPER)


if __name__ == "__main__":
    run()
