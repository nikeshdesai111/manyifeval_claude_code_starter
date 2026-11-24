"""Build the figures and generated tables for the NeurIPS-format draft report.

Reads only design metadata (split S3 assignment and pair coverage) and the TRAINING-CV tables
already committed under results/v2_s3/tables/. No TEST-group outcome, feature or prediction is read.

Run from the repository root:
    python report/neurips_draft/make_assets.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
GEN = OUT / "generated"

# Short display names for the 15 ManyIFEval/IFEval instruction types, grouped by category.
TYPES = [
    ("keywords:existence", "Include keywords"),
    ("keywords:frequency", "Keyword count"),
    ("keywords:forbidden_words", "Forbidden words"),
    ("keywords:letter_frequency", "Letter count"),
    ("length_constraints:number_words", "Word count"),
    ("length_constraints:number_sentences", "Sentence count"),
    ("length_constraints:number_paragraphs", "Paragraph count"),
    ("detectable_content:number_placeholders", "Placeholders"),
    ("detectable_format:number_bullet_lists", "Bullet points"),
    ("detectable_format:title", "Title"),
    ("change_case:english_capital", "All uppercase"),
    ("change_case:english_lowercase", "All lowercase"),
    ("change_case:capital_word_frequency", "Uppercase-word count"),
    ("startend:quotation", "Quotation marks"),
    ("punctuation:no_comma", "No commas"),
]
MUTEX = [
    {"keywords:existence", "keywords:frequency"},
    {"change_case:capital_word_frequency", "change_case:english_capital", "change_case:english_lowercase"},
    {"length_constraints:number_words", "length_constraints:number_sentences", "length_constraints:number_paragraphs"},
]


def pair_heatmap() -> None:
    idx = {t: i for i, (t, _) in enumerate(TYPES)}
    n = len(TYPES)
    test = np.full((n, n), np.nan)
    train = np.full((n, n), np.nan)
    with open(ROOT / "results/split_s3/pair_coverage.csv") as f:
        for row in csv.DictReader(f):
            a, b = idx[row["type_a"]], idx[row["type_b"]]
            i, j = min(a, b), max(a, b)
            test[i, j] = int(row["test_groups_s3"])  # upper triangle: TEST groups
            train[j, i] = int(row["train_groups_s3"])  # lower triangle: TRAIN groups
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.set_facecolor("#ffffff")
    cmap_test = plt.get_cmap("Blues")
    cmap_train = plt.get_cmap("Oranges")
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, color="#f0f0f0"))
                continue
            a, b = TYPES[i][0], TYPES[j][0]
            if any(a in g and b in g for g in MUTEX):
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, color="#d9d9d9", hatch="////", fill=False))
                continue
            v = test[i, j] if j > i else train[i, j]
            if np.isnan(v):
                continue
            cmap, vmax = (cmap_test, 8) if j > i else (cmap_train, 30)
            c = cmap(0.15 + 0.75 * min(v, vmax) / vmax)
            ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, color=c))
            ax.text(j, i, f"{int(v)}", ha="center", va="center", fontsize=6.5,
                    color="white" if (v / vmax) > 0.55 else "black",
                    fontweight="bold" if (v == 0 or (j > i and v == 2)) else "normal")
    ax.set_xlim(-.5, n - .5)
    ax.set_ylim(n - .5, -.5)
    names = [s for _, s in TYPES]
    ax.set_xticks(range(n), names, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(n), names, fontsize=7)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Upper triangle: TEST groups containing the pair (22 total)\n"
                 "Lower triangle: TRAIN groups containing the pair (50 total)", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "pair_coverage_s3.pdf")
    plt.close(fig)


def type_table() -> None:
    d = json.load(open(ROOT / "results/split_s3/proposed_split.json"))
    params = {
        "keywords:existence": "2 keywords",
        "keywords:frequency": r"1 keyword, $N\in[1,3]$, rel.",
        "keywords:forbidden_words": "2 keywords",
        "keywords:letter_frequency": r"letter, $N\in[1,10]$, rel.",
        "length_constraints:number_words": r"$N\in[100,500]$, rel.",
        "length_constraints:number_sentences": r"$N\in[1,20]$, rel.",
        "length_constraints:number_paragraphs": r"$N\in[1,5]$",
        "detectable_content:number_placeholders": r"$N\in[1,4]$",
        "detectable_format:number_bullet_lists": r"$N\in[1,5]$",
        "detectable_format:title": "none",
        "change_case:english_capital": "none",
        "change_case:english_lowercase": "none",
        "change_case:capital_word_frequency": r"$N\in[1,20]$, rel.",
        "startend:quotation": "none",
        "punctuation:no_comma": "none",
    }
    lines = ["% generated by report/neurips_draft/make_assets.py -- do not edit",
             r"\begin{tabular}{lllcc}", r"\toprule",
             r"Rule type & IFEval identifier & Sampled parameters & Train groups & Test groups \\", r"\midrule"]
    for t, name in TYPES:
        ident = t.replace("_", r"\_")
        lines.append(f"{name} & \\texttt{{\\scriptsize {ident}}} & {params[t]} & "
                     f"{d['type_train_groups'][t]} & {d['type_test_groups'][t]} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "tab_types.tex").write_text("\n".join(lines) + "\n")


LABEL = {"B1": r"B1 (S\,+\,P)", "B2": r"B2 (S\,+\,P\,+\,R)", "H1": r"H1\,+\,S", "H2": r"H2\,+\,S",
         "RCAP": r"RCAP (A\,+\,R\,+\,S)", "B1+H1": r"B1\,+\,H1", "B2+H1": r"B2\,+\,H1", "B2+A1": r"B2\,+\,A1",
         "REF_structure": "Structure only (S)", "XH1": r"Other generator's H1\,+\,S"}
ORDER = ["REF_structure", "B1", "B2", "H1", "H2", "RCAP", "B1+H1", "B2+H1", "B2+A1", "XH1"]


def cv_table() -> None:
    rows = {}
    with open(ROOT / "results/v2_s3/tables/selected_settings.csv") as f:
        for r in csv.DictReader(f):
            rows[(r["model"], r["objective"], r["predictor"])] = r
    lines = ["% generated by report/neurips_draft/make_assets.py from results/v2_s3/tables/selected_settings.csv",
             r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
             r" & \multicolumn{4}{c}{Obj.~1: log loss} & \multicolumn{4}{c}{Obj.~2: MSE ($\times 10^{3}$)} \\",
             r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
             r"Predictor & \multicolumn{2}{c}{Llama} & \multicolumn{2}{c}{Qwen} & \multicolumn{2}{c}{Llama} & \multicolumn{2}{c}{Qwen} \\",
             r" & loss & dim. & loss & dim. & MSE & dim. & MSE & dim. \\", r"\midrule"]
    for p in ORDER:
        cells = []
        for o in ("1", "2"):
            for m in ("llama", "qwen"):
                r = rows[(m, o, p)]
                v = float(r["cv_loss"]) * (1 if o == "1" else 1e3)
                cells += [f"{v:.4f}" if o == "1" else f"{v:.3f}", r["n_features"]]
        lines.append(LABEL[p] + " & " + " & ".join(cells) + r" \\")
        if p in ("REF_structure", "RCAP"):
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "tab_cv_selected.tex").write_text("\n".join(lines) + "\n")

    sel = []
    with open(ROOT / "results/v2_s3/tables/layer_selection.csv") as f:
        for r in csv.DictReader(f):
            if r["selected"] == "True" and r["which"].startswith("own"):
                sel.append(r)
    out = {f"{r['model']}_obj{r['objective']}": {"row": int(r["row"]), "depth": float(r["depth"])} for r in sel}
    (GEN / "selected_layers.json").write_text(json.dumps(out, indent=1) + "\n")


def eval2_placeholder() -> None:
    """Empty axes at the final size of main Figure 2 (Evaluation 2 observed effects). No data are drawn."""
    names = [n for _, n in TYPES]
    fixed = {"Title", "All uppercase", "All lowercase", "Quotation marks", "No commas"}
    order = [n for n in names if n in fixed] + [n for n in names if n not in fixed]
    fig = plt.figure(figsize=(5.5, 2.35))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.95], wspace=0.12)
    titles = ["(a) Position", "(b) Companion", "(c) Task"]
    for i, t in enumerate(titles):
        ax = fig.add_subplot(gs[0, i])
        ax.set_ylim(-0.6, len(order) - 0.4)
        ax.set_xlim(0, 1)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order if i == 0 else [], fontsize=5)
        ax.invert_yaxis()
        ax.axhline(4.5, color="0.6", lw=0.5, ls=":")
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "", ""], fontsize=5)
        ax.set_title(t + "\nnoise-adj. RMS (pp)", fontsize=5.5, pad=2)
        ax.text(0.5, 7, "pending", ha="center", va="center", fontsize=6, color="#a33", rotation=90)
        ax.tick_params(length=1.5, width=0.4, pad=1)
        for sp in ax.spines.values():
            sp.set_linewidth(0.4)
    ax = fig.add_subplot(gs[0, 3])
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-1, 1)
    ax.axhline(0, color="0.5", lw=0.5)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["1\u21922", "2\u21923", "3\u21924", "4\u21925"], fontsize=5)
    ax.set_yticks([])
    ax.set_title("(d) Strict append\nretained-rule change (pp)", fontsize=5.5, pad=2)
    ax.text(1.5, 0.5, "pending", ha="center", va="center", fontsize=6, color="#a33")
    ax.tick_params(length=1.5, width=0.4, pad=1)
    for sp in ax.spines.values():
        sp.set_linewidth(0.4)
    fig.savefig(FIG / "eval2_placeholder.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    GEN.mkdir(exist_ok=True)
    pair_heatmap()
    type_table()
    cv_table()
    eval2_placeholder()
    print("ok")
