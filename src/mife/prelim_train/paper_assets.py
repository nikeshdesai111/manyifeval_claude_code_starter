"""PRELIMINARY S3 TRAIN: paper figure (main Figure 2) and LaTeX table bodies, built only from
results/preliminary_train_report/tables/*.csv. Writes results/preliminary_train_report/paper/.
Run: PYTHONPATH=src .venv/bin/python -m mife.prelim_train.paper_assets
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import COMPS
from .matches import FIXED_TYPES
from .paths import EVAL_DIR

T = EVAL_DIR / "tables"
OUT = EVAL_DIR / "paper"
COL = {"llama": "#2a78d6", "qwen": "#eb6834"}
NAMES = {
    "keywords:existence": "Include keywords", "keywords:frequency": "Keyword count",
    "keywords:forbidden_words": "Forbidden words", "length_constraints:number_words": "Word count",
    "length_constraints:number_sentences": "Sentence count", "length_constraints:number_paragraphs": "Paragraph count",
    "detectable_content:number_placeholders": "Placeholders", "detectable_format:number_bullet_lists": "Bullet points",
    "detectable_format:title": "Title", "change_case:english_capital": "All uppercase",
    "change_case:english_lowercase": "All lowercase", "change_case:capital_word_frequency": "Uppercase-word count",
    "startend:quotation": "Quotation marks", "punctuation:no_comma": "No commas"}
SHORT = {"#1": r"1: B1 \cto{} H1", "#2": r"2: B2 \cto{} H2", "#3": r"3: B2 \cto{} A2", "#4": r"4: H2 \cto{} A2",
         "#5": r"5: B1 \cto{} B1+H", "#6": r"6: B2 \cto{} B2+H", "#7": r"7: B2 \cto{} B2+A", "#8": r"8: B2+H \cto{} B2+H+A"}


def r(n):
    return pd.read_csv(T / f"{n}.csv")


def order(types):
    fx = [t for t in types if t in FIXED_TYPES]
    return sorted(fx, key=lambda t: NAMES.get(t, t)) + sorted([t for t in types if t not in FIXED_TYPES],
                                                              key=lambda t: NAMES.get(t, t))


def c(e, lo, hi, s=1.0, d=1):
    f = lambda x: (f"{x * s:.{d}f}" if round(x * s, d) != 0 else f"{0:.{d}f}")
    star = False  # no bolding: intervals are pointwise and unadjusted
    txt = f"{f(e)} [{f(lo)}, {f(hi)}]"
    return r"\textbf{" + txt + "}" if star else txt


def fig2():
    fig = plt.figure(figsize=(5.5, 2.45))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.95], wspace=0.14)
    pos, com, tsk = r("eval2_position_rms_by_type"), r("eval2_companion_rms_by_type"), r("eval2_task_rms_by_type")
    types = order([t for t in pos.type_id.unique() if t in NAMES])
    y = np.arange(len(types))
    xmax = max(x[x.type_id.isin(types)].rms_hi.max() for x in (pos, com, tsk))
    for i, (df, t) in enumerate(((pos, "(a) Position"), (com, "(b) Companion"), (tsk, "(c) Task"))):
        ax = fig.add_subplot(gs[0, i])
        for k, gen in enumerate(("llama", "qwen")):
            s = df[df.generator == gen].set_index("type_id").reindex(types)
            off = -0.17 if k == 0 else 0.17
            ax.errorbar(s.rms_pp, y + off, xerr=[s.rms_pp - s.rms_lo, s.rms_hi - s.rms_pp], fmt="o", ms=1.8,
                        color=COL[gen], elinewidth=0.6, capsize=0, label=("Llama" if k == 0 else "Qwen"))
        ax.set_ylim(-0.6, len(types) - 0.4)
        ax.set_xlim(0, xmax * 1.03)
        ax.set_yticks(y)
        ax.set_yticklabels([NAMES[t] for t in types] if i == 0 else [], fontsize=5)
        ax.invert_yaxis()
        ax.axhline(4.5, color="0.6", lw=0.5, ls=":")
        ax.grid(axis="x", color="0.9", lw=0.4)
        ax.tick_params(length=1.5, width=0.4, pad=1, labelsize=5)
        ax.set_title(t + "\nnoise-adj. RMS (pp)", fontsize=5.5, pad=2)
        for sp in ax.spines.values():
            sp.set_linewidth(0.4)
        if i == 0:
            ax.legend(fontsize=4.5, loc="lower right", handletextpad=0.2, borderpad=0.2, frameon=False)
    ax = fig.add_subplot(gs[0, 3])
    a = r("eval2_append_by_transition")
    trs = ["1->2", "2->3", "3->4", "4->5"]
    for k, gen in enumerate(("llama", "qwen")):
        s = a[a.generator == gen].set_index("transition").reindex(trs)
        x = np.arange(4) + (-0.12 if k == 0 else 0.12)
        ax.errorbar(x, s.effect_pp, yerr=[s.effect_pp - s.lo, s.hi - s.effect_pp], fmt="o", ms=1.8, color=COL[gen],
                    elinewidth=0.6, capsize=0)
    ax.axhline(0, color="0.5", lw=0.5)
    ax.set_xlim(-0.5, 3.5)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["1→2", "2→3", "3→4", "4→5"], fontsize=5)
    ax.yaxis.tick_right()
    ax.tick_params(length=1.5, width=0.4, pad=1, labelsize=5)
    ax.set_title("(d) Strict append\nretained-rule change (pp)", fontsize=5.5, pad=2)
    for sp in ax.spines.values():
        sp.set_linewidth(0.4)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "eval2_prelim_train.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def tab_main():
    o1, o2 = r("eval1_contrasts"), r("obj2_eval1_contrasts")
    lines = []
    for gen, lab in (("llama", "Llama"), ("qwen", "Qwen")):
        lines.append(r"\multirow{7}{*}{\rotatebox{90}{" + lab + "}}")
        for n, (cid, *_rest) in enumerate(COMPS):
            a = o1[(o1.generator == gen) & (o1.comparison == cid)].iloc[0]
            b = o2[(o2.generator == gen) & (o2.comparison == cid)].iloc[0]
            lines.append(" & " + " & ".join([
                SHORT[cid],
                c(a.nll_reduction, a.nll_reduction_lo, a.nll_reduction_hi, 1, 4),
                c(a.auroc_gain, a.auroc_gain_lo, a.auroc_gain_hi, 100),
                f"{int(a.types_improved)}/15",
                c(b.rmse_reduction, b.rmse_reduction_lo, b.rmse_reduction_hi, 100, 2),
                c(b.spearman_gain, b.spearman_gain_lo, b.spearman_gain_hi, 100)]) + r" \\")
            if n == 3:
                lines.append(r"\cmidrule(lr){2-7}")
        lines.append(r"\midrule" if gen == "llama" else "")
    (OUT / "tab_main_rows.tex").write_text("% generated by mife.prelim_train.paper_assets\n" + "\n".join(lines) + "\n")


def tab_eval2():
    sk = {f: r(f"eval2_{f}_skill") for f in ("position", "append", "companion", "task")}
    lines = []
    for gen, lab in (("llama", "Llama"), ("qwen", "Qwen")):
        lines.append(r"\multirow{4}{*}{" + lab + "}")
        for cid in ("#1", "#2", "#3", "#4"):
            cells = []
            for f in ("position", "append", "companion", "task"):
                s = sk[f]
                x = s[(s.generator == gen) & (s.stratum == "macro") & (s.quantity == f"dmse_{cid}")].iloc[0]
                cells.append(c(x.est, x.lo, x.hi, 1e4))
            lines.append(" & " + SHORT[cid] + " & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule" if gen == "llama" else "")
    (OUT / "tab_eval2_rows.tex").write_text("% generated by mife.prelim_train.paper_assets\n" + "\n".join(lines) + "\n")


def tab_app():
    """Appendix: raw Objective 1/2 scores, secondary-comparison Delta-MSE, Objective 2 matched changes."""
    a1 = r("eval1_a1_metrics")
    s2 = r("obj2_eval1_scores")
    lab = {"B1": "B1", "B2": "B2", "H1": "H1+S", "H2": "H2+S", "RCAP": "A2", "B1+H1": "B1+H", "B2+H1": "B2+H",
           "B2+A1": "B2+A", "REF_structure": "Structure only", "REF_trainmean": "Training mean"}
    L = []
    for p in ["REF_structure", "REF_trainmean", "B1", "B2", "H1", "H2", "RCAP", "B1+H1", "B2+H1", "B2+A1", "B2+H1+A1"]:
        row = [lab[p]]
        for gen in ("llama", "qwen"):
            x = a1[(a1.generator == gen) & (a1.predictor == p)]
            if len(x):
                x = x.iloc[0]
                row += [f"{x.nll:.3f}", f"{x.brier:.3f}", f"{100 * x.auroc:.1f}", f"{100 * x.ece:.1f}"]
            else:
                row += ["--"] * 4
            y = s2[(s2.generator == gen) & (s2.predictor == p)].iloc[0]
            row += [f"{100 * y.rmse:.2f}", f"{100 * y.spearman:.1f}"]
        L.append(" & ".join(row) + r" \\")
    row = ["Avg.\\ Obj.~1 H2+S"]
    for gen in ("llama", "qwen"):
        y = s2[(s2.generator == gen) & (s2.predictor == "AVG1_H2")].iloc[0]
        row += ["--"] * 4 + [f"{100 * y.rmse:.2f}", f"{100 * y.spearman:.1f}"]
    L.append(" & ".join(row) + r" \\")
    (OUT / "tab_app_scores_rows.tex").write_text("\n".join(L) + "\n")
    # secondary comparisons Delta-MSE, Obj1
    sk = {f: r(f"eval2_{f}_skill") for f in ("position", "append", "companion", "task")}
    L = []
    for gen, labg in (("llama", "Llama"), ("qwen", "Qwen")):
        for cid in ("#5", "#6", "#7"):
            cells = []
            for f in ("position", "append", "companion", "task"):
                s = sk[f]
                x = s[(s.generator == gen) & (s.stratum == "macro") & (s.quantity == f"dmse_{cid}")].iloc[0]
                cells.append(c(x.est, x.lo, x.hi, 1e4))
            L.append(labg + " & " + SHORT[cid] + " & " + " & ".join(cells) + r" \\")
    (OUT / "tab_app_eval2_secondary_rows.tex").write_text("\n".join(L) + "\n")
    # Objective 2 matched changes
    e = r("obj2_eval2")
    e = e[e.stratum == "macro"]
    L = []
    for gen, labg in (("llama", "Llama"), ("qwen", "Qwen")):
        for q, ql, s_ in (("rms_pp_noise_adjusted", "Observed noise-adj.\\ RMS (pp)", 1),
                          ("dobs", "Observed signed mean (pp)", 100),
                          ("se_zero", r"Zero-change MSE ($\times10^4$)", 1e4),
                          ("dmse_#1", r"$\Delta$-MSE 1 ($\times10^4$)", 1e4), ("dmse_#2", r"$\Delta$-MSE 2 ($\times10^4$)", 1e4),
                          ("dmse_#3", r"$\Delta$-MSE 3 ($\times10^4$)", 1e4), ("dmse_#4", r"$\Delta$-MSE 4 ($\times10^4$)", 1e4)):
            cells = []
            for f in ("position", "append", "companion", "task"):
                x = e[(e.generator == gen) & (e.factor == f) & (e.quantity == q)].iloc[0]
                cells.append(c(x.est, x.lo, x.hi, s_) if not q.startswith(("rms", "se_")) else
                             f"{x.est * s_:.1f} [{x.lo * s_:.1f}, {x.hi * s_:.1f}]")
            L.append(labg + " & " + ql + " & " + " & ".join(cells) + r" \\")
        if gen == "llama":
            L.append(r"\midrule")
    (OUT / "tab_app_obj2_eval2_rows.tex").write_text("\n".join(L) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig2()
    tab_main()
    tab_eval2()
    tab_app()
    print("ok", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
