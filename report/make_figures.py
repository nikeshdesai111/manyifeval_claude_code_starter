"""Figures for report/paper.tex. Every figure is drawn from saved tables (TRAIN groups only):
results/tables/*.csv (written by `mife.cli probe-report`) and report/tables/*.csv (written by analysis_oof.py).

Run: .venv/bin/python report/make_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T, R, F = ROOT / "results" / "tables", ROOT / "report" / "tables", ROOT / "report" / "figures"

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8983", "#e4e3df"
GEN = {"llama": "#2a78d6", "qwen": "#eb6834"}                      # generator colours (validated pair)
GEN_NAME = {"llama": "Llama 3.1 8B", "qwen": "Qwen3-8B"}
FAM = {"probe-only": ("#6250d6", "o"), "B1 family": ("#1baf7a", "^"), "B2 family": ("#eda100", "D")}   # validated triple
REF = "#6b6a66"
CELL = [("llama", 1), ("llama", 2), ("qwen", 1), ("qwen", 2)]
LOSS = {1: "NLL", 2: "MSE"}

TYPE = {
    "change_case:capital_word_frequency": "Uppercase-word count", "change_case:english_capital": "All uppercase",
    "change_case:english_lowercase": "All lowercase", "detectable_content:number_placeholders": "Placeholders",
    "detectable_format:number_bullet_lists": "Bullet points", "detectable_format:title": "Title",
    "keywords:existence": "Include keywords", "keywords:forbidden_words": "Forbidden words",
    "keywords:frequency": "Keyword count", "keywords:letter_frequency": "Letter count",
    "length_constraints:number_paragraphs": "Paragraph count", "length_constraints:number_sentences": "Sentence count",
    "length_constraints:number_words": "Word count", "punctuation:no_comma": "No commas",
    "startend:quotation": "Quotation marks",
}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 7.5,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.5, "axes.labelcolor": INK2, "axes.titlesize": 8, "axes.titlecolor": INK,
    "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "legend.frameon": False, "legend.fontsize": 7.5, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})


def save(fig, name):
    F.mkdir(parents=True, exist_ok=True)
    fig.savefig(F / f"{name}.pdf")
    fig.savefig(F / f"{name}.png", dpi=170)
    plt.close(fig)


def fig_violation():
    K = pd.read_csv(T / "desc_by_k.csv")
    K = K[K.k != "all"].astype({"k": int})
    Y = pd.read_csv(T / "desc_by_type.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 3.0), gridspec_kw={"width_ratios": [1, 1.55], "wspace": 0.62})
    for m, c in GEN.items():
        d = K[K.model == m].sort_values("k")
        a.fill_between(d.k, d.lo, d.hi, color=c, alpha=0.12, lw=0)
        a.plot(d.k, d.est, color=c, lw=1.6, marker="o", ms=4.5, mec="white", mew=0.8, label=GEN_NAME[m])
        for i in (0, -1):
            a.annotate(f"{d.est.iloc[i]:.3f}", (d.k.iloc[i], d.est.iloc[i]), xytext=(0, 6), textcoords="offset points",
                       ha="center", va="bottom", color=INK2, fontsize=7)
    a.set_xlim(0.6, 5.4); a.set_ylim(0, 0.225); a.set_xticks(range(1, 6))
    a.set_xlabel("Number of rules in the prompt (k)"); a.set_ylabel("Violation rate per (response, rule) check")
    a.set_title("(a) By number of rules", loc="left"); a.grid(axis="x", visible=False)
    a.legend(loc="lower right", handlelength=1.4)
    order = Y[Y.model == "llama"].sort_values("est").type_id.tolist()
    for i, t in enumerate(order):
        for m, off in (("llama", 0.17), ("qwen", -0.17)):
            r = Y[(Y.model == m) & (Y.type_id == t)].iloc[0]
            b.plot([r.lo, r.hi], [i + off, i + off], color=GEN[m], lw=1.0, alpha=0.55, solid_capstyle="round")
            b.plot(r.est, i + off, "o", color=GEN[m], ms=4.5, mec="white", mew=0.7, label=GEN_NAME[m] if i == 0 else None)
    b.set_yticks(range(len(order))); b.set_yticklabels([TYPE[t] for t in order]); b.set_ylim(-0.6, len(order) - 0.4)
    b.set_xlim(-0.01, 0.66); b.set_xlabel("Violation rate (95% CI over groups)"); b.grid(axis="y", visible=False)
    b.set_title("(b) By rule type", loc="left"); b.legend(loc="lower right", handletextpad=0.2)
    save(fig, "fig_violation")


def fig_type_by_k():
    D = pd.read_csv(T / "desc_type_by_k_long.csv")
    order = D[(D.model == "llama") & (D.k == 5)].sort_values("est", ascending=False).type_id.tolist()
    fig, axes = plt.subplots(3, 5, figsize=(6.5, 3.9), sharex=True, gridspec_kw={"hspace": 0.5, "wspace": 0.42})
    for ax, t in zip(axes.ravel(), order):
        top = 0
        for m, c in GEN.items():
            d = D[(D.model == m) & (D.type_id == t)].sort_values("k")
            ax.fill_between(d.k, d.lo, d.hi, color=c, alpha=0.12, lw=0)
            ax.plot(d.k, d.est, color=c, lw=1.4, marker="o", ms=2.8, mec="white", mew=0.4, label=GEN_NAME[m])
            top = max(top, d.hi.max())
        ax.set_title(TYPE[t], loc="left", fontsize=7, fontweight="normal", pad=3)
        ax.set_ylim(0, max(0.05, top * 1.08)); ax.set_xticks(range(1, 6)); ax.grid(axis="x", visible=False)
        ax.tick_params(labelsize=6.5); ax.locator_params(axis="y", nbins=4)
    for ax in axes[-1]:
        ax.set_xlabel("k")
    fig.supylabel("Violation rate", fontsize=7.5, color=INK2, x=0.035)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0), handlelength=1.4)
    save(fig, "fig_type_by_k")


def fig_cv_configs():
    S = pd.read_csv(T / "cv_selected.csv")
    cfgs = S[(S.model == "llama") & (S.objective == 1) & (S.family != "reference")]
    rows, y, yt, pos = [], 0.0, [], {}
    for fam in FAM:
        for c in cfgs[cfgs.family == fam].config:
            pos[c] = y; yt.append((y, c)); y += 1
        y += 0.8
    fig, axes = plt.subplots(1, 4, figsize=(6.5, 4.6), sharey=True, gridspec_kw={"wspace": 0.12})
    for ax, (m, o) in zip(axes, CELL):
        d = S[(S.model == m) & (S.objective == o)].set_index("config")
        ref, tk = d.cv_loss["REF_structure"], d.cv_loss["REF_type_k"]
        ax.axvline(ref, color=REF, lw=0.9); ax.axvline(tk, color=REF, lw=0.9, ls=(0, (1, 2)))
        for fam, (c, mk) in FAM.items():
            dd = d[d.family == fam]
            ax.plot(dd.cv_loss, [pos[k] for k in dd.index], mk, color=c, ms=5.2 if mk != "D" else 4.4, mec="white", mew=0.6,
                    label=fam.replace("probe-only", "Probe-only"), ls="none", zorder=3)
        best = d[d.family != "reference"].cv_loss.idxmin()
        ax.plot(d.cv_loss[best], pos[best], "o", ms=10, mfc="none", mec=INK, mew=0.8, zorder=4)
        lo, hi = d[d.family != "reference"].cv_loss.min(), max(tk, d[d.family != "reference"].cv_loss.max())
        pad = (hi - lo) * 0.09
        ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(y - 1.2, -0.8); ax.grid(axis="y", visible=False)
        ax.set_title(f"{GEN_NAME[m]}\nObjective {o}", loc="left", fontsize=7.5)
        ax.set_xlabel(f"CV {LOSS[o]}"); ax.tick_params(axis="x", labelsize=6.5); ax.locator_params(axis="x", nbins=4)
    axes[0].set_yticks([p for p, _ in yt]); axes[0].set_yticklabels([c for _, c in yt], fontsize=7)
    h, l = axes[0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    h += [Line2D([], [], color=REF, lw=0.9), Line2D([], [], color=REF, lw=0.9, ls=(0, (1, 2))),
          Line2D([], [], marker="o", ms=8, mfc="none", mec=INK, mew=0.8, ls="none")]
    l += ["Structure reference", "Type × k reference", "Lowest loss in panel"]
    fig.legend(h, l, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 0.99), handletextpad=0.3, columnspacing=1.0,
               fontsize=6.6, handlelength=1.5)
    save(fig, "fig_cv_configs")


def fig_contrasts():
    C = pd.read_csv(R / "oof_contrasts.csv")
    rows = [("REF_structure", "H1"), ("REF_structure", "H2"), ("REF_structure", "A1"), None,
            ("B1", "H1"), ("B1", "H2"), ("B1", "A1"), None, ("B2", "H1"), ("B2", "H2"), ("B2", "A1"), None, ("B1", "B2")]
    lab = {"REF_structure": "Structure"}
    fig, axes = plt.subplots(1, 4, figsize=(6.5, 3.1), sharey=True, gridspec_kw={"wspace": 0.12})
    for ax, (m, o) in zip(axes, CELL):
        d = C[(C.model == m) & (C.objective == o)].drop_duplicates(["a", "b"]).set_index(["a", "b"])
        ax.axvline(0, color=REF, lw=0.9)
        for i, r in enumerate(rows):
            if r is None:
                continue
            x = d.loc[r]
            sig = (x.hi < 0) or (x.lo > 0)
            ax.plot([x.lo, x.hi], [i, i], color=GEN[m], lw=1.2, solid_capstyle="round", alpha=0.7)
            ax.plot(x["diff"], i, "o", ms=5, color=GEN[m] if sig else "white", mec=GEN[m], mew=1.0, zorder=3)
        ax.set_ylim(len(rows) - 0.4, -0.6); ax.grid(axis="y", visible=False)
        ax.set_title(f"{GEN_NAME[m]}\nObjective {o}", loc="left", fontsize=7.5)
        ax.set_xlabel("Δ NLL" if o == 1 else "Δ RMSE"); ax.tick_params(axis="x", labelsize=6.5)
        ax.locator_params(axis="x", nbins=4)
    axes[0].set_yticks([i for i, r in enumerate(rows) if r]); axes[0].set_yticklabels(
        [f"{lab.get(a, a)} $\\rightarrow$ {b}" for a, b in filter(None, rows)], fontsize=7)
    save(fig, "fig_contrasts")


def fig_per_type():
    P = pd.read_csv(R / "oof_per_type.csv")
    ser = {"REF_structure": (REF, "s", "Structure reference"), "B2": ("#eda100", "D", "B2 (fine-tuned text encoder)"),
           "A1": ("#6250d6", "o", "A1 (attention probe)")}
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.5), sharey=True, gridspec_kw={"wspace": 0.08})
    order = P[(P.model == "llama") & (P.config == "A1")].sort_values("violation_rate").type_id.tolist()
    for ax, m in zip(axes, ("llama", "qwen")):
        d = P[P.model == m]
        for i, t in enumerate(order):
            v = d[d.type_id == t].set_index("config").nll
            ax.plot([v[list(ser)].min(), v[list(ser)].max()], [i, i], color=GRID, lw=2.2, solid_capstyle="round", zorder=1)
            for c, (col, mk, name) in ser.items():
                ax.plot(v[c], i, mk, color=col, ms=5.2 if mk != "D" else 4.4, mec="white", mew=0.6, zorder=3,
                        label=name if i == 0 else None, ls="none")
        ax.set_ylim(-0.6, len(order) - 0.4); ax.set_xlim(-0.02, 0.76); ax.grid(axis="y", visible=False)
        ax.set_title(GEN_NAME[m], loc="left"); ax.set_xlabel("Out-of-fold NLL per check (lower is better)")
    axes[0].set_yticks(range(len(order))); axes[0].set_yticklabels([TYPE[t] for t in order])
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.55, 1.03), handletextpad=0.2, columnspacing=1.6)
    save(fig, "fig_per_type")


def fig_reliability():
    Q = pd.read_csv(R / "oof_reliability.csv")
    ser = {"REF_structure": (REF, "s", "Structure reference"), "B2": ("#eda100", "D", "B2"), "A1": ("#6250d6", "o", "A1")}
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.75), sharey=True, gridspec_kw={"wspace": 0.1})
    for ax, m in zip(axes, ("llama", "qwen")):
        ax.plot([0, 0.85], [0, 0.85], color=MUTED, lw=0.7)
        for c, (col, mk, name) in ser.items():
            d = Q[(Q.model == m) & (Q.config == c)].sort_values("mean_predicted")
            ax.plot(d.mean_predicted, d.observed_rate, color=col, lw=1.3, marker=mk, ms=4.2 if mk != "D" else 3.6,
                    mec="white", mew=0.5, label=name)
        ax.set_xlim(-0.02, 0.85); ax.set_ylim(-0.02, 0.85); ax.set_aspect("equal")
        ax.set_title(GEN_NAME[m], loc="left"); ax.set_xlabel("Mean predicted probability (decile bin)")
    axes[0].set_ylabel("Observed violation rate"); axes[0].legend(loc="upper left", handlelength=1.6)
    save(fig, "fig_reliability")


if __name__ == "__main__":
    for f in (fig_violation, fig_type_by_k, fig_cv_configs, fig_contrasts, fig_per_type, fig_reliability):
        f()
    print("figures:", sorted(p.name for p in F.glob("*.pdf")))
