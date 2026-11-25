"""Figures for the S3 TEST Objective 1 report (static PNGs, built only from results/eval_s3/tables/*.csv)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from .common import COMPS, GEN_LABEL, GENS, PRED_LABEL, short_type
from .matches import FIXED_TYPES
from .paths import EVAL_DIR

T = EVAL_DIR / "tables"
FIG = EVAL_DIR / "figures"
COL = {"llama": "#2a78d6", "qwen": "#eb6834"}
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
DIV = LinearSegmentedColormap.from_list("div", ["#184f95", "#5598e7", "#f0efec", "#ec7b7a", "#a3201f"])
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e87ba4"]

plt.rcParams.update({"font.size": 9, "axes.edgecolor": BASE, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.titlesize": 10, "axes.titlecolor": INK, "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb", "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False})


def r(name):
    return pd.read_csv(T / f"{name}.csv")


def _types(df, col="type_id"):
    ts = sorted(df[col].dropna().unique())
    return [t for t in ts if t in FIXED_TYPES] + [t for t in ts if t not in FIXED_TYPES]


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    return f"figures/{name}.png"


def _sep(ax, n_fixed, horizontal=True, n=15):
    if horizontal:
        ax.axhline(n_fixed - 0.5, color=INK, lw=1.2)
    else:
        ax.axvline(n_fixed - 0.5, color=INK, lw=1.2)


def dot_types(ax, df, est, lo, hi, types, xlabel, title, zero=True):
    y = np.arange(len(types))
    for k, gen in enumerate(GENS):
        s = df[df.generator == gen].set_index("type_id").reindex(types)
        off = -0.17 if k == 0 else 0.17
        ax.errorbar(s[est], y + off, xerr=[s[est] - s[lo], s[hi] - s[est]], fmt="o", ms=4.5, color=COL[gen],
                    elinewidth=1.4, capsize=0, label=GEN_LABEL[gen])
    ax.set_yticks(y, [short_type(t) for t in types])
    ax.invert_yaxis()
    _sep(ax, sum(t in FIXED_TYPES for t in types))
    ax.grid(axis="x", color=GRID, lw=0.6)
    if zero:
        ax.axvline(0, color=BASE, lw=1)
    else:
        ax.set_xlim(left=0)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")


def heat(ax, M, rows, cols, title, vmax, cbar_label, fig, annotate=True, fmt="{:+.1f}", rowlab=None, collab=None,
         n_fixed_rows=None, fs=7):
    norm = TwoSlopeNorm(0, -vmax, vmax)
    im = ax.imshow(np.ma.masked_invalid(M), cmap=DIV, norm=norm, aspect="auto")
    ax.set_facecolor("#ffffff")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]):
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, hatch="////", ec=GRID, lw=0))
            elif annotate:
                ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=fs,
                        color="#ffffff" if abs(M[i, j]) > 0.6 * vmax else INK)
    ax.set_xticks(range(len(cols)), collab or cols, rotation=0 if len(cols) <= 7 else 90)
    ax.set_yticks(range(len(rows)), rowlab or rows)
    if n_fixed_rows:
        _sep(ax, n_fixed_rows)
    ax.set_title(title, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label(cbar_label, color=INK2)
    cb.outline.set_visible(False)
    return im


# ------------------------------------------------------------------------------------------- Evaluation 1
def e1_heatmap():
    d = r("eval1_per_type")
    types = _types(d, "rule_type")
    cols = [c[0] for c in COMPS]
    vmax = np.nanquantile(np.abs(d.nll_reduction), 0.97)
    fig, axes = plt.subplots(1, len(GENS), figsize=(6.5 * len(GENS), 6.2), constrained_layout=True, squeeze=False)
    axes = axes[0]
    for ax, gen in zip(axes, GENS):
        s = d[d.generator == gen].pivot(index="rule_type", columns="comparison", values="nll_reduction").reindex(types)[cols]
        heat(ax, s.values, types, cols, f"{'AB'[GENS.index(gen)]}. {GEN_LABEL[gen]}", vmax,
             "NLL reduction, nats (positive favors candidate)", fig, rowlab=[short_type(t) for t in types],
             n_fixed_rows=sum(t in FIXED_TYPES for t in types), fmt="{:+.3f}", fs=6)
        ax.text(-0.5, -0.9, "fixed (5) above the line · parameterized (10) below", fontsize=7, color=MUTED)
    return _save(fig, "e1_A3_per_type_nll_heatmap")


def e1_calibration():
    c = r("eval1_calibration")
    keys = ["B1", "B2", "H1", "H2", "RCAP"]
    fig, axes = plt.subplots(1, len(GENS), figsize=(5.5 * len(GENS), 4.8), constrained_layout=True, sharey=True, squeeze=False)
    axes = axes[0]
    for ax, gen in zip(axes, GENS):
        ax.plot([0, 1], [0, 1], color=BASE, lw=1, ls="--")
        for p, colr in zip(keys, SERIES):
            s = c[(c.generator == gen) & (c.predictor == p) & (c.n_outcomes > 0)]
            ax.plot(s.mean_pred, s.obs_rate, "-o", color=colr, lw=2, ms=4, label=PRED_LABEL[p])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.grid(color=GRID, lw=0.6)
        ax.set_xlabel("mean predicted violation probability (bin)")
        ax.set_title(f"{'AB'[GENS.index(gen)]}. {GEN_LABEL[gen]}", loc="left")
    axes[0].set_ylabel("observed violation rate (bin)")
    axes[-1].legend(loc="upper left", fontsize=8)
    return _save(fig, "e1_A2_reliability")


# ------------------------------------------------------------------------------------------- Evaluation 2
def e2_pos():
    a = r("eval2_position_rms_by_type")
    b = r("eval2_position_signed_type_k")
    types = _types(a)
    fig = plt.figure(figsize=(5 + 5 * len(GENS), 6.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 1 + len(GENS), width_ratios=[1.15] + [1] * len(GENS))
    ax = fig.add_subplot(gs[0])
    dot_types(ax, a, "rms_pp", "rms_lo", "rms_hi", types, "noise-adjusted RMS change (pp)",
              "A. Position sensitivity, all position pairs", zero=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, fontsize=8)
    vmax = np.nanquantile(np.abs(b.effect_pp), 0.98)
    for k, gen in enumerate(GENS):
        axh = fig.add_subplot(gs[k + 1])
        M = b[b.generator == gen].pivot(index="type_id", columns="k", values="effect_pp").reindex(types).reindex(columns=[2, 3, 4, 5])
        heat(axh, M.values, types, ["k=2", "k=3", "k=4", "k=5"], f"B{k + 1}. last − first, {GEN_LABEL[gen]}", vmax,
             "pp (positive = riskier when last)", fig, rowlab=[short_type(t) for t in types] if k == 0 else [""] * 15,
             n_fixed_rows=sum(t in FIXED_TYPES for t in types))
    return _save(fig, "e2_pos_figure")


def e2_pos_curves():
    s = r("eval2_position_slot_risk")
    p = r("eval2_position_pair_strata")
    fig, axes = plt.subplots(len(GENS), 4, figsize=(14, 3.2 * len(GENS)), constrained_layout=True, sharey=True, squeeze=False)
    for i, gen in enumerate(GENS):
        for j, k in enumerate([2, 3, 4, 5]):
            ax = axes[i, j]
            x = s[(s.generator == gen) & (s.k == k)].sort_values("slot")
            ax.errorbar(x.slot, x.risk_pct, yerr=[x.risk_pct - x.lo, x.hi - x.risk_pct], fmt="-o", color=COL[gen], ms=4,
                        lw=2, elinewidth=1.2)
            e = p[(p.generator == gen) & (p.k == k) & (p.i == 1) & (p.j == k)].iloc[0]
            ax.text(0.03, 0.04, f"last − first: {e.effect_pp:+.1f} pp [{e.lo:+.1f}, {e.hi:+.1f}]", transform=ax.transAxes,
                    fontsize=7.5, color=INK2)
            ax.set_xticks(range(1, k + 1))
            ax.grid(axis="y", color=GRID, lw=0.6)
            ax.set_title(f"{GEN_LABEL[gen].split(' ')[0]} · k={k}", loc="left")
            if j == 0:
                ax.set_ylabel("target violation risk (%)")
            if i == len(GENS) - 1:
                ax.set_xlabel("target slot")
    return _save(fig, "e2_posA_slot_curves")


def e2_pos_curves_by_class():
    """Pos-A2: slot curves split into fixed (5) and parameterised (10) target types (pos_by_class.py table)."""
    s = r("eval2_position_slot_risk_by_class")
    cls_col = {"parameterized": "#6250d6", "fixed": "#1baf7a"}
    fig, axes = plt.subplots(len(GENS), 4, figsize=(14, 3.4 * len(GENS)), constrained_layout=True, sharey="row",
                             squeeze=False)
    for i, gen in enumerate(GENS):
        for j, k in enumerate([2, 3, 4, 5]):
            ax = axes[i, j]
            g = s[(s.generator == gen) & (s.k == k)]
            a = g[(g.rule_class == "all") & (g.slot != "last-first")].assign(slot=lambda x: x.slot.astype(int))
            a = a.sort_values("slot")
            ax.plot(a.slot, a.risk_pct, ls="--", color=MUTED, lw=1.2, label="all 15 types", zorder=1)
            txt = []
            for cls, off in (("parameterized", -0.06), ("fixed", 0.06)):
                x = g[(g.rule_class == cls) & (g.slot != "last-first")].assign(slot=lambda x: x.slot.astype(int))
                x = x.sort_values("slot")
                ax.errorbar(x.slot + off, x.risk_pct, yerr=[x.risk_pct - x.lo, x.hi - x.risk_pct], fmt="-o",
                            color=cls_col[cls], ms=4, lw=2, elinewidth=1,
                            label={"parameterized": "parameterised types (10)", "fixed": "fixed types (5)"}[cls])
                e = g[(g.rule_class == cls) & (g.slot == "last-first")].iloc[0]
                txt.append(f"{cls[:5]}. last − first: {e.risk_pct:+.1f} pp [{e.lo:+.1f}, {e.hi:+.1f}]")
            ax.text(0.03, 0.97, "\n".join(txt), transform=ax.transAxes, va="top", fontsize=7.5, color=INK2)
            ax.set_xticks(range(1, k + 1))
            ax.set_xlim(0.6, k + 0.4)
            ax.set_ylim(0, 31 if gen == "llama" else 27)
            ax.grid(axis="y", color=GRID, lw=0.6)
            ax.set_title(f"{GEN_LABEL[gen].split(' ')[0]} · k={k}", loc="left")
            if j == 0:
                ax.set_ylabel("target violation risk (%)")
            if i == len(GENS) - 1:
                ax.set_xlabel("target slot (1 = first listed)")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05))
    return _save(fig, "e2_posA2_slot_curves_by_class")


def e2_append():
    d = r("eval2_append_by_transition")
    trs = ["1->2", "2->3", "3->4", "4->5"]
    fig, ax = plt.subplots(figsize=(7, 3.8), constrained_layout=True)
    x = np.arange(4)
    for k, gen in enumerate(GENS):
        s = d[d.generator == gen].set_index("transition").reindex(trs)
        ax.errorbar(x + (-0.1 if k == 0 else 0.1), s.effect_pp, yerr=[s.effect_pp - s.lo, s.hi - s.effect_pp], fmt="o",
                    ms=6, color=COL[gen], elinewidth=1.6, label=GEN_LABEL[gen])
    ax.axhline(0, color=BASE, lw=1)
    ax.set_xticks(x, [t.replace("->", "→") for t in trs])
    ax.set_xlabel("strict append transition (k → k+1)")
    ax.set_ylabel("Δ retained-rule violation risk (pp)")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(fontsize=8)
    return _save(fig, "e2_append_figure")


def e2_companion():
    a = r("eval2_companion_rms_by_type")
    m = r("eval2_companion_signed_matrix")
    types = _types(a)
    fig = plt.figure(figsize=(5.5 + 6.8 * len(GENS), 7.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 1 + len(GENS), width_ratios=[0.9] + [1.15] * len(GENS))
    ax = fig.add_subplot(gs[0])
    dot_types(ax, a, "rms_pp", "rms_lo", "rms_hi", types, "noise-adjusted RMS change (pp)",
              "A. Sensitivity to companion identity", zero=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, fontsize=8)
    vmax = np.nanquantile(np.abs(m.effect_pp), 0.97)
    for k, gen in enumerate(GENS):
        axh = fig.add_subplot(gs[k + 1])
        s = m[m.generator == gen]
        M = s.pivot(index="target_type", columns="companion_type", values="effect_pp").reindex(index=types, columns=types)
        low = s.pivot(index="target_type", columns="companion_type", values="n_pools").reindex(index=types, columns=types)
        heat(axh, M.values, types, types, f"B{k + 1}. target × companion, {GEN_LABEL[gen]}", vmax,
             "pp (positive = target riskier with this companion)", fig, annotate=True, fmt="{:+.0f}",
             rowlab=[short_type(t) for t in types] if k == 0 else [""] * 15, collab=[short_type(t) for t in types],
             n_fixed_rows=5, fs=5.5)
        axh.axvline(4.5, color=INK, lw=1.2)
        for i in range(15):
            for j in range(15):
                if np.isfinite(low.values[i, j]) and low.values[i, j] < 3:
                    axh.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec=INK, lw=0.8))
        axh.set_xlabel("companion type (rows: target type)")
    return _save(fig, "e2_companion_figure")


def e2_task():
    a = r("eval2_task_rms_by_type")
    f = r("eval2_task_family")
    types = _types(a)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), constrained_layout=True, gridspec_kw={"width_ratios": [1.2, 1]})
    dot_types(axes[0], a, "rms_pp", "rms_lo", "rms_hi", types, "noise-adjusted RMS change (pp)",
              "A. Sensitivity to task text (same exact rules)", zero=False)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, fontsize=8)
    fams = sorted(f.family.unique())
    ax = axes[1]
    y = np.arange(len(fams))
    for k, gen in enumerate(GENS):
        s = f[f.generator == gen].set_index("family").reindex(fams)
        ax.errorbar(s.effect_pp, y + (-0.15 if k == 0 else 0.15), xerr=[s.effect_pp - s.lo, s.hi - s.effect_pp], fmt="o",
                    ms=5, color=COL[gen], elinewidth=1.4, label=GEN_LABEL[gen])
        if k == len(GENS) - 1:
            for yy, (fam, rr) in zip(y, s.iterrows()):
                ax.text(1.01, yy, f"{int(rr.n_pools)} pools", transform=ax.get_yaxis_transform(), fontsize=7, color=MUTED,
                        va="center")
    ax.axvline(0, color=BASE, lw=1)
    ax.set_yticks(y, fams)
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_xlabel("family task − mean of the other two matched tasks (pp)")
    ax.set_title("B. Task-family signed differences", loc="left")
    return _save(fig, "e2_task_figure")


def e2_task_family_heat():
    m = r("eval2_task_type_family")
    types = _types(m)
    fams = sorted(m.family.unique())
    vmax = np.nanquantile(np.abs(m.effect_pp), 0.97)
    fig, axes = plt.subplots(1, len(GENS), figsize=(6 * len(GENS), 6), constrained_layout=True, squeeze=False)
    axes = axes[0]
    for k, (ax, gen) in enumerate(zip(axes, GENS)):
        M = m[m.generator == gen].pivot(index="type_id", columns="family", values="effect_pp").reindex(index=types, columns=fams)
        heat(ax, M.values, types, fams, f"{'AB'[k]}. {GEN_LABEL[gen]}", vmax, "pp (family task − other two)", fig,
             rowlab=[short_type(t) for t in types], n_fixed_rows=5, collab=[x[:5] for x in fams])
    return _save(fig, "e2_taskDB_type_family_heatmap")


def all_figures() -> dict:
    return {"e1_heat": e1_heatmap(), "e1_cal": e1_calibration(), "pos": e2_pos(), "pos_curves": e2_pos_curves(),
            "pos_curves_by_class": e2_pos_curves_by_class(),
            "append": e2_append(), "companion": e2_companion(), "task": e2_task(), "task_heat": e2_task_family_heat()}
