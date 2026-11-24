"""LaTeX for the paper's Objective 1 test exhibits (main Tables 1, 2, 4 and the plan's Objective 1 appendix), built only
from results/eval_s3[/<sub>]/tables/*.csv for the generators in MIFE_EVAL_GENS. Writes <EVAL_DIR>/paper/test_*.tex.
Run: MIFE_EVAL_SUB=llama_only MIFE_EVAL_GENS=llama PYTHONPATH=src .venv/bin/python -m mife.eval_s3.paper_test
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .common import GENS
from .matches import FIXED_TYPES
from .paths import EVAL_DIR

T = EVAL_DIR / "tables"
OUT = EVAL_DIR / "paper"
FIGDIR = "figures/test_" + "_".join(GENS)
ORDER = ["#1", "#2", "#3", "#4", "#5", "#6", "#7"]
CL = {"#1": r"1: B1 \cto{} H1", "#2": r"2: B2 \cto{} H2", "#3": r"3: B2 \cto{} RCAP", "#4": r"4: H2 \cto{} RCAP",
      "#5": r"5: B1 \cto{} B1+H1", "#6": r"6: B2 \cto{} B2+H1", "#7": r"7: B2 \cto{} B2+A1"}
GL = {"llama": "Llama", "qwen": "Qwen"}
PL = {"B1": "B1", "B2": "B2", "H1": "H1+S", "H2": "H2+S", "RCAP": "RCAP", "B1+H1": "B1+H1", "B2+H1": "B2+H1",
      "B2+A1": "B2+A1", "REF_structure": "Structure only"}
NG = len(GENS)


def r(n):
    return pd.read_csv(T / f"{n}.csv")


def f(x, d=1):
    if x is None or not np.isfinite(float(x)):
        return "n/a"
    v = round(float(x), d)
    return f"{0.0 if v == 0 else v:.{d}f}".replace("-", "$-$")


def ci(e, lo, hi, s=1.0, d=1):
    return f"{f(e * s, d)} [{f(lo * s, d)}, {f(hi * s, d)}]"


def tt(t):
    return r"\texttt{" + t.split(":")[1].replace("_", r"\_") + "}"


def types(ts):
    return [t for t in sorted(ts) if t in FIXED_TYPES] + [t for t in sorted(ts) if t not in FIXED_TYPES]


def tab(caption, label, head, rows, align=None, size=r"\scriptsize", resize=True, mid=None, place="h"):
    align = align or "l" + "c" * (len(head) - 1)
    body = []
    for i, row in enumerate(rows):
        if mid and i in mid:
            body.append(r"\midrule")
        body.append(" & ".join(str(c) for c in row) + r" \\")
    t = "\n".join([r"\begin{tabular}{" + align + "}", r"\toprule", " & ".join(head) + r" \\", r"\midrule", *body,
                   r"\bottomrule", r"\end{tabular}"])
    if resize:
        t = r"\resizebox{\linewidth}{!}{" + "\n" + t + "\n}"
    return "\n".join([r"\begin{table}[" + place + "]", r"\centering" + size, r"\caption{" + caption + "}",
                      (r"\label{" + label + "}") if label else "", t, r"\end{table}", ""])


def fig(path, cap, label=None, width=r"\linewidth"):
    lab = (r"\label{" + label + "}") if label else ""
    return (r"\begin{figure}[h]\centering\includegraphics[width=" + width + "]{" + f"{FIGDIR}/{path}" + "}"
            + r"\caption{" + cap + "}" + lab + r"\end{figure}")


def gl_join():
    return " and ".join(GL[g] for g in GENS)


# ------------------------------------------------------------------------------------------------- main
def main_tables():
    con = r("eval1_contrasts")
    sk = {x: r(f"eval2_{x}_skill") for x in ("position", "append", "companion", "task")}
    qc = json.loads((T / "eval1_qc.json").read_text())
    cov = json.loads((T / "match_coverage.json").read_text())
    rows = [[GL[q["generator"]], q["pools"], q["tasks"], f"{q['prompts']:,}", f"{q['rows']:,}", f"{q['prompts'] * 20:,}",
             f"{100 * q['violations'] / q['outcomes']:.1f}\\%"] for q in qc]
    t1 = tab(r"Held-out evaluation inventory: the 22 S3 test groups (all 456 ordered configurations per task). Responses: "
             r"20 verified per prompt, all present. Matched target contrasts per generator (metadata only, equal to the "
             f"design counts): position {cov['position_pairs']:,}, strict append {cov['append_targets']:,}, companion "
             f"replacement {cov['companion_targets']:,}, task {cov['task_targets']:,}.",
             "tab:inventory", ["Generator", "Pools", "Tasks", "Prompts", "Prompt--rule rows", "Responses", "Violation rate"],
             rows, resize=False, size=r"\footnotesize", place="t")
    rows = []
    for cid in ORDER:
        row = [CL[cid]]
        for g in GENS:
            x = con[(con.generator == g) & (con.comparison == cid)].iloc[0]
            row += [ci(x.nll_reduction, x.nll_reduction_lo, x.nll_reduction_hi, 1, 4),
                    ci(x.auroc_gain, x.auroc_gain_lo, x.auroc_gain_hi, 100), f"{int(x.types_improved)}/15"]
        rows.append(row)
    pending = (r" Qwen: test features still being extracted at the time of this version." if GENS == ("llama",) else "")
    t2 = tab(r"Evaluation~1, Objective~1 on the 22 held-out S3 test groups: rule-level violation-risk prediction "
             r"(115,236 prompt--rule rows and 2,304,720 response $\times$ rule outcomes per generator). NLL reduction "
             r"$=$ NLL(comparator) $-$ NLL(candidate), in nats per outcome; AUROC gain $\times100$; Types $=$ "
             r"rule types with strictly lower NLL (descriptive). Positive favours the candidate. Paired, pointwise 95\% "
             r"intervals from a 22-pool cluster bootstrap (10,000 draws). Primary comparisons above the rule, secondary "
             r"below." + pending,
             "tab:e1o1", ["Comparison"] + ["NLL red.", "AUROC gain", "Types"] * NG, rows,
             align="l" + "ccc" * NG, mid=[4], size=r"\footnotesize", place="t", resize=NG > 1)
    if NG > 1:
        t2 = t2.replace(r"\toprule", r"\toprule" + "\n" + " & " + " & ".join(
            r"\multicolumn{3}{c}{" + GL[g] + "}" for g in GENS) + r" \\" + "\n" + "".join(
            r"\cmidrule(lr){" + f"{2 + 3 * i}-{4 + 3 * i}" + "}" for i in range(NG)), 1)
    rows, mids = [], []
    for gi, g in enumerate(GENS):
        if gi:
            mids.append(len(rows))
        for cid in ORDER:
            row = [GL[g] if cid == "#1" else "", CL[cid]]
            for fac in sk:
                s = sk[fac]
                x = s[(s.generator == g) & (s.stratum == "macro") & (s.quantity == f"dmse_{cid}")].iloc[0]
                row.append(ci(x.est, x.lo, x.hi, 1e4))
            rows.append(row)
        row = ["", "zero-change MSE"]
        for fac in sk:
            s = sk[fac]
            row.append(f(s[(s.generator == g) & (s.stratum == "macro") & (s.quantity == "se_zero")].est.iloc[0] * 1e4))
        rows.append(row)
    t4 = tab(r"Evaluation~2, Objective~1 predictor skill on the 22 held-out test groups: paired $\Delta$-MSE reduction "
             r"($\times10^{4}$, i.e.\ squared points) [95\% interval] of the predicted versus observed change in a rule's "
             r"violation probability, MSE(comparator) $-$ MSE(candidate); positive favours the candidate. Position: "
             r"57,420 all-pair contrasts; strict append: 69,300 retained-rule contrasts; companion: 177,408 retained-target "
             r"contrasts; task: 115,236 exact-rule task-pair contrasts (per generator). Equal weight per $k$ (position, "
             r"companion, task) or transition (append), then pool, task and context. The zero-change row is the MSE of "
             r"predicting no change. By-stratum values and $\Delta$-MAE in Appendices Pos-B, K-B, C-B and D-C." + pending,
             "tab:eval2", ["", "Comparison", "Position", "Strict append", "Companion", "Task"], rows, align="llcccc",
             mid=mids, size=r"\footnotesize", place="t")
    return {"t1": t1, "t2": t2, "t4": t4}


def by_gen_cols(prefix=""):
    return [GL[g] for g in GENS]


# ------------------------------------------------------------------------------------------------- appendix
def appendix():
    out = []
    w = out.append
    a1, con = r("eval1_a1_metrics"), r("eval1_contrasts")
    rows, mids = [], []
    for gi, g in enumerate(GENS):
        if gi:
            mids.append(len(rows))
        for p in ["REF_structure", "B1", "B2", "H1", "H2", "RCAP", "B1+H1", "B2+H1", "B2+A1"]:
            x = a1[(a1.generator == g) & (a1.predictor == p)].iloc[0]
            rows.append([GL[g], PL[p], ci(x.nll, x.nll_lo, x.nll_hi, 1, 4), ci(x.brier, x.brier_lo, x.brier_hi, 1, 4),
                         ci(x.auroc, x.auroc_lo, x.auroc_hi, 100), ci(x.ece, x.ece_lo, x.ece_hi, 100, 2)])
    w(r"\subsection*{A1 --- Complete overall metrics (Objective 1)}")
    w(tab(r"A1. Pooled test scores (22 held-out groups; unit: one response $\times$ rule outcome) with 95\% pool-bootstrap "
          r"intervals. ECE uses ten bins fixed from each predictor's training out-of-fold deciles.", "tab:A1",
          ["", "Predictor", "NLL", "Brier", r"AUROC $\times100$", r"ECE $\times100$"], rows, align="llcccc", mid=mids))
    rows = []
    for g in GENS:
        for cid in ORDER:
            x = con[(con.generator == g) & (con.comparison == cid)].iloc[0]
            nc = a1[(a1.generator == g) & (a1.predictor == x.comparator)].iloc[0]
            nk = a1[(a1.generator == g) & (a1.predictor == x.candidate)].iloc[0]
            rows.append([GL[g], CL[cid], f"{f(nc.nll, 4)} / {f(nk.nll, 4)}",
                         ci(x.brier_reduction, x.brier_reduction_lo, x.brier_reduction_hi, 1e3, 2),
                         f"{f(nc.auroc * 100)} / {f(nk.auroc * 100)}",
                         f"[{f(x.nll_t_lo, 4)}, {f(x.nll_t_hi, 4)}]",
                         f"{f(x.nll_loo_min, 4)} to {f(x.nll_loo_max, 4)}"])
    w(tab(r"A1 (contrasts) and A4 (sensitivity). Raw NLL (nats) and AUROC ($\times100$), comparator / candidate; "
          r"Brier reduction $\times10^3$ with paired interval; NLL-reduction $t_{21}$ jackknife interval and "
          r"leave-one-pool-out range (nats).", "tab:A1c",
          ["", "Comparison", "NLL comp./cand.", "Brier reduction", "AUROC comp./cand.", r"NLL $t_{21}$ interval",
           "NLL leave-one-pool-out"], rows, align="llccccc", mid=[7 * i for i in range(1, NG)]))
    w(r"\subsection*{A2 --- Calibration (Objective 1)}")
    w(fig("e1_A2_reliability.png", r"A2. Reliability curves on the test groups for five key predictors; bins are deciles of "
          r"each predictor's \emph{training} out-of-fold predictions (fixed before test scoring); unit: response "
          r"$\times$ rule outcome. Bin support: \texttt{tables/eval1\_calibration.csv}.", width=r"\linewidth" if NG > 1 else r"0.62\linewidth"))
    w(r"\subsection*{A3 --- Per-type NLL reduction}")
    w(fig("e1_A3_per_type_nll_heatmap.png", r"A3. Per-type test NLL reduction (nats) for the seven comparisons; five "
          r"fixed-wording types above the line, ten parameterised types below; positive (red) favours the candidate. "
          r"All 210 values per generator with paired intervals and support: \texttt{tables/eval1\_per\_type.csv}."))
    fp = r("eval1_fixed_vs_param")
    rows, mids = [], []
    for gi, g in enumerate(GENS):
        if gi:
            mids.append(len(rows))
        for cid in ORDER:
            x = fp[(fp.generator == g) & (fp.comparison == cid)].iloc[0]
            rows.append([GL[g] if cid == "#1" else "", CL[cid], ci(x.fixed_dnll, x.fixed_lo, x.fixed_hi, 1, 4),
                         ci(x.parameterized_dnll, x.parameterized_lo, x.parameterized_hi, 1, 4),
                         ci(x["diff_dnll"], x.diff_lo, x.diff_hi, 1, 4),
                         f"{f(x.fixed_typeeq_dnll, 4)} / {f(x.param_typeeq_dnll, 4)}"])
    rates = "; ".join(f"{GL[g]}: fixed {100 * fp[fp.generator == g].fixed_viol_rate.iloc[0]:.1f}\\% "
                      f"({int(fp[fp.generator == g].fixed_outcomes.iloc[0]):,} outcomes), parameterised "
                      f"{100 * fp[fp.generator == g].parameterized_viol_rate.iloc[0]:.1f}\\% "
                      f"({int(fp[fp.generator == g].parameterized_outcomes.iloc[0]):,})" for g in GENS)
    w(r"\subsection*{A3b --- Fixed versus parameterised rule types (Objective 1 only)}")
    w(tab(r"A3b. NLL reductions of the same seven predictor comparisons, calculated separately on rule-level outcomes from "
          r"the five fixed and ten parameterised ManyIFEval rule types (nats). Positive improvement favours the "
          r"candidate (right of the arrow). The fixed-minus-parameterised column compares improvements between subgroups, "
          r"not raw violation rates. Intervals are paired pointwise 95\% intervals from resampling the 22 test rule pools "
          r"(identical draws for both classes); all seven subgroup differences are exploratory. Last column: "
          r"type-equal-weight sensitivity (simple mean of per-type reductions). "
          f"Observed violation rates: {rates}.", "tab:A3b",
          ["", "Comparison", "Fixed $\\Delta$NLL", "Parameterised $\\Delta$NLL", "Fixed $-$ param.", "Type-equal (F / P)"],
          rows, align="llcccc", mid=mids))
    qc = json.loads((T / "eval1_qc.json").read_text())
    pt = r("eval1_per_type")
    low = pt[pt.low_support_flag].drop_duplicates(["generator", "rule_type"])
    lows = ", ".join(f"{tt(t)} ({n})" for t, n in zip(low.rule_type, low.n_pools)) or "none"
    chk = pd.concat([pd.read_csv(p) for p in sorted(T.glob("checks_*.csv"))]) if list(T.glob("checks_*.csv")) else None
    w(r"\subsection*{A4 --- Technical uncertainty and QC}")
    w("Probabilities are clipped to $[10^{-6}, 1-10^{-6}]$ before the logarithm (rows clipped: "
      + "; ".join(f"{GL[q['generator']]} {sum(q['clipped_rows'].values())}" for q in qc)
      + "). Every test prompt has 20 distinct verified responses (originals from run \\texttt{main}, expansion prompts "
        "from run \\texttt{expansion\\_s3\\_20261123}; one decoding-settings hash per generator). The frozen S3 refits were "
        "applied once; no sparse-type fallback head fired. New test hidden states and rule-span attention sums equal the "
        "frozen V1 features bit for bit on every original prompt checked, and the recomputed BGE-M3 embeddings equal the "
        "V1 cache. Bootstrap: 10,000 draws over the 22 test pools, seed 20261203, identical for every predictor, contrast "
        "and subgroup; intervals are pointwise and unadjusted. $t_{21}$ jackknife intervals and leave-one-pool-out ranges "
        f"are in Table~\\ref{{tab:A1c}}. Rule types with fewer than 8 contributing test pools: {lows}."
        + (f" Independent recomputation checks ({int(chk['pass'].sum())}/{len(chk)} passed): direct NLL and "
           "\\texttt{sklearn} AUROC on expanded binary outcomes; explicit-weight $\\Delta$-MSE; structure block "
           "reproduces the stored training cache; test and training prompts disjoint." if chk is not None else "") + "\n")
    # ---- Position
    sk = {x: r(f"eval2_{x}_skill") for x in ("position", "append", "companion", "task")}
    w(r"\subsection*{Pos-A --- Position profiles and all 20 pair strata}")
    w(fig("e2_posA_slot_curves.png", r"Pos-A. Observed target violation risk at every slot by $k$ (COA-balanced across exact "
          r"matched contexts, 95\% pool intervals) with the paired last $-$ first difference. Observed outcomes only."))
    ps = r("eval2_position_pair_strata")
    rows = []
    for (k, i, j), x in ps.groupby(["k", "i", "j"]):
        row = [k, f"{i}$\\to${j}"]
        for g in GENS:
            y = x[x.generator == g].iloc[0]
            row.append(ci(y.effect_pp, y.lo, y.hi, 1, 2))
        row.append(f"{int(x.n_contexts.iloc[0]):,}")
        rows.append(row)
    w(tab("Pos-A. Signed change (later slot $-$ earlier slot, points) for all 20 position-pair strata, 22 test pools.",
          "tab:posA", ["$k$", "Slots"] + by_gen_cols() + ["Contexts"], rows, align="cc" + "c" * NG + "c", resize=False))
    for fac, name, lab in (("position", "Pos-B", "$k$"), ("append", "K-B", "transition"), ("companion", "C-B", "$k$"),
                           ("task", "D-C", "$k$")):
        s = sk[fac]
        strata = sorted(x for x in s.stratum.unique() if x != "macro")
        rows, mids = [], []
        for gi, g in enumerate(GENS):
            if gi:
                mids.append(len(rows))
            for cid in ORDER:
                row = [GL[g] if cid == "#1" else "", CL[cid]]
                for st in strata:
                    x = s[(s.generator == g) & (s.stratum == st) & (s.quantity == f"dmse_{cid}")].iloc[0]
                    row.append(ci(x.est, x.lo, x.hi, 1e4))
                rows.append(row)
            rows.append(["", "zero-change MSE"] + [f(s[(s.generator == g) & (s.stratum == st) & (s.quantity == "se_zero")].est.iloc[0] * 1e4)
                                                   for st in strata])
            rows.append(["", r"$\Delta$-MAE \#2 (pts)"] + [ci(*s[(s.generator == g) & (s.stratum == st) & (s.quantity == "dmae_#2")][["est", "lo", "hi"]].iloc[0], 100, 2)
                                                           for st in strata])
        if name == "Pos-B":
            w(r"\subsection*{Pos-B --- Predicting position changes by $k$}")
        if name == "K-B":
            ka = r("eval2_append_by_slot")
            rows2 = []
            for (tr_, sl), x in ka.groupby(["transition", "slot"]):
                row = [tr_.replace("->", r"$\to$"), sl]
                for g in GENS:
                    y = x[x.generator == g].iloc[0]
                    row.append(ci(y.effect_pp, y.lo, y.hi, 1, 2))
                row.append(f"{int(x.n_prompt_pairs.iloc[0]):,} / {int(x.n_pools.iloc[0])}")
                rows2.append(row)
            w(r"\subsection*{K-A --- Observed append effect by retained slot and transition}")
            w(tab("K-A. Mean change in retained-rule violation risk (points) by transition and retained slot $j$ (its "
                  "absolute position, unchanged by the append); the new rule is excluded; 95\\% pool intervals. "
                  "Coverage: realised strict prefix pairs equal the design (1,980 / 7,920 / 11,880 / 3,960 prompt pairs; "
                  "1,980 / 15,840 / 35,640 / 15,840 retained targets); 3$\\to$4 and 4$\\to$5 cover only COA orders with an "
                  "exact prefix. Weights: retained targets equal within a pair, then pairs, tasks and pools.",
                  "tab:KA", ["Transition", "Slot $j$"] + by_gen_cols() + ["Pairs / pools"], rows2,
                  align="cc" + "c" * NG + "c", resize=False))
            w(r"\subsection*{K-B --- Predicting append changes by transition}")
        if name == "C-B":
            ko = r("eval2_companion_rms_k_order")
            rows2 = []
            for (k, ro), x in ko.groupby(["k", "relative_order"], sort=False):
                row = [k, ro]
                for g in GENS:
                    y = x[x.generator == g].iloc[0]
                    row.append(ci(y.rms_pp, y.lo, y.hi))
                row.append(f"{int(x.n_comparisons.iloc[0]):,}")
                rows2.append(row)
            w(r"\subsection*{C-A --- Companion effects: heterogeneity and numerical support}")
            w(tab("C-A. Noise-adjusted RMS companion-replacement sensitivity (points) by $k$ and whether the replaced "
                  "companion is earlier or later than the target (all target types). Signed target $\\times$ companion "
                  "cells (Figure~\\ref{fig:e2comp}B) with intervals, pools, tasks, contexts and alternatives: "
                  "\\texttt{tables/eval2\\_companion\\_signed\\_matrix.csv}.", "tab:CA",
                  ["$k$", "Companion"] + by_gen_cols() + ["Comparisons"], rows2, align="cc" + "c" * NG + "c", resize=False))
            m = r("eval2_companion_signed_matrix")
            txt = []
            for g in GENS:
                x = m[m.generator == g]
                y = x[x.n_pools >= 3]
                ex = int(((y.lo > 0) | (y.hi < 0)).sum())
                txt.append(f"{GL[g]}: {len(x)} of 210 directed off-diagonal cells have strict matches, {int((x.n_pools < 3).sum())} "
                           f"with fewer than 3 pools; {ex} of the {len(y)} cells with at least 3 pools have a pointwise "
                           f"interval excluding zero (about {0.05 * len(y):.0f} expected by chance if all were null).")
            w(" ".join(txt) + "\n")
            w(rms_detail("eval2_companion_rms_by_type", "C-A (by target type, Panel A numbers). Noise-adjusted RMS (points) "
                         "[95\\% interval], corrected squared change and binomial noise floor ($\\times10^4$), baseline risk (\\%) "
                         "and supporting pools.", "tab:CAt"))
            w(r"\subsection*{C-B --- Predicting companion changes by $k$}")
        if name == "D-C":
            tk = r("eval2_task_rms_type_k")
            rows2 = []
            for t in types(tk.type_id.unique()):
                row = [tt(t)]
                for g in GENS:
                    for k in (1, 2, 3, 4, 5):
                        x = tk[(tk.generator == g) & (tk.type_id == t) & (tk.k == k)]
                        row.append(ci(x.rms_pp.iloc[0], x.lo.iloc[0], x.hi.iloc[0]) if len(x) and x.status.iloc[0] == "ok" else "--")
                rows2.append(row)
            w(r"\subsection*{D-A --- Task sensitivity by target type and $k$}")
            w(tab("D-A. Noise-adjusted RMS task sensitivity (points) [95\\% interval] by target type and $k$; baseline rates, "
                  "corrected squares and support in \\texttt{tables/eval2\\_task\\_rms\\_type\\_k.csv}.", "tab:DA",
                  ["Type"] + [f"{GL[g][0]} $k{{=}}{k}$" for g in GENS for k in range(1, 6)], rows2))
            w(rms_detail("eval2_task_rms_by_type", "D-A (by target type, Panel A numbers). Noise-adjusted RMS (points) [95\\% "
                         "interval], corrected squared change and noise floor ($\\times10^4$), baseline risk (\\%) and pools.",
                         "tab:DAt"))
            w(r"\subsection*{D-B --- Rule type $\times$ task family}")
            w(fig("e2_taskDB_type_family_heatmap.png", r"D-B. Signed family-task minus mean of the other two matched tasks "
                  r"(points) by target type and family; hatched = unsupported. Descriptive associations; family is "
                  r"confounded with task text. Intervals and support: \texttt{tables/eval2\_task\_type\_family.csv}."))
            fpairs = r("eval2_task_family_pairs")
            rows3 = []
            for (f1, f2), x in fpairs.groupby(["family_1", "family_2"]):
                row = [f"{f1} $-$ {f2}"]
                for g in GENS:
                    y = x[x.generator == g]
                    row.append(ci(y.effect_pp.iloc[0], y.lo.iloc[0], y.hi.iloc[0], 1, 2) if len(y) else "--")
                row.append(int(x.n_pools.iloc[0]))
                rows3.append(row)
            if rows3:
                w(tab("D-B (family pairs). Within-pool directed family-pair differences (points), shown where at least "
                      "3 pools contain both families.", "tab:DBp", ["Family pair"] + by_gen_cols() + ["Pools"], rows3,
                      align="l" + "c" * NG + "c", resize=False))
            w(r"\subsection*{D-C --- Predicting task changes by $k$}")
        w(tab(f"{name}. Paired $\\Delta$-MSE reduction ($\\times10^4$) [95\\% interval] by {lab} on the test groups; "
              f"positive favours the candidate. Also: zero-change MSE and the $\\Delta$-MAE reduction (points) for "
              f"comparison 2; all $\\Delta$-MAE values: \\texttt{{tables/eval2\\_{fac}\\_skill.csv}}.", f"tab:{name}",
              ["", "Comparison"] + [f"{lab} {x.replace('->', '$\\to$')}" for x in strata], rows,
              align="ll" + "c" * len(strata), mid=mids))
        if name == "Pos-B":
            w(r"\subsection*{Pos-C --- Numerical position estimates and noise diagnostics}")
            w(rms_detail("eval2_position_rms_by_type", "Pos-C (Panel A numbers). Noise-adjusted RMS (points) [95\\% "
                         "interval], corrected squared change and binomial noise floor ($\\times10^4$), baseline risk (\\%) "
                         "and pools, by target type.", "tab:posC"))
            st = r("eval2_position_signed_type_k")
            rows2 = []
            for t in types(st.type_id.unique()):
                row = [tt(t)]
                for g in GENS:
                    for k in (2, 3, 4, 5):
                        x = st[(st.generator == g) & (st.type_id == t) & (st.k == k)]
                        row.append(ci(x.effect_pp.iloc[0], x.lo.iloc[0], x.hi.iloc[0]) + f" ({int(x.n_pools.iloc[0])})"
                                   if len(x) else "NA")
                rows2.append(row)
            w(tab("Pos-C (Panel B numbers). Signed last $-$ first change (points) [pointwise 95\\% interval] (supporting "
                  "pools) by target type and $k$.", "tab:posCs",
                  ["Type"] + [f"{GL[g][0]} $k{{=}}{k}$" for g in GENS for k in range(2, 6)], rows2))
    nv = r("noise_validation_synthetic")
    w(r"\subsection*{C-C and D-D --- Matching, weighting and validation}")
    w("Matched pairs are built from manifest metadata only, before outcomes are read, and their counts equal the design "
      "counts for every factor and $k$. Companion: same pool, task, $k$, target rule and slot, every other slot identical, "
      "exactly one companion differs (earlier and later replacements both included; 7,920 / 35,640 / 22,176 / 7,920 "
      "prompt pairs at $k=2..5$). Task: each pool's 456 configurations appear with its three tasks, which belong to three "
      "different families, giving three unordered task pairs per target; no within-family matched comparison exists. "
      "Weights: equal per $k$ or transition, then pool, task, context and pair (the plan's proposed recipe). The "
      "noise-adjusted RMS estimator and its 22-pool interval were checked by simulation before test scoring "
      f"(\\texttt{{tables/noise\\_validation\\_synthetic.csv}}, {len(nv)} scenarios): the corrected square is unbiased, "
      f"interval coverage is {100 * nv.coverage_rms_95.min():.0f}--{100 * nv.coverage_rms_95.max():.0f}\\%, and at an "
      "exact null the clipped root is biased upward by 0.5--1.1 points, so RMS values below about 1.5 points are not "
      "distinguishable from zero. Intervals are pointwise and unadjusted.\n")
    # Factor E
    e = r("factor_e_terciles")
    rows = []
    for gi, g in enumerate(GENS):
        for terc in ("easy", "medium", "hard"):
            x = e[(e.generator == g) & (e.tercile == terc)]
            rows.append([GL[g] if terc == "easy" else "", terc,
                         f"{100 * x.train_k1_rate.min():.1f}--{100 * x.train_k1_rate.max():.1f}",
                         f"{100 * x.test_rate.mean():.1f}", f(x.position_rms.median()), f(x.companion_rms.median()),
                         f(x.task_rms.median())])
    w(r"\subsection*{Factor E --- Training-defined difficulty terciles}")
    w(tab("Factor E. Types grouped by their $k=1$ violation rate on the 50 training groups (5 per tercile, frozen before "
          "test scoring); mean test violation rate (\\%) and median test noise-adjusted RMS (points) within tercile. "
          "Descriptive stratification, not a manipulated factor.", "tab:E",
          ["", "Tercile", "Train $k{=}1$ rate (\\%)", "Test rate", "Position", "Companion", "Task"], rows,
          align="llccccc", resize=False, mid=[3 * i for i in range(1, NG)]))
    return "\n".join(out)


def rms_detail(name, cap, label):
    df = r(name)
    rows = []
    for t in types(df.type_id.unique()):
        row = [tt(t)]
        for g in GENS:
            x = df[(df.generator == g) & (df.type_id == t)]
            if len(x) and x.status.iloc[0] == "ok":
                x = x.iloc[0]
                row += [ci(x.rms_pp, x.rms_lo, x.rms_hi), f(x.corrected_sq_x1e4), f(x.noise_floor_x1e4),
                        f(100 * x.baseline_risk), int(x.n_pools)]
            else:
                row += ["--"] * 5
        rows.append(row)
    head = ["Type"] + [f"{GL[g][0]} {c}" for g in GENS for c in ("RMS", "corr.", "noise", "base", "pools")]
    return tab(cap, label, head, rows)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    for k, v in main_tables().items():
        (OUT / f"test_{k}.tex").write_text(v + "\n")
    (OUT / "test_appendix.tex").write_text(appendix() + "\n")
    print("ok", OUT)


if __name__ == "__main__":
    run()
