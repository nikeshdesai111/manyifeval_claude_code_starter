"""PRELIMINARY S3 TRAIN: LaTeX for the paper laid out as the evaluation plan prescribes (main Tables 1-4, appendix
A1-A7, Pos-A/B/C, K-A/B, C-A/B/C, D-A/B/C/D, Factor E, Objective 2 Evaluation 2), built only from
results/preliminary_train_report/tables/*.csv. Writes results/preliminary_train_report/paper/plan_*.tex.
Run: PYTHONPATH=src .venv/bin/python -m mife.prelim_train.paper_plan
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .matches import FIXED_TYPES
from .paths import EVAL_DIR

T = EVAL_DIR / "tables"
OUT = EVAL_DIR / "paper"
ORDER = ["#1", "#2", "#3", "#4", "#5", "#6", "#7"]
CL = {"#1": r"1: B1 \cto{} H1", "#2": r"2: B2 \cto{} H2", "#3": r"3: B2 \cto{} A2", "#4": r"4: H2 \cto{} A2",
      "#5": r"5: B1 \cto{} B1+H", "#6": r"6: B2 \cto{} B2+H", "#7": r"7: B2 \cto{} B2+A"}
GL = {"llama": "Llama", "qwen": "Qwen"}
PL = {"B1": "B1", "B2": "B2", "H1": "H1+S", "H2": "H2+S", "RCAP": "A2", "B1+H1": "B1+H", "B2+H1": "B2+H",
      "B2+A1": "B2+A", "REF_structure": "Structure only", "REF_trainmean": "Training-fold mean",
      "AVG1_H2": r"Mean of Obj.~1 H2", "AVG1_B2": r"Mean of Obj.~1 B2", "AVG1_RCAP": r"Mean of Obj.~1 A2"}


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


def tab(caption, label, head, rows, align=None, size=r"\scriptsize", resize=True, mid=None):
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
    return "\n".join([r"\begin{table}[h]", r"\centering" + size, r"\caption{" + caption + "}",
                      (r"\label{" + label + "}") if label else "", t, r"\end{table}", ""])


# ------------------------------------------------------------------------------------------------- main
def main_tables():
    con, c2, sk = r("eval1_contrasts"), r("obj2_eval1_contrasts"), {x: r(f"eval2_{x}_skill") for x in
                                                                      ("position", "append", "companion", "task")}
    qc = json.loads((T / "eval1_qc.json").read_text())
    cov = json.loads((T / "match_coverage.json").read_text())
    # Table 1 inventory
    rows = [[GL[q["generator"]], q["pools"], q["tasks"], f"{q['prompts']:,}", f"{q['rows']:,}", f"{q['prompts'] * 20:,}",
             f"{100 * q['violations'] / q['outcomes']:.1f}\\%"] for q in qc]
    t1 = tab(r"\emph{Preliminary} evaluation inventory: S3 training groups (out-of-fold). Responses: 20 verified per "
             r"prompt, all present. Matched target contrasts per generator (metadata only): position "
             f"{cov['position_pairs']:,}, strict append {cov['append_targets']:,}, companion replacement "
             f"{cov['companion_targets']:,} ($k=2,3,5$; none at $k=4$ in the training design), task {cov['task_targets']:,}. "
             r"The 22-group test design gives 57,420, 69,300, 177,408 and 115,236.",
             "tab:inventory", ["Generator", "Pools", "Tasks", "Prompts", "Prompt--rule rows", "Responses", "Violation rate"],
             rows, resize=False, size=r"\footnotesize")
    # Table 2 Obj1
    rows = []
    for cid in ORDER:
        row = [CL[cid]]
        for g in ("llama", "qwen"):
            x = con[(con.generator == g) & (con.comparison == cid)].iloc[0]
            row += [ci(x.nll_reduction, x.nll_reduction_lo, x.nll_reduction_hi, 1, 4),
                    ci(x.auroc_gain, x.auroc_gain_lo, x.auroc_gain_hi, 100), f"{int(x.types_improved)}/15"]
        rows.append(row)
    t2 = tab(r"\emph{Preliminary} Evaluation~1, Objective~1: rule-level violation-risk prediction, out-of-fold on the "
             r"50 S3 training groups (153,900 prompt--rule rows and 3,078,000 response $\times$ rule outcomes per "
             r"generator; not held-out test performance). NLL reduction $=$ NLL(comparator) $-$ NLL(candidate), "
             r"in nats per outcome; AUROC gain $\times100$; Types $=$ rule types with strictly lower NLL "
             r"(descriptive). Positive favours the candidate. Paired, pointwise 95\% intervals from a 50-pool cluster "
             r"bootstrap. Primary comparisons above the rule, secondary below.",
             "tab:e1o1", ["Comparison", "NLL red.", "AUROC gain", "Types", "NLL red.", "AUROC gain", "Types"], rows,
             align="lcccccc", mid=[4], size=r"\footnotesize")
    t2 = t2.replace(r"\toprule", r"\toprule" + "\n" + r" & \multicolumn{3}{c}{Llama} & \multicolumn{3}{c}{Qwen} \\"
                    + "\n" + r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}", 1)
    # Table 3 Obj2
    rows = []
    for cid in ORDER:
        row = [CL[cid]]
        for g in ("llama", "qwen"):
            x = c2[(c2.generator == g) & (c2.comparison == cid)].iloc[0]
            row += [ci(x.rmse_reduction, x.rmse_reduction_lo, x.rmse_reduction_hi, 100, 2),
                    ci(x.spearman_gain, x.spearman_gain_lo, x.spearman_gain_hi, 100)]
        rows.append(row)
    t3 = tab(r"\emph{Preliminary} Evaluation~1, Objective~2: prompt-level expected fraction of rules violated, "
             r"out-of-fold on 36,900 training prompts per generator. RMSE reduction in percentage points; Spearman gain "
             r"$\times100$. Positive favours the candidate. Paired 95\% 50-pool bootstrap intervals (2,000 draws).",
             "tab:e1o2", ["Comparison", "RMSE red. (pp)", "Spearman gain", "RMSE red. (pp)", "Spearman gain"], rows,
             align="lcccc", mid=[4], size=r"\footnotesize")
    t3 = t3.replace(r"\toprule", r"\toprule" + "\n" + r" & \multicolumn{2}{c}{Llama} & \multicolumn{2}{c}{Qwen} \\"
                    + "\n" + r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}", 1)
    # Table 4
    rows, mids = [], []
    for g in ("llama", "qwen"):
        if g == "qwen":
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
    t4 = tab(r"\emph{Preliminary} Evaluation~2 predictor skill (out-of-fold, 50 training groups): paired $\Delta$-MSE "
             r"reduction ($\times10^{4}$) [95\% interval] of the predicted versus observed change in a rule's violation "
             r"probability, MSE(comparator) $-$ MSE(candidate); positive favours the candidate. Equal weight per $k$ "
             r"(position, companion, task) or transition (append), then pool, task and context. The zero-change row is "
             r"the MSE of predicting no change. By-stratum values and $\Delta$-MAE in Appendices Pos-B, K-B, C-B and D-C.",
             "tab:eval2", ["", "Comparison", "Position", "Strict append", "Companion", "Task"], rows, align="llcccc",
             mid=mids, size=r"\footnotesize")
    return {"t1": t1, "t2": t2, "t3": t3, "t4": t4}


# ------------------------------------------------------------------------------------------------- appendix
def appendix():
    out = []
    w = out.append
    a1, con = r("eval1_a1_metrics"), r("eval1_contrasts")
    rows, mids = [], []
    for g in ("llama", "qwen"):
        if g == "qwen":
            mids.append(len(rows))
        for p in ["REF_structure", "B1", "B2", "H1", "H2", "RCAP", "B1+H1", "B2+H1", "B2+A1"]:
            x = a1[(a1.generator == g) & (a1.predictor == p)].iloc[0]
            rows.append([GL[g], PL[p], ci(x.nll, x.nll_lo, x.nll_hi, 1, 4), ci(x.brier, x.brier_lo, x.brier_hi, 1, 4),
                         ci(x.auroc, x.auroc_lo, x.auroc_hi, 100), ci(x.ece, x.ece_lo, x.ece_hi, 100, 2)])
    w(r"\subsection*{A1 --- Complete overall metrics (Objective 1)}")
    w(tab("A1. Pooled out-of-fold scores, 50 training groups, with 95\\% pool-bootstrap intervals.", "tab:A1",
          ["", "Predictor", "NLL", "Brier", r"AUROC $\times100$", r"ECE $\times100$"], rows, align="llcccc", mid=mids))
    rows = []
    for g in ("llama", "qwen"):
        for cid in ORDER:
            x = con[(con.generator == g) & (con.comparison == cid)].iloc[0]
            rows.append([GL[g], CL[cid], ci(x.brier_reduction, x.brier_reduction_lo, x.brier_reduction_hi, 1, 4),
                         f"[{f(x.nll_t_lo, 4)}, {f(x.nll_t_hi, 4)}]",
                         f"{f(x.nll_loo_min, 4)} to {f(x.nll_loo_max, 4)}"])
    w(tab(r"A1 (contrasts) and A4 (sensitivity). Brier reduction; NLL reduction $t_{49}$ interval and "
          r"leave-one-pool-out range (nats).", "tab:A1c",
          ["", "Comparison", "Brier reduction", r"NLL $t_{49}$ interval", "NLL leave-one-pool-out"], rows, align="llccc", mid=[7]))
    w(r"\subsection*{A2 --- Calibration (Objective 1)}")
    w(r"\begin{figure}[h]\centering\includegraphics[width=\linewidth]{figures/prelim/e1_A2_reliability.png}"
      r"\caption{A2. Reliability curves (out-of-fold, decile bins of each predictor's own predictions).}\end{figure}")
    w(r"\subsection*{A3 --- Per-type NLL reduction}")
    w(r"\begin{figure}[h]\centering\includegraphics[width=\linewidth]{figures/prelim/e1_A3_per_type_nll_heatmap.png}"
      r"\caption{A3. Per-type NLL reduction for the seven comparisons; five fixed-wording types above the line. All 210 "
      r"values with paired intervals and support: \texttt{tables/eval1\_per\_type.csv}.}\end{figure}")
    fp = r("eval1_fixed_vs_param")
    rows, mids = [], []
    for g in ("llama", "qwen"):
        if g == "qwen":
            mids.append(len(rows))
        for cid in ORDER:
            x = fp[(fp.generator == g) & (fp.comparison == cid)].iloc[0]
            rows.append([GL[g] if cid == "#1" else "", CL[cid], ci(x.fixed_dnll, x.fixed_lo, x.fixed_hi, 1, 4),
                         ci(x.parameterized_dnll, x.parameterized_lo, x.parameterized_hi, 1, 4),
                         ci(x["diff_dnll"], x.diff_lo, x.diff_hi, 1, 4)])
    rates = "; ".join(f"{GL[g]}: fixed {100 * fp[fp.generator == g].fixed_viol_rate.iloc[0]:.1f}\\%, parameterised "
                      f"{100 * fp[fp.generator == g].parameterized_viol_rate.iloc[0]:.1f}\\%" for g in ("llama", "qwen"))
    w(r"\subsection*{A3b --- Fixed versus parameterised rule types (Objective 1 only)}")
    w(tab(r"A3b. NLL reductions (nats) of the seven comparisons within the five fixed and ten parameterised types, "
          r"and their difference (fixed $-$ parameterised improvement); identical 50-pool resamples; exploratory. "
          f"Observed violation rates: {rates}.", "tab:A3b",
          ["", "Comparison", "Fixed $\\Delta$NLL", "Parameterised $\\Delta$NLL", "Fixed $-$ param."], rows, align="llccc", mid=mids))
    qc = json.loads((T / "eval1_qc.json").read_text())
    w(r"\subsection*{A4 --- Technical uncertainty and QC}")
    w("Probabilities are clipped to $[10^{-6}, 1-10^{-6}]$ (rows clipped, Llama/Qwen: " + "; ".join(
        f"{p} {qc[0]['clipped_rows'][p]}/{qc[1]['clipped_rows'][p]}" for p in qc[0]["clipped_rows"]) +
      "). Every prompt has 20 verified responses. For every row and predictor the stored out-of-fold fold equals the fold "
      "of the row's pool, so each prediction comes from a model fitted on the other 40 pools. Bootstrap: 10,000 draws, "
      "seed 20261203, identical for all predictors and contrasts. Sensitivity intervals are in Table~\\ref{tab:A1c}.\n")
    # A5-A7
    s2, c2, cal2 = r("obj2_eval1_scores"), r("obj2_eval1_contrasts"), r("obj2_eval1_calibration")
    rows, mids = [], []
    for g in ("llama", "qwen"):
        if g == "qwen":
            mids.append(len(rows))
        for p in ["B1", "B2", "H1", "H2", "RCAP", "B1+H1", "B2+H1", "B2+A1", "REF_structure"]:
            x = s2[(s2.generator == g) & (s2.predictor == p)].iloc[0]
            rows.append([GL[g], PL[p], f(x.rmse * 100, 2), f(x.mae * 100, 2), f(x.mse * 1e4), f(x.pearson * 100),
                         f(x.spearman * 100), f(x.ece * 100, 2)])
    w(r"\subsection*{A5 --- Complete overall metrics (Objective 2)}")
    w(tab("A5. Out-of-fold prompt-level scores (36,900 prompts). RMSE, MAE and ECE in points; MSE $\\times10^4$; "
          "correlations $\\times100$. Intervals: \\texttt{tables/obj2\\_eval1\\_scores.csv}.", "tab:A5",
          ["", "Predictor", "RMSE", "MAE", "MSE", "Pearson", "Spearman", "ECE"], rows, align="llcccccc", mid=mids))
    rows = []
    for g in ("llama", "qwen"):
        for cid in ORDER:
            x = c2[(c2.generator == g) & (c2.comparison == cid)].iloc[0]
            rows.append([GL[g], CL[cid], ci(x.mae_reduction, x.mae_reduction_lo, x.mae_reduction_hi, 100, 2),
                         ci(x.mse_reduction, x.mse_reduction_lo, x.mse_reduction_hi, 1e4),
                         ci(x.pearson_gain, x.pearson_gain_lo, x.pearson_gain_hi, 100)])
    w(tab("A5 (contrasts). MAE reduction (points), MSE reduction ($\\times10^4$) and Pearson gain ($\\times100$).", "tab:A5c",
          ["", "Comparison", "MAE reduction", "MSE reduction", "Pearson gain"], rows, align="llccc", mid=[7]))
    rows = []
    for g in ("llama", "qwen"):
        for p in ["B2", "H2", "RCAP"]:
            x = cal2[(cal2.generator == g) & (cal2.predictor == p)]
            rows.append([GL[g], PL[p], ", ".join(f"{100 * a:.0f}/{100 * b:.0f}" for a, b in zip(x.mean_pred, x.mean_obs))])
    w(r"\subsection*{A6 --- Fraction calibration (Objective 2)}")
    w(tab("A6. Mean predicted / mean observed fraction violated (points) in decile bins of each predictor's out-of-fold "
          "predictions ($\\approx$3,690 prompts per bin). All predictors: \\texttt{tables/obj2\\_eval1\\_calibration.csv}.",
          "tab:A6", ["", "Predictor", "Bins 1--10: predicted/observed"], rows, align="llp{0.62\\linewidth}", resize=False))
    rows = []
    for g in ("llama", "qwen"):
        for p in ["REF_trainmean", "AVG1_B2", "AVG1_H2", "AVG1_RCAP", "H2"]:
            x = s2[(s2.generator == g) & (s2.predictor == p)].iloc[0]
            rows.append([GL[g], PL.get(p, "Obj.~2 head H2+S") if p != "H2" else "Obj.~2 head H2+S", f(x.rmse * 100, 2),
                         "undefined (constant per fold)" if p == "REF_trainmean" else f(x.spearman * 100)])
    w(r"\subsection*{A7 --- Reference predictions (Objective 2)}")
    w(tab("A7. Training-fold mean and averaged Objective~1 predictions as references (not among the seven comparisons).",
          "tab:A7", ["", "Reference", "RMSE (pp)", r"Spearman $\times100$"], rows, align="llcc", resize=False, mid=[5]))
    # ---- Position
    sk = {x: r(f"eval2_{x}_skill") for x in ("position", "append", "companion", "task")}
    w(r"\subsection*{Pos-A --- Position profiles and all 20 pair strata}")
    w(r"\begin{figure}[h]\centering\includegraphics[width=\linewidth]{figures/prelim/e2_posA_slot_curves.png}"
      r"\caption{Pos-A. Observed violation risk at every slot, by $k$ (COA-balanced, 95\% pool intervals).}\end{figure}")
    ps = r("eval2_position_pair_strata")
    rows = []
    for (k, i, j), x in ps.groupby(["k", "i", "j"]):
        row = [k, f"{i}$\\to${j}"]
        for g in ("llama", "qwen"):
            y = x[x.generator == g].iloc[0]
            row.append(ci(y.effect_pp, y.lo, y.hi, 1, 2))
        row.append(int(x.n_contexts.iloc[0]))
        rows.append(row)
    w(tab("Pos-A. Signed change (later slot $-$ earlier slot, points) for all 20 position-pair strata, 50 pools.",
          "tab:posA", ["$k$", "Slots", "Llama", "Qwen", "Contexts"], rows, align="ccccc", resize=False))
    for fac, name, lab in (("position", "Pos-B", "$k$"), ("append", "K-B", "transition"), ("companion", "C-B", "$k$"),
                           ("task", "D-C", "$k$")):
        s = sk[fac]
        strata = sorted(x for x in s.stratum.unique() if x != "macro")
        rows, mids = [], []
        for g in ("llama", "qwen"):
            if g == "qwen":
                mids.append(len(rows))
            for cid in ORDER:
                row = [GL[g] if cid == "#1" else "", CL[cid]]
                for st in strata:
                    x = s[(s.generator == g) & (s.stratum == st) & (s.quantity == f"dmse_{cid}")].iloc[0]
                    row.append(ci(x.est, x.lo, x.hi, 1e4))
                rows.append(row)
            for q, ql, sc in (("se_zero", "zero-change MSE", 1e4),):
                rows.append(["", ql] + [f(s[(s.generator == g) & (s.stratum == st) & (s.quantity == q)].est.iloc[0] * sc)
                                        for st in strata])
        if name == "Pos-B":
            w(r"\subsection*{Pos-B --- Predicting position changes by $k$}")
        if name == "K-B":
            ka = r("eval2_append_by_slot")
            rows2 = []
            for (tr_, sl), x in ka.groupby(["transition", "slot"]):
                row = [tr_.replace("->", r"$\to$"), sl]
                for g in ("llama", "qwen"):
                    y = x[x.generator == g].iloc[0]
                    row.append(ci(y.effect_pp, y.lo, y.hi, 1, 2))
                row.append(f"{int(x.n_targets.iloc[0]):,}")
                rows2.append(row)
            w(r"\subsection*{K-A --- Observed append effect by retained slot and transition}")
            w(tab("K-A. Mean change in retained-rule violation risk (points) by transition and retained slot; the new rule "
                  "is excluded. 1$\\to$2 and 2$\\to$3 cover every chain prefix; 3$\\to$4 and 4$\\to$5 only COA orders with an "
                  "exact prefix.", "tab:KA", ["Transition", "Slot", "Llama", "Qwen", "Targets"], rows2, align="ccccc", resize=False))
            w(r"\subsection*{K-B --- Predicting append changes by transition}")
        if name == "C-B":
            ko = r("eval2_companion_rms_k_order")
            rows2 = []
            for (k, ro), x in ko.groupby(["k", "relative_order"], sort=False):
                row = [k, ro]
                for g in ("llama", "qwen"):
                    y = x[x.generator == g].iloc[0]
                    row.append(ci(y.rms_pp, y.lo, y.hi))
                row.append(f"{int(x.n_comparisons.iloc[0]):,}")
                rows2.append(row)
            w(r"\subsection*{C-A --- Companion effects by $k$ and placement; numerical support}")
            w(tab("C-A. Noise-adjusted RMS companion-replacement sensitivity (points) by $k$ and whether the replaced "
                  "companion is earlier or later than the target. Signed target $\\times$ companion cells with intervals "
                  "and support: \\texttt{tables/eval2\\_companion\\_signed\\_matrix.csv}.", "tab:CA",
                  ["$k$", "Companion", "Llama", "Qwen", "Comparisons"], rows2, align="ccccc", resize=False))
            w(rms_detail("eval2_companion_rms_by_type", "C-A (by target type). Corrected and raw squared changes, binomial "
                         "noise floor ($\\times10^4$), baseline risk and support.", "tab:CAt"))
            w(r"\subsection*{C-B --- Predicting companion changes by $k$}")
        if name == "D-C":
            tk = r("eval2_task_rms_type_k")
            rows2 = []
            for t in types(tk.type_id.unique()):
                row = [tt(t)]
                for g in ("llama", "qwen"):
                    for k in (1, 2, 3, 4, 5):
                        x = tk[(tk.generator == g) & (tk.type_id == t) & (tk.k == k)]
                        row.append(f(x.rms_pp.iloc[0]) if len(x) and x.status.iloc[0] == "ok" else "--")
                rows2.append(row)
            w(r"\subsection*{D-A --- Task sensitivity by target type and $k$}")
            w(tab("D-A. Noise-adjusted RMS task sensitivity (points) by type and $k$ (intervals in "
                  "\\texttt{tables/eval2\\_task\\_rms\\_type\\_k.csv}).", "tab:DA",
                  ["Type"] + [f"L $k{{=}}{k}$" for k in range(1, 6)] + [f"Q $k{{=}}{k}$" for k in range(1, 6)], rows2))
            w(r"\subsection*{D-B --- Rule type $\times$ task family}")
            w(r"\begin{figure}[h]\centering\includegraphics[width=\linewidth]{figures/prelim/e2_taskDB_type_family_heatmap.png}"
              r"\caption{D-B. Signed family-task minus mean of the other two matched tasks (points), by target type and "
              r"family. Descriptive associations; family is confounded with task text.}\end{figure}")
            w(r"\subsection*{D-C --- Predicting task changes by $k$}")
        w(tab(f"{name}. Paired $\\Delta$-MSE reduction ($\\times10^4$) [95\\% interval] by {lab}; positive favours the "
              f"candidate. $\\Delta$-MAE: \\texttt{{tables/eval2\\_{fac}\\_skill.csv}}.", f"tab:{name}",
              ["", "Comparison"] + [f"{lab} {x.replace('->', '$\\to$')}" for x in strata], rows, align="ll" + "c" * len(strata), mid=mids))
        if name == "Pos-B":
            w(r"\subsection*{Pos-C --- Numerical position estimates and noise diagnostics}")
            w(rms_detail("eval2_position_rms_by_type", "Pos-C. Position: corrected and raw squared changes, binomial noise "
                         "floor ($\\times10^4$), baseline risk and support by target type. Signed type $\\times k$ values "
                         "with intervals: \\texttt{tables/eval2\\_position\\_signed\\_type\\_k.csv}.", "tab:posC"))
    w(r"\subsection*{C-C and D-D --- Matching, weighting and validation}")
    w("Matched pairs are built from manifest metadata only. Companion: same pool, task, $k$, target rule and slot, every "
      "other slot identical, one companion differs, earlier and later replacements both included; the training design "
      "yields such pairs only at $k=2,3,5$. Task: each pool's 246 configurations appear with its three tasks (one per "
      "family), giving three unordered task pairs per target. Weights: equal per $k$ or transition, then pool, task, "
      "context and pair. The noise-adjusted RMS estimator and its 50-pool interval were checked by simulation "
      "(\\texttt{tables/noise\\_validation\\_synthetic.csv}): mean estimate within 0.1 points of the true RMS whenever the "
      "true RMS is at least 3 points, an upward bias of 0.5--1 point at an exact null (truncation at zero), and 91--100\\% "
      "coverage. Intervals are pointwise and unadjusted.\n")
    w(rms_detail("eval2_task_rms_by_type", "D-D. Task: corrected and raw squared changes, noise floor, baseline risk and "
                 "support by target type.", "tab:DD"))
    # Factor E
    try:
        from ..probing import data as pdata
        pr, cr, tr = r("eval2_position_rms_by_type"), r("eval2_companion_rms_by_type"), r("eval2_task_rms_by_type")
        rows = []
        for g in ("llama", "qwen"):
            Lb = pdata.labels_prompt_rule(g, "train").merge(pdata.prompts("train")[["prompt_id", "k"]], on="prompt_id")
            Lb = Lb[Lb.k == 1]
            rate = (Lb.groupby("type_id").v.sum() / Lb.groupby("type_id").n.sum()).rename("rate").reset_index()
            rate = rate.sort_values(["rate", "type_id"]).reset_index(drop=True)
            rate["terc"] = ["easy"] * 5 + ["medium"] * 5 + ["hard"] * 5
            for terc, x in rate.groupby("terc", sort=False):
                ids = set(x.type_id)
                row = [GL[g], terc, f"{100 * x.rate.min():.1f}--{100 * x.rate.max():.1f}"]
                for df in (pr, cr, tr):
                    d = df[(df.generator == g) & df.type_id.isin(ids)]
                    row.append(f(d.rms_pp.median()))
                rows.append(row)
        w(r"\subsection*{Factor E --- Training-defined difficulty terciles}")
        w(tab("Factor E. Types grouped by their $k=1$ training violation rate (5 per tercile); median noise-adjusted RMS "
              "(points) within tercile. On training data the terciles are defined on the data they stratify.", "tab:E",
              ["", "Tercile", "$k{=}1$ rate (\\%)", "Position", "Companion", "Task"], rows, align="llcccc", resize=False, mid=[3]))
    except Exception as e:  # pragma: no cover
        w(f"% Factor E not computed: {e}")
    # Obj2 Eval2
    e, sup = r("obj2_eval2"), r("obj2_eval2_support")
    rows, mids = [], []
    for g in ("llama", "qwen"):
        if g == "qwen":
            mids.append(len(rows))
        for fac in ("position", "append", "companion", "task"):
            x = e[(e.generator == g) & (e.factor == fac) & (e.stratum == "macro")].set_index("quantity")
            q = lambda n, s=1e4, d=1: ci(x.loc[n, "est"], x.loc[n, "lo"], x.loc[n, "hi"], s, d)
            rows.append([GL[g] if fac == "position" else "", fac,
                         f"{int(sup[(sup.generator == g) & (sup.factor == fac)].n_prompt_pairs.iloc[0]):,}",
                         q("rms_pp_noise_adjusted", 1), q("dobs", 100, 2), f(x.loc["se_zero", "est"] * 1e4),
                         q("dmse_#1"), q("dmse_#2"), q("dmse_#3"), q("dmse_#4")])
    w(r"\subsection*{Evaluation 2, Objective 2 (preliminary design, not fixed by the plan)}")
    w(tab("Objective~2 matched changes: prompt-level pairs, outcome = fraction of rules violated. Observed noise-adjusted RMS "
          "and signed mean (points; append includes the new rule, so the denominator changes); zero-change MSE and "
          "$\\Delta$-MSE ($\\times10^4$) for the primary comparisons. Secondary comparisons: \\texttt{tables/obj2\\_eval2.csv}.",
          "tab:o2e2", ["", "Factor", "Pairs", "RMS", "Signed mean", "Zero MSE", "1", "2", "3", "4"], rows,
          align="llcccccccc", mid=mids))
    return "\n".join(out)


def rms_detail(name, cap, label):
    df = r(name)
    rows = []
    for t in types(df.type_id.unique()):
        row = [tt(t)]
        for g in ("llama", "qwen"):
            x = df[(df.generator == g) & (df.type_id == t)]
            if len(x) and x.status.iloc[0] == "ok":
                x = x.iloc[0]
                row += [ci(x.rms_pp, x.rms_lo, x.rms_hi), f(x.corrected_sq_x1e4), f(x.noise_floor_x1e4),
                        f(100 * x.baseline_risk), int(x.n_pools)]
            else:
                row += ["--"] * 5
        rows.append(row)
    t = tab(cap, label, ["Type", "L RMS", "L corr.", "L noise", "L base", "L pools", "Q RMS", "Q corr.", "Q noise",
                         "Q base", "Q pools"], rows)
    return t


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    for k, v in main_tables().items():
        (OUT / f"plan_{k}.tex").write_text(v + "\n")
    (OUT / "plan_appendix.tex").write_text(appendix() + "\n")
    print("ok")


if __name__ == "__main__":
    run()
