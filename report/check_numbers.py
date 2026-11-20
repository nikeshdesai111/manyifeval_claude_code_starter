"""Check the numbers written in the prose of report/paper.tex against the saved tables.

1. Every decimal or percentage in the prose must equal (after rounding to the precision written) a value in
   results/tables/*.csv or report/tables/*.csv, a value derived below from those tables, or a listed design constant.
2. Derived statements (relative reductions, counts) are recomputed here and asserted.

Run: .venv/bin/python report/check_numbers.py
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T, R = ROOT / "results" / "tables", ROOT / "report" / "tables"

# ---------------------------------------------------------------------------------------------- pool of table values
pool: list[float] = []
for d in (T, R):
    for f in sorted(d.glob("*.csv")):
        df = pd.read_csv(f)
        for c in df.select_dtypes("number"):
            pool += df[c].dropna().abs().tolist()
pool_arr = np.array(sorted(set(pool)))

S = pd.read_csv(T / "cv_selected.csv")
C = pd.read_csv(R / "oof_contrasts.csv")
M = pd.read_csv(R / "oof_metrics.csv")
PT = pd.read_csv(R / "oof_per_type.csv")
PD_ = pd.read_csv(R / "oof_per_type_diff.csv")


def cv(m, o, c):
    return float(S[(S.model == m) & (S.objective == o) & (S.config == c)].cv_loss.iloc[0])


derived: dict[str, float] = {}
for m in ("llama", "qwen"):
    for o in (1, 2):
        d = S[(S.model == m) & (S.objective == o)].set_index("config")
        c0 = d.cv_loss["REF_constant"]
        po = d[d.family == "probe-only"].cv_loss.min()
        best = d[d.family != "reference"].cv_loss.min()
        for name, v in (("type_k", d.cv_loss["REF_type_k"]), ("structure", d.cv_loss["REF_structure"])):
            derived[f"{m}{o} skill {name}"] = 100 * (1 - v / c0)
        for a in ("REF_structure", "B1", "B2"):
            derived[f"{m}{o} best probe-only vs {a}"] = 100 * (1 - po / d.cv_loss[a])
        derived[f"{m}{o} best vs best probe-only"] = 100 * (1 - best / po)
        derived[f"{m}{o} B1 vs structure"] = 100 * (d.cv_loss["B1"] / d.cv_loss["REF_structure"] - 1)
        strongest = min(d.cv_loss[a] for a in ("REF_structure", "B1", "B2"))
        derived[f"{m}{o} best probe-only vs strongest comparator"] = 100 * (1 - po / strongest)
        for c in d.index:  # MSE in units of 1e-2
            if o == 2:
                derived[f"{m}{o} {c} mse x100"] = 100 * d.cv_loss[c]
Cmp = pd.read_csv(T / "cv_complementarity.csv")
for r in Cmp.itertuples():
    derived[f"comp {r.model}{r.objective} {r.hidden}"] = abs(100 * r.combined_minus_better_single / min(r.cv_loss_hidden, r.cv_loss_A1))
for (m, c), g in PT.groupby(["model", "config"]):
    derived[f"macro auroc {m} {c}"] = g.auroc.mean()
E = pd.read_csv(T / "cv_l2_edges.csv")
for r in E[E.config == "REF_structure"].itertuples():
    derived[f"l2 gap {r.model}{r.objective}"] = r.gap_pct
derived["se phat max"] = float(np.sqrt(0.25 / 20))
A = pd.read_csv(T / "audit_ec_langdetect_only_assessment.csv")
for m, g in A.groupby("model"):
    derived[f"ec lower en {m}"] = g.langdetect_on_lowercased_is_en.sum()
    derived[f"ec loops {m}"] = g.repetition_loop.sum()
derived_arr = np.array(list(derived.values()))

# design constants and counts stated in the text (manifest, configuration, plan); not outcome data
CONST = {
    216, 72, 15, 57, 53136, 2.1, 20, 23, 37, 8, 3, 6, 5, 2160, 432, 246, 1026, 738, 3078, 171, 45, 42066, 11070, 175446,
    46170, 221616, 4920, 14760, 841320, 221400, 1062720, 20000, 50, 90, 0.6, 0.9, 0.7, 0.8, 2048, 4096, 32, 36, 1024,
    1152, 0.998, 75, 175, 128, 0.01, 0.1, 1, 30, 16, 80, 12, 11, 2000, 10000, 0.001, 10, 95, 0.05, 22950, 2700, 25380,
    1080, 7560, 500, 13, 100, 4, 7, 9, 0.03, 2, 14, 1000, 256, 2026, 2025, 2024, 2023, 2022, 2021, 2019, 2017, 2016,
    2010, 2000, 1979, 4.4, 2.2, 22, 3.1, 0.0004,
}


def ok(x: float, dec: int) -> bool:
    tol = 0.5 * 10 ** (-dec) + 1e-12
    for arr in (pool_arr, derived_arr):
        for scale in (1.0, 100.0):
            if np.any(np.abs(arr * scale - x) <= tol):
                return True
    return any(abs(c - x) <= 1e-9 for c in CONST)


# ---------------------------------------------------------------------------------------------- prose numbers
tex = (ROOT / "report" / "paper.tex").read_text()
body = tex[tex.index("\\begin{abstract}"):tex.index("\\bibliographystyle")] + tex[tex.index("\\appendix"):]
body = re.sub(r"(?m)^%.*$", "", body)
body = re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", "", body, flags=re.S)
body = re.sub(r"\\(label|ref|input|includegraphics(\[[^\]]*\])?|citep|citet|usepackage|vspace|setlength)\{[^}]*\}", "", body)
body = re.sub(r"\\begin\{(tabular|table|figure|quote|description|itemize)\}(\[[^\]]*\])?(\{[^}]*\})?", "", body)
body = re.sub(r"p\{[\d.]+cm\}|\[\d+pt\]|width=[\d.]+\\linewidth|\{[\d.]+em\}", "", body)
nums = re.findall(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+\.\d+|\d+)(?![\w])", body)
bad = []
for s in nums:
    x = float(s.replace(",", ""))
    dec = len(s.split(".")[1]) if "." in s else 0
    if not ok(x, dec):
        bad.append(s)
print(f"prose numbers checked: {len(nums)}; unmatched: {sorted(set(bad))}")

# ---------------------------------------------------------------------------------------------- explicit statements
def has(text: str) -> None:
    assert re.sub(r"\s+", " ", text) in re.sub(r"\s+", " ", tex), f"statement not found in paper: {text}"


def close(a: float, b: float, tol: float) -> None:
    assert abs(a - b) <= tol, (a, b)


close(derived["llama1 skill type_k"], 11.8, 0.05); close(derived["qwen1 skill type_k"], 15.3, 0.05)
close(derived["llama1 skill structure"], 19.7, 0.05); close(derived["qwen1 skill structure"], 21.0, 0.05)
has("by 11.8\\% for Llama and 15.3\\% for Qwen, and the structure head by 19.7\\% and 21.0\\%")
close(derived["llama1 best probe-only vs REF_structure"], 12.3, 0.05); close(derived["qwen1 best probe-only vs REF_structure"], 8.4, 0.05)
close(derived["llama1 best probe-only vs B2"], 10.4, 0.05); close(derived["qwen1 best probe-only vs B2"], 9.5, 0.05)
has("lowers the loss of the structure head by 12.3\\% (Llama) and 8.4\\% (Qwen), and that of B2 by 10.4\\% and 9.5\\%")
close(derived["qwen1 best vs best probe-only"], 1.4, 0.05); has("1.4\\% below the best probe-only configuration")
close(derived["llama2 B1 vs structure"], -0.5, 0.05); has("only 0.5\\% lower in Objective~2")
v = sorted(derived[f"{m}{o} best probe-only vs strongest comparator"] for m in ("llama", "qwen") for o in (1, 2))
assert 1.5 < v[0] < 2 and 8 <= v[1] and v[3] <= 14, v
has("by 8 to 14\\% in three of the four")
for m in ("llama", "qwen"):  # structure head never below the grid's lowest value
    assert (E[(E.model == m) & (E.config == "REF_structure")].edge == "lowest l2").all()
g = [derived[f"l2 gap {m}{o}"] for m in ("llama", "qwen") for o in (1, 2)]
close(min(g), 7.9, 0.05); close(max(g), 11.1, 0.05); has("is 7.9 to 11.1\\% higher")
for m, n_b2, s_b2, n_rs, s_rs in (("llama", 13, 8, 11, 8), ("qwen", 12, 5, 14, 5)):
    for a, n, sg in (("B2", n_b2, s_b2), ("REF_structure", n_rs, s_rs)):
        d = PD_[(PD_.model == m) & (PD_.a == a) & (PD_.b == "A1")]
        assert (d["diff"] < 0).sum() == n and (d.hi < 0).sum() == sg, (m, a, (d["diff"] < 0).sum(), (d.hi < 0).sum())
has("on 13 of the 15 types (with an interval excluding zero on 8) and a lower loss than the structure head on 11 (8). "
    "For Qwen the counts are 12 (5) and 14 (5)")
top = PD_[(PD_.model == "llama") & (PD_.a == "REF_structure") & (PD_.b == "A1")].sort_values("diff").type_id.head(5).tolist()
assert [t.split(":")[1] for t in top] == ["letter_frequency", "number_paragraphs", "number_bullet_lists", "title", "quotation"], top
for m in ("llama", "qwen"):  # letter count: B2 lowest, interval vs A1 includes 0
    x = PT[(PT.model == m) & (PT.type_id == "keywords:letter_frequency")].set_index("config").nll
    assert x["B2"] == x[["REF_structure", "B1", "B2", "H1", "H2", "A1"]].min()
    d = PD_[(PD_.model == m) & (PD_.a == "B2") & (PD_.b == "A1") & (PD_.type_id == "keywords:letter_frequency")].iloc[0]
    assert d.lo < 0 < d.hi
close(derived["macro auroc llama A1"], 0.73, 0.005); close(derived["macro auroc llama REF_structure"], 0.63, 0.005)
close(derived["macro auroc llama B2"], 0.59, 0.005); close(derived["macro auroc qwen A1"], 0.56, 0.005)
close(derived["macro auroc qwen REF_structure"], 0.54, 0.005); close(derived["macro auroc qwen B2"], 0.58, 0.005)
assert (PT[(PT.model == "qwen") & (PT.config == "A1")].violation_rate < 0.01).sum() == 3
ece = M[(M.objective == 1) & (M.metric == "ece")]
show = {"llama": ["REF_type_k", "REF_structure", "B1", "B2", "H1", "H2", "A1"],
        "qwen": ["REF_type_k", "REF_structure", "B1", "B2", "H1", "H2", "A1", "H1+H2+A1", "B2+H1+H2+A1"]}
for m, cs in show.items():
    assert ece[(ece.model == m) & ece.config.isin(cs)].est.max() < 0.03
# single-component contrasts quoted in the text
def con(m, o, a, b):
    return C[(C.model == m) & (C.objective == o) & (C.a == a) & (C.b == b)].iloc[0]
for m in ("llama", "qwen"):
    for a in ("REF_structure", "B2"):
        r = con(m, 1, a, "H1"); assert r.lo < 0 < r.hi
    r = con(m, 1, "B2", "A1"); assert r.hi < 0 and r.folds_b_lower == 5
    r = con(m, 1, "B2", "H2"); assert r.hi < 0
    r = con(m, 2, "H1", "H2"); assert r.lo < 0 < r.hi
    for a in ("B1", "B2"):
        for b in (f"{a}+H1", f"{a}+H2", f"{a}+A1"):
            assert con(m, 1, a, b).hi < 0
    assert con(m, 2, "B1", "B1+H1").hi < 0 and con(m, 2, "B1", "B1+H2").hi < 0
    for b in ("B2+H1", "B2+H2", "B2+A1"):
        r = con(m, 2, "B2", b); assert r.lo < 0 < r.hi
for b in ("H1", "H2", "A1"):
    r = con("llama", 2, "B2", b); assert r.lo < 0 < r.hi
r = con("qwen", 1, "B1", "B2"); assert r.lo < 0 < r.hi
r = con("qwen", 2, "B1", "B2"); assert r.lo < 0 < r.hi
# counts below the structure head, by family, Qwen Objective 2
d = S[(S.model == "qwen") & (S.objective == 2)].set_index("config")
below = d[d.cv_loss < d.cv_loss["REF_structure"]]
assert (below.family == "probe-only").sum() == 7 and (below.family == "B1 family").sum() == 6 and (below.family == "B2 family").sum() == 0
# layer selection
Ls = pd.read_csv(T / "cv_layer_selection.csv")
assert (Ls[Ls.selected].depth == "50%").sum() == 3
assert all(g.sort_values("best_cv_loss").depth.iloc[-1].startswith("100%") for _, g in Ls.groupby(["model", "objective"]))
assert pd.read_csv(T / "cv_pooling_selection.csv").query("selected").pooling.eq("sum").all()
assert derived["ec lower en qwen"] == 9839 and derived["ec lower en llama"] == 1263 and derived["ec loops qwen"] == 3520
print("explicit statements: all assertions passed")
