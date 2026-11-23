"""Fill the V2 CV point cache in parallel pieces (cache only; cv.run makes every selection, expansion and refit).

Usage: prewarm_s3.py MODEL OBJ PART
  PART = h1rows     H1 + structure at every candidate row (base grid)
         xh1rows    other generator's H1 + structure at every candidate row (base grid)
         text       structure-only, B1 and B2 (base grids)
         hdep       after h1rows and text: H2 at the selected row; B1+H1 and B2+H1 against the base-round selections
         attn       after the max-attention extraction and text: RCAP (base grid); B2+A1 against B2's base selection
"""
import sys

from mife.probing.v2 import cv

m, o, part = sys.argv[1], int(sys.argv[2]), sys.argv[3]
E = cv.Engine(m, o)
R = E.R
g = {"P": list(R["prompt_lambda_grid"]), "R": list(R["rule_lambda_grid"])}
gp, gpr = cv._grid(g, ("P",)), cv._grid(g, ("P", "R"))
rows = R["hidden_rows"][m]
xrows = R["hidden_rows"][cv.OTHER[m]]
S = cv.make_specs(m, o, None, None)


def h1(r, x=False):
    return cv.Spec("XH1" if x else "H1", ((f"{'X' if x else ''}H{r}", "P"), ("S", "S")))


def best_row(rs, x=False):
    res = {r: E.select(h1(r, x), gp)["loss"] for r in rs}
    return min(rs, key=lambda r: (round(res[r], E.prec), rs.index(r)))


if part == "h1rows":
    for r in rows:
        E.evaluate(h1(r), gp)
elif part == "xh1rows":
    for r in xrows:
        E.evaluate(h1(r, True), gp)
elif part == "text":
    E.evaluate(S["REF_structure"], [{}])
    E.evaluate(S["B1"], gp)
    E.evaluate(S["B2"], gpr)
elif part == "hdep":
    row = best_row(rows)
    Sr = cv.make_specs(m, o, row, None)
    E.evaluate(Sr["H2"], gpr)
    for name, base, gb in (("B1+H1", "B1", gp), ("B2+H1", "B2", gpr)):
        sel = E.select(S[base], gb)["setting"]
        E.evaluate(cv.incremental_spec(name, S[base], sel, f"H{row}"), [{"X": float(v)} for v in g["P"]])
elif part.startswith("h2r"):  # H2 at the selected row, only for the R values listed: h2r=100,1000
    row = best_row(rows)
    rv = [float(x) for x in part.split("=")[1].split(",")]
    Sr = cv.make_specs(m, o, row, None)
    E.evaluate(Sr["H2"], [st for st in gpr if st["R"] in rv])
elif part == "hinc":  # the incremental half of hdep, so it can run in parallel with H2
    row = best_row(rows)
    for name, base, gb in (("B1+H1", "B1", gp), ("B2+H1", "B2", gpr)):
        sel = E.select(S[base], gb)["setting"]
        E.evaluate(cv.incremental_spec(name, S[base], sel, f"H{row}"), [{"X": float(v)} for v in g["P"]])
elif part == "attn":
    E.evaluate(S["RCAP"], gpr)
    sel = E.select(S["B2"], gpr)["setting"]
    E.evaluate(cv.incremental_spec("B2+A1", S["B2"], sel, "A1" if o == 1 else "A1mean"), [{"X": float(v)} for v in g["P"]])
print({"done": [m, o, part]}, flush=True)
