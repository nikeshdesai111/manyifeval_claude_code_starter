"""Fill the CV point cache for the Objective 2 ordered-slot B2o / H2o predictors (no attention needed) using the
warm-up's grids and row. Cache only: the selection itself is made later by cv.run_ordered from the final selection."""
import sys
from mife.probing.v2 import cv
m, row, ext = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
E = cv.Engine(m, 2)
g = {"P": list(E.R["prompt_lambda_grid"]), "R": list(E.R["rule_lambda_grid"])}
for e in ext:  # e.g. P:high
    s, d = e.split(":")
    g[s] = sorted(g[s] + [min(g[s]) / 10 if d == "low" else max(g[s]) * 10])
S = cv.make_specs(m, 2, row, None, ordered=True)
for name in ("B2o", "H2o"):
    E.evaluate(S[name], cv._grid(g, S[name].tuned))
