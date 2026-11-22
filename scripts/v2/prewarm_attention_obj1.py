"""Fill the CV point cache for Objective 1 RCAP (base 6x6 grid) and B2+A1 (against B2's base-round selection) while
the warm-up process finishes the non-attention predictors. Cache only; cv.run makes every selection."""
import sys
from mife.probing.v2 import cv
m = sys.argv[1]
E = cv.Engine(m, 1)
g = {"P": list(E.R["prompt_lambda_grid"]), "R": list(E.R["rule_lambda_grid"])}
S = cv.make_specs(m, 1, None, None)
E.evaluate(S["RCAP"], cv._grid(g, ("P", "R")))
gb = cv._grid(g, ("P", "R"))
E.evaluate(S["B2"], gb)  # cached by the warm-up
b2 = E.select(S["B2"], gb)
sp = cv.incremental_spec("B2+A1", S["B2"], b2["setting"], "A1")
E.evaluate(sp, [{"X": float(v)} for v in g["P"]])
