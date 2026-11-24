"""Additional incremental predictor requested by the project owner on 2026-11-24: B2+H+A (code name B2+H1+A1).

Does rule attention add information beyond the hidden state, on top of B2? Same recipe as the other incremental
predictors (TRAIN only, S3 folds): the baseline B2+H (code B2+H1) is frozen at its CV-selected settings (its prompt,
rule and hidden-state penalties become fixed slots F_P, F_R, F_X), and only the added attention block's penalty is
tuned on the same grid as the other incremental blocks, with "off" (= the baseline's predictions) as an option. The
selection rule, folds, fallbacks and refit are the ones in cv.py. Writes the OOF parquet, the refit model and
<cv>/selection_extra.json under results/v2_s3/. Run: python -m mife.probing.v2.extra <model> <objective>
"""
from __future__ import annotations

import json
import sys

from . import cv
from .cv import Engine, Spec, incremental_spec, setting_key
from .versioning import RESULTS_V2, write_versioned_json

NAME = "B2+H1+A1"


def _parse(s: str) -> dict:
    return {} if s in ("none", "off") else {k: float(v) for k, v in (x.split("=") for x in s.split(","))}


def run(model: str, o: int) -> dict:
    E = Engine(model, o)
    sel_all = json.loads((E.root / "selection.json").read_text())
    grids = sel_all["final_grids"]
    row = sel_all["hidden_row"]
    S = sel_all["selected"]
    ER, A = ("ER", "A1") if o == 1 else ("ERmean", "A1mean")
    b2 = Spec("B2", (("EP", "P"), (ER, "R"), ("S", "S")))
    b2_set = _parse(S["B2"]["setting"])
    b2h = incremental_spec("B2+H1", b2, b2_set, f"H{row}")
    assert [list(b) for b in b2h.blocks] == S["B2+H1"]["blocks"], "B2+H1 spec does not reproduce the stored selection"
    b2h_set = _parse(S["B2+H1"]["setting"])
    if b2h_set:  # hidden-state block selected: freeze it at its penalty
        blocks = tuple((b, "F_X" if sl == "X" else sl) for b, sl in b2h.blocks) + ((A, "X"),)
        fixed = tuple(sorted(dict(b2h.fixed, F_X=b2h_set["X"]).items()))
    else:  # B2+H1 chose "off": its baseline is B2 itself
        blocks = tuple((b, sl) for b, sl in b2h.blocks if sl != "X") + ((A, "X"),)
        fixed = b2h.fixed
    sp = Spec(NAME, blocks, fixed, baseline="B2+H1")
    g = [{"X": float(v)} for v in grids["P"]]
    E.evaluate(sp, g)
    off = E.oof(b2h, b2h_set) if b2h_set else E.oof(b2, b2_set)
    s = E.select(sp, g, off=off)
    # OOF parquet with the same layout as cv.run
    pred_dir = RESULTS_V2 / "predictions" / model / f"obj{o}"
    keys = ["prompt_id", "position", "rule_id", "type_id"] if o == 1 else ["prompt_id"]
    df = E.D.U[keys].copy()
    df["fold"] = E.D.fold
    for gpt in s["grid"]:
        col = "off" if gpt["setting"] is None else setting_key(gpt["setting"])
        df[col] = off if gpt["setting"] is None else E.oof(sp, gpt["setting"])
    scol = "off" if s["setting"] is None else setting_key(s["setting"])
    df["selected"] = df[scol]
    df.to_parquet(pred_dir / f"{NAME}.parquet")
    specs = {"B2+H1": b2h if b2h_set else b2}
    sel = {"B2+H1": {"setting": b2h_set or b2_set}}
    cv.refit(E, {NAME: {"spec": sp, "setting": s["setting"]}}, specs, sel)
    rec = {"predictor": NAME, "label": "B2+H+A", "baseline": "B2+H1", "baseline_setting": S["B2+H1"]["setting"],
           "blocks": [list(b) for b in sp.blocks], "fixed": dict(sp.fixed), "setting": scol, "cv_loss": s["loss"],
           "grid": [{"setting": None if x["setting"] is None else setting_key(x["setting"]), "cv_loss": x["loss"]}
                    for x in s["grid"]],
           "fold_losses": E.fold_scores(df["selected"].values),
           "requested": "project owner, 2026-11-24 (added after the evaluation plan; not part of the original seven)"}
    write_versioned_json(rec, E.root / "selection_extra.json")
    return rec


if __name__ == "__main__":
    for m in sys.argv[1].split(","):
        for o in (int(x) for x in sys.argv[2].split(",")):
            r = run(m, o)
            print(m, o, r["setting"], r["cv_loss"], flush=True)
