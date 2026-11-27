"""Split K10 Objective-1 CV fitting into (fold, type) units that can run elsewhere (Betty), without changing results.

The selection logic stays in cv.run / extra.run on the Mac. `plan` runs them with Engine.evaluate patched: a spec
whose point cache is complete passes through; a spec whose units have all come back is merged into the normal cache
(points/<spec>/<setting>.npy, fits.jsonl, spec.json, exactly as Engine.evaluate writes them); otherwise the spec and
its `todo` settings (the same list and order evaluate would fit) are written as a fit request and planning stops.

    python -m mife.k10paper.unitfit plan   <model>                       -> results/k10_paper/v2/unitfit/<model>/next.json
    python -m mife.k10paper.unitfit export <model> <request.json> <pack_dir> [--types i,j,...]
    python -m mife.k10paper.unitfit fit    <pack_dir> <units_dir> [--types i,j,...] [--folds 0,1,...]   (Betty)
    python -m mife.k10paper.unitfit merge  <model> <request.json> <units_dir>

A pack holds only TRAIN rows of the requested types (rule-row blocks sliced to those rows, prompt blocks to the
prompts they use, ER whole), their labels, folds and rule indices. `fit` runs, per unit, the same code as
Engine.evaluate: Engine.prepare (standardisation, ER basis), ridge.Gram + ridge.fit warm-started strong -> weak, and
writes unit_f<fold>_t<type index>.npz atomically, so a stopped job resumes. Fallback units are never shipped: merge
fills them from the full labels exactly as evaluate does.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from ..probing.v2 import ridge
from . import cv
from . import features as F

TYPES = list(F.TYPES)
UNITROOT = cv.RESULTS_V2 / "unitfit"


class NeedFit(Exception):
    pass


def _spec_from(d: dict) -> cv.Spec:
    return cv.Spec(d["name"], tuple(tuple(b) for b in d["blocks"]), tuple(tuple(f) for f in d["fixed"]), d.get("baseline"))


def _spec_dict(sp: cv.Spec) -> dict:
    return {"name": sp.name, "blocks": [list(b) for b in sp.blocks], "fixed": [list(f) for f in sp.fixed],
            "baseline": sp.baseline, "key": sp.key}


def _order(settings: list[dict]) -> list[dict]:  # identical to Engine.evaluate
    return sorted(settings, key=lambda s: tuple(-s.get(k, 0) for k in ("R", "X", "P")))


def request_id(model: str, sp: cv.Spec, todo: list[dict]) -> str:
    h = hashlib.sha1(json.dumps([sp.key, [cv.setting_key(s) for s in todo]]).encode()).hexdigest()[:8]
    return f"{model}__{sp.key}__{h}"


# ------------------------------------------------------------------------------------------------ Mac side
def _units_complete(E: cv.Engine, units_dir: Path, nset: int) -> bool:
    for f in range(5):
        for ti, t in enumerate(TYPES):
            if E.fallback[(f, t)][0]:
                continue
            if not (units_dir / f"unit_f{f}_t{ti:02d}.npz").exists():
                return False
    return True


def merge(E: cv.Engine, sp: cv.Spec, todo: list[dict], units_dir: Path) -> None:
    """Write the point cache for `todo` from unit files + fallbacks, as Engine.evaluate does."""
    order = _order(todo)
    keys = [cv.setting_key(s) for s in order]
    out = {k: np.full(len(E.D.v), np.nan) for k in keys}
    diag = []
    for f in range(5):
        for ti, t in enumerate(TYPES):
            fit, held = E.rows(f, t)
            if E.fallback[(f, t)][0]:
                _, v, n = E.fallback[(f, t)]
                for k in keys:
                    out[k][held] = (v + 0.5) / (n + 1)
                continue
            z = np.load(units_dir / f"unit_f{f}_t{ti:02d}.npz", allow_pickle=False)
            assert list(z["settings"]) == keys, (f, t)
            assert np.array_equal(z["held_global"], held), (f, t)
            for i, k in enumerate(keys):
                out[k][held] = z["preds"][i]
            diag += json.loads(str(z["diag"]))
    for k, p in out.items():
        assert np.isfinite(p).all(), (sp.name, k)
        np.save(E.pdir(sp) / f"{k}.npy", p)
    with open(E.pdir(sp) / "fits.jsonl", "a") as fh:
        for r in diag:
            fh.write(json.dumps(r) + "\n")
    blocks = [b for b, _ in sp.blocks]
    cv.write_json({"spec": sp.name, "blocks": sp.blocks, "fixed": sp.fixed, "widths": E.D.widths(blocks),
                   "version": cv.VERSION, "fitted_by": "unitfit", "units_dir": str(units_dir)}, E.pdir(sp) / "spec.json")
    print(json.dumps({"model": E.model, "obj": E.o, "spec": sp.name, "n_settings": len(order), "merged_from": str(units_dir),
                      "not_converged": sum(not r["converged"] for r in diag)}), flush=True)


def plan(model: str) -> dict:
    """Run cv.run then extra.run with evaluate patched; return {'status': 'need', request} or {'status': 'done'}."""
    orig = cv.Engine.evaluate
    req_dir = UNITROOT / model
    req_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self, spec, settings):
        todo = [s for s in settings if not self.cached(spec, s)]
        if not todo:
            return
        rid = request_id(self.model, spec, todo)
        udir = req_dir / rid / "units"
        if udir.exists() and _units_complete(self, udir, len(todo)):
            merge(self, spec, todo, udir)
            return
        req = {"request_id": rid, "model": self.model, "objective": self.o, "spec": _spec_dict(spec),
               "todo": _order(todo), "settings_keys": [cv.setting_key(s) for s in _order(todo)],
               "structure_lambda": float(self.R["structure_lambda"]),
               "fallback_units": [[f, TYPES.index(t)] for (f, t), v in self.fallback.items() if f != "final" and v[0]],
               "widths": self.D.widths([b for b, _ in spec.blocks]), "created": time.time()}
        (req_dir / rid).mkdir(parents=True, exist_ok=True)
        (req_dir / rid / "request.json").write_text(json.dumps(req, indent=1))
        (req_dir / "next.json").write_text(json.dumps(req, indent=1))
        raise NeedFit(rid)

    cv.Engine.evaluate = evaluate
    try:
        try:
            cv.record_recipe()
            cv.run(model, 1)
            from . import extra
            if not (cv.RESULTS_V2 / "cv" / model / "obj1" / "selection_extra.json").exists():
                extra.run(model, 1)
        except NeedFit as e:
            return {"status": "need", "request_id": str(e), "request": str(req_dir / str(e) / "request.json")}
    finally:
        cv.Engine.evaluate = orig
    (req_dir / "next.json").write_text(json.dumps({"status": "done"}))
    return {"status": "done"}


def export(model: str, req: dict, pack_dir: Path, types: list[int] | None = None) -> dict:
    """Write a fitting pack for the request's spec, restricted to `types` (indices into TYPES)."""
    E_D = cv.Data(model, 1)
    sp = _spec_from(req["spec"])
    types = list(range(len(TYPES))) if types is None else types
    tnames = [TYPES[i] for i in types]
    R = np.nonzero(np.isin(E_D.types, tnames))[0]
    P = np.unique(E_D.pix[R])
    ploc = np.full(int(E_D.pix.max()) + 1, -1, np.int64)
    ploc[P] = np.arange(len(P))
    pack_dir.mkdir(parents=True, exist_ok=True)
    arrays = {"rows_global": R, "v": E_D.v[R], "n": E_D.n[R], "rix": E_D.rix[R].astype(np.int64),
              "fold": E_D.fold[R].astype(np.int64), "type_ix": np.array([TYPES.index(t) for t in E_D.types[R]], np.int64),
              "pix_local": ploc[E_D.pix[R]]}
    for k, a in arrays.items():
        np.save(pack_dir / f"_{k}.npy", a)
    for b, _ in sp.blocks:
        a = E_D._prompt_block(b)
        if b in ("S", "A1"):
            x = np.asarray(a[R])
        elif b in cv.Engine.RULE_BLOCKS:
            x = np.asarray(a)
        else:
            x = np.asarray(a[P])
        np.save(pack_dir / f"B_{b}.npy", x)
    meta = {**req, "types": types, "n_rows": int(len(R)), "n_prompts": int(len(P))}
    (pack_dir / "pack.json").write_text(json.dumps(meta, indent=1))
    md5 = {}
    for f in sorted(pack_dir.glob("*.npy")) + [pack_dir / "pack.json"]:
        h = hashlib.md5()
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                h.update(chunk)
        md5[f.name] = h.hexdigest()
    (pack_dir / "MD5SUMS").write_text("".join(f"{v}  {k}\n" for k, v in md5.items()))
    return {"pack": str(pack_dir), "rows": int(len(R)), "prompts": int(len(P)),
            "bytes": int(sum(f.stat().st_size for f in pack_dir.iterdir()))}


# ---------------------------------------------------------------------------------------------- fit side
class _PackData:
    """The subset of cv.Data that Engine.prepare / block_rows / widths use, backed by a pack."""
    block_rows = cv.Data.block_rows
    widths = cv.Data.widths
    X = cv.Data.X

    def __init__(self, pack_dir: Path):
        L = lambda k: np.load(pack_dir / f"_{k}.npy")
        self.rows_global, self.v, self.n, self.rix = L("rows_global"), L("v"), L("n"), L("rix")
        self.fold, self.type_ix, self.pix = L("fold"), L("type_ix"), L("pix_local")
        self._dir = pack_dir
        self._b = {}

    def _prompt_block(self, name: str):
        if name not in self._b:
            self._b[name] = np.load(self._dir / f"B_{name}.npy", mmap_mode="r")
        return self._b[name]


class _PackEngine:
    RULE_BLOCKS = cv.Engine.RULE_BLOCKS
    prepare = cv.Engine.prepare
    lam_vector = cv.Engine.lam_vector

    def __init__(self, D, structure_lambda):
        self.D, self.R = D, {"structure_lambda": structure_lambda}


def fit(pack_dir: Path, units_dir: Path, types: list[int] | None = None, folds: list[int] | None = None) -> dict:
    meta = json.loads((pack_dir / "pack.json").read_text())
    D = _PackData(pack_dir)
    E = _PackEngine(D, meta["structure_lambda"])
    sp = _spec_from(meta["spec"])
    order = meta["todo"]
    keys = meta["settings_keys"]
    assert [cv.setting_key(s) for s in order] == keys
    blocks = [b for b, _ in sp.blocks]
    widths = D.widths(blocks)
    assert widths == meta["widths"], (widths, meta["widths"])
    fb = {tuple(x) for x in meta["fallback_units"]}
    units_dir.mkdir(parents=True, exist_ok=True)
    types = meta["types"] if types is None else types
    folds = list(range(5)) if folds is None else folds
    done, t_all = [], time.time()
    for f in folds:
        for ti in types:
            if (f, ti) in fb:
                continue
            out = units_dir / f"unit_f{f}_t{ti:02d}.npz"
            if out.exists():
                continue
            t0 = time.time()
            sel = D.type_ix == ti
            fit_, held = np.nonzero(sel & (D.fold != f))[0], np.nonzero(sel & (D.fold == f))[0]
            Xf, Xh, mu, sd, weff, _ = E.prepare(blocks, widths, fit_, held, (f, TYPES[ti]))
            gram = ridge.Gram(Xf)
            init, preds, diag = None, [], []
            for s, k in zip(order, keys):
                h = ridge.fit(Xf, D.v[fit_], D.n[fit_], E.lam_vector(sp, s, weff), mu, sd, init, gram)
                init = np.r_[h.w, h.b]
                preds.append(ridge.expit(Xh @ h.w + h.b))
                diag.append({"fold": f, "type_id": TYPES[ti], "setting": k, "n_iter": h.n_iter, "converged": h.converged,
                             "hessian_jitter_steps": len(ridge.JITTER_EVENTS)})
                ridge.JITTER_EVENTS.clear()
            tmp = out.with_suffix(".tmp.npz")
            np.savez(tmp, held_global=D.rows_global[held], preds=np.array(preds), settings=np.array(keys),
                     diag=np.array(json.dumps(diag)))
            os.replace(tmp, out)
            dt = time.time() - t0
            done.append({"fold": f, "type": ti, "rows_fit": int(len(fit_)), "d": int(Xf.shape[1]), "s": round(dt, 1),
                         "s_per_fit": round(dt / len(order), 2)})
            print(json.dumps({"unit": [f, ti], **done[-1], "dev": os.environ.get("MIFE_V2_HESS", "cpu")}), flush=True)
            del Xf, Xh, gram
    return {"units": len(done), "wall_s": round(time.time() - t_all, 1)}


def _ints(s: str | None):
    return None if not s else [int(x) for x in s.split(",")]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "export", "fit", "merge"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--types")
    ap.add_argument("--folds")
    a = ap.parse_args()
    if a.cmd == "plan":
        print(json.dumps(plan(a.args[0])))
    elif a.cmd == "export":
        print(json.dumps(export(a.args[0], json.loads(Path(a.args[1]).read_text()), Path(a.args[2]), _ints(a.types))))
    elif a.cmd == "fit":
        print(json.dumps(fit(Path(a.args[0]), Path(a.args[1]), _ints(a.types), _ints(a.folds))))
    elif a.cmd == "merge":
        req = json.loads(Path(a.args[1]).read_text())
        E = cv.Engine(a.args[0], 1)
        merge(E, _spec_from(req["spec"]), req["todo"], Path(a.args[2]))
