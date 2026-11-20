"""V2 training-only grouped CV, selection, boundary expansion, incremental comparisons and final refits.

Everything is TRAIN-only (data.prompts("train")); folds = results/cv/folds.json (unchanged). The recipe is read
from configs/v2_recipe.yaml and must have been recorded (results/v2/recipe_recorded.json, same sha256) before any
selection runs. Each (predictor, setting) out-of-fold vector is cached under results/v2/cv/<model>/obj<o>/points/, so
the procedure is resumable and later stages reuse earlier fits exactly.

Predictor names (Objective 1 and 2 share them; Objective 2 blocks are the prompt-level versions):
  REF_structure  S                         B1    EP(P) + S            B2    EP(P) + ER(R) + S
  H1             H(P) + S                  H2    H(P) + ER(R) + S     RCAP  A1max(P) + ER(R) + S  (rule-conditioned attention probe)
  B1+H1, B2+H1, B2+A1  baseline frozen + added block (X slot: prompt-grid values + off)
  XH1            other generator's H(P) + target generator's S    (reference)
  Objective 2 ordered-slot sensitivity: B2o, H2o, RCAPo, B2o+H1, B2o+A1 (ER5 / A5 ordered slots instead of means)
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yaml

from ...config import ROOT
from ...storage import sha256_file, write_json
from .. import data
from . import features as F
from . import maxattn, ridge
from .versioning import RESULTS_V2, VERSION, read_versioned_json, write_versioned_json

RECIPE = ROOT / "configs" / "v2_recipe.yaml"
OTHER = {"llama": "qwen", "qwen": "llama"}


def recipe() -> dict:
    r = yaml.safe_load(RECIPE.read_text())
    assert r["version"] == VERSION
    return r


def record_recipe() -> dict:
    """Record the recipe before selection (idempotent; refuses if the file changed after recording)."""
    out = RESULTS_V2 / "recipe_recorded.json"
    sha = sha256_file(RECIPE)
    if out.exists():
        prev = read_versioned_json(out)
        if prev["sha256"] != sha:
            raise RuntimeError("configs/v2_recipe.yaml changed after it was recorded; record a dated change instead")
        return prev
    rec = {"sha256": sha, "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "recipe": recipe()}
    write_versioned_json(rec, out)
    return rec


def _require_recorded():
    out = RESULTS_V2 / "recipe_recorded.json"
    if not out.exists() or read_versioned_json(out)["sha256"] != sha256_file(RECIPE):
        raise RuntimeError("record the V2 recipe (cv.record_recipe()) before running selection")


# ------------------------------------------------------------------------------------------------- data
class Data:
    """Row-aligned TRAIN labels, folds and feature blocks for one label generator and objective."""

    def __init__(self, model: str, objective: int):
        self.model, self.o = model, objective
        self.P = data.prompts("train")
        pid = {x: i for i, x in enumerate(self.P["prompt_id"])}
        fog = json.loads((data.RESULTS / "cv" / "folds.json").read_text())["fold_of_group"]
        if objective == 1:
            PR = data.prompt_rules("train")[["prompt_id", "position", "rule_id", "type_id", "group_id"]]
            L = data.labels_prompt_rule(model, "train")[["prompt_id", "position", "rule_id", "v", "n"]]
            self.U = PR.merge(L, on=["prompt_id", "position", "rule_id"], how="left", validate="one_to_one")
            assert self.U["n"].notna().all() and (self.U["n"] == 20).all()
            self.v, self.n = self.U["v"].values.astype(np.float64), self.U["n"].values.astype(np.float64)
            self.pix = self.U["prompt_id"].map(pid).values
            self.rix = self.U["rule_id"].map(F.rule_index()).values
            self.slot = self.U["position"].values - 1
            self.types = self.U["type_id"].values
        else:
            L = data.labels_prompt(model, "train")[["prompt_id", "q_hat"]]
            self.U = self.P[["prompt_id", "group_id", "k"]].merge(L, on="prompt_id", how="left", validate="one_to_one")
            assert self.U["q_hat"].notna().all()
            self.v, self.n = self.U["q_hat"].values.astype(np.float64), np.ones(len(self.U))
            self.types = None
        self.groups = self.U["group_id"].values
        self.fold = np.array([fog[g] for g in self.groups])
        self._b: dict[str, np.ndarray] = {}

    # ---- blocks (Objective 1: row-level views are built lazily per head with `rows`)
    def _prompt_block(self, name: str) -> np.ndarray:
        if name in self._b:
            return self._b[name]
        m = self.model
        if name == "EP":
            a = F.load_shared("EP")
        elif name.startswith("XH"):
            a = F.load_model_block(OTHER[m], name[1:])
        elif name.startswith("H"):
            a = F.load_model_block(m, name)
        elif name in ("A1", "A1mean", "A5"):
            A, K = maxattn.load(m, self.P["prompt_id"].tolist())  # (n, L, H, 5)
            if name == "A1":
                a = A  # Objective 1: indexed by (prompt, slot) per row
            elif name == "A1mean":
                a = (A * K[:, None, None, :]).sum(-1) / K.sum(1)[:, None, None]
                a = a.reshape(len(A), -1)
            else:
                a = np.moveaxis(A * K[:, None, None, :], 3, 1).reshape(len(A), -1)  # slot-major, absent = 0
        elif name in ("ERmean", "ER5"):
            ER = F.load_shared("ER")
            PR = data.prompt_rules("train")
            pid = {x: i for i, x in enumerate(self.P["prompt_id"])}
            out = np.zeros((len(self.P), 5, ER.shape[1]), dtype=np.float32)
            out[PR["prompt_id"].map(pid).values, PR["position"].values - 1] = ER[PR["rule_id"].map(F.rule_index()).values]
            a = out.sum(1) / self.P["k"].values[:, None] if name == "ERmean" else out.reshape(len(self.P), -1)
        elif name == "S":
            a = F.load_model_block(m, "S1" if self.o == 1 else "S2")
        elif name == "ER":
            a = F.load_shared("ER")
        else:
            raise KeyError(name)
        self._b[name] = a
        return a

    def X(self, blocks: list[str], rows: np.ndarray) -> np.ndarray:
        parts = []
        for b in blocks:
            a = self._prompt_block(b)
            if self.o == 1:
                if b == "S":
                    parts.append(np.asarray(a[rows]))
                elif b == "ER":
                    parts.append(a[self.rix[rows]])
                elif b == "A1":
                    parts.append(np.asarray(a[self.pix[rows], :, :, self.slot[rows]]).reshape(len(rows), -1))
                else:
                    parts.append(np.asarray(a[self.pix[rows]]))
            else:
                parts.append(np.asarray(a[rows]))
        return np.concatenate(parts, axis=1).astype(np.float64)

    def widths(self, blocks: list[str]) -> list[int]:
        return [self.X([b], np.arange(1)).shape[1] for b in blocks]

    def drop(self, keep=()):
        for k in list(self._b):
            if k not in keep:
                del self._b[k]


# --------------------------------------------------------------------------------------------- predictors
@dataclass(frozen=True)
class Spec:
    name: str
    blocks: tuple            # ((block, slot), ...), slot in S (fixed structure), P, R (tuned), F_P / F_R (frozen), X (added)
    fixed: tuple = ()        # (("F_P", lam), ("F_R", lam)) for incremental baselines
    baseline: str | None = None

    @property
    def key(self) -> str:
        b = "+".join(f"{bl}:{sl}" for bl, sl in self.blocks)
        fx = ",".join(f"{s}={l:g}" for s, l in self.fixed)
        return f"{self.name}__" + hashlib.sha1(f"{b}|{fx}".encode()).hexdigest()[:10]

    @property
    def tuned(self) -> tuple:
        return tuple(s for s in ("P", "R", "X") if any(sl == s for _, sl in self.blocks))


def setting_key(st: dict) -> str:
    return ",".join(f"{k}={st[k]:g}" for k in sorted(st)) or "none"


def make_specs(model: str, o: int, row: int | None, xrow: int | None, sel: dict | None = None, ordered=False) -> dict:
    """Head-to-head predictors (+ references). Incremental ones are built by `incremental_spec`."""
    ER = "ER" if o == 1 else ("ER5" if ordered else "ERmean")
    A = "A1" if o == 1 else ("A5" if ordered else "A1mean")
    s = {"REF_structure": Spec("REF_structure", (("S", "S"),))}
    sfx = "o" if ordered else ""
    if not ordered:
        s["B1"] = Spec("B1", (("EP", "P"), ("S", "S")))
    s["B2" + sfx] = Spec("B2" + sfx, (("EP", "P"), (ER, "R"), ("S", "S")))
    if row is not None:
        if not ordered:
            s["H1"] = Spec("H1", ((f"H{row}", "P"), ("S", "S")))
        s["H2" + sfx] = Spec("H2" + sfx, ((f"H{row}", "P"), (ER, "R"), ("S", "S")))
    s["RCAP" + sfx] = Spec("RCAP" + sfx, ((A, "P"), (ER, "R"), ("S", "S")))
    if xrow is not None and not ordered:
        s["XH1"] = Spec("XH1", ((f"XH{xrow}", "P"), ("S", "S")))
    return s


def incremental_spec(name: str, base: Spec, base_setting: dict, added_block: str) -> Spec:
    blocks = tuple((b, {"P": "F_P", "R": "F_R"}.get(sl, sl)) for b, sl in base.blocks) + ((added_block, "X"),)
    fixed = tuple((f"F_{k}", float(v)) for k, v in sorted(base_setting.items()))
    return Spec(name, blocks, fixed, baseline=base.name)


# ------------------------------------------------------------------------------------------------ engine
class Engine:
    def __init__(self, model: str, o: int):
        _require_recorded()
        self.model, self.o = model, o
        self.R = recipe()[f"objective{o}"]
        self.D = Data(model, o)
        self.root = RESULTS_V2 / "cv" / model / f"obj{o}"
        self.root.mkdir(parents=True, exist_ok=True)
        self.prec = int(round(-np.log10(self.R["tie_precision"])))
        self.fallback = self._fallbacks() if o == 1 else {}

    # ---- sparse-head fallback (Objective 1): decided from labels of the fitting rows, identical for every method
    def _fallbacks(self) -> dict:
        fb = self.R["fallback"]
        out, rows = {}, []
        D = self.D
        for f in list(range(5)) + ["final"]:
            for t in F.TYPES:
                m = (D.types == t) & ((D.fold != f) if f != "final" else True)
                v, n = D.v[m].sum(), D.n[m].sum()
                gv = len(set(D.groups[m & (D.v > 0)]))
                fire = bool(v < fb["min_violated"] or (n - v) < fb["min_compliant"] or gv < fb["min_groups_with_violation"])
                out[(f, t)] = (fire, float(v), float(n))
                rows.append({"model": self.model, "fold": f, "type_id": t, "violated": v, "compliant": n - v,
                             "groups_with_violation": gv, "fallback": fire,
                             "reason": ";".join(r for r, c in (("violated<50", v < fb["min_violated"]),
                                                              ("compliant<50", (n - v) < fb["min_compliant"]),
                                                              ("groups_with_violation<5", gv < fb["min_groups_with_violation"])) if c)})
        self.fallback_table = pd.DataFrame(rows)
        return out

    def units(self):
        return [None] if self.o == 2 else F.TYPES

    def rows(self, f, t):
        D = self.D
        sel = np.ones(len(D.v), bool) if t is None else (D.types == t)
        if f == "final":
            return np.nonzero(sel)[0], np.zeros(0, int)
        return np.nonzero(sel & (D.fold != f))[0], np.nonzero(sel & (D.fold == f))[0]

    def lam_vector(self, spec: Spec, st: dict, widths) -> np.ndarray:
        fixed = dict(spec.fixed)
        lam = []
        for (b, sl), w in zip(spec.blocks, widths):
            v = self.R["structure_lambda"] if sl == "S" else fixed[sl] if sl.startswith("F_") else st[sl]
            lam.append(np.full(w, float(v)))
        return np.concatenate(lam)

    # ---- point cache
    def pdir(self, spec: Spec):
        d = self.root / "points" / spec.key
        d.mkdir(parents=True, exist_ok=True)
        return d

    def cached(self, spec: Spec, st: dict) -> bool:
        return (self.pdir(spec) / f"{setting_key(st)}.npy").exists()

    def oof(self, spec: Spec, st: dict | None) -> np.ndarray:
        if st is None:  # "off": the frozen baseline's predictions
            raise RuntimeError("off is resolved by the caller")
        return np.load(self.pdir(spec) / f"{setting_key(st)}.npy")

    def evaluate(self, spec: Spec, settings: list[dict]) -> None:
        todo = [s for s in settings if not self.cached(spec, s)]
        if not todo:
            return
        t0 = time.time()
        order = sorted(todo, key=lambda s: tuple(-s.get(k, 0) for k in ("R", "X", "P")))  # strong -> weak
        blocks = [b for b, _ in spec.blocks]
        widths = self.D.widths(blocks)
        out = {setting_key(s): np.full(len(self.D.v), np.nan) for s in order}
        diag = []
        for f in range(5):
            for t in self.units():
                fit, held = self.rows(f, t)
                if self.o == 1 and self.fallback[(f, t)][0]:
                    _, v, n = self.fallback[(f, t)]
                    for s in order:
                        out[setting_key(s)][held] = (v + 0.5) / (n + 1)
                    continue
                Xf = self.D.X(blocks, fit)
                mu, sd = ridge.standardise(Xf)
                Xf = (Xf - mu) / sd
                Xh = (self.D.X(blocks, held) - mu) / sd
                Xf32 = Xf.astype(np.float32)
                init = None
                for s in order:
                    h = ridge.fit(Xf, self.D.v[fit], self.D.n[fit], self.lam_vector(spec, s, widths), mu, sd, init, Xf32)
                    init = np.r_[h.w, h.b]
                    out[setting_key(s)][held] = ridge.expit(Xh @ h.w + h.b)
                    diag.append({"fold": f, "type_id": t, "setting": setting_key(s), "n_iter": h.n_iter, "converged": h.converged})
                del Xf, Xh, Xf32
        for k, p in out.items():
            assert np.isfinite(p).all(), (spec.name, k)
            np.save(self.pdir(spec) / f"{k}.npy", p)
        dj = self.pdir(spec) / "fits.jsonl"
        with open(dj, "a") as fh:
            for r in diag:
                fh.write(json.dumps(r) + "\n")
        write_json({"spec": spec.name, "blocks": spec.blocks, "fixed": spec.fixed, "widths": widths,
                    "version": VERSION}, self.pdir(spec) / "spec.json")
        print(json.dumps({"model": self.model, "obj": self.o, "spec": spec.name, "n_settings": len(order),
                          "wall_s": round(time.time() - t0, 1),
                          "not_converged": sum(not r["converged"] for r in diag)}), flush=True)

    # ---- scoring and selection
    def score(self, p: np.ndarray) -> float:
        if self.o == 1:
            s, n = ridge.log_loss(p, self.D.v, self.D.n)
            return s / n
        return float(((p - self.D.v) ** 2).mean())

    def fold_scores(self, p):
        out = []
        for f in range(5):
            m = self.D.fold == f
            if self.o == 1:
                s, n = ridge.log_loss(p[m], self.D.v[m], self.D.n[m])
                out.append(s / n)
            else:
                out.append(float(((p[m] - self.D.v[m]) ** 2).mean()))
        return out

    def select(self, spec: Spec, settings: list[dict], off: np.ndarray | None = None) -> dict:
        """Tie rule: rounded loss, then largest sum of log10 tuned penalties (off = +inf), then larger P (or X)."""
        cands = []
        for s in settings:
            p = self.oof(spec, s)
            loss = self.score(p)
            cands.append({"setting": s, "loss": loss, "rounded": round(loss, self.prec),
                          "logsum": sum(np.log10(v) for v in s.values()), "pen": s.get("P", s.get("X", 0.0))})
        if off is not None:
            loss = self.score(off)
            cands.append({"setting": None, "loss": loss, "rounded": round(loss, self.prec), "logsum": np.inf, "pen": np.inf})
        best = min(cands, key=lambda c: (c["rounded"], -c["logsum"], -c["pen"]))
        return {"setting": best["setting"], "loss": best["loss"],
                "grid": [{"setting": c["setting"], "loss": c["loss"]} for c in cands]}


# ---------------------------------------------------------------------------------------------- orchestration
def _grid(values, slots):
    if not slots:
        return [{}]
    out = [{}]
    for s in slots:
        out = [{**o, s: float(v)} for o in out for v in values[s]]
    return out


def _edge(sel_setting: dict | None, grids: dict, slot: str) -> str | None:
    if not sel_setting or slot not in sel_setting:
        return None
    v = sel_setting[slot]
    g = grids["X" if slot == "X" else slot]
    if v == min(g):
        return "low"
    if v == max(g) and slot != "X":
        return "high"
    return None


def run(model: str, o: int, with_attention: bool = True, final: bool = True) -> dict:
    E = Engine(model, o)
    R = E.R
    base = {"P": list(R["prompt_lambda_grid"]), "R": list(R["rule_lambda_grid"])}
    state_path = E.root / "selection.json"
    log = {"model": model, "objective": o, "version": VERSION, "base_grids": base, "with_attention": with_attention}

    def layer_select(grids, prefix, rows_):
        res = {}
        for r in rows_:
            sp = Spec(prefix, ((f"{'X' if prefix == 'XH1' else ''}H{r}", "P"), ("S", "S")))
            E.evaluate(sp, _grid(grids, ("P",)))
            res[r] = E.select(sp, _grid(grids, ("P",)))
        best = min(rows_, key=lambda r: (round(res[r]["loss"], E.prec), rows_.index(r)))  # tie -> earlier (shallower) row
        return best, res

    def stage(grids):
        rows_ = R["hidden_rows"][model]
        xrows = R["hidden_rows"][OTHER[model]]
        row, lres = layer_select(grids, "H1", rows_)
        xrow, xres = layer_select(grids, "XH1", xrows)
        E.D.drop(keep={"S", "EP", "ER", f"H{row}", f"XH{xrow}"})
        specs = make_specs(model, o, row, xrow)
        if not with_attention:
            specs.pop("RCAP")
        sel = {}
        for name, sp in specs.items():
            g = _grid(grids, sp.tuned)
            E.evaluate(sp, g)
            sel[name] = {"spec": sp, **E.select(sp, g)}
        return row, xrow, lres, xres, specs, sel

    def incremental(grids, specs, sel, row):
        out = {}
        inc = [("B1+H1", "B1", f"H{row}"), ("B2+H1", "B2", f"H{row}")]
        if with_attention:
            inc.append(("B2+A1", "B2", "A1" if o == 1 else "A1mean"))
        for name, bname, blk in inc:
            sp = incremental_spec(name, specs[bname], sel[bname]["setting"], blk)
            g = [{"X": float(v)} for v in grids["P"]]
            E.evaluate(sp, g)
            out[name] = {"spec": sp, **E.select(sp, g, off=E.oof(specs[bname], sel[bname]["setting"]))}
        return out

    # (1) + (2) base round
    grids = {k: list(v) for k, v in base.items()}
    row, xrow, lres, xres, specs, sel = stage(grids)
    inc = incremental(grids, specs, sel, row)
    # (3) boundary expansion, once
    hits = []
    for name, s in {**sel, **inc}.items():
        if name == "REF_structure":
            continue
        for slot in ("P", "R", "X"):
            e = _edge(s["setting"], {**grids, "X": grids["P"]}, slot)
            if e:
                hits.append({"predictor": name, "slot": "P" if slot == "X" else slot, "direction": e,
                             "selected": s["setting"][slot]})
    expansions = sorted({(h["slot"], h["direction"]) for h in hits})
    for slot, d in expansions:
        g = grids[slot]
        grids[slot] = sorted(g + [min(g) / 10 if d == "low" else max(g) * 10])
    if expansions:
        # (4) evaluate extra values, reselect; (5) incremental against the reselected baselines
        row, xrow, lres, xres, specs, sel = stage(grids)
        inc = incremental(grids, specs, sel, row)
    if not with_attention:
        return {"warmup": True, "hidden_row": row, "expansions": expansions}  # cache filling only; nothing written
    log.update({"final_grids": grids, "boundary_hits_base_round": hits,
                "expansions": [{"slot": s, "direction": d, "factor": 10} for s, d in expansions],
                "hidden_row": row, "hidden_row_scores": {str(r): v["loss"] for r, v in lres.items()},
                "other_generator_row": xrow, "other_generator_row_scores": {str(r): v["loss"] for r, v in xres.items()},
                "row_convention": F.row_convention(model), "other_row_convention": F.row_convention(OTHER[model])})
    # edges after expansion (reported, never expanded again)
    post = []
    for name, s in {**sel, **inc}.items():
        for slot in ("P", "R", "X"):
            e = _edge(s["setting"], {**grids, "X": grids["P"]}, slot)
            if e:
                post.append({"predictor": name, "slot": slot, "direction": e, "note": "best among the tested values"})
    log["edges_after_expansion"] = post
    # selections, OOF parquet, tables
    pred_dir = RESULTS_V2 / "predictions" / model / f"obj{o}"
    pred_dir.mkdir(parents=True, exist_ok=True)
    keys = ["prompt_id", "position", "rule_id", "type_id"] if o == 1 else ["prompt_id"]
    selected, grid_rows = {}, []
    for name, s in {**sel, **inc}.items():
        sp = s["spec"]
        df = E.D.U[keys].copy()
        df["fold"] = E.D.fold
        for gpt in s["grid"]:
            col = "off" if gpt["setting"] is None else setting_key(gpt["setting"])
            df[col] = E.oof(specs[sp.baseline], sel[sp.baseline]["setting"]) if gpt["setting"] is None else E.oof(sp, gpt["setting"])
            grid_rows.append({"model": model, "objective": o, "predictor": name, "setting": col, "cv_loss": gpt["loss"]})
        scol = "off" if s["setting"] is None else setting_key(s["setting"])
        df["selected"] = df[scol]
        df.to_parquet(pred_dir / f"{name}.parquet")
        selected[name] = {"setting": scol, "cv_loss": s["loss"], "fold_losses": E.fold_scores(df["selected"].values),
                          "blocks": [list(b) for b in sp.blocks], "fixed": dict(sp.fixed), "baseline": sp.baseline,
                          "n_features": int(sum(E.D.widths([b for b, _ in sp.blocks]))),
                          "hidden_row": row if name in ("H1", "H2", "B1+H1", "B2+H1") else xrow if name == "XH1" else None,
                          "note": ("CV did not select the block, so the gain is 0 by construction." if s["setting"] is None else None)}
    log["selected"] = selected
    if o == 1:
        fbt = E.fallback_table
        log["fallbacks_fired"] = fbt[fbt.fallback].to_dict("records")
    write_versioned_json(log, state_path)
    t = RESULTS_V2 / "tables"
    t.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(grid_rows).to_csv(t / f"cv_grid_{model}_obj{o}.csv", index=False)
    if o == 1:
        E.fallback_table.to_csv(t / f"fallbacks_{model}_obj1.csv", index=False)
    if final:
        refit(E, {**sel, **inc}, specs, sel)
    return log


def refit(E: Engine, allsel: dict, specs: dict, sel: dict) -> None:
    """Refit each selected predictor on all 57 TRAIN groups (fallback per type decided on all 57)."""
    mdir = RESULTS_V2 / "models" / E.model / f"obj{E.o}"
    mdir.mkdir(parents=True, exist_ok=True)
    for name, s in allsel.items():
        sp = s["spec"]
        st = s["setting"]
        if st is None:
            sp, st = specs[sp.baseline], sel[sp.baseline]["setting"]
        blocks = [b for b, _ in sp.blocks]
        widths = E.D.widths(blocks)
        heads = {}
        for t in E.units():
            fit, _ = E.rows("final", t)
            key = t or "shared"
            if E.o == 1 and E.fallback[("final", t)][0]:
                _, v, n = E.fallback[("final", t)]
                heads[key] = {"fallback": True, "p": (v + 0.5) / (n + 1)}
                continue
            Xf = E.D.X(blocks, fit)
            mu, sd = ridge.standardise(Xf)
            h = ridge.fit((Xf - mu) / sd, E.D.v[fit], E.D.n[fit], E.lam_vector(sp, st, widths), mu, sd)
            heads[key] = {"fallback": False, "mean": mu, "scale": sd, "w": h.w, "b": h.b, "converged": h.converged}
        np.savez(mdir / f"{name}.npz", **{f"{k}__{a}": np.asarray(v) for k, d in heads.items() for a, v in d.items()})
        write_versioned_json({"predictor": name, "setting": setting_key(st) if s["setting"] is not None else "off",
                              "fitted_as": sp.name, "blocks": [list(b) for b in sp.blocks], "fixed": dict(sp.fixed),
                              "widths": widths, "n_fitting_groups": 57,
                              "fallback_types": [k for k, d in heads.items() if d["fallback"]]}, mdir / f"{name}.json")


# ------------------------------------------------------------------------------- Objective 2 extras
def run_ordered(model: str) -> dict:
    """Objective 2 ordered-slot sensitivity: B2o, H2o, RCAPo, B2o+H1, B2o+A1 (ER5 / A5 in their five slots).
    Uses the main analysis' final grids and selected hidden row; no further boundary expansion."""
    main = read_versioned_json(RESULTS_V2 / "cv" / model / "obj2" / "selection.json")
    E = Engine(model, 2)
    grids, row = main["final_grids"], main["hidden_row"]
    specs = make_specs(model, 2, row, None, ordered=True)
    specs.pop("REF_structure")
    sel = {}
    for name, sp in specs.items():
        g = _grid(grids, sp.tuned)
        E.evaluate(sp, g)
        sel[name] = {"spec": sp, **E.select(sp, g)}
    for name, blk in (("B2o+H1", f"H{row}"), ("B2o+A1", "A5")):
        sp = incremental_spec(name, specs["B2o"], sel["B2o"]["setting"], blk)
        g = [{"X": float(v)} for v in grids["P"]]
        E.evaluate(sp, g)
        sel[name] = {"spec": sp, **E.select(sp, g, off=E.oof(specs["B2o"], sel["B2o"]["setting"]))}
    pred_dir = RESULTS_V2 / "predictions" / model / "obj2" / "ordered"
    pred_dir.mkdir(parents=True, exist_ok=True)
    out, rows = {}, []
    for name, s in sel.items():
        sp = s["spec"]
        df = E.D.U[["prompt_id"]].copy()
        df["fold"] = E.D.fold
        for gpt in s["grid"]:
            col = "off" if gpt["setting"] is None else setting_key(gpt["setting"])
            df[col] = E.oof(specs[sp.baseline], sel[sp.baseline]["setting"]) if gpt["setting"] is None else E.oof(sp, gpt["setting"])
            rows.append({"model": model, "objective": 2, "predictor": name, "setting": col, "cv_loss": gpt["loss"]})
        scol = "off" if s["setting"] is None else setting_key(s["setting"])
        df["selected"] = df[scol]
        df.to_parquet(pred_dir / f"{name}.parquet")
        out[name] = {"setting": scol, "cv_loss": s["loss"], "fold_losses": E.fold_scores(df["selected"].values),
                     "n_features": int(sum(E.D.widths([b for b, _ in sp.blocks]))), "baseline": sp.baseline,
                     "edges": [sl for sl in ("P", "R", "X") if _edge(s["setting"], {**grids, "X": grids["P"]}, sl)],
                     "note": ("CV did not select the block, so the gain is 0 by construction." if s["setting"] is None else None)}
    res = {"model": model, "objective": 2, "analysis": "ordered-slot sensitivity", "grids": grids, "hidden_row": row,
           "selected": out}
    write_versioned_json(res, RESULTS_V2 / "cv" / model / "obj2" / "selection_ordered.json")
    pd.DataFrame(rows).to_csv(RESULTS_V2 / "tables" / f"cv_grid_{model}_obj2_ordered.csv", index=False)
    refit(E, sel, specs, sel)
    return res


def avg_obj1_reference(model: str) -> dict:
    """For each Objective 1 predictor: mean of its selected OOF per-rule predictions over each prompt's rules, scored
    against q_hat by the Objective 2 metric on the same prompts and folds."""
    D = Data(model, 2)
    src = RESULTS_V2 / "predictions" / model / "obj1"
    out_dir = RESULTS_V2 / "predictions" / model / "obj2" / "avg_obj1"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {}
    for f in sorted(src.glob("*.parquet")):
        p1 = pd.read_parquet(f, columns=["prompt_id", "fold", "selected"])
        m = p1.groupby("prompt_id").agg(pred=("selected", "mean"), fold=("fold", "first"))
        df = D.U[["prompt_id", "q_hat"]].merge(m, left_on="prompt_id", right_index=True, how="left", validate="one_to_one")
        assert df["pred"].notna().all() and (df["fold"].values == D.fold).all()
        df.to_parquet(out_dir / f"{f.stem}.parquet")
        err = (df["pred"] - df["q_hat"]) ** 2
        res[f.stem] = {"cv_mse": float(err.mean()),
                       "fold_mse": [float(err[D.fold == k].mean()) for k in range(5)]}
    write_versioned_json({"model": model, "reference": "averaged Objective 1 OOF predictions", "results": res},
                         RESULTS_V2 / "cv" / model / "obj2" / "avg_obj1_reference.json")
    return res
