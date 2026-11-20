"""B2: BGE-M3 adapted to each objective (full fine-tuning or LoRA), trained jointly with a linear head.

Protocol (tracker §3A, docx §9.1.1): the encoder is adapted only on a fold's fitting groups (never pre-fitted on all
57 TRAIN groups before CV); the LR x epoch grid is chosen on the B2-only configuration by held-out-fold loss. The
selected fold encoder is then frozen and its dense embeddings become the B2 feature block, on which B2 and every
B2+probe configuration fit the common head grid (heads.py), exactly like B1 on the frozen encoder.

Objective 1 row: [e(P), e(R), S] -> linear -> logit, binomial NLL with n = 20 (mean per completion).
Objective 2 prompt: [e(P), e(R_1..5) zero-padded, mask, S] -> linear -> sigmoid, MSE on q_hat.
Optimiser: AdamW, weight decay 0.01 on the encoder, head LR 1e-3, linear warmup over 5% of steps then linear decay.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
import torch
from torch import nn

from ..config import load_config
from . import encoder as enc

HEAD_LR = 1e-3
WARMUP = 0.05
BATCH_PROMPTS = 32


def b2cfg() -> dict:
    return load_config()["probing"]["b2"]


def make_encoder(adapt: str, dev: str):
    m = enc.load_model(dev)
    if adapt == "lora":
        from peft import LoraConfig, get_peft_model
        lc = b2cfg()["lora"]
        m = get_peft_model(m, LoraConfig(r=lc["rank"], lora_alpha=2 * lc["rank"], lora_dropout=lc["dropout"],
                                         target_modules=["query", "key", "value", "dense"], bias="none"))
    elif adapt != "full":
        raise ValueError(adapt)
    return m


class B2Model(nn.Module):
    """Encoder + linear head for one objective. `n_struct`: width of the standardized structure block."""

    def __init__(self, encoder, objective: int, n_struct: int):
        super().__init__()
        self.encoder, self.objective = encoder, objective
        d = 1024
        width = (2 * d if objective == 1 else d + 5 * d + 5) + n_struct
        self.head = nn.Linear(width, 1)

    def embed(self, texts: list[str], dev: str) -> torch.Tensor:
        b = enc.tokenizer()(texts, padding=True, truncation=False, return_tensors="pt").to(dev)
        return enc.dense(self.encoder, b)

    def forward_obj1(self, prompt_texts, rule_texts, row_prompt, row_rule, S, dev):
        eP, eR = self.embed(prompt_texts, dev), self.embed(rule_texts, dev)
        x = torch.cat([eP[row_prompt], eR[row_rule], S], dim=1)
        return self.head(x).squeeze(-1)

    def forward_obj2(self, prompt_texts, rule_texts, slot_rule, mask, S, dev):
        """slot_rule: (B, 5) index into rule_texts, -1 for empty slots; mask: (B, 5) float."""
        eP, eR = self.embed(prompt_texts, dev), self.embed(rule_texts, dev)
        eR = torch.cat([eR, torch.zeros(1, eR.shape[1], device=dev)])  # index -1 -> zero padding
        slots = eR[slot_rule].reshape(len(prompt_texts), -1)
        x = torch.cat([eP, slots, mask, S], dim=1)
        return torch.sigmoid(self.head(x).squeeze(-1))


def binom_nll(logit: torch.Tensor, v: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """Mean per-completion binomial NLL (the constant log C(n, v) is omitted; it does not depend on the model)."""
    return (-(v * nn.functional.logsigmoid(logit) + (n - v) * nn.functional.logsigmoid(-logit))).sum() / n.sum()


def optimizer(model: B2Model, lr: float, total_steps: int):
    wd = b2cfg()["weight_decay"]
    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": lr, "weight_decay": wd},
                             {"params": model.head.parameters(), "lr": HEAD_LR, "weight_decay": 0.0}])
    warm = max(1, int(WARMUP * total_steps))
    decay = max(1, total_steps - warm)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total_steps - s) / decay))
    return opt, sched


def benchmark(adapt: str, texts: list[str], rule_texts: list[str], n_struct: int = 60, steps: int = 12,
              dev: str | None = None) -> dict:
    """Timed Objective 1 training steps (forward + backward + AdamW) and inference on real prompt texts."""
    dev = dev or enc.device()
    torch.manual_seed(0)
    m = B2Model(make_encoder(adapt, dev), 1, n_struct).to(dev)
    m.train()
    opt, sched = optimizer(m, 1e-5 if adapt == "full" else 1e-4, steps)
    rng = np.random.default_rng(0)
    times = []
    for s in range(steps + 2):
        pi = rng.choice(len(texts), BATCH_PROMPTS, replace=False)
        k = rng.integers(1, 6, BATCH_PROMPTS)
        rows_p = torch.tensor(np.repeat(np.arange(BATCH_PROMPTS), k), device=dev)
        ur = rng.choice(len(rule_texts), min(len(rule_texts), 64), replace=False)
        rows_r = torch.tensor(rng.integers(0, len(ur), len(rows_p)), device=dev)
        S = torch.randn(len(rows_p), n_struct, device=dev)
        v = torch.tensor(rng.integers(0, 21, len(rows_p)), device=dev, dtype=torch.float32)
        n = torch.full_like(v, 20.0)
        t0 = time.time()
        loss = binom_nll(m.forward_obj1([texts[i] for i in pi], [rule_texts[i] for i in ur], rows_p, rows_r, S, dev), v, n)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if dev == "mps":
            torch.mps.synchronize()
        if s >= 2:  # skip warm-up steps
            times.append(time.time() - t0)
    train_s_per_prompt = float(np.median(times)) / BATCH_PROMPTS
    m.eval()
    t0 = time.time()
    with torch.no_grad():
        sub = texts[:512]
        for s in range(0, len(sub), 64):
            m.embed(sub[s:s + 64], dev)
        if dev == "mps":
            torch.mps.synchronize()
    infer_s_per_prompt = (time.time() - t0) / len(sub)
    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    peak = torch.mps.driver_allocated_memory() / 1e9 if dev == "mps" else None
    del m, opt
    if dev == "mps":
        torch.mps.empty_cache()
    return {"adapt": adapt, "device": dev, "batch_prompts": BATCH_PROMPTS, "train_s_per_prompt": train_s_per_prompt,
            "train_step_s_median": float(np.median(times)), "infer_s_per_prompt": infer_s_per_prompt,
            "trainable_params": int(n_train), "mps_driver_allocated_gb": peak}


def project(bench: dict, n_fit_prompts: float, n_all_train: int, n_all: int, epoch_units_per_fold: float,
            n_grid_points: int, refit_epochs: int = 3) -> dict:
    """Wall-time projection (hours) of the B2 protocol at the benchmarked speed.

    epoch_units_per_fold: training epochs run per fold over the whole grid (8 for LR {2} x epochs {1, 3} as separate
    runs; 6 if the 1-epoch point is a snapshot of the 3-epoch run). n_grid_points: held-out evaluations per fold."""
    tr, inf = bench["train_s_per_prompt"], bench["infer_s_per_prompt"]
    per_fold = (epoch_units_per_fold * n_fit_prompts * tr + n_grid_points * (n_all_train - n_fit_prompts) * inf
                + n_all_train * inf)                                     # selected fold encoder -> all TRAIN features
    cv = 5 * per_fold
    refit = refit_epochs * n_all_train * tr + n_all * inf                # worst case: 3 epochs selected
    return {"hours_per_objective_generator": (cv + refit) / 3600, "hours_total": 4 * (cv + refit) / 3600,
            "hours_cv_part": 4 * cv / 3600, "hours_refits": 4 * refit / 3600}


SCENARIOS = [  # (id, adapt, description, epoch_units_per_fold, grid_points, refit_epochs)
    ("spec_full", "full", "As specified: full FT, LR {1e-5, 3e-5} x epochs {1, 3}, separate runs", 8, 4, 3),
    ("spec_lora", "lora", "As specified: LoRA, LR {1e-4, 2e-4} x epochs {1, 3}, separate runs", 8, 4, 3),
    ("spec_full_snapshot", "full", "Same 4 grid points, but the 1-epoch point is a snapshot of the 3-epoch run "
     "(constant LR after warm-up instead of linear decay)", 6, 4, 3),
    ("small_full_1lr", "full", "Smaller grid frozen in advance: full FT, LR 1e-5 only, epochs {1, 3} (snapshot)", 3, 2, 3),
    ("small_full_1pt", "full", "Smaller grid frozen in advance: full FT, LR 1e-5, 1 epoch (no B2 search)", 1, 1, 1),
]


def run_benchmark(bench: list[dict] | None = None) -> dict:
    """Benchmark (or record a given benchmark of) full and LoRA B2 on this machine; write the tables."""
    import pandas as pd
    from ..config import ROOT
    from . import data
    p = data.prompts("train")
    if bench is None:
        r = data.prompt_rules("train").drop_duplicates("rule_id")
        bench = [benchmark(a, p["prompt_text"].tolist(), r["rule_text"].tolist()) for a in ("full", "lora")]
    by = {b["adapt"]: b for b in bench}
    n_train = len(p)
    n_fit = n_train * (57 - 57 / 5) / 57                                   # 45.6 fitting groups on average
    rows = []
    for sid, adapt, desc, units, pts, rep in SCENARIOS:
        rows.append({"scenario": sid, "adapt": adapt, "description": desc,
                     **project(by[adapt], n_fit, n_train, 53136, units, pts, rep)})
    t = ROOT / "results" / "tables"
    t.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(bench).to_csv(t / "b2_benchmark.csv", index=False)
    pd.DataFrame(rows).to_csv(t / "b2_projection.csv", index=False)
    return {"benchmark": bench, "projection": rows}


# ------------------------------------------------------------------------------------------- B2 training runs
# Inputs travel as a TRAIN-only bundle (no TEST prompt, label or feature), so a remote job cannot touch the test set.
def bundle_dir():
    from ..config import DATA_DIR
    return DATA_DIR / "probing_features" / "b2_bundle"


def make_bundle(model: str) -> dict:
    """Write data/probing_features/b2_bundle/{shared_prompts.parquet, rules.parquet, <model>.npz} (TRAIN only)."""
    import pandas as pd
    from . import cv, data, features
    d = bundle_dir()
    d.mkdir(parents=True, exist_ok=True)
    P = data.prompts("train")[["prompt_id", "group_id", "prompt_text"]]
    rk = pd.read_parquet(features.cache_dir() / "ER_rules_train_keys.parquet")
    P.to_parquet(d / "shared_prompts.parquet")
    rk.to_parquet(d / "rules.parquet")
    s1, s2 = cv.Store(model, 1), cv.Store(model, 2)
    slot = np.full((len(P), 5), -1, dtype=np.int64)
    slot[s1.pix, s1.U["position"].values - 1] = s1.rix
    np.savez(d / f"{model}.npz", o1_pix=s1.pix, o1_rix=s1.rix, o1_v=s1.y, o1_n=s1.n, o1_S=features.load_rule_block("S1"),
             o2_slot=slot, o2_q=s2.y, o2_S=np.asarray(features.load_prompt_block("S2")),
             groups=np.array(P["group_id"].tolist(), dtype="U32"), folds=np.array(json.dumps(cv.load_folds()["folds"])))
    return {"bundle": str(d), "model": model, "n_prompts": len(P), "n_rules": len(rk), "n_obj1_rows": len(s1.y)}


def _fit_groups(groups: np.ndarray, folds: list, fold: int) -> np.ndarray:
    held = set(folds[fold]) if fold >= 0 else set()
    return np.array([g not in held for g in groups])


def _standardise(S: np.ndarray, fit_rows: np.ndarray) -> np.ndarray:
    mu, sd = S[fit_rows].mean(0), S[fit_rows].std(0)
    sd[sd < 1e-8] = 1.0
    return ((S - mu) / sd).astype(np.float32)


@torch.no_grad()
def _embed_all(model: B2Model, texts: list[str], dev: str, bs: int = 128) -> np.ndarray:
    model.eval()
    order = np.argsort([len(t) for t in texts])[::-1]
    out = np.empty((len(texts), 1024), dtype=np.float32)
    for s in range(0, len(texts), bs):
        ix = order[s:s + bs]
        out[ix] = model.embed([texts[i] for i in ix], dev).float().cpu().numpy()
    return out


def train_run(model_key: str, objective: int, fold: int, adapt: str, lr: float, epochs: int, out_dir,
              save_embeddings: bool = False, max_prompts: int | None = None, dev: str | None = None, seed: int | None = None,
              save_adapter: bool = False) -> dict:
    """Adapt BGE-M3 + linear head on the fold's fitting groups (fold = -1: all TRAIN groups) and score the held-out
    groups. Writes result.json (+ heldout.npz, + embeddings_fp16.npz if save_embeddings) to out_dir."""
    import pandas as pd
    from pathlib import Path
    from ..config import derive_seed
    dev = dev or ("cuda" if torch.cuda.is_available() else enc.device())
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = seed if seed is not None else derive_seed(load_config()["probing"]["seeds"]["b2"], model_key, objective, fold, adapt, lr, epochs) % (1 << 31)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    d = bundle_dir()
    P = pd.read_parquet(d / "shared_prompts.parquet")
    R = pd.read_parquet(d / "rules.parquet")
    B = np.load(d / f"{model_key}.npz", allow_pickle=False)
    folds = json.loads(str(B["folds"]))
    groups = B["groups"]
    texts, rtexts = P["prompt_text"].tolist(), R["rule_text"].tolist()
    fit_p = _fit_groups(groups, folds, fold)
    if max_prompts:  # smoke tests only
        keep = np.zeros(len(P), bool)
        keep[rng.choice(np.nonzero(fit_p)[0], max_prompts, replace=False)] = True
        if fold >= 0:
            keep[rng.choice(np.nonzero(~fit_p)[0], min(max_prompts, int((~fit_p).sum())), replace=False)] = True
    else:
        keep = np.ones(len(P), bool)
    if objective == 1:
        pix, rix, v, n = B["o1_pix"], B["o1_rix"], B["o1_v"].astype(np.float32), B["o1_n"].astype(np.float32)
        S = _standardise(B["o1_S"], fit_p[pix])
        rows_of = pd.Series(np.arange(len(pix))).groupby(pix).apply(np.asarray).to_dict()
    else:
        slot, q = B["o2_slot"], B["o2_q"].astype(np.float32)
        S = _standardise(B["o2_S"], fit_p)
    m = B2Model(make_encoder(adapt, dev), objective, S.shape[1]).to(dev)
    fit_ids = np.nonzero(fit_p & keep)[0]
    steps_per_epoch = int(np.ceil(len(fit_ids) / BATCH_PROMPTS))
    opt, sched = optimizer(m, lr, steps_per_epoch * epochs)
    t0, losses = time.time(), []

    def batch_loss(pb: np.ndarray, train: bool):
        if objective == 1:
            rows = np.concatenate([rows_of[i] for i in pb])
            ur, rinv = np.unique(rix[rows], return_inverse=True)
            pmap = {p: j for j, p in enumerate(pb)}
            z = m.forward_obj1([texts[i] for i in pb], [rtexts[i] for i in ur],
                               torch.tensor([pmap[p] for p in pix[rows]], device=dev), torch.tensor(rinv, device=dev),
                               torch.from_numpy(S[rows]).to(dev), dev)
            vt, nt = torch.from_numpy(v[rows]).to(dev), torch.from_numpy(n[rows]).to(dev)
            return binom_nll(z, vt, nt), (rows, torch.sigmoid(z).detach().float().cpu().numpy())
        sl = slot[pb]
        ur, inv = np.unique(sl[sl >= 0], return_inverse=True)
        si = np.full(sl.shape, -1, dtype=np.int64)
        si[sl >= 0] = inv
        mask = torch.from_numpy((sl >= 0).astype(np.float32)).to(dev)
        p = m.forward_obj2([texts[i] for i in pb], [rtexts[i] for i in ur], torch.from_numpy(si).to(dev), mask,
                           torch.from_numpy(S[pb]).to(dev), dev)
        return ((p - torch.from_numpy(q[pb]).to(dev)) ** 2).mean(), (pb, p.detach().float().cpu().numpy())

    for ep in range(epochs):
        m.train()
        perm = rng.permutation(fit_ids)
        for s in range(0, len(perm), BATCH_PROMPTS):
            loss, _ = batch_loss(perm[s:s + BATCH_PROMPTS], True)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(float(loss))
    train_s = time.time() - t0
    res = {"model": model_key, "objective": objective, "fold": fold, "adapt": adapt, "lr": lr, "epochs": epochs,
           "seed": seed, "device": dev, "n_fit_prompts": int(len(fit_ids)), "steps": len(losses), "train_s": train_s,
           "train_loss_last_epoch": float(np.mean(losses[-steps_per_epoch:])) if losses else None,
           "torch": torch.__version__, "encoder": load_config()["probing"]["encoder"]}
    if fold >= 0:
        m.eval()
        held = np.nonzero(~fit_p & keep)[0]
        preds_rows, preds = [], []
        with torch.no_grad():
            for s in range(0, len(held), 128):
                _, (rows, p) = batch_loss(held[s:s + 128], False)
                preds_rows.append(rows)
                preds.append(p)
        rows, p = np.concatenate(preds_rows), np.concatenate(preds)
        if objective == 1:
            pc = np.clip(p, 1e-6, 1 - 1e-6)
            res["heldout_loss"] = float(-(v[rows] * np.log(pc) + (n[rows] - v[rows]) * np.log1p(-pc)).sum() / n[rows].sum())
        else:
            res["heldout_loss"] = float(((p - q[rows]) ** 2).mean())
        np.savez(out_dir / "heldout.npz", rows=rows, pred=p)
    if save_embeddings:
        t1 = time.time()
        np.savez(out_dir / "embeddings_fp16.npz", EP=_embed_all(m, texts, dev).astype(np.float16),
                 ER=_embed_all(m, rtexts, dev).astype(np.float16))
        res["embed_s"] = time.time() - t1
    if save_adapter:
        if adapt != "lora":
            raise ValueError("only LoRA adapters are saved (full fine-tuning would copy the whole encoder)")
        m.encoder.save_pretrained(out_dir / "adapter")
        res["adapter"] = str(out_dir / "adapter")
    res["wall_s"] = time.time() - t0
    (out_dir / "result.json").write_text(json.dumps(res, indent=1))
    return res


def grid(adapt: str | None = None) -> list[tuple[str, float, int]]:
    c = b2cfg()
    return [(a, float(lr), int(e)) for a in (("full", "lora") if adapt is None else (adapt,))
            for lr in c[f"{a}_lr"] for e in c["epochs"]]


# Full vs LoRA (training-only check, rule fixed before any B2 result is seen): Llama, Objective 1, CV fold 0, one
# epoch at each method's lower learning rate (full 1e-5, LoRA 1e-4). The method with the lower held-out-fold
# binomial NLL is used for every B2 run (both generators, both objectives); an exact tie goes to full fine-tuning.
CHECK = ("llama", 1, 0, 1)
GRID_TASKS = [(m, o, f) for m in ("llama", "qwen") for o in (1, 2) for f in range(5)]  # 20 array tasks


def run_dir(out_root, model, objective, fold, adapt, lr, epochs):
    from pathlib import Path
    return Path(out_root) / "grid" / f"{model}_o{objective}_f{fold}_{adapt}_lr{lr:g}_ep{epochs}"


def _done(d) -> bool:
    return (d / "result.json").exists() and (d / "embeddings_fp16.npz").exists()


def check(out_root) -> dict:
    from pathlib import Path
    m, o, f, ep = CHECK
    res = {}
    for adapt in ("full", "lora"):
        lr = float(min(b2cfg()[f"{adapt}_lr"]))
        d = run_dir(out_root, m, o, f, adapt, lr, ep)
        r = json.loads((d / "result.json").read_text()) if _done(d) else train_run(m, o, f, adapt, lr, ep, d, save_embeddings=True)
        res[adapt] = {"lr": lr, "heldout_loss": r["heldout_loss"], "train_s": r["train_s"], "wall_s": r["wall_s"]}
    method = "lora" if res["lora"]["heldout_loss"] < res["full"]["heldout_loss"] else "full"
    out = {"rule": "lower held-out-fold NLL on (llama, objective 1, fold 0, 1 epoch, lower LR); tie -> full",
           "results": res, "method": method}
    Path(out_root).mkdir(parents=True, exist_ok=True)
    (Path(out_root) / "method.json").write_text(json.dumps(out, indent=1))
    return out


def grid_task(task: int, out_root) -> list[dict]:
    """One array task: every grid point of the chosen method for one (generator, objective, fold); resumable."""
    from pathlib import Path
    method = json.loads((Path(out_root) / "method.json").read_text())["method"]
    m, o, f = GRID_TASKS[task]
    out = []
    for adapt, lr, ep in grid(method):
        d = run_dir(out_root, m, o, f, adapt, lr, ep)
        out.append(json.loads((d / "result.json").read_text()) if _done(d)
                   else train_run(m, o, f, adapt, lr, ep, d, save_embeddings=True))
    return out


def runs_root():
    from ..config import DATA_DIR
    return DATA_DIR / "probing_features" / "b2_runs"


def select(model: str, objective: int) -> dict:
    """Choose the B2 learning rate x epochs by the mean (equal fold weights) held-out-fold loss of the jointly
    trained B2 head over the 5 folds; record every grid point. Writes results/cv/<model>/obj<o>/b2_selection.json."""
    import pandas as pd
    from .data import RESULTS
    from ..storage import write_json
    method = json.loads((runs_root() / "method.json").read_text())
    rows = []
    for adapt, lr, ep in grid(method["method"]):
        fl = []
        for f in range(5):
            p = run_dir(runs_root(), model, objective, f, adapt, lr, ep) / "result.json"
            fl.append(json.loads(p.read_text())["heldout_loss"] if p.exists() else np.nan)
        rows.append({"model": model, "objective": objective, "adapt": adapt, "lr": lr, "epochs": ep,
                     "fold_losses": fl, "cv_loss": float(np.mean(fl)) if not np.isnan(fl).any() else np.nan})
    complete = [r for r in rows if not np.isnan(r["cv_loss"])]
    if len(complete) != len(rows):
        raise RuntimeError(f"B2 grid incomplete for {model} obj{objective}: {[r for r in rows if np.isnan(r['cv_loss'])]}")
    best = min(rows, key=lambda r: r["cv_loss"])
    out = {"method_check": method, "grid": rows, "selected": {k: best[k] for k in ("adapt", "lr", "epochs")},
           "selected_cv_loss": best["cv_loss"],
           "fold_dirs": {str(f): str(run_dir(runs_root(), model, objective, f, best["adapt"], best["lr"], best["epochs"]))
                         for f in range(5)}}
    write_json(out, RESULTS / "cv" / model / f"obj{objective}" / "b2_selection.json")
    return out


def selected_run_dir(model: str, objective: int, fold):
    """Run directory holding the selected B2 encoder's embeddings for CV fold `fold` (or 'final')."""
    from pathlib import Path
    from .data import RESULTS
    s = json.loads((RESULTS / "cv" / model / f"obj{objective}" / "b2_selection.json").read_text())
    key = str(fold)
    if key == "final":
        sel = s["selected"]
        return runs_root() / "final" / f"{model}_o{objective}_{sel['adapt']}_lr{sel['lr']:g}_ep{sel['epochs']}"
    return Path(s["fold_dirs"][key])


def final_tasks_file():
    return runs_root() / "final_tasks.json"


def write_final_tasks() -> list:
    """One final refit per (generator, objective) at its CV-selected setting (from b2_selection.json)."""
    from .data import RESULTS
    tasks = []
    for m in ("llama", "qwen"):
        for o in (1, 2):
            s = json.loads((RESULTS / "cv" / m / f"obj{o}" / "b2_selection.json").read_text())["selected"]
            tasks.append({"model": m, "objective": o, "adapt": s["adapt"], "lr": s["lr"], "epochs": s["epochs"]})
    final_tasks_file().write_text(json.dumps(tasks, indent=1))
    return tasks


def final_task(task: int, out_root) -> dict:
    """Refit the selected B2 setting on all 57 TRAIN groups; save TRAIN embeddings and the LoRA adapter (so TEST
    prompts can be embedded with the same frozen encoder after unblinding, on the same backend)."""
    from pathlib import Path
    t = json.loads((Path(out_root) / "final_tasks.json").read_text())[task]
    d = Path(out_root) / "final" / f"{t['model']}_o{t['objective']}_{t['adapt']}_lr{t['lr']:g}_ep{t['epochs']}"
    if _done(d) and (d / "adapter").exists():
        return json.loads((d / "result.json").read_text())
    return train_run(t["model"], t["objective"], -1, t["adapt"], t["lr"], t["epochs"], d, save_embeddings=True, save_adapter=True)


def main(argv=None):
    """Remote entry points (Betty):
      python -m mife.probing.b2 run MODEL OBJ FOLD ADAPT LR EPOCHS OUT [--embeddings] [--max-prompts N]
      python -m mife.probing.b2 check OUT_ROOT          full-vs-LoRA check (+ benchmark); writes OUT_ROOT/method.json
      python -m mife.probing.b2 grid TASK OUT_ROOT      one of the 20 (generator, objective, fold) grid tasks"""
    import argparse
    import sys
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "check":
        print(json.dumps(check(argv[1])))
        return
    if argv and argv[0] == "final":
        print(json.dumps(final_task(int(argv[1]), argv[2]), default=str))
        return
    if argv and argv[0] == "grid":
        print(json.dumps(grid_task(int(argv[1]), argv[2]), default=str))
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("model"); ap.add_argument("objective", type=int); ap.add_argument("fold", type=int)
    ap.add_argument("adapt"); ap.add_argument("lr", type=float); ap.add_argument("epochs", type=int); ap.add_argument("out")
    ap.add_argument("--embeddings", action="store_true"); ap.add_argument("--max-prompts", type=int, default=None)
    a = ap.parse_args(argv)
    print(json.dumps(train_run(a.model, a.objective, a.fold, a.adapt, a.lr, a.epochs, a.out, a.embeddings, a.max_prompts)))


if __name__ == "__main__":
    main()
