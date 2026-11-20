"""Probing-phase unit tests. Everything here except the real-data structure checks uses SYNTHETIC data, which is for
tests only and never written to results/."""
import json
import math

import numpy as np
import pandas as pd
import pytest
import torch

from mife.config import DATA_DIR
from mife.probing import configs as C
from mife.probing import cv, heads, stats

HAVE_DATA = (DATA_DIR / "manifest" / "prompts.parquet").exists()


# --------------------------------------------------------------------------------------------------- losses
def test_binomial_nll_matches_hand_computation():
    p = np.array([0.2, 0.5, 0.9])
    v = np.array([3.0, 10.0, 20.0])
    n = np.array([20.0, 20.0, 20.0])
    hand = -sum(vi * math.log(pi) + (ni - vi) * math.log(1 - pi) for pi, vi, ni in zip(p, v, n)) / 60.0
    assert heads.binom_nll_np(p, v, n) == pytest.approx(hand, rel=1e-12)
    z = torch.tensor(np.log(p / (1 - p)))
    t = heads._loss_t(z, torch.tensor(v), torch.tensor(n), 1)
    assert float(t) == pytest.approx(hand, rel=1e-9)
    from mife.probing import b2
    assert float(b2.binom_nll(z, torch.tensor(v), torch.tensor(n))) == pytest.approx(hand, rel=1e-9)


def test_objective2_loss_is_sigmoid_mse():
    z = torch.tensor([0.0, 2.0])
    q = torch.tensor([0.5, 0.1])
    expect = ((0.5 - 0.5) ** 2 + (1 / (1 + math.exp(-2)) - 0.1) ** 2) / 2
    assert float(heads._loss_t(z, q, None, 2)) == pytest.approx(expect, rel=1e-6)


# ------------------------------------------------------------------------------------------------ configurations
def test_23_configurations_and_routing_rules():
    assert len(C.CONFIGS) == 23 and len(set(C.CONFIGS)) == 23
    assert len(C.PROBE_ONLY) == 7 and len(C.B1_FAMILY) == 8 and len(C.B2_FAMILY) == 8
    for c in C.CONFIGS:
        assert C.per_type(c, 1) == any(x in ("H1", "H2", "A1") for x in c.split("+"))
        assert not C.per_type(c, 2)
    assert not C.per_type("B1", 1) and not C.per_type("B2", 1) and not C.per_type("REF_structure", 1)
    assert C.blocks("B1+H2", 1) == ["EP", "ER", "S1", "H"]           # frozen rule embedding appears once
    assert C.blocks("B2+H2", 1) == ["EP_B2", "ER_B2", "S1", "H", "ER"]  # adapted and frozen are different blocks
    assert C.blocks("B1+H1", 2) == ["EP", "ER5", "S2", "H"]          # M2 is a column subset of S2
    assert C.blocks("A1", 2) == ["A5", "KM"] and C.blocks("H1+A1", 2) == ["H", "M2", "A5"]
    assert all("ID" not in b and "rule_id" not in b for c in C.CONFIGS for b in C.blocks(c, 1))


def test_per_type_head_routing_sees_only_its_type():
    rng = np.random.default_rng(0)
    types = rng.choice(["a", "b", "c"], 500)
    fold = rng.integers(0, 5, 500)
    covered = np.zeros(500, int)
    for f in range(5):
        for t in ("a", "b", "c"):
            fit, held = cv.route(types, fold, f, t)
            assert set(types[fit]) <= {t} and set(types[held]) <= {t}
            assert not set(fit) & set(held) and (fold[fit] != f).all() and (fold[held] == f).all()
            covered[held] += 1
    assert (covered == 1).all()  # every row is scored exactly once, by its own type's head


# --------------------------------------------------------------------------------------------------- folds
@pytest.mark.skipif(not HAVE_DATA, reason="needs the frozen manifest")
def test_folds_grouped_deterministic_and_sized():
    a, b = cv.make_folds(write=False), cv.make_folds(write=False)
    assert a["folds"] == b["folds"]
    sizes = sorted(len(f) for f in a["folds"])
    assert sizes == [11, 11, 11, 12, 12]
    allg = [g for f in a["folds"] for g in f]
    assert len(allg) == len(set(allg)) == 57
    for f in range(5):  # every type has fitting rows in every fold
        assert min(a["fitting_groups_per_type"][str(f)].values()) > 0


# ------------------------------------------------------------------------------------------------- bootstrap
def test_bootstrap_resamples_groups_not_prompts():
    W = stats.group_weights(7, 200, 1)
    assert W.shape == (200, 7) and (W.sum(1) == 7).all()
    # two groups, one always violated and one never: resampling GROUPS can only give 0, 1/2 or 1
    df = pd.DataFrame({"group_id": ["g1"] * 50 + ["g2"] * 50, "v": [1.0] * 50 + [0.0] * 50, "n": 1.0})
    r = stats.ratio_ci(df, "v", "n", [], B=4000, seed=3)
    assert r["est"][0] == 0.5 and r["lo"][0] == 0.0 and r["hi"][0] == 1.0
    # a prompt-level bootstrap would give a CI of roughly 0.5 +- 0.1
    est, lo, hi = stats.statistic_ci(df, lambda d: d.v.mean(), B=400, seed=3)
    assert lo == 0.0 and hi == 1.0


# --------------------------------------------------------------------------------------------------- heads
def test_linear_head_recovers_signal_and_scaler_uses_fitting_rows():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 5)).astype(np.float32)
    p = 1 / (1 + np.exp(-(1.5 * X[:, 0] - 1.0)))
    v = rng.binomial(20, p).astype(float)
    n = np.full(4000, 20.0)
    h = heads.fit_linear(X[:3000], v[:3000], n[:3000], 1, 0.01)
    assert np.allclose(h.scaler.mean, X[:3000].mean(0), atol=1e-5)
    pr = h.predict(X[3000:])
    const = np.full(1000, v[:3000].sum() / n[:3000].sum())
    assert heads.binom_nll_np(pr, v[3000:], n[3000:]) < heads.binom_nll_np(const, v[3000:], n[3000:]) - 0.05


# ---------------------------------------------------------------------------------- Objective 2 slots / masks
@pytest.mark.skipif(not (DATA_DIR / "probing_features" / "shared" / "S2_train.npy").exists(), reason="needs caches")
def test_objective2_slots_follow_prompt_order_with_masks():
    from mife.probing import data, features
    P = data.prompts("train")
    S2 = np.load(features.cache_dir() / "S2_train.npy")
    cols = json.loads((features.cache_dir() / "S2_columns.json").read_text())["columns"]
    ix = np.random.default_rng(0).choice(len(P), 300, replace=False)
    for i in ix:
        row = dict(zip(cols, S2[i]))
        types = list(P.ordered_type_ids.iloc[i])
        for s in range(5):
            present = row[f"slot{s + 1}_present"]
            assert present == (1.0 if s < len(types) else 0.0)
            hot = [t for t in features.TYPES if row[f"slot{s + 1}_type:{t}"] == 1.0]
            assert hot == ([types[s]] if s < len(types) else [])
    ER5 = features.load_prompt_block("ER5").reshape(len(P), 5, -1)
    er = np.load(features.cache_dir() / "ER_rules_train.npy")
    rk = pd.read_parquet(features.cache_dir() / "ER_rules_train_keys.parquet")
    rix = {r: j for j, r in enumerate(rk.rule_id)}
    for i in ix[:50]:
        rules = list(P.ordered_rule_ids.iloc[i])
        for s in range(5):
            expect = er[rix[rules[s]]] if s < len(rules) else np.zeros(er.shape[1])
            assert np.array_equal(ER5[i, s], expect)


# ------------------------------------------------------------------------------- tiny synthetic end-to-end run
class _FakeStore:
    """SYNTHETIC store: 10 groups x 40 prompts x k rules; labels depend on one feature column and the type."""

    def __init__(self, objective, rng):
        self.model, self.objective, self.split = "synthetic", objective, "train"
        g = np.repeat([f"s{i:02d}" for i in range(10)], 40 * (3 if objective == 1 else 1))
        self.groups = g
        n = len(g)
        self.Xall = rng.normal(size=(n, 6)).astype(np.float32)
        self.types = rng.choice(["t1", "t2"], n) if objective == 1 else None
        lin = 1.2 * self.Xall[:, 0] - 1.0 + (0.8 if objective == 1 else 0) * (self.types == "t1" if objective == 1 else 0)
        p = 1 / (1 + np.exp(-lin))
        if objective == 1:
            self.n = np.full(n, 20.0)
            self.y = rng.binomial(20, p).astype(float)
            self.U = pd.DataFrame({"prompt_id": [f"p{i}" for i in range(n)], "position": 1, "rule_id": "r", "type_id": self.types})
        else:
            self.n, self.y = None, p + rng.normal(0, 0.02, n).clip(-p, 1 - p)
            self.U = pd.DataFrame({"prompt_id": [f"p{i}" for i in range(n)]})

    def X(self, blocks, rows, hrow=None, pool=None):
        return self.Xall[rows]


@pytest.mark.parametrize("objective,config", [(1, "H1"), (1, "B1"), (2, "H1")])
def test_tiny_synthetic_end_to_end(tmp_path, monkeypatch, objective, config):
    rng = np.random.default_rng(1)
    st = _FakeStore(objective, rng)
    groups = sorted(set(st.groups))
    folds = {"folds": [groups[i::5] for i in range(5)]}
    folds["fold_of_group"] = {g: i for i, f in enumerate(folds["folds"]) for g in f}
    monkeypatch.setattr(cv, "load_folds", lambda: folds)
    monkeypatch.setattr(cv, "CV_DIR", tmp_path / "cv")
    monkeypatch.setattr(cv, "RESULTS", tmp_path)
    monkeypatch.setattr(cv, "MIN_VIOLATED", 0)
    monkeypatch.setattr(cv, "pcfg", lambda: {"heads": {"kinds": ["linear", "mlp"], "l2": [0.01, 0.1], "mlp_epochs": 3,
                                                       "mlp_lr": 1e-2, "mlp_batch": 256}, "seeds": {"heads": 1}})
    out = cv.run_config(st, config, hrow=1, pool="sum")
    assert out["per_type"] == (objective == 1 and config == "H1")
    assert len(out["grid"]) == 4 and out["selected"] in {r["setting"] for r in out["grid"]}
    oof = pd.read_parquet(next((tmp_path / "predictions").rglob("*.parquet")))
    assert len(oof) == len(st.y) and oof.drop(columns=[c for c in oof if not c.startswith(("linear", "mlp"))]).notna().all().all()
    const = st.y.sum() / st.n.sum() if objective == 1 else st.y.mean()
    base = heads.score(np.full(len(st.y), const), st.y, st.n, objective)
    assert out["selected_cv_loss"] < base
    assert (tmp_path / "cv" / "synthetic").exists() and not (DATA_DIR.parent / "results" / "cv" / "synthetic").exists()


# ------------------------------------------------------------------------------------------------- metrics
def test_metrics_match_direct_computation():
    from sklearn.metrics import roc_auc_score
    from mife.probing import metrics as M
    rng = np.random.default_rng(5)
    n_rows = 600
    p = rng.uniform(0.01, 0.99, n_rows).round(2)  # rounding creates ties
    n = np.full(n_rows, 20.0)
    v = rng.binomial(20, p).astype(float)
    groups = rng.choice([f"g{i}" for i in range(6)], n_rows)
    st = M.Obj1Stats(p, v, n, groups, M.bins_from(rng.uniform(0, 1, 1000)))
    m = {k: float(x[0]) for k, x in st.metrics(np.ones(6)).items()}
    assert m["nll"] == pytest.approx(heads.binom_nll_np(p, v, n), rel=1e-9)
    y = np.concatenate([np.ones(int(v.sum())), np.zeros(int((n - v).sum()))])
    s = np.concatenate([np.repeat(p, v.astype(int)), np.repeat(p, (n - v).astype(int))])
    assert m["auroc"] == pytest.approx(roc_auc_score(y, s), rel=1e-9)
    assert m["brier"] == pytest.approx(((s - y) ** 2).mean(), rel=1e-9)
    # a group weight of 2 equals duplicating that group's rows
    w = np.ones(6); w[2] = 2
    dup = np.concatenate([np.arange(n_rows), np.nonzero(groups == "g2")[0]])
    st2 = M.Obj1Stats(p[dup], v[dup], n[dup], groups[dup], st_edges := M.bins_from(rng.uniform(0, 1, 10)))
    st1 = M.Obj1Stats(p, v, n, groups, st_edges)
    a, b = st1.metrics(w), st2.metrics(np.ones(6))
    for k in ("nll", "brier", "auroc", "ece"):
        assert float(a[k][0]) == pytest.approx(float(b[k][0]), rel=1e-9), k
    # risk deciles: thresholds come from the full evaluation set and stay fixed within the bootstrap
    top = p >= np.quantile(p, 0.9)
    assert m["risk_top"] == pytest.approx(v[top].sum() / n[top].sum(), rel=1e-12)
    q = rng.uniform(0, 0.5, 300); pr = (q + rng.normal(0, 0.05, 300)).clip(0, 1)
    g2 = rng.choice(["a", "b", "c"], 300)
    o2 = M.Obj2Stats(pr, q, g2, np.array([0.1, 0.2, 0.3]))
    mm = o2.metrics(np.ones(3))
    assert float(mm["pearson"][0]) == pytest.approx(np.corrcoef(pr, q)[0, 1], rel=1e-9)
    assert float(mm["mae"][0]) == pytest.approx(np.abs(pr - q).mean(), rel=1e-12)


def test_reliability_rules():
    from mife.probing import controlled as K
    assert K.agresti_caffo_reliable(np.array([0.0]), np.array([20.0]), np.array([12.0]), np.array([20.0]))[0]
    assert not K.agresti_caffo_reliable(np.array([5.0]), np.array([20.0]), np.array([7.0]), np.array([20.0]))[0]
