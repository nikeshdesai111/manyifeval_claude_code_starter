"""V2 (2026-11-20 amendment) tests: versioned caches, ridge recipe, tie rule, max-attention helpers, test-set gate."""
import json

import numpy as np
import pytest

from mife.config import DATA_DIR
from mife.probing.v2 import ridge, versioning
from mife.probing.v2.versioning import VersionMismatch

needs_data = pytest.mark.skipif(not (DATA_DIR / "manifest" / "prompts.parquet").exists(), reason="needs the frozen manifest")


# ------------------------------------------------------------------------------------------- versioning
def test_v2_roundtrip_and_refusals(tmp_path):
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    p = tmp_path / "B1_block.npy"
    versioning.save_array(a, p, "B1")
    assert (versioning.load_array(p, "B1") == a).all()
    with pytest.raises(VersionMismatch):  # loaded under another block name
        versioning.load_array(p, "B2")
    m = json.loads(versioning.meta_path(p).read_text())
    for field, bad in (("version", "v1"), ("definition", "B1 = frozen embedding + rule embedding + S1 (V1)")):
        mm = dict(m, **{field: bad})
        versioning.meta_path(p).write_text(json.dumps(mm))
        with pytest.raises(VersionMismatch):
            versioning.load_array(p, "B1")


def test_unversioned_v1_array_refused(tmp_path):
    p = tmp_path / "S1_train.npy"  # a V1-style cache: plain .npy, no sidecar
    np.save(p, np.zeros((2, 2)))
    with pytest.raises(VersionMismatch):
        versioning.load_array(p, "S1v2")


@needs_data
@pytest.mark.parametrize("rel,kind", [("data/probing_features/shared/S1_train.npy", "S1v2"),
                                      ("data/probing_features/shared/EP_train.npy", "B1"),
                                      ("data/probing_features/llama/Asum_train.npy", "A1max")])
def test_real_v1_caches_refused(rel, kind):
    from mife.config import ROOT
    p = ROOT / rel
    if not p.exists():
        pytest.skip("V1 cache absent")
    with pytest.raises(VersionMismatch):
        versioning.load_array(p, kind)


@needs_data
def test_v1_results_refused_as_v2():
    from mife.config import ROOT
    for p in (ROOT / "results/cv/llama/obj1/selection.json", ROOT / "results/cv/qwen/obj2/selection.json"):
        if p.exists():
            with pytest.raises(VersionMismatch):
                versioning.read_versioned_json(p)


# ------------------------------------------------------------------------------------------------ ridge
def _problem(seed=0, n=400, d=30, counts=20):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    z = X @ (rng.standard_normal(d) * 0.3) - 0.4
    v = rng.binomial(counts, 1 / (1 + np.exp(-z))).astype(float)
    return X, v, np.full(n, float(counts))


def test_summed_loss_stationarity():
    """lam/N implementation = minimiser of the SUMMED objective: X'(n p - v) + lam w = 0 and sum(n p - v) = 0."""
    X, v, n = _problem()
    lam = np.r_[np.full(10, 1.0), np.full(20, 300.0)]  # block-specific penalties
    h = ridge.fit(X, v, n, lam, 0, 1)
    p = ridge.expit(X @ h.w + h.b)
    r = n * p - v
    assert np.abs(X.T @ r + lam * h.w).max() < 1e-4
    assert abs(r.sum()) < 1e-2


def test_warm_start_same_optimum():
    X, v, n = _problem(1)
    lam = np.full(X.shape[1], 50.0)
    a = ridge.fit(X, v, n, lam, 0, 1)
    w0 = ridge.fit(X, v, n, np.full(X.shape[1], 5000.0), 0, 1)
    b = ridge.fit(X, v, n, lam, 0, 1, init=np.r_[w0.w, w0.b])
    assert a.converged and b.converged and np.allclose(a.w, b.w, atol=1e-6) and abs(a.b - b.b) < 1e-6


def test_fractional_logit_soft_labels():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((300, 5))
    q = 1 / (1 + np.exp(-(X[:, 0] - 1)))
    h = ridge.fit(X, q, np.ones(300), np.full(5, 1e-3), 0, 1)
    assert abs(h.w[0] - 1) < 0.05 and abs(h.b + 1) < 0.05  # exact soft labels -> recovers the generating logit


# -------------------------------------------------------------------------------------------- tie rule
class _FakeEngine:
    prec = 6
    o = 1

    def __init__(self, losses):
        self.losses = losses

    def oof(self, spec, st):
        return self.losses[tuple(sorted(st.items()))]

    def score(self, p):
        return float(p)


def test_tie_rule():
    from mife.probing.v2.cv import Engine
    L = {(("P", 1e3), ("R", 1e4)): 0.3000001, (("P", 1e4), ("R", 1e3)): 0.3000002, (("P", 1e2), ("R", 1e2)): 0.31,
         (("P", 1e5), ("R", 1e2)): 0.3000004}
    fe = _FakeEngine(L)
    sts = [dict(k) for k in L]
    s = Engine.select(fe, None, sts)
    # 0.3000001/2/4 all round to 0.300000; log sums 7, 7, 7 -> larger P wins
    assert s["setting"] == {"P": 1e5, "R": 1e2}
    # "off" is the strongest penalty: it wins a tie
    s2 = Engine.select(fe, None, [{"P": 1e5, "R": 1e2}], off=np.array(0.3000003))
    assert s2["setting"] is None


# ------------------------------------------------------------------------------------ max attention helpers
def test_span_max_and_bits():
    torch = pytest.importorskip("torch")
    from mife.introspect import bits_to_float32
    from mife.probing.v2 import maxattn
    g = torch.Generator().manual_seed(0)
    att = torch.softmax(torch.randn(3, 4, 40, generator=g), -1).to(torch.bfloat16)
    spans = [(2, 9), (15, 16), (30, 40)]
    mx, sm = maxattn.span_stats(att, spans)
    full = att.float().numpy()
    got = bits_to_float32(maxattn._bits(mx))
    for j, (a, b) in enumerate(spans):
        assert np.array_equal(got[:, :, j], full[:, :, a:b].max(-1))
        assert np.allclose(sm[:, :, j].numpy(), full[:, :, a:b].sum(-1), atol=1e-6)
    assert (got[:, :, 3:] == 0).all()


@needs_data
def test_maxattn_refuses_test_prompts(monkeypatch):
    import pandas as pd
    from mife.probing import data
    from mife.probing.v2 import maxattn
    monkeypatch.setattr(data, "test_unblinded", lambda: False)
    m = pd.read_parquet(DATA_DIR / "manifest" / "prompts.parquet", columns=["prompt_id", "split"])
    tid = m.prompt_id[m.split == "test"].tolist()[:2]
    with pytest.raises(data.TestSetLocked):
        maxattn.run("llama", ids=tid)
    with pytest.raises(data.TestSetLocked):
        maxattn.load("llama", tid)


@needs_data
def test_v2_caches_train_only():
    import pandas as pd
    from mife.probing.v2.versioning import FEAT_V2, RESULTS_V2
    m = pd.read_parquet(DATA_DIR / "manifest" / "prompts.parquet", columns=["prompt_id", "split"])
    test = set(m.prompt_id[m.split == "test"])
    for f in list(FEAT_V2.rglob("*.parquet")) + list(RESULTS_V2.rglob("*.parquet")):
        df = pd.read_parquet(f)
        if "prompt_id" in df:
            assert not (set(df.prompt_id) & test), f


def test_low_rank_rule_block_is_exact():
    """Replacing a low-rank block by its row-space coordinates leaves ridge predictions unchanged."""
    rng = np.random.default_rng(3)
    n = 600
    C = rng.standard_normal((7, 40))                      # 7 distinct "rules"
    B = C[rng.integers(0, 7, n)]
    A = rng.standard_normal((n, 15))
    X = np.concatenate([A, B], 1)
    X = (X - X.mean(0)) / X.std(0)
    z = A[:, 0] + B[:, 0] - 0.3
    v = rng.binomial(20, 1 / (1 + np.exp(-z))).astype(float)
    nn = np.full(n, 20.0)
    lam = np.r_[np.full(15, 30.0), np.full(40, 500.0)]
    full = ridge.fit(X, v, nn, lam, 0, 1)
    U = np.unique(X[:, 15:], axis=0)
    _, sv, Vt = np.linalg.svd(U, full_matrices=False)
    V = Vt[sv > sv[0] * 1e-10].T
    assert V.shape[1] <= 7
    Xr = np.concatenate([X[:, :15], X[:, 15:] @ V], 1)
    red = ridge.fit(Xr, v, nn, np.r_[np.full(15, 30.0), np.full(V.shape[1], 500.0)], 0, 1)
    assert np.allclose(X @ full.w + full.b, Xr @ red.w + red.b, atol=1e-6)
    assert np.allclose(full.w[15:], V @ red.w[15:], atol=1e-6)


def test_freeze_and_v1_untouched():
    """The 2026-11-14 freeze still matches the config and folds it hashed, and unblinding is not approved."""
    from mife.config import ROOT
    from mife.storage import sha256_file
    fz = json.loads((ROOT / "results/prespec/FROZEN.json").read_text())
    assert fz["unblind_approved"] is False and fz["test_runs"] == []
    assert sha256_file(ROOT / "configs/experiment.yaml") == fz["config_sha256"]
    assert sha256_file(ROOT / "results/cv/folds.json") == fz["folds_sha256"]
    assert sha256_file(ROOT / "results/prespec/analysis_plan.md") == fz["analysis_plan_sha256"]
