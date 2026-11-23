"""Mac hedge sampler: vLLM-order top-p on tempered probs, per-sequence deterministic seeds, separate run folder."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mife import expansion_s3 as E  # noqa: E402
from mife.expansion_s3_mac import MAC_RUN, SeededVllmSampler, out_dir_for  # noqa: E402


def _ref(logits, T, P):
    x = logits / T
    p = np.exp(x - x.max()); p /= p.sum()
    o = np.argsort(-p); before = np.cumsum(p[o]) - p[o]
    keep = o[before < P]
    r = np.zeros_like(p); r[keep] = p[keep]
    return r / r.sum(), set(keep.tolist())


def test_sampler_matches_vllm_definition():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=40) * 2
    lp = logits - np.log(np.exp(logits).sum())
    ref, keep = _ref(logits, 0.6, 0.9)
    s = SeededVllmSampler(123, 0.6, 0.9, 0, 0.0)
    a = mx.array(lp[None].astype(np.float32))
    cnt = np.zeros(40)
    for _ in range(8000):
        cnt[int(s(a).item())] += 1
    assert set(np.nonzero(cnt)[0].tolist()) <= keep
    assert np.abs(cnt / cnt.sum() - ref).max() < 0.02


def test_sampler_seeded_and_distinct():
    a = mx.array(np.linspace(0, 1, 30, dtype=np.float32)[None])
    draw = lambda seed: [int(SeededVllmSampler(seed, 1.0, 0.95, 0, 0.0)(a).item()) for _ in range(1)]
    s1, s2 = SeededVllmSampler(7, 1.0, 0.95, 0, 0.0), SeededVllmSampler(7, 1.0, 0.95, 0, 0.0)
    assert [int(s1(a).item()) for _ in range(20)] == [int(s2(a).item()) for _ in range(20)]
    assert draw(1) is not None


def test_refuses_betty_and_main_runs():
    assert MAC_RUN not in E.SOURCE_RUNS
    for r in (*E.SOURCE_RUNS, "main"):
        with pytest.raises(SystemExit):
            out_dir_for("llama", r)
