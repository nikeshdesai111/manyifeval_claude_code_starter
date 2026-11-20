"""CUDA-backend helpers that can be tested without a GPU or vLLM.

Arrays/tables built here are SYNTHETIC UNIT-TEST FIXTURES; nothing is written under data/.
"""
import numpy as np
import pandas as pd
import pytest
import torch

from mife import generate_vllm as GV
from mife import sampler_semantics as SS
from mife import xbackend as XB
from mife.config import DATA_DIR
from mife.consistency_cuda import compare_topk
from mife.generate_mlx import gen_settings, settings_hash


def _logits(seed=0, v=500):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(8, v, generator=g) * 3


def test_sampler_definitions_agree_without_temperature_or_topk():
    lg = _logits()
    a = SS.mlx_distribution(lg, 1.0, 0.9, 0, 0.0)
    b = SS.vllm_distribution(lg, 1.0, 0.9, 0, 0.0)
    assert torch.allclose(a, b, atol=1e-6)


def test_mlx_nucleus_on_untempered_probs_is_wider():
    lg = _logits(1)
    a = SS.mlx_distribution(lg, 0.6, 0.9, 0, 0.0)
    b = SS.vllm_distribution(lg, 0.6, 0.9, 0, 0.0)
    assert ((b > 0) <= (a > 0)).all()          # vLLM support is a subset of MLX support
    assert (a > 0).sum() > (b > 0).sum()       # and strictly smaller here
    for d in (a, b):
        assert torch.allclose(d.sum(-1), torch.ones(8), atol=1e-5)


def test_topk_only_is_order_invariant():
    lg = _logits(2)
    a = SS.mlx_distribution(lg, 0.7, 1.0, 20, 0.0)
    b = SS.vllm_distribution(lg, 0.7, 1.0, 20, 0.0)
    assert torch.allclose(a, b, atol=1e-6) and ((a > 0).sum(-1) == 20).all()


def test_sampling_kwargs_and_stop_stripping():
    st = {"temperature": 0.6, "top_p": 0.9, "top_k": 0, "min_p": 0.0, "max_new_tokens": 2048}
    kw = GV.sampling_kwargs(st)
    assert kw["top_k"] == -1 and kw["max_tokens"] == 2048 and kw["repetition_penalty"] == 1.0
    assert GV.sampling_kwargs({**st, "top_k": 20})["top_k"] == 20
    assert GV.strip_stop([5, 6, 128009], "stop", {128009}) == [5, 6]
    assert GV.strip_stop([5, 6, 7], "length", {128009}) == [5, 6, 7]


def test_mac_pilot_run_is_protected():
    with pytest.raises(SystemExit, match="protected"):
        GV.check_run_name("llama", "pilot")


def test_sample_seeds_deterministic_and_distinct():
    s = {GV.sample_seed("llama", "r", "p1", i) for i in range(20)}
    assert len(s) == 20 and GV.sample_seed("llama", "r", "p1", 3) == GV.sample_seed("llama", "r", "p1", 3)


def test_vllm_settings_record_sampler_and_keep_mlx_hash():
    v = gen_settings("llama", "vllm")
    assert v["backend"] == "vllm" and "sampler_semantics" in v
    assert "sampler_semantics" not in gen_settings("llama")
    shard = DATA_DIR / "generations" / "llama" / "pilot" / "shard_00000.parquet"
    if shard.exists():  # MLX hash must be unchanged so the Mac pilot stays consistent
        assert set(pd.read_parquet(shard, columns=["settings_hash"]).settings_hash) == {settings_hash("llama")}


def test_compare_hidden_identity_and_perturbation():
    rng = np.random.default_rng(0)
    H = rng.normal(size=(4, 3, 64)).astype(np.float32)
    d = XB.compare_hidden(H, H.copy())
    assert np.allclose(d.cos_min, 1, atol=1e-6) and (d.max_abs_diff_max == 0).all()
    d2 = XB.compare_hidden(H, H + rng.normal(scale=0.5, size=H.shape).astype(np.float32))
    assert (d2.cos_min < XB.COS_FLAG).all()


def test_bootstrap_diff_detects_only_real_shift():
    rows = [{"prompt_id": f"p{i}", "k": 1 + i % 5, "v": i % 4, "n": 20} for i in range(100)]
    a = pd.DataFrame(rows)
    same = XB.bootstrap_diff(a, a.copy(), "k", n_boot=200)
    assert np.allclose(same["diff"], 0) and not same.ci_excludes_0.any()
    b = a.assign(v=a.v + 8)
    shifted = XB.bootstrap_diff(a, b, "k", n_boot=200)
    assert shifted.ci_excludes_0.all() and (shifted["diff"] > 0).all()


def test_compare_topk():
    lp = np.log(np.array([0.5, 0.3, 0.15, 0.05]))
    r = compare_topk(lp, [0, 1], [float(lp[0]), float(lp[1])])
    assert r["top1_agree"] and r["max_abs_logprob_diff_on_vllm_topk"] == 0


@pytest.mark.skipif(not (DATA_DIR / "tokenized" / "llama.parquet").exists(), reason="tokenized data not built")
def test_retokenization_matches_frozen_tokens_on_this_machine():
    pids = list(pd.read_parquet(DATA_DIR / "pilot" / "pilot_prompts.parquet").prompt_id)[:5]
    for m in ("llama", "qwen"):
        r = XB.check_tokens(m, pids)
        assert r["identical_token_ids_and_spans"] and r["eos_identical"], r


def test_chunk_ids_partition_is_exact_and_deterministic():
    ids = [f"p{i}" for i in range(53136)]
    parts = [GV.chunk_ids(ids, c, 12) for c in range(12)]
    assert sum(parts, []) == ids and {len(x) for x in parts} == {4428}
    assert GV.chunk_ids(ids, 3, 12) == parts[3]
    with pytest.raises(ValueError):
        GV.chunk_ids(ids, 12, 12)


def test_import_vllm_pilot_and_refuse_backend_mixing(tmp_path, fake_manifest):
    import json

    from mife import generate_mlx as G
    fake_manifest(["p0", "p1"])
    src, dst = tmp_path / "pilot_betty", tmp_path / "main"
    src.mkdir()
    rows = [{"response_id": f"llama:p{p}:s{i:02d}", "prompt_id": f"p{p}", "sample_index": i,
             "response_text": "FIXTURE", "settings_hash": "h", "rendered_sha256": f"R-p{p}"}
            for p in range(2) for i in range(20)]
    pd.DataFrame(rows).to_parquet(src / "shard_00000.parquet", index=False)
    (src / "run_summary_1.json").write_text(json.dumps({"settings": gen_settings("llama", "vllm")}))
    info = G.import_pilot("llama", src, dst)  # settings inferred from the pilot's backend (vllm)
    assert info["n_rows_kept"] == 40 and info["main_settings"]["backend"] == "vllm"
    # an MLX pilot must not be imported into a main run that already holds vLLM generations
    (dst / "pilot_import.json").unlink()
    (dst / "run_summary_x.json").write_text(json.dumps({"settings": {"backend": "vllm"}}))
    mlx_src = tmp_path / "pilot_mac"
    mlx_src.mkdir()
    pd.DataFrame(rows).to_parquet(mlx_src / "shard_00000.parquet", index=False)
    (mlx_src / "run_summary_1.json").write_text(json.dumps({"settings": gen_settings("llama")}))
    with pytest.raises(RuntimeError, match="refusing to import"):
        G.import_pilot("llama", mlx_src, dst)


def test_fp32_reference_summary_ranks_the_closer_backend():
    """SYNTHETIC: backend 'near' has half the noise of 'far'; the summary must say so."""
    from mife import fp32_reference as FR
    rng = np.random.default_rng(0)
    R = rng.normal(size=(30, 4, 256)).astype(np.float32)
    R[:, 0] = FR.bf16_round(R[:, 0])          # row 0 exactly representable, as embeddings are
    near = FR.bf16_round(R + rng.normal(scale=0.01, size=R.shape).astype(np.float32))
    far = FR.bf16_round(R + rng.normal(scale=0.02, size=R.shape).astype(np.float32))
    near[:, 0] = far[:, 0] = R[:, 0]
    A = {k: rng.random((30, 3, 2, 5)).astype(np.float32) for k in ("rule_sum", "rule_mean")}
    A["task_sum"] = rng.random((30, 3, 2)).astype(np.float32)
    s = FR.summarize(R, A, {"near": (near, A), "far": (far, A)})
    assert s["paired"]["frac_near_closer"] > 0.95 and 0.4 < s["paired"]["median_ratio_near_over_far"] < 0.6
    assert s["per_row"][0]["near_rel_err_median"] == 0 and s["attention"]["near"]["rule_sum"]["max_abs_err"] == 0
    assert s["bf16_rounding_floor_median_all_rows"] < s["near_rel_err_median_all_rows"]
    assert s["between_prompt_rel_l2_median_all_rows"] > 1.0
