"""Resumable feature extraction, on a TINY RANDOM Llama on CPU.

Everything here is a SYNTHETIC UNIT-TEST FIXTURE (random weights, made-up token IDs); nothing is written under data/.
"""
import json

import numpy as np
import pandas as pd
import pytest
import torch

from mife import introspect as I


@pytest.fixture
def tiny(tmp_path, monkeypatch, fake_manifest):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    model = LlamaForCausalLM(cfg).to(torch.bfloat16).eval()
    model.config._attn_implementation = "eager"
    rng = np.random.default_rng(0)
    rows = []
    for n in range(7):
        L = 12 + n
        rows.append({"prompt_id": f"p{n}", "input_ids": rng.integers(0, 64, L).tolist(), "n_input_tokens": L,
                     "rule_token_spans": [[2, 4], [5, 8]][: 1 + n % 2], "task_token_span": [8, 11]})
    fake_manifest([r["prompt_id"] for r in rows])
    for r in rows:
        r["prompt_sha256"], r["rendered_sha256"] = f"P-{r['prompt_id']}", f"R-{r['prompt_id']}"
    (tmp_path / "tokenized").mkdir()
    pd.DataFrame(rows).to_parquet(tmp_path / "tokenized" / "llama.parquet")
    monkeypatch.setattr(I, "DATA_DIR", tmp_path)
    monkeypatch.setattr(I, "load_torch_model", lambda m, d=None: model)
    monkeypatch.setenv("MIFE_TORCH_DEVICE", "cpu")
    return tmp_path, [r["prompt_id"] for r in rows]


def _load(out):
    idx = pd.read_parquet(out / "index.parquet")
    H = {int(n): np.load(out / f"hidden_{int(n):05d}.npy") for n in idx.shard.unique()}
    A = {int(n): np.load(out / f"attn_{int(n):05d}.npz") for n in idx.shard.unique()}
    hid = {r.prompt_id: H[r.shard][r.row] for r in idx.itertuples()}
    att = {r.prompt_id: A[r.shard]["rule_sum"][r.row] for r in idx.itertuples()}
    return idx, hid, att


def test_interrupted_run_resumes_to_identical_features(tiny):
    root, pids = tiny
    I.run("llama", "full", pids, n_diag=2, shard_size=3)
    m1 = I.run("llama", "resumed", pids, n_diag=2, shard_size=3, max_prompts=4)   # "interrupted" after 4
    assert m1["n_prompts"] == 4 and m1["n_requested_missing"] == 3
    m2 = I.run("llama", "resumed", pids, n_diag=2, shard_size=3)                    # resume
    assert m2["n_prompts"] == 7 and m2["n_requested_missing"] == 0
    assert len(m2["sessions"]) == 2 and m2["sessions"][1]["n_prompts_skipped_done"] == 4
    a_idx, a_h, a_att = _load(root / "features" / "llama" / "full")
    b_idx, b_h, b_att = _load(root / "features" / "llama" / "resumed")
    assert sorted(b_idx.prompt_id) == sorted(pids) and b_idx.prompt_id.is_unique
    for p in pids:
        assert np.array_equal(a_h[p], b_h[p])
        assert np.allclose(a_att[p], b_att[p], equal_nan=True)
    # third call is a no-op and keeps the metadata
    m3 = I.run("llama", "resumed", pids, n_diag=2, shard_size=3)
    assert m3["n_prompts"] == 7 and len(m3["sessions"]) == 2
    assert sorted(m3["diag_full_attention_prompts"]) == ["p0", "p1"]


def test_uncommitted_shard_is_recomputed(tiny):
    root, pids = tiny
    out = root / "features" / "llama" / "r"
    I.run("llama", "r", pids, shard_size=3, max_prompts=6)       # shards 0 and 1
    (out / "index_00001.parquet").unlink()                        # crash after writing arrays, before commit
    done, nxt = I.done_features(out, "llama")
    assert done == set(pids[:3]) and nxt == 1
    m = I.run("llama", "r", pids, shard_size=3)
    assert m["n_prompts"] == 7 and pd.read_parquet(out / "index.parquet").prompt_id.is_unique


def test_refuses_to_mix_backends(tiny):
    root, pids = tiny
    out = root / "features" / "llama" / "x"
    out.mkdir(parents=True)
    (out / "features_backend.json").write_text(json.dumps({"backend": "transformers+torch-cuda"}))
    with pytest.raises(SystemExit, match="refusing"):
        I.run("llama", "x", pids)


def test_features_from_another_manifest_are_refused(tiny, fake_manifest):
    from mife.integrity import StaleDataError
    root, pids = tiny
    out = root / "features" / "llama" / "r"
    I.run("llama", "r", pids, shard_size=3, max_prompts=3)
    idx = pd.read_parquet(out / "index_00000.parquet")
    idx.loc[0, "rendered_sha256"] = "R-OLD"
    idx.to_parquet(out / "index_00000.parquet")
    with pytest.raises(StaleDataError, match="stale"):
        I.run("llama", "r", pids, shard_size=3)


def test_index_without_per_shard_hashes_is_refused(tmp_path):
    from mife.integrity import StaleDataError
    pd.DataFrame({"prompt_id": ["a", "b"], "shard": [0, 0]}).to_parquet(tmp_path / "index.parquet")
    with pytest.raises(StaleDataError):
        I.done_features(tmp_path, "llama")
