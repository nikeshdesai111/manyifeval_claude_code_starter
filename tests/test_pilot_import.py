"""Importing pilot samples 0..n-1 into the main run (n=20 by design; n<pilot count also tested).

All data here is a SYNTHETIC UNIT-TEST FIXTURE (fake response text); it is never written under data/.
"""
import json

import pandas as pd
import pytest

from mife import generate_mlx as G
from mife.verify_stage import expected_samples_for

SETTINGS = {"model_key": "llama", "temperature": 0.6, "top_p": 0.9, "n_samples": 10}


@pytest.fixture(autouse=True)
def _hashes(fake_manifest):
    fake_manifest(["p1", "p2", "p3"])


def _fixture(src, n_pilot=20, prompts=("p1", "p2", "p3"), settings=None):
    src.mkdir(parents=True)
    rows = [{"response_id": f"llama:{p}:s{i:02d}", "prompt_id": p, "sample_index": i,
             "response_text": f"FIXTURE {p} {i}", "settings_hash": "pilothash", "rendered_sha256": f"R-{p}"}
            for p in prompts for i in range(n_pilot)]
    df = pd.DataFrame(rows)
    df.iloc[: len(df) // 2].to_parquet(src / "shard_00000.parquet", index=False)
    df.iloc[len(df) // 2:].to_parquet(src / "shard_00001.parquet", index=False)
    (src / "run_summary_1.json").write_text(json.dumps({"settings": settings or {**SETTINGS, "n_samples": 20}}))


def test_import_keeps_first_n_and_is_idempotent(tmp_path):
    src, dst = tmp_path / "pilot", tmp_path / "main"
    _fixture(src)
    info = G.import_pilot("llama", src, dst, SETTINGS)
    assert info["n_rows_kept"] == 30 and info["n_rows_excluded"] == 30 and info["n_prompts"] == 3
    got = pd.concat([pd.read_parquet(p) for p in sorted(dst.glob("shard_*.parquet"))])
    assert sorted(got["sample_index"].unique()) == list(range(10))
    assert (got.groupby("prompt_id").size() == 10).all()
    assert (got["source_run"] == "pilot").all() and got["response_id"].is_unique
    assert G.done_prompts(dst, 10, "llama") == {"p1", "p2", "p3"}
    n_files = len(list(dst.glob("shard_*.parquet")))
    assert G.import_pilot("llama", src, dst, SETTINGS)["status"] == "already imported"
    assert len(list(dst.glob("shard_*.parquet"))) == n_files


def test_import_refuses_changed_decoding(tmp_path):
    src = tmp_path / "pilot"
    _fixture(src, settings={**SETTINGS, "n_samples": 20, "temperature": 0.7})
    with pytest.raises(RuntimeError, match="decoding settings differ"):
        G.import_pilot("llama", src, tmp_path / "main", SETTINGS)


def test_import_refuses_missing_indices(tmp_path):
    src = tmp_path / "pilot"
    _fixture(src, n_pilot=8)
    with pytest.raises(RuntimeError, match="lack sample indices"):
        G.import_pilot("llama", src, tmp_path / "main", SETTINGS)


def test_expected_samples_per_run():
    assert expected_samples_for("pilot") == 20
    assert expected_samples_for("main") == 20


def test_stale_generations_are_refused(tmp_path, fake_manifest):
    """A shard whose rendered hash differs from the current tokenization must never be resumed or imported."""
    from mife.integrity import StaleDataError
    src, dst = tmp_path / "pilot", tmp_path / "main"
    _fixture(src)
    df = pd.read_parquet(src / "shard_00000.parquet")
    df.loc[df.prompt_id == "p1", "rendered_sha256"] = "R-OLD"
    df.to_parquet(src / "shard_00000.parquet", index=False)
    with pytest.raises(StaleDataError, match="stale"):
        G.done_prompts(src, 20, "llama")
    with pytest.raises(StaleDataError, match="stale"):
        G.import_pilot("llama", src, dst, {**SETTINGS, "n_samples": 20})
    fake_manifest(["p2", "p3"])            # p1 no longer in the manifest at all
    with pytest.raises(StaleDataError, match="not in the current manifest"):
        G.done_prompts(src, 20, "llama")


def test_shard_without_hash_is_refused(tmp_path):
    from mife.integrity import StaleDataError
    src = tmp_path / "pilot"
    _fixture(src)
    for f in src.glob("shard_*.parquet"):
        pd.read_parquet(f).drop(columns="rendered_sha256").to_parquet(f, index=False)
    with pytest.raises(Exception):
        G.done_prompts(src, 20, "llama")


def test_stale_tokenization_and_verification_are_refused(tmp_path, fake_manifest, monkeypatch):
    from mife import integrity
    from mife.integrity import StaleDataError
    fake_manifest(["p1", "p2"])
    root = integrity.DATA_DIR
    tok = pd.read_parquet(root / "tokenized" / "llama.parquet")
    tok.loc[0, "prompt_sha256"] = "P-OLD"
    tok.to_parquet(root / "tokenized" / "llama.parquet")
    with pytest.raises(StaleDataError, match="re-run"):
        integrity.tokenized_hashes("llama")
    fake_manifest(["p1", "p2"])
    v = root / "verification" / "llama" / "pilot"
    v.mkdir(parents=True)
    pd.DataFrame({"prompt_id": ["p1", "p2"], "prompt_sha256": ["P-p1", "P-OLD"],
                  "rendered_sha256": ["R-p1", "R-p2"]}).to_parquet(v / "labels_prompt.parquet")
    with pytest.raises(StaleDataError, match="stale"):
        integrity.read_verification("llama", "pilot", "labels_prompt.parquet")
