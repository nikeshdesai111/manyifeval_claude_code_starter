"""Integration checks on the built manifest (skipped until `python -m mife.cli manifest` has run)."""
import json

import numpy as np
import pandas as pd
import pytest

from mife import manifest, storage
from mife.config import DATA_DIR

M = DATA_DIR / "manifest"
built = pytest.mark.skipif(not (M / "prompts.parquet").exists(), reason="manifest not built")


@built
def test_manifest_counts_and_split():
    p = pd.read_parquet(M / "prompts.parquet")
    r = pd.read_parquet(M / "prompt_rules.parquet")
    assert len(p) == 53136 and len(r) == 221616
    assert p.groupby("split")["group_id"].nunique().to_dict() == {"train": 57, "test": 15}
    assert p.groupby("split").size().to_dict() == {"train": 42066, "test": 11070}
    assert p.task_key.nunique() == 216
    assert (r.groupby("prompt_id").size() == p.set_index("prompt_id")["k"].reindex(r.groupby("prompt_id").size().index)).all()
    for row in p.sample(200, random_state=0).itertuples():  # char spans point at the exact rule text
        rr = r[r.prompt_id == row.prompt_id].sort_values("position")
        assert [row.prompt_text[a:b] for a, b in zip(rr.char_start, rr.char_end)] == list(rr.rule_text)


@built
def test_manifest_rebuild_is_deterministic(tmp_path):
    s = manifest.build(out_dir=tmp_path)
    ref = json.loads((M / "manifest_summary.json").read_text())
    assert s["manifest_sha256"] == ref["manifest_sha256"]


@built
def test_tokenized_spans_decode_to_rule_text():
    from mife import chat
    r = pd.read_parquet(M / "prompt_rules.parquet")
    for mk in ("llama", "qwen"):
        tok = chat.tokenizer(mk)
        t = pd.read_parquet(DATA_DIR / "tokenized" / f"{mk}.parquet").sample(100, random_state=1)
        for row in t.itertuples():
            texts = list(r[r.prompt_id == row.prompt_id].sort_values("position").rule_text)
            for (a, b), txt in zip(row.rule_token_spans, texts):
                assert tok.decode(list(row.input_ids[a:b])).strip() == txt


def test_atomic_writes_and_bf16_roundtrip(tmp_path):
    storage.write_json({"a": 1}, tmp_path / "x.json")
    assert json.loads((tmp_path / "x.json").read_text()) == {"a": 1}
    assert not list(tmp_path.glob(".*tmp"))
    torch = pytest.importorskip("torch")
    from mife.introspect import _bf16_bits, bits_to_float32
    x = torch.randn(64, dtype=torch.float32).to(torch.bfloat16)
    assert np.array_equal(bits_to_float32(_bf16_bits(x)), x.float().numpy())


@built
def test_pilot_selection_train_only():
    sel = pd.read_parquet(DATA_DIR / "pilot" / "pilot_prompts.parquet")
    p = pd.read_parquet(M / "prompts.parquet").set_index("prompt_id")
    assert len(sel) == 100 and (p.loc[sel.prompt_id, "split"] == "train").all()
    assert sel.groupby("k").size().tolist() == [20] * 5
