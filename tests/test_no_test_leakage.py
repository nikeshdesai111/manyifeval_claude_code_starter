"""The 15 TEST groups must not be read by any CV / tuning code path before the pre-specification is frozen and
unblinding is approved. These tests fail if a loader, cache, bundle or saved CV output contains a TEST prompt_id."""
import json

import numpy as np
import pandas as pd
import pytest

from mife.config import DATA_DIR
from mife.probing import data

pytestmark = pytest.mark.skipif(not (DATA_DIR / "manifest" / "prompts.parquet").exists(), reason="needs the frozen manifest")


@pytest.fixture(scope="module")
def split_ids():
    m = pd.read_parquet(DATA_DIR / "manifest" / "prompts.parquet", columns=["prompt_id", "split"])
    return set(m.prompt_id[m.split == "train"]), set(m.prompt_id[m.split == "test"])


@pytest.fixture
def locked(monkeypatch):
    monkeypatch.setattr(data, "test_unblinded", lambda: False)


def test_test_split_refused_while_locked(locked):
    for split in ("test", "all"):
        with pytest.raises(data.TestSetLocked):
            data.prompts(split)
        with pytest.raises(data.TestSetLocked):
            data.labels_prompt_rule("llama", split)


def test_feature_readers_refuse_test_ids(locked, split_ids):
    _, test = split_ids
    tid = sorted(test)[:3]
    with pytest.raises(data.TestSetLocked):
        data.hidden_states("llama", tid, [16])
    with pytest.raises(data.TestSetLocked):
        data.rule_attention("qwen", tid)


def test_cv_store_reads_train_only(locked, split_ids, monkeypatch):
    """Every prompt_id the CV store asks the label readers for is a TRAIN id."""
    from mife.probing import cv
    train, test = split_ids
    seen = []
    orig = data._read_filtered

    def spy(path, ids, columns=None):
        seen.extend(ids)
        return orig(path, ids, columns)
    monkeypatch.setattr(data, "_read_filtered", spy)
    for o in (1, 2):
        st = cv.Store("llama", o)
        assert set(st.U.prompt_id) <= train
        assert not set(st.U.prompt_id) & test
    assert seen and set(seen) <= train


def test_cached_features_and_bundle_are_train_only(split_ids):
    from mife.probing import features
    train, test = split_ids
    root = data.PROBE_FEAT
    for p in root.rglob("*_keys*.parquet"):
        ids = set(pd.read_parquet(p).get("prompt_id", pd.Series(dtype=str)))
        assert not ids & test, p
    if not data.test_unblinded():
        assert not list(root.rglob("*_test*")), "test-split feature caches exist before unblinding"
    b = root / "b2_bundle" / "shared_prompts.parquet"
    if b.exists():
        ids = set(pd.read_parquet(b).prompt_id)
        assert ids <= train and len(ids) == len(train)


def test_saved_cv_outputs_are_train_only(split_ids):
    _, test = split_ids
    for p in (data.RESULTS / "predictions").glob("*/obj*/cv/*.parquet"):
        assert not set(pd.read_parquet(p, columns=["prompt_id"]).prompt_id) & test, p
    folds = data.RESULTS / "cv" / "folds.json"
    if folds.exists():
        f = json.loads(folds.read_text())
        test_groups = set(pd.read_parquet(DATA_DIR / "manifest" / "prompts.parquet", columns=["group_id", "split"])
                          .query("split == 'test'").group_id)
        assert not {g for fold in f["folds"] for g in fold} & test_groups
