"""TEST groups must not be read by any CV / tuning code path before unblinding is approved. These tests fail if a
loader, cache, bundle or saved CV output contains a TEST prompt_id.

Since amendment S3 (2026-11-23) the loaders use the active split (`mife.splits`, 22 TEST groups). Artifacts built
under the 2026-11-14 split (V1 caches, predictions and results/cv/folds.json) are checked against that split, which
the manifest `split` column still records."""
import json

import numpy as np
import pandas as pd
import pytest

from mife import splits
from mife.config import DATA_DIR
from mife.probing import data

pytestmark = pytest.mark.skipif(not (DATA_DIR / "manifest" / "prompts.parquet").exists(), reason="needs the frozen manifest")


@pytest.fixture(scope="module")
def split_ids():
    """(train, test) prompt ids of the ACTIVE split."""
    m = splits.apply(pd.read_parquet(DATA_DIR / "manifest" / "prompts.parquet", columns=["prompt_id", "group_id", "split"]))
    return set(m.prompt_id[m.split == "train"]), set(m.prompt_id[m.split == "test"])


@pytest.fixture(scope="module")
def v1_split_ids():
    """(train, test) prompt ids of the 2026-11-14 split, for artifacts built under it."""
    m = pd.read_parquet(DATA_DIR / "manifest" / "prompts.parquet", columns=["prompt_id", "split"])
    return set(m.prompt_id[m.split == "train"]), set(m.prompt_id[m.split == "test"])


def test_loaders_use_active_split(split_ids):
    train, _ = split_ids
    assert set(data.prompt_ids("train")) == train
    assert set(data.prompts("train").group_id).isdisjoint(splits.test_groups())


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


def test_cached_features_and_bundle_are_train_only(v1_split_ids):
    from mife.probing import features
    train, test = v1_split_ids  # V1 caches were built under the 2026-11-14 split
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


def test_saved_cv_outputs_are_train_only(v1_split_ids):
    _, test = v1_split_ids  # V1 predictions and folds were built under the 2026-11-14 split
    for p in (data.RESULTS / "predictions").glob("*/obj*/cv/*.parquet"):
        assert not set(pd.read_parquet(p, columns=["prompt_id"]).prompt_id) & test, p
    folds = data.RESULTS / "cv" / "folds.json"
    if folds.exists():
        f = json.loads(folds.read_text())
        test_groups = set(pd.read_parquet(DATA_DIR / "manifest" / "prompts.parquet", columns=["group_id", "split"])
                          .query("split == 'test'").group_id)
        assert not {g for fold in f["folds"] for g in fold} & test_groups


def test_active_split_folds_are_train_only():
    folds = splits.cv_dir() / "folds.json"
    if splits.version() == splits.MANIFEST_VERSION or not folds.exists():
        pytest.skip("no folds for a non-manifest active split")
    f = json.loads(folds.read_text())
    assert f["split_version"] == splits.version()
    in_folds = {g for fold in f["folds"] for g in fold}
    assert not in_folds & splits.test_groups()
    assert in_folds == set(data.prompts("train").group_id)
