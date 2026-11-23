"""Split amendment S3 (pair-coverage split). Synthetic pool metadata except the last test, which needs the manifest."""
import json
import random

import pandas as pd
import pytest

from mife import source as S, split_v3 as V, tasks as T
from mife.config import DATA_DIR, ROOT


def _synthetic_meta(n_groups=36, seed=0):
    rng = random.Random(seed)
    fams = ["A", "B", "C", "D"]
    out = []
    for i in range(n_groups):
        types = []
        while len(types) < 6:
            t = rng.choice(S.TYPES)
            if all(not S.conflicts(t, u) for u in types):
                types.append(t)
        out.append({"group_id": f"g{i + 1:02d}", "task_keys": [10 * i + 1, 10 * i + 2, 10 * i + 3],
                    "families": [rng.choice(fams) for _ in range(3)], "types": sorted(types)})
    return out


def test_compatible_pairs_exclude_exclusive_groups():
    pairs = V.compatible_type_pairs()
    assert len(S.TYPES) == 15 and len(pairs) == 98  # 105 pairs minus 1 + 3 + 3 mutually exclusive
    assert not any(S.conflicts(a, b) for a, b in pairs)


def test_related_units_match_original_split_components():
    meta = _synthetic_meta(72)
    meta[0]["task_keys"][0], meta[5]["task_keys"][0] = 1001, 1092  # japan cluster -> g01 + g06
    units = V.related_units(meta)
    assert ["g01", "g06"] in units and sum(len(u) for u in units) == 72


def test_pair_coverage_split_satisfies_constraints_and_is_deterministic():
    meta = _synthetic_meta(72, seed=3)
    pairs_all = V.pair_coverage(meta, {g["group_id"] for g in meta})
    m = min(2, min(pairs_all.values()))
    test, info = V.split_groups_pair_coverage(meta, 24, seed=7, min_pair_test_groups=m, n_restarts=50)
    test2, _ = V.split_groups_pair_coverage(meta, 24, seed=7, min_pair_test_groups=m, n_restarts=50)
    assert test == test2 and len(test) == 24
    cov = V.pair_coverage(meta, test)
    assert min(cov.values()) >= m and info["min_pair_coverage"] == min(cov.values())
    assert min(info["type_test_groups"].values()) >= 1 and min(info["type_train_groups"].values()) >= 5


def test_infeasible_threshold_raises():
    meta = _synthetic_meta(72, seed=3)
    worst = min(V.pair_coverage(meta, {g["group_id"] for g in meta}).values())
    with pytest.raises(ValueError, match="infeasible"):
        V.split_groups_pair_coverage(meta, 24, seed=7, min_pair_test_groups=worst + 1, n_restarts=5)


@pytest.mark.skipif(not (DATA_DIR / "manifest" / "prompts.parquet").exists(), reason="needs the frozen manifest")
def test_real_manifest_reproduces_recorded_proposal_and_old_split():
    import yaml

    cols = ["group_id", "split", "task_key", "task_family", "ordered_type_ids"]
    prompts = pd.read_parquet(DATA_DIR / "manifest" / "prompts.parquet", columns=cols)
    meta = V.group_meta_from_prompts(prompts)
    old = sorted(prompts.loc[prompts["split"] == "test", "group_id"].unique())
    assert sorted(T.split_groups(meta, 15, 20261113)[0]) == old  # rebuilt metadata reproduces the 2026-11-14 split
    cfg = yaml.safe_load((ROOT / "configs" / "split_s3.yaml").read_text())
    test, info = V.split_groups_pair_coverage(
        meta, cfg["n_test_groups"], cfg["seed"], min_pair_test_groups=cfg["min_pair_test_groups"],
        min_type_test_groups=cfg["min_type_test_groups"], min_type_train_groups=cfg["min_type_train_groups"],
        n_restarts=cfg["n_restarts"], max_steps=cfg["max_steps"])
    recorded = json.loads((ROOT / "results" / "split_s3" / "proposed_split.json").read_text())
    assert sorted(test) == recorded["test_groups"] and info["min_pair_coverage"] >= 2
