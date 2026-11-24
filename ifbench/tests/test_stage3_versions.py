"""The version rule: value lists match the released generators; pair categories add up."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import stage3_audit as A  # noqa: E402
import stage3_versions as V  # noqa: E402
from mife_ifb import inventory, source  # noqa: E402


def test_small_value_lists_match_default_generators():
    for t, vals in V.SMALL_VALUES.items():
        seen = set()
        for seed in range(20000):
            seen.add(json.dumps(source.sample_default_kwargs(t, rng_seed=seed), sort_keys=True))
        assert seen == {json.dumps(v, sort_keys=True) for v in vals}, t


def test_every_other_parameterised_type_has_at_least_50_values():
    for t in set(inventory.parameterized()) - set(V.SMALL_VALUES):
        seen = {json.dumps(source.sample_default_kwargs(t, rng_seed=s), sort_keys=True) for s in range(400)}
        assert len(seen) >= 50, t


def test_categories():
    from collections import Counter
    c = Counter(V.category(p) for p in A.all_pairs())
    assert c == {"fixed": 351, "all_combinations": 172, "draws": 338}
