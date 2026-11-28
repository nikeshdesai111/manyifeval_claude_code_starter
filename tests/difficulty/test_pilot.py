"""Calibration-pilot helpers on SYNTHETIC data (no model, no API)."""
import numpy as np
import pandas as pd

from mife.difficulty import pilot as PL


def test_54_rule_instances_one_per_type_and_level():
    r = PL.rule_instances(1)
    assert len(r) == 54 and len({(x["type_id"], x["designed_difficulty"]) for x in r}) == 54
    assert PL.rule_instances(1) == r  # deterministic


def test_analyze_criteria_on_synthetic_rates(tmp_path, monkeypatch):
    monkeypatch.setattr(PL, "PILOT", tmp_path)
    rng = np.random.default_rng(0)
    rows = []
    true = {"good": {"easy": 0.1, "medium": 0.4, "hard": 0.7}, "flat": {"easy": 0.3, "medium": 0.32, "hard": 0.35}}
    for t, rates in true.items():
        for m in ("gemma2b", "llama", "gemma9b"):
            for task in range(48):
                for lvl, p in rates.items():
                    for s in range(5):
                        rows.append({"model_key": m, "task_id": f"t{task}", "type_id": t, "level": lvl,
                                     "passed": rng.random() > p})
    (tmp_path / "round1").mkdir()
    pd.DataFrame(rows).to_parquet(tmp_path / "round1" / "verification.parquet")
    A = PL.analyze(1, n_boot=300).set_index("type_id")
    assert A.loc["good", "passes"] and not A.loc["flat", "passes"]
    assert not A.loc["flat", "gap_em"]
