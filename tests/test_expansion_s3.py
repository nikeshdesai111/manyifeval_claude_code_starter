"""S3 TEST expansion: design reconstruction, worklist completeness, per-sample resume and seeding (synthetic data only)."""
from __future__ import annotations

import math

import pandas as pd
import pytest

from mife import design
from mife import expansion_s3 as X

ORDER = [f"g99-r{i}" for i in (4, 1, 6, 2, 5, 3)]  # a shuffled chain order, as in the manifest


def test_expanded_design_contains_frozen_246_exactly():
    cfgs = X.expanded_configs(ORDER)
    frozen = {tuple(c["ordered_rule_labels"]) for c in design.configurations(ORDER)}
    counts = X.check_expanded(cfgs, frozen)
    assert counts == {"new_k1": 0, "new_k2": 18, "new_k3": 84, "new_k4": 108, "new_k5": 0,
                      "expanded": 456, "original": 246}
    # original configurations keep their chain index and COA row
    ref = {(tuple(c["ordered_rule_labels"])): (c["chain_index"], c["coa_row"]) for c in design.configurations(ORDER)}
    for c in cfgs:
        if c["is_original"]:
            assert ref[c["order"]] == (c["chain_index"], c["coa_row"])


def test_k4_needs_the_frozen_base_order():
    """Re-basing a frozen 4-subset on chain-label order would select a different 12 of 24 orders."""
    frozen = {tuple(c["ordered_rule_labels"]) for c in design.configurations(ORDER) if c["k"] == 4}
    rank = {r: i for i, r in enumerate(ORDER)}
    naive = set()
    for c in design.configurations(ORDER):
        if c["k"] == 4:
            base = sorted(c["subset_chain_order"], key=rank.get)
            naive |= {tuple(base[x] for x in s) for s in design.coa_template(4)}
    assert naive != frozen


def test_new_subset_ids_cannot_collide_with_frozen_ids():
    ids = {X.subset_id("g99", c) for c in X.expanded_configs(ORDER)}
    assert len(ids) == sum(math.comb(6, k) for k in range(1, 6)) == 62
    assert all(("-c" in i) != ("-s" in i) for i in ids)


def _gen(rows):
    return pd.DataFrame(rows, columns=["prompt_id", "sample_index", "model_key", "settings_hash", "rendered_sha256",
                                       "finish_reason", "source_run", "n_output_tokens", "batch_seed", "backend"])


def _expected(pids):
    return pd.DataFrame({"prompt_id": pids, "rendered_sha256": [f"h{p}" for p in pids]}).set_index("prompt_id")


def test_sample_index_completeness_and_partial_prompts():
    exp = _expected(["a", "b", "c"])
    rows = [("a", s, "llama", "H", "ha", "stop", "main", 5, 1, "vllm") for s in range(20)]
    rows += [("b", s, "llama", "H", "hb", "stop", "main", 5, 1, "vllm") for s in range(0, 20, 2)]  # partial
    have, stats = X.valid_records(_gen(rows), exp, "llama", "H")
    work = X.outstanding(list(exp.index), have)
    assert len(work) == 10 + 20
    assert set(work[work.prompt_id == "b"].sample_index) == set(range(1, 20, 2))
    assert 20 * len(exp) - len(have) == len(work)


def test_invalid_and_duplicate_records_are_not_reused():
    exp = _expected(["a"])
    rows = [("a", 0, "llama", "H", "ha", "stop", "main", 5, 1, "vllm"),
            ("a", 0, "llama", "H", "ha", "stop", "main", 6, 1, "vllm"),          # duplicate
            ("a", 1, "llama", "OTHER", "ha", "stop", "main", 5, 1, "vllm"),      # different settings
            ("a", 2, "llama", "H", "stale", "stop", "main", 5, 1, "vllm"),       # different rendered prompt
            ("a", 3, "qwen", "H", "ha", "stop", "main", 5, 1, "vllm"),           # other model
            ("a", 4, "llama", "H", "ha", "stop", "pilot", 5, 1, "vllm"),         # other run
            ("a", 25, "llama", "H", "ha", "stop", "main", 5, 1, "vllm"),         # bad index
            ("zz", 0, "llama", "H", "hz", "stop", "main", 5, 1, "vllm")]         # not in the expanded set
    have, stats = X.valid_records(_gen(rows), exp, "llama", "H")
    assert list(have.sample_index) == [0]
    assert stats["rows_duplicate_prompt_sample"] == 1 and stats["rows_invalid_provenance"] == 5
    assert stats["rows_other_prompts"] == 1


def test_worklist_is_idempotent():
    exp = _expected(["a", "b"])
    have, _ = X.valid_records(_gen([]), exp, "llama", "H")
    w1 = X.outstanding(list(exp.index), have)
    w2 = X.outstanding(list(exp.index), have)
    assert w1.equals(w2) and len(w1) == 40
    done = _gen([(p, int(s), "llama", "H", f"h{p}", "stop", X.RUN_NAME, 3, 0, "vllm") for p, s in zip(w1.prompt_id, w1.sample_index)])
    have2, _ = X.valid_records(done, exp, "llama", "H")
    assert X.outstanding(list(exp.index), have2).empty


def test_chunks_keep_each_prompt_together_and_cover_everything():
    work = pd.DataFrame([(f"p{i}", s) for i in range(13) for s in range(20)], columns=["prompt_id", "sample_index"])
    ch = X.assign_chunks(work, 6)
    assert work.assign(c=ch).groupby("prompt_id").c.nunique().max() == 1
    assert sorted(ch.unique()) == list(range(ch.max() + 1)) and ch.max() < 6


def test_seed_rule_matches_generate_vllm():
    from mife import generate_vllm
    assert X.sample_seed("qwen", "g03-sABD-k3-o02-t1", 7) == generate_vllm.sample_seed("qwen", X.RUN_NAME, "g03-sABD-k3-o02-t1", 7)
    assert X.sample_seed("qwen", "p", 0) != X.sample_seed("qwen", "p", 1) != X.sample_seed("llama", "p", 1)
    assert 0 <= X.sample_seed("llama", "p", 19) < 2 ** 31


def test_expansion_never_targets_main():
    assert X.RUN_NAME != "main" and "main" in X.SOURCE_RUNS
    with pytest.raises(SystemExit):
        X.generate("llama", 0, run_name="main")
