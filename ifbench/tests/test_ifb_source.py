import random
import sys

import pytest

from mife_ifb import inventory, plain_meaning as pm, source
from mife_ifb.config import load_config, nltk_data_dir, source_dir


def test_pinned_checkout_test_file_and_nltk():
    out = source.verify_pinned()
    assert out["commit"] == load_config()["source"]["commit"]
    assert set(out["nltk"]) == set(load_config()["nltk"]["packages"])


def test_verifier_package_versions_match_pins():
    source.check_package_pins()


def test_ifbench_imported_from_pinned_checkout():
    source._bootstrap()
    mod = sys.modules["ifbench"]
    assert str(source_dir().resolve()) in str(mod.__file__)


def test_nltk_uses_only_vendored_data_and_never_downloads():
    import nltk
    source._bootstrap()
    assert nltk.data.path == [str(nltk_data_dir())]
    with pytest.raises(source.NetworkDisabled):
        nltk.download("punkt")


def test_registry_split():
    assert len(source.ood_types()) == 58
    assert len(source.classic_types()) == 25


def test_rendered_wording_is_official_and_single_line():
    for t in inventory.candidates():
        kw = source.sample_default_kwargs(t, rng_seed=7)
        kw.pop("reference_text", None)
        text = source.render_rule_text(t, kw)
        inst = source.instruction(t)
        if t == "ratio:overlap":
            kw["reference_text"] = "x"
        assert text == inst.build_description(**kw)
        assert "\n" not in text


def test_default_draws_are_seeded_and_reproducible():
    for t in inventory.parameterized():
        assert source.sample_default_kwargs(t, 11) == source.sample_default_kwargs(t, 11)


def test_words_japanese_shim_matches_released_default_draw():
    for seed in range(50):
        random.seed(seed)
        want = random.randint(1, 30)
        assert source.sample_default_kwargs("count:words_japanese", seed) == {"N": want}
    with pytest.raises(TypeError):  # the released default really does crash
        source.instruction("count:words_japanese").build_description()


def test_blank_response_fails_every_candidate():
    for t in inventory.candidates():
        ex = pm.CHECKS[t].ok
        assert source.official_check(t, ex.kwargs, "", ex.task_text) is False
        assert source.official_check(t, ex.kwargs, "  \n ", ex.task_text) is False


def test_overlap_needs_task_text():
    with pytest.raises(ValueError):
        source.official_check("ratio:overlap", {"percentage": 50}, "some words here")
    assert source.types_reading_task_text() == ["ratio:overlap"]


def test_strict_and_loose_match_evaluation_lib():
    sys.path.insert(0, str(source_dir()))
    import evaluation_lib as el

    for t in inventory.candidates():
        for ex in (pm.CHECKS[t].ok, pm.CHECKS[t].loophole):
            if ex is None:
                continue
            kw = dict(ex.kwargs)
            if ex.task_text is not None:
                kw["reference_text"] = ex.task_text
            for resp in (ex.response, "intro line\n" + ex.response + "\n*end*"):
                mk = lambda: el.InputExample(key=0, instruction_id_list=[t], prompt="p", kwargs=[dict(kw)])
                strict = el.test_instruction_following_strict(mk(), {"p": resp}).follow_instruction_list[0]
                loose = el.test_instruction_following_loose(mk(), {"p": resp}).follow_instruction_list[0]
                assert source.official_check(t, ex.kwargs, resp, ex.task_text) == strict, (t, resp)
                assert source.official_check_loose(t, ex.kwargs, resp, ex.task_text) == loose, (t, resp)


def test_serialize_prompt_uses_manyifeval_layout():
    p = source.serialize_prompt("Write a poem.", ["Rule one.", "Rule two."])
    assert p == "Write a poem.\n\nYour response should follow the instructions below:\n- Rule one.\n- Rule two."


def test_released_prompt_drift_audit():
    d = source.released_prompt_drift()
    assert d["rows"] == 300
    assert d["rows_verbatim"] == 242
    assert d["constraints_per_row"] == {1: 256, 2: 44}
