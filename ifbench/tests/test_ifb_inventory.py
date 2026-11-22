from mife_ifb import inventory, source

PARAMETERIZED = {
    "count:word_count_range", "count:unique_word_count", "count:conjunctions", "count:person_names",
    "count:numbers", "count:pronouns", "count:keywords_multiple", "ratio:stop_words",
    "ratio:overlap", "sentence:keyword", "sentence:increment", "words:repeats", "words:words_position",
    "words:keywords_specific_position", "format:list",
}


def test_exclusions_are_the_decided_fifteen():
    ex = set(inventory.EXCLUDED)
    assert len(ex) == 16 and ex <= set(source.ood_types())
    assert {t for t in ex if t.startswith("custom:")} == {t for t in source.ood_types() if t.startswith("custom:")}
    assert {t for t in ex if t.startswith("repeat:")} == {t for t in source.ood_types() if t.startswith("repeat:")}
    assert "format:options" in ex and "count:words_japanese" in ex and "ratio:overlap" not in ex
    assert all(r for r in inventory.EXCLUDED.values())


def test_forty_two_candidates_fifteen_parameterized():
    c = inventory.candidates()
    assert len(c) == 42 and len(set(c)) == 42
    assert set(inventory.parameterized()) == PARAMETERIZED
    assert len(inventory.fixed()) == 27
    assert set(inventory.parameterized()) | set(inventory.fixed()) == set(c)


def test_only_overlap_reads_task_text_and_no_candidate_reads_the_prompt():
    assert source.types_reading_task_text() == ["ratio:overlap"]
    for t in inventory.candidates():
        assert "prompt_to_repeat" not in (source.instruction(t).get_instruction_args_keys() or [])
