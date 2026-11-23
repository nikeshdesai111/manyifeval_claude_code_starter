"""Tests for the rule-based task-family rubric (mife.family_rules)."""
import csv
from pathlib import Path

import pytest

from mife.family_rules import PRECEDENCE, classify

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("text,family", [
    # Rubric rule 1: supplied text to transform, or underspecified.
    ('Can you elaborate on "I froze when I was jogging"?', "TRANSFORM"),
    ('Complete the following sentence: "The cat sat."', "TRANSFORM"),
    ("Brainstorm a name for a bakery.", "TRANSFORM"),
    # Rule 2: form beats topic (the rubric's own example).
    ('Write a startup pitch for "X". Make it a weird poem that explains why the pitch is good.', "CREATIVE"),
    ("Write a rubric in the form of a poem for how to evaluate a poem.", "CREATIVE"),
    # Rule 3.
    ("Write an advertisement for a perfume.", "PERSUASIVE"),
    ('Write an article with title "Springfield is the best city in the world"', "PERSUASIVE"),
    # Rule 4, incl. correspondence-to-friends policy.
    ("Write a resume for a nurse.", "PRACTICAL"),
    ("Write a letter to your friend who recently moved away.", "PRACTICAL"),
    # Rule 5.
    ("What are the pros and cons of kotlin vs java?", "ANALYTICAL"),
    ("Critique the following ad copy: \"Buy now!\"", "ANALYTICAL"),
    # Rule 6: default.
    ("Who built the first artificial ice rink?", "INFO"),
])
def test_rubric_examples(text, family):
    assert classify(text).family == family


def test_tone_does_not_change_family():
    for tone in ["", "funny ", "weird ", "very angry ", "casual "]:
        assert classify(f"Write a {tone}poem about rain.").family == "CREATIVE"
        assert classify(f"Write a {tone}resume for a chef.").family == "PRACTICAL"


def test_precedence_order_and_determinism():
    assert PRECEDENCE == ["TRANSFORM", "CREATIVE", "PERSUASIVE", "PRACTICAL", "ANALYTICAL", "INFO"]
    a = classify("Write a funny song for a proposal to build a playground.")
    assert a.family == "CREATIVE" and a.all_families[0] == "CREATIVE" and "PRACTICAL" in a.all_families
    assert classify("Write a poem.") == classify("  Write a   POEM. ")


def test_hand_label_file_is_unchanged_input():
    """The rule labels are an audit only; the pipeline's CSV keeps its schema and 216 rows."""
    with open(ROOT / "configs" / "task_families_v1.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 216
    assert set(rows[0]) == {"task_key", "family", "ambiguous", "alt_family", "note"}
