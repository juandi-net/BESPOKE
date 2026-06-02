# tests/test_extract_feedback.py
"""Deterministic feedback classification from the user's next message."""
import pytest

from bespoke.extract.feedback import classify_feedback


@pytest.mark.parametrize("text,expected", [
    ("", "neutral"),
    ("   ", "neutral"),
    (None, "neutral"),
    ("perfect, do it", "strong_accept"),
    ("ok thanks that works", "accept"),
    ("yes go ahead", "accept"),
    ("no that's wrong, fix it", "reject"),
    ("this is completely wrong", "strong_reject"),
    ("what about caching though?", "neutral"),
    ("can you also add tests?", "neutral"),
])
def test_classify(text, expected):
    assert classify_feedback(text) == expected


def test_mixed_signal_is_neutral():
    # both accept and reject cues present -> ambiguous -> neutral
    assert classify_feedback("yes but no that's not right") == "neutral"
