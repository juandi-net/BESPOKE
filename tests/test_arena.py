"""Tests for the blind thesis-eval arena (pure logic: assembly + scoring)."""
from bespoke.eval.arena import assemble_contestants, score_verdicts

RESP = {"base": "base answer", "adapter": "adapter answer", "frontier": "frontier answer"}


def test_assemble_is_deterministic_with_seed():
    l1, k1 = assemble_contestants(RESP, seed=42)
    l2, k2 = assemble_contestants(RESP, seed=42)
    assert k1 == k2
    assert [x["label"] for x in l1] == ["A", "B", "C"]


def test_assemble_covers_each_model_exactly_once_and_text_matches_key():
    labeled, key = assemble_contestants(RESP, seed=1)
    assert set(key.keys()) == {"A", "B", "C"}
    assert set(key.values()) == {"base", "adapter", "frontier"}
    for item in labeled:
        assert item["text"] == RESP[key[item["label"]]]


def test_assemble_shuffles_across_seeds():
    keys = {tuple(sorted(assemble_contestants(RESP, seed=s)[1].items())) for s in range(25)}
    assert len(keys) > 1  # blinding actually varies the order


def test_score_counts_wins_by_model():
    items = [
        {"key": {"A": "base", "B": "adapter", "C": "frontier"}, "verdict": "B"},   # adapter
        {"key": {"A": "adapter", "B": "frontier", "C": "base"}, "verdict": "B"},   # frontier
        {"key": {"A": "frontier", "B": "base", "C": "adapter"}, "verdict": "C"},   # adapter
        {"key": {"A": "base", "B": "adapter", "C": "frontier"}, "verdict": None},  # unrated
    ]
    rep = score_verdicts(items)
    assert rep["n_rated"] == 3
    assert rep["wins"] == {"adapter": 2, "frontier": 1, "base": 0}
    assert abs(rep["win_rate"]["adapter"] - 2 / 3) < 1e-9


def test_score_handles_none_acceptable_verdict():
    items = [
        {"key": {"A": "base", "B": "adapter", "C": "frontier"}, "verdict": "none"},
        {"key": {"A": "base", "B": "adapter", "C": "frontier"}, "verdict": "A"},
    ]
    rep = score_verdicts(items)
    assert rep["n_rated"] == 2            # "none" counts as rated
    assert rep["wins"]["base"] == 1
    assert rep["none_acceptable"] == 1


def test_score_no_ratings():
    rep = score_verdicts([{"key": {"A": "base"}, "verdict": None}])
    assert rep["n_rated"] == 0
