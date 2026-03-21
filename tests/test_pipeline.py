"""Tests for the refactored capture_interaction function."""

from bespoke.capture.parsers import Interaction
from bespoke.capture.pipeline import capture_interaction


def _make_interaction(**overrides):
    defaults = {
        "provider": "claude",
        "model": "claude-sonnet-4-6",
        "source": "test",
        "session_id": "test-session-1",
        "system_prompt": None,
        "user_message": "Hello",
        "assistant_response": "Hi there",
        "input_tokens": None,
        "output_tokens": None,
        "timestamp": "2026-03-19T12:00:00",
    }
    defaults.update(overrides)
    return Interaction(**defaults)


def test_capture_interaction_without_embedding_or_stats(db):
    """capture_interaction works with embedding_svc=None and stats=None."""
    interaction = _make_interaction()
    rowid = capture_interaction(db, interaction)
    assert rowid is not None
    assert rowid > 0

    row = db.execute("SELECT * FROM interactions WHERE id = ?", (rowid,)).fetchone()
    assert row["source"] == "test"
    assert row["user_message"] == "Hello"


def test_capture_interaction_dedup(db):
    """Duplicate interactions are skipped via content_hash."""
    interaction = _make_interaction()
    rowid1 = capture_interaction(db, interaction)
    rowid2 = capture_interaction(db, interaction)
    assert rowid1 is not None
    assert rowid2 is None  # skipped


def test_capture_interaction_with_stats(db):
    """Stats dict is updated when provided."""
    stats = {"interactions_captured": 0, "interactions_skipped": 0}
    interaction = _make_interaction()
    capture_interaction(db, interaction, stats=stats)
    assert stats["interactions_captured"] == 1

    capture_interaction(db, interaction, stats=stats)
    assert stats["interactions_skipped"] == 1
