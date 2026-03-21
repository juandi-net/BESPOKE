from datetime import datetime

from bespoke.benchmark.prescan import generate_prescan_summary


def _insert(conn, id, session_id, user_msg, asst_resp, followup=None):
    """Insert a test interaction."""
    conn.execute("""
        INSERT INTO interactions (id, provider, model, source, session_id,
            user_message, assistant_response, user_followup, captured_at)
        VALUES (?, 'claude', 'sonnet', 'claude-code', ?, ?, ?, ?, ?)
    """, (id, session_id, user_msg, asst_resp, followup,
          datetime.now().isoformat()))
    conn.commit()


def test_prescan_basic_counts(db):
    """Prescan returns correct total counts and date range."""
    _insert(db, 1, "sess-a", "Write a function", "Here's the function")
    _insert(db, 2, "sess-a", "Now add tests", "Here are tests")
    _insert(db, 3, "sess-b", "Deploy this", "Deployed")

    summary = generate_prescan_summary(conn=db)

    assert summary["total_interactions"] == 3
    assert summary["total_sessions"] == 2
    assert "earliest" in summary["date_range"]
    assert "latest" in summary["date_range"]


def test_prescan_message_length_stats(db):
    """Prescan computes message length statistics correctly."""
    _insert(db, 1, "s1", "x" * 50, "y" * 200)   # short
    _insert(db, 2, "s2", "x" * 150, "y" * 300)  # medium
    _insert(db, 3, "s3", "x" * 600, "y" * 1000) # long

    summary = generate_prescan_summary(conn=db)

    assert summary["avg_user_message_length"] > 0
    assert summary["avg_assistant_response_length"] > 0
    assert summary["median_user_message_length"] > 0
    # 1 of 3 is >500 chars
    assert abs(summary["pct_long_form_prompts"] - 0.333) < 0.01
    # 1 of 3 is <100 chars
    assert abs(summary["pct_short_directives"] - 0.333) < 0.01


def test_prescan_followup_signals(db):
    """Prescan detects positive and negative followup signals."""
    _insert(db, 1, "s1", "Write code", "Here's code", "perfect, exactly what I needed")
    _insert(db, 2, "s2", "Deploy", "Deployed", "no that's wrong, try again")
    _insert(db, 3, "s3", "Help", "Sure", None)  # no followup

    summary = generate_prescan_summary(conn=db)

    assert summary["pct_with_followup"] == round(2/3, 3)
    assert summary["pct_positive_followup"] == round(1/3, 3)
    assert summary["pct_negative_followup"] == round(1/3, 3)


def test_prescan_session_patterns(db):
    """Prescan computes session statistics."""
    # Session with 1 interaction (single-turn)
    _insert(db, 1, "s1", "hi", "hello")
    # Session with 3 interactions
    _insert(db, 2, "s2", "a", "b")
    _insert(db, 3, "s2", "c", "d")
    _insert(db, 4, "s2", "e", "f")

    summary = generate_prescan_summary(conn=db)

    assert summary["avg_interactions_per_session"] == 2.0
    assert summary["pct_single_turn_sessions"] == 0.5
    assert summary["pct_long_sessions"] == 0.0


def test_prescan_sample_prompts_variety(db):
    """Sample prompts include messages from different length buckets."""
    _insert(db, 1, "s1", "do it", "ok")  # short
    _insert(db, 2, "s2", "x" * 200, "y" * 200)  # medium
    _insert(db, 3, "s3", "x" * 600, "y" * 600)  # long

    summary = generate_prescan_summary(conn=db)

    assert len(summary["sample_prompts"]) >= 3
    lengths = [s["length"] for s in summary["sample_prompts"]]
    assert any(l < 100 for l in lengths), "Should include short prompts"
    assert any(100 <= l < 500 for l in lengths) or any(l >= 500 for l in lengths), \
        "Should include medium or long prompts"


def test_prescan_empty_db(db):
    """Prescan returns zero-value summary on empty database."""
    summary = generate_prescan_summary(conn=db)
    assert summary["total_interactions"] == 0
    assert summary["sample_prompts"] == []
