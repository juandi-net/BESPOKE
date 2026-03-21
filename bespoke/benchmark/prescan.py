"""Pre-scan: statistical summary of raw interaction data for the benchmark interview.

Queries the interactions table directly. No LLM involved. Pure SQL + Python.
Called before the interview so the LLM can generate domain-appropriate scenarios.
"""

import statistics
from typing import Optional

from bespoke.db.init import get_connection


def generate_prescan_summary(
    limit: int = 50,
    conn=None,
) -> dict:
    """Generate a statistical summary of captured interactions.

    Args:
        limit: Max sample prompts to consider (not max interactions).
        conn: Optional DB connection (for testing). If None, opens one.
    """
    own_conn = conn is None
    if own_conn:
        conn = get_connection()

    try:
        return _build_summary(conn, limit)
    finally:
        if own_conn:
            conn.close()


def _build_summary(conn, limit: int) -> dict:
    """Build the full prescan summary dict."""
    row = conn.execute("""
        SELECT
            COUNT(*) as total,
            COUNT(DISTINCT session_id) as sessions,
            MIN(captured_at) as earliest,
            MAX(captured_at) as latest
        FROM interactions
    """).fetchone()

    total = row["total"]
    if total == 0:
        return _empty_summary()

    summary = {
        "total_interactions": total,
        "total_sessions": row["sessions"],
        "date_range": {
            "earliest": row["earliest"],
            "latest": row["latest"],
        },
    }

    # Message length stats
    lengths = conn.execute("""
        SELECT
            LENGTH(user_message) as user_len,
            LENGTH(assistant_response) as asst_len
        FROM interactions
    """).fetchall()

    user_lens = [r["user_len"] for r in lengths]
    asst_lens = [r["asst_len"] for r in lengths]

    summary["avg_user_message_length"] = int(statistics.mean(user_lens))
    summary["avg_assistant_response_length"] = int(statistics.mean(asst_lens))
    summary["median_user_message_length"] = int(statistics.median(user_lens))
    summary["pct_long_form_prompts"] = round(
        sum(1 for l in user_lens if l > 500) / total, 3
    )
    summary["pct_short_directives"] = round(
        sum(1 for l in user_lens if l < 100) / total, 3
    )

    # Followup signals
    followup_stats = _compute_followup_stats(conn, total)
    summary.update(followup_stats)

    # Sample prompts
    summary["sample_prompts"] = _select_sample_prompts(conn, limit)

    # Session patterns
    session_stats = _compute_session_stats(conn)
    summary.update(session_stats)

    return summary


def _compute_followup_stats(conn, total: int) -> dict:
    """Compute behavioral signals from user_followup column."""
    POSITIVE_SIGNALS = [
        "perfect", "exactly", "that works", "ship it", "do it",
        "yes", "correct", "bingo", "nice",
    ]
    NEGATIVE_SIGNALS = [
        "no", "wrong", "that's not", "try again", "redo",
        "not what i", "missing",
    ]

    rows = conn.execute(
        "SELECT user_followup FROM interactions WHERE user_followup IS NOT NULL"
    ).fetchall()

    with_followup = len(rows)
    positive = 0
    negative = 0

    for row in rows:
        text = row["user_followup"].lower()
        if any(sig in text for sig in POSITIVE_SIGNALS):
            positive += 1
        if any(sig in text for sig in NEGATIVE_SIGNALS):
            negative += 1

    return {
        "pct_with_followup": round(with_followup / total, 3) if total else 0.0,
        "pct_positive_followup": round(positive / total, 3) if total else 0.0,
        "pct_negative_followup": round(negative / total, 3) if total else 0.0,
    }


def _select_sample_prompts(conn, limit: int) -> list[dict]:
    """Select 10-15 representative user messages across length buckets.

    Picks 3-4 short (<100 chars), 3-4 medium (100-500), 3-4 long (>500).
    Avoids duplicates from the same session.
    """
    buckets = {
        "short": {"max": 100, "target": 4, "items": []},
        "medium": {"max": 500, "target": 5, "items": []},
        "long": {"max": 999999, "target": 5, "items": []},
    }

    rows = conn.execute("""
        SELECT user_message, LENGTH(user_message) as msg_len,
               user_followup IS NOT NULL as has_followup, session_id
        FROM interactions
        ORDER BY RANDOM()
        LIMIT ?
    """, (limit * 3,)).fetchall()

    seen_sessions = set()
    for row in rows:
        sid = row["session_id"]
        if sid and sid in seen_sessions:
            continue
        if sid:
            seen_sessions.add(sid)

        msg_len = row["msg_len"]
        if msg_len < 100:
            bucket = buckets["short"]
        elif msg_len < 500:
            bucket = buckets["medium"]
        else:
            bucket = buckets["long"]

        if len(bucket["items"]) < bucket["target"]:
            bucket["items"].append({
                "message": row["user_message"],
                "length": msg_len,
                "has_followup": bool(row["has_followup"]),
            })

    samples = []
    for bucket in buckets.values():
        samples.extend(bucket["items"])

    return samples[:15]


def _compute_session_stats(conn) -> dict:
    """Compute session pattern statistics."""
    rows = conn.execute("""
        SELECT session_id, COUNT(*) as cnt
        FROM interactions
        WHERE session_id IS NOT NULL AND session_id != ''
        GROUP BY session_id
    """).fetchall()

    if not rows:
        return {
            "avg_interactions_per_session": 0.0,
            "pct_single_turn_sessions": 0.0,
            "pct_long_sessions": 0.0,
        }

    counts = [r["cnt"] for r in rows]
    total_sessions = len(counts)

    return {
        "avg_interactions_per_session": round(statistics.mean(counts), 1),
        "pct_single_turn_sessions": round(
            sum(1 for c in counts if c == 1) / total_sessions, 3
        ),
        "pct_long_sessions": round(
            sum(1 for c in counts if c >= 10) / total_sessions, 3
        ),
    }


def _empty_summary() -> dict:
    """Return an empty summary when no interactions exist."""
    return {
        "total_interactions": 0,
        "total_sessions": 0,
        "date_range": {"earliest": None, "latest": None},
        "avg_user_message_length": 0,
        "avg_assistant_response_length": 0,
        "median_user_message_length": 0,
        "pct_long_form_prompts": 0.0,
        "pct_short_directives": 0.0,
        "pct_with_followup": 0.0,
        "pct_positive_followup": 0.0,
        "pct_negative_followup": 0.0,
        "sample_prompts": [],
        "avg_interactions_per_session": 0.0,
        "pct_single_turn_sessions": 0.0,
        "pct_long_sessions": 0.0,
    }
