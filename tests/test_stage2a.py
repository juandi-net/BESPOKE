from datetime import datetime

from bespoke.teach.stage2a import (
    group_sessions,
    build_turn_map,
    compute_context_only_turns,
    apply_extractions,
)


def _insert_interaction(conn, id, session_id, user_msg="hi", asst_resp="hello",
                         processed=False):
    """Insert a fake interaction into the test DB."""
    conn.execute("""
        INSERT INTO interactions (id, provider, model, source, session_id,
            user_message, assistant_response, captured_at, processed_2a_at)
        VALUES (?, 'claude', 'sonnet', 'claude-code', ?, ?, ?, ?, ?)
    """, (id, session_id, user_msg, asst_resp,
          datetime.now().isoformat(),
          datetime.now().isoformat() if processed else None))
    conn.commit()


def test_group_sessions_basic(db):
    """Interactions group by session_id."""
    _insert_interaction(db, 1, "sess-a", processed=False)
    _insert_interaction(db, 2, "sess-a", processed=False)
    _insert_interaction(db, 3, "sess-b", processed=False)
    sessions = group_sessions(db)
    assert set(sessions.keys()) == {"sess-a", "sess-b"}
    assert len(sessions["sess-a"]) == 2
    assert len(sessions["sess-b"]) == 1


def test_group_sessions_null_session_id(db):
    """NULL/empty session_ids become individual single-interaction sessions."""
    _insert_interaction(db, 1, "", processed=False)
    _insert_interaction(db, 2, None, processed=False)
    _insert_interaction(db, 3, "real-session", processed=False)
    sessions = group_sessions(db)
    assert "real-session" in sessions
    solo_sessions = {k: v for k, v in sessions.items() if k != "real-session"}
    assert len(solo_sessions) == 2
    for v in solo_sessions.values():
        assert len(v) == 1


def test_group_sessions_includes_processed_for_context(db):
    """Sessions with some processed interactions include ALL interactions for context."""
    _insert_interaction(db, 1, "sess-a", processed=True)
    _insert_interaction(db, 2, "sess-a", processed=True)
    _insert_interaction(db, 3, "sess-a", processed=False)
    sessions = group_sessions(db)
    assert "sess-a" in sessions
    assert len(sessions["sess-a"]) == 3  # all 3, not just the unprocessed one


def test_group_sessions_excludes_fully_processed(db):
    """Sessions where ALL interactions are processed are excluded."""
    _insert_interaction(db, 1, "sess-done", processed=True)
    _insert_interaction(db, 2, "sess-done", processed=True)
    _insert_interaction(db, 3, "sess-new", processed=False)
    sessions = group_sessions(db)
    assert "sess-done" not in sessions
    assert "sess-new" in sessions


def test_build_turn_map():
    """Turn numbers map to interaction IDs in order."""
    rows = [{"id": 10}, {"id": 20}, {"id": 30}]
    turn_map = build_turn_map(rows)
    assert turn_map == {1: 10, 2: 20, 3: 30}


def test_compute_context_only_already_processed(db):
    """Already-processed interactions become context-only turns."""
    _insert_interaction(db, 1, "sess-a", processed=True)
    _insert_interaction(db, 2, "sess-a", processed=True)
    _insert_interaction(db, 3, "sess-a", processed=False)
    rows = db.execute(
        "SELECT * FROM interactions WHERE session_id = 'sess-a' ORDER BY captured_at, id"
    ).fetchall()
    context_only = compute_context_only_turns(rows)
    assert context_only == {1, 2}


def test_compute_context_only_none_processed(db):
    """When no interactions are processed, context_only is empty."""
    _insert_interaction(db, 1, "sess-a", processed=False)
    _insert_interaction(db, 2, "sess-a", processed=False)
    rows = db.execute(
        "SELECT * FROM interactions WHERE session_id = 'sess-a' ORDER BY captured_at, id"
    ).fetchall()
    context_only = compute_context_only_turns(rows)
    assert context_only == set()


def test_apply_extractions_marks_all_processed(db):
    """All turns get processed_2a_at, even those not in the extraction results."""
    _insert_interaction(db, 1, "sess-a", "fix bug", "fixed it", processed=False)
    _insert_interaction(db, 2, "sess-a", "thanks", "you're welcome", processed=False)
    _insert_interaction(db, 3, "sess-a", "now deploy", "deployed", processed=False)

    turn_map = {1: 1, 2: 2, 3: 3}
    extractions = [
        {"turn_number": 1, "domain": "code", "quality_score": "high",
         "reasoning_primitives": [], "feedback_class": "accept",
         "feedback_reasoning": "user said thanks", "training_pair": None,
         "dpo_pair": None, "skip_reason": None},
        {"turn_number": 3, "domain": "code", "quality_score": "medium",
         "reasoning_primitives": [], "feedback_class": "unknown",
         "feedback_reasoning": "last turn", "training_pair": None,
         "dpo_pair": None, "skip_reason": None},
    ]

    apply_extractions(db, turn_map, extractions, extractable_turns={1, 2, 3})
    db.commit()

    processed = db.execute(
        "SELECT COUNT(*) FROM interactions WHERE processed_2a_at IS NOT NULL"
    ).fetchone()[0]
    assert processed == 3

    domains = db.execute(
        "SELECT id, domain FROM interactions ORDER BY id"
    ).fetchall()
    assert domains[0]["domain"] == "code"
    assert domains[1]["domain"] is None  # skipped trivial turn
    assert domains[2]["domain"] == "code"


def test_apply_extractions_respects_extractable_filter(db):
    """Extractions for non-extractable turns are ignored (dedup safety)."""
    _insert_interaction(db, 1, "sess-a", "old msg", "old resp", processed=False)
    _insert_interaction(db, 2, "sess-a", "new msg", "new resp", processed=False)

    turn_map = {1: 1, 2: 2}
    # LLM produced an extraction for turn 1, but turn 1 is context-only
    extractions = [
        {"turn_number": 1, "domain": "code", "quality_score": "high",
         "reasoning_primitives": [], "feedback_class": "accept",
         "feedback_reasoning": "test", "training_pair": None,
         "dpo_pair": None, "skip_reason": None},
        {"turn_number": 2, "domain": "code", "quality_score": "medium",
         "reasoning_primitives": [], "feedback_class": "unknown",
         "feedback_reasoning": "test", "training_pair": None,
         "dpo_pair": None, "skip_reason": None},
    ]

    # Only turn 2 is extractable (turn 1 would be context-only in a real chunk)
    counts = apply_extractions(db, turn_map, extractions, extractable_turns={2})
    db.commit()

    assert counts["succeeded"] == 1  # only turn 2

    domains = db.execute(
        "SELECT id, domain FROM interactions ORDER BY id"
    ).fetchall()
    assert domains[0]["domain"] is None  # turn 1 filtered out
    assert domains[1]["domain"] == "code"  # turn 2 extracted


def test_chunk_to_prompt_to_response_integration():
    """Integration test: chunk context_only → prompt context markers → response filtering.

    Verifies the seam between chunking (Task 4) and prompt building (Task 3)
    and the client-side dedup filter (Task 5).
    """
    from bespoke.teach.chunking import build_chunks
    from bespoke.teach.prompts import make_session_extraction_prompt

    # Simulate a session that would be chunked
    turns = [
        {"user_message": f"Message {i}", "assistant_response": f"Response {i}" * 1000}
        for i in range(20)
    ]

    # Force chunking with a low threshold
    chunks = build_chunks(turns, threshold=50_000)
    assert len(chunks) >= 2, "Test needs multiple chunks to verify integration"

    # For each chunk, verify the prompt correctly marks context-only turns
    for chunk in chunks:
        chunk_start = chunk["start"]
        chunk_end = chunk["end"]
        chunk_context = chunk["context_only"]

        chunk_turns = turns[chunk_start - 1 : chunk_end]

        # Remap to relative numbering
        chunk_relative_context = set()
        for abs_turn in range(chunk_start, chunk_end + 1):
            rel_turn = abs_turn - chunk_start + 1
            if abs_turn in chunk_context:
                chunk_relative_context.add(rel_turn)

        messages = make_session_extraction_prompt(
            turns=chunk_turns,
            context_only_turns=chunk_relative_context,
        )

        prompt_content = messages[1]["content"]

        # Verify context-only turns are marked in the prompt
        for rel_turn in chunk_relative_context:
            assert f"Turn {rel_turn} [CONTEXT ONLY" in prompt_content

        # Verify extractable turns are NOT marked context-only
        for rel_turn in range(1, len(chunk_turns) + 1):
            if rel_turn not in chunk_relative_context:
                assert f"Turn {rel_turn} [CONTEXT ONLY" not in prompt_content
                assert f"--- Turn {rel_turn} ---" in prompt_content

    # Simulate LLM response for chunk 2 and verify dedup filtering
    chunk2 = chunks[1]
    extractable_in_chunk2 = (
        set(range(chunk2["start"], chunk2["end"] + 1))
        - chunk2["context_only"]
    )

    # Fake LLM response that includes a context-only turn (LLM misbehaved)
    fake_response = []
    for abs_turn in range(chunk2["start"], chunk2["end"] + 1):
        rel_turn = abs_turn - chunk2["start"] + 1
        fake_response.append({"turn_number": rel_turn, "domain": "code"})

    # Remap relative → absolute (like process_session does)
    for ext in fake_response:
        ext["turn_number"] = ext["turn_number"] + chunk2["start"] - 1

    # Apply client-side dedup filter
    filtered = [
        ext for ext in fake_response
        if ext.get("turn_number") in extractable_in_chunk2
    ]

    # Context-only turns should be filtered out
    filtered_turns = {ext["turn_number"] for ext in filtered}
    assert not (filtered_turns & chunk2["context_only"]), \
        "Context-only turns should have been filtered out"
    assert filtered_turns == extractable_in_chunk2, \
        "All extractable turns should be present"
