# tests/test_web_capture.py
"""Tests for claude.ai web conversation capture."""

import pytest
from unittest.mock import patch, MagicMock

from bespoke.capture.web_capture import parse_conversation, run_web_capture, get_org_id, fetch_conversation


def _make_message(uuid, sender, text_or_content, parent_uuid=None, created_at="2026-03-19T12:00:00Z", index=0):
    """Helper to build a claude.ai message dict."""
    msg = {
        "uuid": uuid,
        "sender": sender,
        "created_at": created_at,
        "index": index,
    }
    if parent_uuid:
        msg["parent_message_uuid"] = parent_uuid
    # Human messages have "text", assistant messages have "content" blocks
    if sender == "human":
        msg["text"] = text_or_content
    else:
        if isinstance(text_or_content, str):
            msg["content"] = [{"type": "text", "text": text_or_content}]
        else:
            msg["content"] = text_or_content  # allow passing raw content blocks
    return msg


def test_parse_linear_conversation():
    """Simple 2-turn conversation produces 2 Interactions."""
    conv = {
        "uuid": "conv-1",
        "model": "claude-sonnet-4-6",
        "current_leaf_message_uuid": "m4",
        "chat_messages": [
            _make_message("m1", "human", "Hello", index=0, created_at="2026-03-19T12:00:00Z"),
            _make_message("m2", "assistant", "Hi there!", parent_uuid="m1", index=1, created_at="2026-03-19T12:00:01Z"),
            _make_message("m3", "human", "How are you?", parent_uuid="m2", index=2, created_at="2026-03-19T12:01:00Z"),
            _make_message("m4", "assistant", "I'm good!", parent_uuid="m3", index=3, created_at="2026-03-19T12:01:01Z"),
        ],
    }
    interactions = parse_conversation(conv)
    assert len(interactions) == 2
    assert interactions[0].user_message == "Hello"
    assert interactions[0].assistant_response == "Hi there!"
    assert interactions[0].source == "claude-web"
    assert interactions[0].provider == "claude"
    assert interactions[0].model == "claude-sonnet-4-6"
    assert interactions[0].session_id == "conv-1"
    assert interactions[0].timestamp == "2026-03-19T12:00:00Z"
    assert interactions[1].user_message == "How are you?"
    assert interactions[1].assistant_response == "I'm good!"


def test_parse_branched_conversation_follows_active_branch():
    """Only follows the active branch via current_leaf_message_uuid."""
    conv = {
        "uuid": "conv-2",
        "model": "claude-sonnet-4-6",
        "current_leaf_message_uuid": "m4",
        "chat_messages": [
            _make_message("m1", "human", "Hello", index=0),
            _make_message("m2", "assistant", "Hi!", parent_uuid="m1", index=1),
            # Dead branch
            _make_message("m3-dead", "human", "Dead branch message", parent_uuid="m2", index=2),
            _make_message("m4-dead", "assistant", "Dead branch reply", parent_uuid="m3-dead", index=3),
            # Active branch
            _make_message("m3", "human", "Active question", parent_uuid="m2", index=2),
            _make_message("m4", "assistant", "Active answer", parent_uuid="m3", index=3),
        ],
    }
    interactions = parse_conversation(conv)
    assert len(interactions) == 2
    assert interactions[1].user_message == "Active question"
    assert interactions[1].assistant_response == "Active answer"


def test_parse_empty_conversation():
    """Empty conversation returns no Interactions."""
    conv = {"uuid": "conv-3", "model": "test", "chat_messages": []}
    assert parse_conversation(conv) == []


def test_parse_single_turn():
    """Single human+assistant pair produces 1 Interaction."""
    conv = {
        "uuid": "conv-4",
        "model": "test",
        "current_leaf_message_uuid": "m2",
        "chat_messages": [
            _make_message("m1", "human", "One question", index=0),
            _make_message("m2", "assistant", "One answer", parent_uuid="m1", index=1),
        ],
    }
    interactions = parse_conversation(conv)
    assert len(interactions) == 1


def test_user_followup_backfill():
    """user_followup is set to the next interaction's user_message."""
    conv = {
        "uuid": "conv-5",
        "model": "test",
        "current_leaf_message_uuid": "m4",
        "chat_messages": [
            _make_message("m1", "human", "First question", index=0),
            _make_message("m2", "assistant", "First answer", parent_uuid="m1", index=1),
            _make_message("m3", "human", "Follow up", parent_uuid="m2", index=2),
            _make_message("m4", "assistant", "Follow up answer", parent_uuid="m3", index=3),
        ],
    }
    interactions = parse_conversation(conv)
    assert interactions[0].user_followup == "Follow up"
    assert interactions[1].user_followup is None  # last interaction


def test_assistant_content_blocks():
    """Assistant messages with content blocks are extracted correctly."""
    conv = {
        "uuid": "conv-6",
        "model": "test",
        "current_leaf_message_uuid": "m2",
        "chat_messages": [
            _make_message("m1", "human", "Tell me two things", index=0),
            _make_message("m2", "assistant", [
                {"type": "text", "text": "First thing."},
                {"type": "text", "text": "Second thing."},
            ], parent_uuid="m1", index=1),
        ],
    }
    interactions = parse_conversation(conv)
    assert len(interactions) == 1
    assert "First thing." in interactions[0].assistant_response
    assert "Second thing." in interactions[0].assistant_response


def test_fallback_to_index_order():
    """Falls back to index order when no current_leaf_message_uuid."""
    conv = {
        "uuid": "conv-7",
        "model": "test",
        "chat_messages": [
            _make_message("m1", "human", "Question", index=0),
            _make_message("m2", "assistant", "Answer", index=1),
        ],
    }
    interactions = parse_conversation(conv)
    assert len(interactions) == 1
    assert interactions[0].user_message == "Question"


def test_unpaired_human_message_skipped():
    """A human message without a following assistant message is skipped."""
    conv = {
        "uuid": "conv-8",
        "model": "test",
        "current_leaf_message_uuid": "m3",
        "chat_messages": [
            _make_message("m1", "human", "First", index=0),
            _make_message("m2", "assistant", "Reply", parent_uuid="m1", index=1),
            _make_message("m3", "human", "No reply yet", parent_uuid="m2", index=2),
        ],
    }
    interactions = parse_conversation(conv)
    assert len(interactions) == 1
    assert interactions[0].user_message == "First"


class _NoCloseConnection:
    """Wraps a sqlite3.Connection, delegating everything except close().

    Used to prevent run_web_capture from closing the test fixture's connection.
    """

    def __init__(self, conn):
        self._conn = conn

    def close(self):
        pass  # no-op — fixture teardown handles this

    def __getattr__(self, name):
        return getattr(self._conn, name)


def _web_capture_patches(db, conversations, full_conv):
    """Helper: return a context manager that patches all external dependencies.

    Wraps db in _NoCloseConnection so run_web_capture's conn.close() is a no-op.
    """
    wrapped = _NoCloseConnection(db)

    class _Patches:
        def __enter__(self):
            self._patchers = [
                patch("bespoke.capture.web_capture.get_session_cookie", return_value={"sessionKey": "fake"}),
                patch("bespoke.capture.web_capture.get_org_id", return_value="org-1"),
                patch("bespoke.capture.web_capture.list_conversations", return_value=conversations),
                patch("bespoke.capture.web_capture.fetch_conversation", return_value=full_conv),
                patch("bespoke.capture.web_capture.get_connection", return_value=wrapped),
                patch("bespoke.capture.web_capture.time"),
            ]
            for p in self._patchers:
                p.start()
            return self

        def __exit__(self, *args):
            for p in self._patchers:
                p.stop()

    return _Patches()


def test_run_web_capture_incremental(db):
    """Only fetches conversations updated after last run."""
    # Seed pipeline state with a past run (UTC)
    db.execute(
        "INSERT OR REPLACE INTO pipeline_state (stage, last_successful_run, metadata) "
        "VALUES ('web_capture', '2026-03-18T00:00:00+00:00', '{}')"
    )
    db.commit()

    conversations = [
        {"uuid": "c1", "updated_at": "2026-03-19T10:00:00Z", "name": "New conv"},
        {"uuid": "c2", "updated_at": "2026-03-17T10:00:00Z", "name": "Old conv"},
    ]
    full_conv = {
        "uuid": "c1",
        "model": "claude-sonnet-4-6",
        "current_leaf_message_uuid": "m2",
        "chat_messages": [
            {"uuid": "m1", "sender": "human", "text": "Hello", "created_at": "2026-03-19T10:00:00Z", "index": 0},
            {"uuid": "m2", "sender": "assistant", "content": [{"type": "text", "text": "Hi!"}],
             "parent_message_uuid": "m1", "created_at": "2026-03-19T10:00:01Z", "index": 1},
        ],
    }

    with _web_capture_patches(db, conversations, full_conv):
        stats = run_web_capture()

    assert stats["conversations_checked"] == 2
    assert stats["conversations_fetched"] == 1  # only c1 (c2 is old)
    assert stats["interactions_captured"] == 1


def test_run_web_capture_dedup(db):
    """Duplicate interactions are skipped on second run.

    updated_at is set far in the future so the incremental filter never skips
    the conversation — both runs fetch it, but the second deduplicates via
    content_hash.
    """
    conversations = [
        {"uuid": "c1", "updated_at": "2099-12-31T23:59:59Z"},
    ]
    full_conv = {
        "uuid": "c1",
        "model": "test",
        "current_leaf_message_uuid": "m2",
        "chat_messages": [
            {"uuid": "m1", "sender": "human", "text": "Hello", "created_at": "2026-03-19T10:00:00Z", "index": 0},
            {"uuid": "m2", "sender": "assistant", "content": [{"type": "text", "text": "Hi!"}],
             "parent_message_uuid": "m1", "created_at": "2026-03-19T10:00:01Z", "index": 1},
        ],
    }

    with _web_capture_patches(db, conversations, full_conv):
        stats1 = run_web_capture()
        stats2 = run_web_capture()

    assert stats1["interactions_captured"] == 1
    # Second run: conversation still shows as updated (mock doesn't change),
    # but the interaction is deduped via content_hash
    assert stats2["interactions_skipped"] == 1


# ── Error path tests ──────────────────────────────────────────────────


def test_get_org_id_401_raises():
    """401 from API raises with session-expired message."""
    mock_response = MagicMock()
    mock_response.status_code = 401
    with patch("bespoke.capture.web_capture.requests.get", return_value=mock_response):
        with pytest.raises(RuntimeError, match="Session expired"):
            get_org_id({"sessionKey": "fake"})


def test_fetch_conversation_429_retries():
    """429 response triggers exponential backoff retries."""
    mock_429 = MagicMock()
    mock_429.status_code = 429
    mock_200 = MagicMock()
    mock_200.status_code = 200
    mock_200.json.return_value = {"uuid": "c1", "chat_messages": []}
    mock_200.raise_for_status = MagicMock()

    with patch("bespoke.capture.web_capture.requests.get", side_effect=[mock_429, mock_200]), \
         patch("bespoke.capture.web_capture.time") as mock_time:
        result = fetch_conversation({"sessionKey": "fake"}, "org", "conv-1")

    assert result == {"uuid": "c1", "chat_messages": []}
    mock_time.sleep.assert_called_once_with(2)  # first retry: 2s delay


def test_get_session_cookie_missing_raises():
    """Missing sessionKey cookie raises RuntimeError."""
    from bespoke.capture.web_capture import get_session_cookie
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "fake-password\n"
    with patch("subprocess.run", return_value=mock_result), \
         patch("pycookiecheat.chrome_cookies", return_value={}):
        with pytest.raises(RuntimeError, match="No sessionKey cookie found"):
            get_session_cookie()
