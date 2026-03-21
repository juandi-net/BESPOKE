from bespoke.teach.chunking import estimate_session_tokens, build_chunks, compute_max_tokens


def _make_turn(user_len=100, assistant_len=400):
    """Create a fake turn dict with content of specified char lengths."""
    return {
        "user_message": "x" * user_len,
        "assistant_response": "y" * assistant_len,
    }


def test_estimate_tokens_basic():
    turns = [_make_turn(100, 400)]  # 500 chars / 4 = 125 tokens + 2K overhead
    est = estimate_session_tokens(turns)
    assert est == 125 + 2000


def test_estimate_tokens_multiple():
    turns = [_make_turn(100, 400)] * 10  # 10 * 125 + 2000 = 3250
    est = estimate_session_tokens(turns)
    assert est == 3250


def test_build_chunks_small_session():
    """Session under threshold → single chunk, no context-only turns."""
    turns = [_make_turn(100, 400)] * 5
    chunks = build_chunks(turns, threshold=600_000)
    assert len(chunks) == 1
    assert chunks[0]["start"] == 1
    assert chunks[0]["end"] == 5
    assert chunks[0]["context_only"] == set()


def test_build_chunks_large_session():
    """Session over threshold → multiple chunks with overlap."""
    # Each turn ~25K chars = ~6250 tokens. 100 turns = 625K + 2K overhead.
    turns = [_make_turn(5000, 20000)] * 100
    chunks = build_chunks(turns, threshold=300_000)  # low threshold for testing
    assert len(chunks) >= 2
    # First chunk has no context-only turns
    assert chunks[0]["context_only"] == set()
    # Second chunk has overlap turns as context-only
    assert len(chunks[1]["context_only"]) > 0
    # No gaps — every turn number appears in exactly one chunk's extractable range
    _assert_no_gaps(chunks, 100)


def test_build_chunks_single_oversized_turn():
    """A single turn exceeding the threshold still produces one chunk."""
    turns = [_make_turn(500_000, 2_000_000)]  # ~625K tokens
    chunks = build_chunks(turns, threshold=100_000)
    assert len(chunks) == 1
    assert chunks[0]["start"] == 1
    assert chunks[0]["end"] == 1
    assert chunks[0]["context_only"] == set()


def test_build_chunks_overlap_with_large_turns():
    """When individual turns are large, overlap may include fewer than OVERLAP_TURNS."""
    # Each turn ~50K tokens. Overlap budget is 50K tokens, so at most 1 overlap turn.
    turns = [_make_turn(50_000, 150_000)] * 20
    chunks = build_chunks(turns, threshold=200_000)
    assert len(chunks) >= 2
    _assert_no_gaps(chunks, 20)


def test_compute_max_tokens():
    # Formula: min(max(8192, turns * 1500), 64000)
    assert compute_max_tokens(3) == 8192  # floor: 3 * 1500 = 4500 < 8192
    assert compute_max_tokens(10) == 15000  # mid: 10 * 1500
    assert compute_max_tokens(50) == 64000  # ceiling: 50 * 1500 = 75000 > 64000


def _assert_no_gaps(chunks, total_turns):
    """Verify every turn 1..total_turns appears in exactly one chunk's extractable range."""
    all_extractable = set()
    for chunk in chunks:
        extractable = set(range(chunk["start"], chunk["end"] + 1)) - chunk["context_only"]
        overlap = extractable & all_extractable
        assert not overlap, f"Duplicate extractable turns: {overlap}"
        all_extractable |= extractable
    expected = set(range(1, total_turns + 1))
    missing = expected - all_extractable
    assert not missing, f"Missing turns: {missing}"
