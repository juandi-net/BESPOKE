"""Token estimation and session chunking for Stage 2a."""

PROMPT_OVERHEAD_TOKENS = 2000
CHARS_PER_TOKEN = 4
CHUNK_THRESHOLD = 600_000  # estimated tokens before chunking kicks in
OVERLAP_TURNS = 5
OVERLAP_MAX_TOKENS = 50_000


def estimate_session_tokens(turns: list[dict]) -> int:
    """Estimate total tokens for a session's turns + prompt overhead."""
    char_count = sum(
        len(t.get("user_message", "")) + len(t.get("assistant_response", ""))
        for t in turns
    )
    return char_count // CHARS_PER_TOKEN + PROMPT_OVERHEAD_TOKENS


def estimate_turn_tokens(turn: dict) -> int:
    """Estimate tokens for a single turn."""
    chars = len(turn.get("user_message", "")) + len(turn.get("assistant_response", ""))
    return chars // CHARS_PER_TOKEN


def compute_max_tokens(extractable_turn_count: int) -> int:
    """Compute max_tokens for the LLM call based on extractable turns."""
    return min(max(8192, extractable_turn_count * 1500), 64000)


def build_chunks(
    turns: list[dict],
    threshold: int = CHUNK_THRESHOLD,
) -> list[dict]:
    """Split a session into chunks if it exceeds the token threshold.

    Returns a list of chunk dicts:
        {
            "start": int,          # 1-based first turn number in this chunk
            "end": int,            # 1-based last turn number in this chunk
            "context_only": set,   # 1-based turn numbers that are context-only
        }

    For sessions under the threshold, returns a single chunk with no context-only turns.
    """
    n = len(turns)
    total_est = estimate_session_tokens(turns)

    if total_est <= threshold:
        return [{"start": 1, "end": n, "context_only": set()}]

    chunks = []
    cursor = 0  # 0-based index: next turn to start a new chunk from

    while cursor < n:
        context_only = set()

        # Determine overlap context from previous chunk
        if chunks:
            prev_end_idx = chunks[-1]["end"] - 1  # convert 1-based end to 0-based
            # Walk backwards from prev_end to find overlap turns within budget
            overlap_tokens = 0
            overlap_start_idx = prev_end_idx + 1  # default: no overlap

            for i in range(prev_end_idx, max(-1, prev_end_idx - OVERLAP_TURNS), -1):
                t_tokens = estimate_turn_tokens(turns[i])
                if overlap_tokens + t_tokens > OVERLAP_MAX_TOKENS:
                    break
                overlap_tokens += t_tokens
                overlap_start_idx = i

            # Only include overlap if it actually goes before cursor
            if overlap_start_idx <= prev_end_idx:
                cursor = overlap_start_idx
                # Mark overlap turns as context-only (1-based)
                for i in range(overlap_start_idx, prev_end_idx + 1):
                    context_only.add(i + 1)

        chunk_start = cursor  # 0-based
        chunk_tokens = PROMPT_OVERHEAD_TOKENS

        # Add context-only turns' tokens first
        for i in range(chunk_start, min(chunk_start + len(context_only), n)):
            chunk_tokens += estimate_turn_tokens(turns[i])

        # Fill with extractable turns up to threshold
        end_idx = chunk_start + len(context_only)
        while end_idx < n:
            t_tokens = estimate_turn_tokens(turns[end_idx])
            if chunk_tokens + t_tokens > threshold and end_idx > chunk_start + len(context_only):
                # Don't break if we haven't added any extractable turns yet
                break
            chunk_tokens += t_tokens
            end_idx += 1

        # Must make progress — at least one extractable turn
        if end_idx <= chunk_start + len(context_only):
            end_idx = min(chunk_start + len(context_only) + 1, n)

        chunks.append({
            "start": chunk_start + 1,  # 1-based
            "end": end_idx,            # 1-based (inclusive)
            "context_only": context_only,
        })

        cursor = end_idx

    # Runtime safety net: verify no-gaps invariant
    _assert_no_gaps(chunks, n)

    return chunks


def _assert_no_gaps(chunks: list[dict], total_turns: int) -> None:
    """Verify every turn 1..total_turns appears in exactly one chunk's extractable range."""
    all_extractable = set()
    for chunk in chunks:
        extractable = set(range(chunk["start"], chunk["end"] + 1)) - chunk["context_only"]
        overlap = extractable & all_extractable
        assert not overlap, (
            f"build_chunks bug: duplicate extractable turns {overlap}"
        )
        all_extractable |= extractable

    expected = set(range(1, total_turns + 1))
    missing = expected - all_extractable
    assert not missing, (
        f"build_chunks bug: missing turns {missing}"
    )
