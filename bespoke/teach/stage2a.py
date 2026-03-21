"""Stage 2a: Per-session extraction pipeline (nightly).

Groups interactions by session_id, sends full conversations to the teacher
LLM for extraction with conversation context, and writes results back to
the DB on the main thread.

Public API (used by tests and other modules):
    group_sessions, build_turn_map, compute_context_only_turns, apply_extractions
"""

import json
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional

from tqdm import tqdm

from bespoke.config import config
from bespoke.db.init import get_connection
from bespoke.teach.chunking import (
    CHUNK_THRESHOLD,
    build_chunks,
    compute_max_tokens,
    estimate_session_tokens,
)
from bespoke.teach.llm_client import chat_json
from bespoke.teach.prompts import get_benchmark_context, make_session_extraction_prompt

# Larger payloads now (full sessions), so fewer concurrent requests.
MAX_CONCURRENT = 5

# Set on KeyboardInterrupt to suppress output from in-flight threads.
_cancelled = threading.Event()


# ── Public helpers ──────────────────────────────────────────────────


def group_sessions(conn) -> dict[str, list]:
    """Query unprocessed interactions and group by session_id.

    Interactions with NULL/empty session_id get synthetic keys like
    ``__solo_{id}`` so they are processed as single-interaction sessions.

    Returns dict mapping session_key -> list of sqlite3.Row, ordered by
    captured_at within each session.
    """
    rows = conn.execute("""
        SELECT * FROM interactions
        WHERE COALESCE(stage2a_fail_count, 0) < 3
          AND (
              -- Session has at least one unprocessed interaction
              session_id IN (
                  SELECT DISTINCT session_id FROM interactions
                  WHERE processed_2a_at IS NULL
                    AND COALESCE(stage2a_fail_count, 0) < 3
                    AND session_id IS NOT NULL AND session_id != ''
              )
              OR
              -- Solo interactions (no session_id) that are unprocessed
              (
                  (session_id IS NULL OR session_id = '')
                  AND processed_2a_at IS NULL
              )
          )
        ORDER BY captured_at ASC
    """).fetchall()

    sessions: dict[str, list] = defaultdict(list)
    for row in rows:
        sid = row["session_id"]
        if sid is None or sid == "":
            key = f"__solo_{row['id']}"
        else:
            key = sid
        sessions[key].append(row)

    return dict(sessions)


def build_turn_map(rows: list) -> dict[int, int]:
    """Build a mapping from 1-based turn number to interaction id.

    Turns are numbered in the order the rows appear (assumed sorted by
    captured_at already).
    """
    return {i + 1: row["id"] for i, row in enumerate(rows)}


def compute_context_only_turns(rows: list) -> set[int]:
    """Return 1-based turn numbers for already-processed interactions.

    These are interactions where ``processed_2a_at IS NOT NULL`` — they
    serve as context-only turns for incremental processing of sessions
    that got new interactions since the last run.
    """
    return {
        i + 1
        for i, row in enumerate(rows)
        if row["processed_2a_at"] is not None
    }


def apply_extractions(
    conn,
    turn_map: dict[int, int],
    extractions: list[dict],
    extractable_turns: set[int],
) -> dict:
    """Write extraction results to DB. Returns counts dict.

    For each extraction in *extractions*, maps turn_number -> interaction_id
    via *turn_map* and calls ``update_interaction()``.

    Then sets ``processed_2a_at`` on ALL interactions in *turn_map* whose
    turn_number is in *extractable_turns* (including trivial turns that the
    LLM skipped).

    All writes happen in the caller's transaction.
    """
    now = datetime.now().isoformat()
    succeeded = 0
    skipped = 0

    # Build lookup of extractions by turn_number
    extraction_by_turn: dict[int, dict] = {}
    for ext in extractions:
        tn = ext.get("turn_number")
        if tn is not None:
            extraction_by_turn[tn] = ext

    # Write extractions for turns that got results
    for turn_num, ext in extraction_by_turn.items():
        iid = turn_map.get(turn_num)
        if iid is None:
            continue
        if turn_num not in extractable_turns:
            continue
        if ext.get("skip_reason"):
            skipped += 1
        else:
            update_interaction(conn, iid, ext)
            succeeded += 1

    # Mark ALL extractable turns as processed (including trivial turns the LLM
    # omitted from the array and explicit skips).
    for turn_num in extractable_turns:
        iid = turn_map.get(turn_num)
        if iid is None:
            continue
        conn.execute(
            "UPDATE interactions SET processed_2a_at = ? WHERE id = ? AND processed_2a_at IS NULL",
            (now, iid),
        )

    return {"succeeded": succeeded, "skipped": skipped}


# ── DB write helper (unchanged from v1) ────────────────────────────


def update_interaction(conn, interaction_id: int, result: dict) -> None:
    """Write Stage 2a extraction results back to the interaction record."""
    conn.execute("""
        UPDATE interactions SET
            domain = ?,
            quality_score = ?,
            reasoning_primitives = ?,
            feedback_class = ?,
            feedback_raw = ?,
            processed_2a_at = ?
        WHERE id = ?
    """, (
        result.get("domain"),
        result.get("quality_score"),
        json.dumps(result.get("reasoning_primitives", [])),
        result.get("feedback_class"),
        result.get("feedback_reasoning"),
        datetime.now().isoformat(),
        interaction_id,
    ))

    # Insert training pair if quality is sufficient
    quality = result.get("quality_score", "exclude")
    if quality in ("high", "medium") and result.get("training_pair"):
        tp = result["training_pair"]
        if tp.get("instruction") and tp.get("response"):
            conn.execute("""
                INSERT INTO training_pairs (
                    interaction_id, pair_type, domain,
                    instruction, response,
                    quality_score, reasoning_primitives
                ) VALUES (?, 'sft', ?, ?, ?, ?, ?)
            """, (
                interaction_id,
                result.get("domain", "general"),
                tp["instruction"],
                tp["response"],
                quality,
                json.dumps(result.get("reasoning_primitives", [])),
            ))

    # Insert DPO pair if present
    if result.get("dpo_pair"):
        dp = result["dpo_pair"]
        if dp.get("prompt") and dp.get("chosen") and dp.get("rejected"):
            conn.execute("""
                INSERT INTO training_pairs (
                    interaction_id, pair_type, domain,
                    prompt, chosen, rejected,
                    quality_score, reasoning_primitives
                ) VALUES (?, 'dpo', ?, ?, ?, ?, ?, ?)
            """, (
                interaction_id,
                result.get("domain", "general"),
                dp["prompt"],
                dp["chosen"],
                dp["rejected"],
                quality,
                json.dumps(result.get("reasoning_primitives", [])),
            ))


# ── Thread-pool worker (pure I/O, no DB) ───────────────────────────


def process_session(
    session_id: str,
    turns: list[dict],
    context_only_turns: set[int],
    benchmark_context: str | None = None,
) -> tuple[str, list[dict]]:
    """Call the teacher LLM for a full session. Handles chunking internally.

    Pure I/O — no DB access. Thread-safe.

    Returns ``(session_id, extractions)`` where *extractions* is a list of
    dicts each containing at least ``turn_number``.

    For chunked sessions, processes chunks sequentially within this thread,
    filtering each chunk's results to only keep turns in ``extractable_turns``
    (i.e., not in ``context_only_turns`` and within the chunk's range).
    """
    all_extractions: list[dict] = []
    chunks = build_chunks(turns, CHUNK_THRESHOLD)

    for chunk in chunks:
        chunk_start = chunk["start"]
        chunk_end = chunk["end"]
        chunk_context = chunk["context_only"] | context_only_turns

        # Slice turns for this chunk (0-based indexing)
        chunk_turns = turns[chunk_start - 1 : chunk_end]

        # The prompt builder numbers turns 1..len(chunk_turns) internally.
        # Remap absolute context-only turn numbers to chunk-relative numbering.
        chunk_relative_context = set()
        for abs_turn in range(chunk_start, chunk_end + 1):
            rel_turn = abs_turn - chunk_start + 1
            if abs_turn in chunk_context:
                chunk_relative_context.add(rel_turn)

        # Extractable turns for this chunk (absolute numbers)
        extractable_in_chunk = (
            set(range(chunk_start, chunk_end + 1))
            - chunk_context
        )
        extractable_count = len(extractable_in_chunk)

        if extractable_count == 0:
            continue

        max_tokens = compute_max_tokens(extractable_count)

        messages = make_session_extraction_prompt(
            turns=[dict(t) for t in chunk_turns],
            context_only_turns=chunk_relative_context,
            benchmark_context=benchmark_context,
        )

        result = chat_json(
            messages=messages,
            model=config.llm.model_stage_2a,
            temperature=0.2,
            max_tokens=max_tokens,
        )

        # Normalize: if the LLM returned a dict instead of a list, wrap it
        if isinstance(result, dict):
            result = [result]

        # Remap relative turn numbers back to absolute
        for ext in result:
            rel_tn = ext.get("turn_number")
            if rel_tn is not None:
                abs_tn = rel_tn + chunk_start - 1
                ext["turn_number"] = abs_tn

        # Client-side dedup: only keep turns in extractable_in_chunk
        filtered = [
            ext for ext in result
            if ext.get("turn_number") in extractable_in_chunk
        ]
        all_extractions.extend(filtered)

    return (session_id, all_extractions)


# ── Main orchestrator ───────────────────────────────────────────────


def run_stage_2a() -> dict:
    """Run Stage 2a on all unprocessed interactions, grouped by session.

    LLM calls run concurrently (MAX_CONCURRENT threads, one session per thread).
    DB writes are serial on the main thread.
    """
    _cancelled.clear()
    conn = get_connection()

    sessions = group_sessions(conn)
    benchmark_context = get_benchmark_context()

    stats = {"processed": 0, "succeeded": 0, "failed": 0, "skipped": 0}

    if not sessions:
        print("Nothing to extract.")
    else:
        # Count total interactions for the summary line
        total_interactions = sum(len(turns) for turns in sessions.values())
        print(f"{len(sessions)} sessions ({total_interactions} interactions)")

        pbar = tqdm(
            total=len(sessions),
            unit="sess",
            dynamic_ncols=True,
            ascii="░▏▎▍▌▋▊▉█",
            bar_format=(
                "  {percentage:3.0f}% {bar:30} {n_fmt}/{total_fmt} sessions | "
                "{desc} [{elapsed}<{remaining}]"
            ),
        )
        pbar.set_description_str("starting")

        executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT)
        try:
            # Precompute per-session metadata and submit to thread pool
            future_to_meta: dict = {}
            for sid, rows in sessions.items():
                turns = [dict(r) for r in rows]
                turn_map = build_turn_map(rows)
                ctx_only = compute_context_only_turns(rows)

                future = executor.submit(process_session, sid, turns, ctx_only, benchmark_context)
                future_to_meta[future] = {
                    "session_id": sid,
                    "rows": rows,
                    "turn_map": turn_map,
                    "context_only_turns": ctx_only,
                }

            for future in as_completed(future_to_meta):
                meta = future_to_meta[future]
                sid = meta["session_id"]
                turn_map = meta["turn_map"]
                ctx_only = meta["context_only_turns"]
                rows = meta["rows"]
                stats["processed"] += 1

                # Extractable turns = all turns minus context-only
                extractable_turns = set(turn_map.keys()) - ctx_only

                try:
                    _returned_sid, extractions = future.result()
                except Exception as e:
                    extractions = None
                    if not _cancelled.is_set():
                        tqdm.write(f"  [FAIL] session {sid}: {e}")

                if extractions is None:
                    # Session-level failure: increment fail count on all
                    # unprocessed interactions in this session.
                    stats["failed"] += 1
                    for turn_num in extractable_turns:
                        iid = turn_map.get(turn_num)
                        if iid is not None:
                            conn.execute("""
                                UPDATE interactions
                                SET stage2a_fail_count = COALESCE(stage2a_fail_count, 0) + 1
                                WHERE id = ?
                            """, (iid,))
                    conn.commit()
                else:
                    # Apply extractions in one transaction
                    counts = apply_extractions(
                        conn, turn_map, extractions, extractable_turns
                    )
                    stats["succeeded"] += counts["succeeded"]
                    stats["skipped"] += counts["skipped"]
                    conn.commit()

                # Update progress bar
                desc_parts = [f"{stats['succeeded']} extracted"]
                if stats["skipped"]:
                    desc_parts.append(f"{stats['skipped']} skipped")
                if stats["failed"]:
                    desc_parts.append(f"{stats['failed']} failed")
                pbar.set_description_str(", ".join(desc_parts))
                pbar.update(1)

        except KeyboardInterrupt:
            _cancelled.set()
            executor.shutdown(wait=False, cancel_futures=True)
            pbar.close()
            conn.commit()
            raise

        executor.shutdown(wait=False)
        pbar.close()

    # Update pipeline state
    conn.execute("""
        UPDATE pipeline_state
        SET last_successful_run = ?, metadata = ?
        WHERE stage = 'stage2a'
    """, (datetime.now().isoformat(), json.dumps(stats)))

    conn.commit()
    conn.close()

    return stats


def main():
    """CLI entry point for Stage 2a."""
    run_stage_2a()


if __name__ == "__main__":
    main()
