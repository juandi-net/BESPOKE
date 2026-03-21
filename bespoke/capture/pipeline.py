"""Stage 1: Capture pipeline — parse sessions, embed, write to DB."""

import hashlib
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm
from sqlite_vec import serialize_float32

from bespoke.db.init import get_connection, DB_PATH
from bespoke.capture.parsers import (
    Interaction,
    ingest_extraction_jsonl,
    ingest_claude_code_direct,
    find_claude_code_sessions,
)
from bespoke.capture.embeddings import EmbeddingService
from bespoke.config import config


def capture_interaction(
    conn: sqlite3.Connection,
    interaction: Interaction,
    embedding_svc: Optional[EmbeddingService] = None,
    stats: Optional[dict] = None,
) -> Optional[int]:
    """Hash, optionally embed, and insert a single interaction.

    Uses INSERT OR IGNORE with the content_hash unique index for dedup.

    Args:
        conn: Database connection.
        interaction: The interaction to capture.
        embedding_svc: If provided, compute and store embedding. If None, skip.
        stats: If provided, increment 'interactions_captured' or 'interactions_skipped'.

    Returns:
        The rowid of the inserted row, or None if skipped (dedup).
    """
    content_hash = hashlib.sha256(
        f"{interaction.session_id or ''}|{interaction.user_message}|{interaction.assistant_response}".encode()
    ).hexdigest()

    if embedding_svc is not None:
        try:
            embed_text = f"{interaction.user_message}\n{interaction.assistant_response}"
            embedding, num_chunks = embedding_svc.embed(embed_text)
        except Exception as e:
            if stats is not None:
                stats["interactions_failed"] = stats.get("interactions_failed", 0) + 1
            tqdm.write(f"  [FAIL] session={interaction.session_id} "
                       f"user={interaction.user_message[:80]!r}... — {e}")
            return None

    cursor = conn.execute("""
        INSERT OR IGNORE INTO interactions (
            provider, model, source, session_id,
            system_prompt, user_message, assistant_response,
            user_followup,
            input_tokens, output_tokens, captured_at, content_hash
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        interaction.provider,
        interaction.model,
        interaction.source,
        interaction.session_id,
        interaction.system_prompt,
        interaction.user_message,
        interaction.assistant_response,
        interaction.user_followup,
        interaction.input_tokens,
        interaction.output_tokens,
        interaction.timestamp,
        content_hash,
    ))

    if cursor.rowcount == 1:
        if embedding_svc is not None:
            conn.execute("""
                INSERT INTO vec_interactions (rowid, interaction_embedding)
                VALUES (?, ?)
            """, (cursor.lastrowid, serialize_float32(embedding.tolist())))
        if stats is not None:
            stats["interactions_captured"] = stats.get("interactions_captured", 0) + 1
            if embedding_svc is not None and num_chunks > 1:
                stats["interactions_chunked"] = stats.get("interactions_chunked", 0) + 1
                stats["total_chunks"] = stats.get("total_chunks", 0) + num_chunks
        return cursor.lastrowid
    else:
        if stats is not None:
            stats["interactions_skipped"] = stats.get("interactions_skipped", 0) + 1
        return None


def run_capture(
    session_dir: Optional[Path] = None,
    jsonl_path: Optional[Path] = None,
    since: Optional[datetime] = None,
) -> dict:
    """Run the capture pipeline.

    Two ingestion paths:
    - jsonl_path: Ingest from ai-data-extraction JSONL export (backfill or batch)
    - session_dir/since: Parse Claude Code session files directly (ongoing capture)

    Returns stats dict with counts.
    """
    conn = get_connection()
    embedding_svc = EmbeddingService.get()

    stats = {"sessions_processed": 0, "sessions_skipped": 0,
             "interactions_captured": 0, "interactions_skipped": 0,
             "interactions_failed": 0, "interactions_chunked": 0, "total_chunks": 0}

    if jsonl_path and jsonl_path.exists():
        print(f"Ingesting from JSONL: {jsonl_path}")
        interactions = ingest_extraction_jsonl(jsonl_path)
        stats["sessions_processed"] = 1

        for interaction in interactions:
            capture_interaction(conn, interaction, embedding_svc, stats)
        conn.commit()
    else:
        if since is None:
            row = conn.execute(
                "SELECT last_successful_run FROM pipeline_state WHERE stage = 'stage1'"
            ).fetchone()
            if row and row["last_successful_run"]:
                since = datetime.fromisoformat(row["last_successful_run"])

        # Load known session IDs to skip already-captured sessions
        known_sessions = {row[0] for row in conn.execute(
            "SELECT DISTINCT session_id FROM interactions"
        ).fetchall()}

        session_paths = list(find_claude_code_sessions(session_dir, since=since))
        to_process = [p for p in session_paths if p.stem not in known_sessions]
        stats["sessions_skipped"] = len(session_paths) - len(to_process)

        # Summary line
        if stats["sessions_skipped"]:
            print(f"{len(session_paths)} sessions found, "
                  f"{stats['sessions_skipped']} cached, "
                  f"{len(to_process)} to process")
        else:
            print(f"{len(session_paths)} sessions to process")

        if not to_process:
            print("Nothing to do.")
        else:
            pbar = tqdm(to_process, unit="sess", dynamic_ncols=True,
                        ascii="░▏▎▍▌▋▊▉█",
                        bar_format="  {percentage:3.0f}% {bar:30} {n_fmt}/{total_fmt} sessions | "
                                   "{desc} [{elapsed}<{remaining}]")
            pbar.set_description_str("starting")

            for session_path in pbar:
                interactions = ingest_claude_code_direct(session_path)
                stats["sessions_processed"] += 1
                n = len(interactions)

                for i, interaction in enumerate(interactions):
                    capture_interaction(conn, interaction, embedding_svc, stats)
                    if n > 3:
                        pbar.set_description_str(
                            f"{stats['interactions_captured']} new interactions "
                            f"({i+1}/{n} in session)")

                conn.commit()
                known_sessions.add(session_path.stem)

                pbar.set_description_str(
                    f"{stats['interactions_captured']} new interactions")

    # Update pipeline state
    conn.execute("""
        UPDATE pipeline_state
        SET last_successful_run = ?, metadata = ?
        WHERE stage = 'stage1'
    """, (datetime.now().isoformat(), json.dumps(stats)))

    conn.commit()
    conn.close()

    EmbeddingService.unload()
    return stats


def main():
    """CLI entry point for Stage 1 capture."""
    import argparse

    parser = argparse.ArgumentParser(description="BESPOKE Stage 1: Capture")
    parser.add_argument("--session-dir", type=Path, help="Override session directory")
    parser.add_argument("--jsonl", type=Path, help="Ingest from ai-data-extraction JSONL export")
    parser.add_argument("--since", type=str, help="Only process sessions modified after this ISO datetime")
    parser.add_argument("--backfill", action="store_true", help="Process all sessions (ignore last run time)")
    args = parser.parse_args()

    since = None
    if args.since:
        since = datetime.fromisoformat(args.since)
    elif args.backfill:
        since = datetime(2020, 1, 1)  # Process everything

    print("bespoke capture")

    stats = run_capture(
        session_dir=args.session_dir,
        jsonl_path=args.jsonl,
        since=since,
    )

    if stats["interactions_captured"] or stats["sessions_processed"]:
        print(f"\nDone: {stats['interactions_captured']} interactions "
              f"from {stats['sessions_processed']} sessions"
              + (f" ({stats['interactions_failed']} failed)" if stats["interactions_failed"] else ""))


if __name__ == "__main__":
    main()
