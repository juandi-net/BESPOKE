"""Session-level representation: assemble conversations, embed them, extract trajectory features.

The session-level latent is the cheap geometric substitute for the ONE thing that made
Stage 2a send whole sessions to the frontier LLM — whole-conversation context. Used for
ROUTING and archetypes (not fine work): a single vector for a long, topic-shifting session
gets muddy, so the turn-level methods still do pair selection.
"""
import numpy as np
from datetime import datetime
from collections import defaultdict

ACCEPT = {"accept", "strong_accept"}


def assemble_sessions(conn):
    """Group all interactions into conversations by session_id (with __solo_ fallback).

    Returns list of {session_id, turns:[...]} where each turn carries id, ts (captured_at),
    feedback_class, cache_read, cache_creation, user_message, assistant_response, emb (np 768
    or None). Turns are ordered by captured_at.
    """
    rows = conn.execute("""
        SELECT i.id AS id, i.session_id AS sid, i.captured_at AS ts,
               i.feedback_class AS fc, i.cache_read_tokens AS cr,
               i.cache_creation_tokens AS cc, i.user_message AS um,
               i.assistant_response AS ar, v.interaction_embedding AS emb
        FROM interactions i
        LEFT JOIN vec_interactions v ON v.rowid = i.id
        ORDER BY i.captured_at ASC, i.id ASC
    """).fetchall()

    groups = defaultdict(list)
    for r in rows:
        key = r["sid"] if r["sid"] not in (None, "") else f"__solo_{r['id']}"
        emb = np.frombuffer(r["emb"], dtype=np.float32) if r["emb"] is not None else None
        groups[key].append({
            "id": r["id"], "ts": r["ts"], "feedback_class": r["fc"],
            "cache_read": r["cr"], "cache_creation": r["cc"],
            "user_message": r["um"], "assistant_response": r["ar"], "emb": emb,
        })
    return [{"session_id": k, "turns": v} for k, v in groups.items()]


def session_embedding(turns):
    """Mean-pool the turn embeddings into one L2-normalized session vector (768-dim)."""
    embs = [t["emb"] for t in turns if t.get("emb") is not None]
    if not embs:
        return np.zeros(768, dtype=np.float32)
    m = np.mean(np.vstack(embs), axis=0).astype(np.float32)
    norm = float(np.linalg.norm(m))
    return (m / norm).astype(np.float32) if norm > 0 else m


def _parse_ts(ts):
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def trajectory_features(turns):
    """Shape signals a pure embedding misses: length, did it converge, pacing, presence."""
    n = len(turns)
    ended_in_accept = bool(turns and turns[-1].get("feedback_class") in ACCEPT)

    times = [t for t in (_parse_ts(x.get("ts")) for x in turns) if t is not None]
    gaps = [(times[i] - times[i - 1]).total_seconds() for i in range(1, len(times))]
    mean_gap_s = float(np.mean(gaps)) if gaps else 0.0

    cache_reads = sum(1 for t in turns if (t.get("cache_read") or 0) > 0)
    cache_read_ratio = cache_reads / n if n else 0.0

    return {"n_turns": n, "ended_in_accept": ended_in_accept,
            "mean_gap_s": mean_gap_s, "cache_read_ratio": cache_read_ratio}
