"""Read-only Track 2 preview — validate segmentation + mechanical extraction on the real
warehouse WITHOUT modifying anything or touching the live pipeline.

Run:  python -m bespoke.extract.preview [num_samples]

Prints: segmentation stats over all sessions, a few sample conversations with boundaries
marked, tangled-routing counts, sample mechanical pairs, and a mechanical-vs-old-LLM pair
count comparison. Nothing is written to the DB.
"""
import sys
from collections import defaultdict

import numpy as np

from bespoke.db.init import get_connection
from bespoke.extract.conversations import (
    segment_conversation, trajectory_features, _cosine_dist, _gap_seconds,
)
from bespoke.extract.mechanical import is_tangled, extract_sft_pairs


def _load_light(conn):
    """Per-session turns with metadata + embedding (NO text, to stay light on 40k rows)."""
    rows = conn.execute("""
        SELECT i.id AS id, i.session_id AS sid, i.captured_at AS ts, i.feedback_class AS fc,
               i.cache_read_tokens AS cr, i.cache_creation_tokens AS cc,
               v.interaction_embedding AS emb
        FROM interactions i LEFT JOIN vec_interactions v ON v.rowid = i.id
        ORDER BY i.captured_at ASC, i.id ASC
    """).fetchall()
    groups = defaultdict(list)
    for r in rows:
        key = r["sid"] if r["sid"] not in (None, "") else f"__solo_{r['id']}"
        emb = np.frombuffer(r["emb"], dtype=np.float32) if r["emb"] is not None else None
        groups[key].append({
            "id": r["id"], "ts": r["ts"], "feedback_class": r["fc"],
            "cache_read": r["cr"], "cache_creation": r["cc"], "emb": emb,
        })
    return groups


def _fetch_text(conn, ids):
    if not ids:
        return {}
    qmarks = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT id, user_message, assistant_response FROM interactions WHERE id IN ({qmarks})",
        list(ids)).fetchall()
    return {r["id"]: (r["user_message"] or "", r["assistant_response"] or "") for r in rows}


def _trunc(s, n=78):
    s = " ".join((s or "").split())
    return s[:n] + ("…" if len(s) > n else "")


def main():
    n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    conn = get_connection()

    print("=" * 70)
    print("TRACK 2 PREVIEW — segmentation + mechanical extraction (READ-ONLY)")
    print("=" * 70)

    groups = _load_light(conn)
    seg_counts, total_convos, split_sessions, tangled = [], 0, 0, 0
    multi = []  # (n_turns, sid) for multi-turn sessions, for sampling
    for sid, turns in groups.items():
        segs = segment_conversation(turns)
        seg_counts.append(len(segs))
        total_convos += len(segs)
        if len(segs) > 1:
            split_sessions += 1
        if is_tangled(turns):
            tangled += 1
        if len(turns) >= 4 and not sid.startswith("__solo_"):
            multi.append((len(turns), sid))

    n_sess = len(groups)
    print(f"\nSessions: {n_sess}")
    print(f"Conversations after segmentation: {total_convos}  "
          f"(avg {total_convos / n_sess:.2f} per session)")
    print(f"Sessions that split into >1 conversation: {split_sessions} "
          f"({100 * split_sessions / n_sess:.1f}%)")
    print(f"Sessions flagged TANGLED (→ route to distiller): {tangled} "
          f"({100 * tangled / n_sess:.1f}%)")
    avg_turns = np.mean([len(t) for t in groups.values()])
    print(f"Avg turns/session: {avg_turns:.1f}")

    # ---- sample conversations with boundaries marked ----
    multi.sort(reverse=True)
    sample_sids = [sid for _, sid in multi[:n_samples]]
    sample_ids = [t["id"] for sid in sample_sids for t in groups[sid]]
    text = _fetch_text(conn, sample_ids)

    print("\n" + "=" * 70)
    print(f"SAMPLE CONVERSATIONS (top {len(sample_sids)} by length) — boundaries marked")
    print("=" * 70)
    for sid in sample_sids:
        turns = groups[sid]
        segs = segment_conversation(turns)
        seg_starts = {s[0]["id"] for s in segs}  # first turn id of each segment
        print(f"\n── session {sid[:16]}…  ({len(turns)} turns → {len(segs)} conversations) "
              f"tangled={is_tangled(turns)} ──")
        for i, t in enumerate(turns):
            um, _ = text.get(t["id"], ("", ""))
            if i > 0 and t["id"] in seg_starts:
                gap = _gap_seconds(turns[i - 1], t)
                dist = _cosine_dist(turns[i - 1].get("emb"), t.get("emb"))
                print(f"      ════ NEW CONVERSATION  (gap {gap/60:.0f} min, emb-dist {dist:.2f}) ════")
            print(f"   {i+1:>2}. {_trunc(um)}")

    # ---- mechanical pairs sample + comparison ----
    print("\n" + "=" * 70)
    print("MECHANICAL PAIRS — sample + count comparison vs old LLM extract")
    print("=" * 70)
    sample_pairs = []
    for sid in sample_sids:
        turns_with_text = [{**t, "user_message": text.get(t["id"], ("", ""))[0],
                            "assistant_response": text.get(t["id"], ("", ""))[1]}
                           for t in groups[sid]]
        sample_pairs += extract_sft_pairs(turns_with_text)
    for p in sample_pairs[:5]:
        print(f"  • q: {_trunc(p['instruction'], 60)}")
        print(f"    a: {_trunc(p['response'], 60)}   [quality {p['quality']:.2f}]")

    # global candidate count: non-reject, non-empty interactions
    mech_candidates = conn.execute("""
        SELECT COUNT(*) FROM interactions
        WHERE COALESCE(feedback_class,'') NOT IN ('reject','strong_reject')
          AND TRIM(COALESCE(user_message,'')) != ''
          AND TRIM(COALESCE(assistant_response,'')) != ''
    """).fetchone()[0]
    old_pairs = conn.execute("SELECT COUNT(*) FROM training_pairs").fetchone()[0]
    print(f"\nCandidate mechanical SFT pairs (whole warehouse): {mech_candidates}")
    print(f"Old LLM-extract training_pairs:                    {old_pairs}")

    print("\n" + "=" * 70)
    print("Nothing was written. Tune thresholds in bespoke/extract/ then re-run.")
    print("Cache signal not exercised (0 rows have cache data — needs fresh captures).")
    print("=" * 70)
    conn.close()


if __name__ == "__main__":
    main()
