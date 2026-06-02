# tests/test_extract_run.py
"""Track 2 / go-local: geometric extract orchestrator end-to-end (no cloud)."""
from sqlite_vec import serialize_float32


def _ins(db, sid, ts, um, ar, followup, emb):
    cur = db.execute(
        "INSERT INTO interactions (provider,model,source,session_id,user_message,"
        "assistant_response,user_followup,captured_at) VALUES ('c','m','s',?,?,?,?,?)",
        (sid, um, ar, followup, ts))
    rid = cur.lastrowid
    db.execute("INSERT INTO vec_interactions (rowid,interaction_embedding) VALUES (?,?)",
               (rid, serialize_float32(emb)))
    return rid


GOOD = [1.0] + [0.0] * 767
BAD = [-1.0] + [0.0] * 767


class TestGeometricExtract:
    def test_end_to_end_local(self, db):
        from bespoke.extract.run import run_geometric_extract
        for i in range(6):
            _ins(db, "s1", f"2026-06-02T00:0{i}:00Z", "q", "a great answer", "perfect, do it", GOOD)
        for i in range(6):
            _ins(db, "s2", f"2026-06-02T01:0{i}:00Z", "q2", "a bad answer", "no that's wrong, fix it", BAD)

        stats = run_geometric_extract(conn=db)

        accepts = db.execute("SELECT COUNT(*) FROM interactions WHERE feedback_class IN "
                             "('accept','strong_accept')").fetchone()[0]
        rejects = db.execute("SELECT COUNT(*) FROM interactions WHERE feedback_class IN "
                             "('reject','strong_reject')").fetchone()[0]
        assert accepts == 6 and rejects == 6
        # every interaction got a quality_score + domain + processed
        assert db.execute("SELECT COUNT(*) FROM interactions WHERE quality_score IS NULL").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM interactions WHERE domain IS NULL").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM interactions WHERE processed_2a_at IS NULL").fetchone()[0] == 0
        # pairs only from the 6 non-rejected (rejects excluded)
        assert db.execute("SELECT COUNT(*) FROM training_pairs").fetchone()[0] == 6
        assert stats["pairs_written"] == 6
        assert stats["probe_trained"] is True

    def test_incremental_no_duplicate_pairs(self, db):
        from bespoke.extract.run import run_geometric_extract
        for i in range(5):
            _ins(db, "s1", f"2026-06-02T00:0{i}:00Z", "q", "a", "ok thanks", GOOD)
        run_geometric_extract(conn=db)
        p1 = db.execute("SELECT COUNT(*) FROM training_pairs").fetchone()[0]
        run_geometric_extract(conn=db)  # nothing unprocessed now
        p2 = db.execute("SELECT COUNT(*) FROM training_pairs").fetchone()[0]
        assert p1 == p2 and p1 == 5
