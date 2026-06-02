# tests/test_extract_conversations.py
"""Track 2: session representation."""
import numpy as np
from sqlite_vec import serialize_float32


class TestSessionEmbedding:
    def test_mean_pool_normalized(self):
        from bespoke.extract.conversations import session_embedding
        turns = [{"emb": np.array([3.0] + [0.0] * 767, np.float32)},
                 {"emb": np.array([0.0, 4.0] + [0.0] * 766, np.float32)}]
        v = session_embedding(turns)
        assert v.shape == (768,)
        assert abs(np.linalg.norm(v) - 1.0) < 1e-5

    def test_empty_returns_zeros(self):
        from bespoke.extract.conversations import session_embedding
        v = session_embedding([{"emb": None}])
        assert v.shape == (768,) and not v.any()


class TestTrajectory:
    def test_features(self):
        from bespoke.extract.conversations import trajectory_features
        turns = [
            {"ts": "2026-06-02T00:00:00", "feedback_class": "neutral", "cache_read": 10},
            {"ts": "2026-06-02T00:01:00", "feedback_class": "accept", "cache_read": 0},
        ]
        f = trajectory_features(turns)
        assert f["n_turns"] == 2
        assert f["ended_in_accept"] is True
        assert abs(f["mean_gap_s"] - 60.0) < 1e-6
        assert abs(f["cache_read_ratio"] - 0.5) < 1e-6


class TestAssemble:
    def test_groups_by_session(self, db):
        from bespoke.extract.conversations import assemble_sessions

        def ins(sid, ts):
            cur = db.execute(
                "INSERT INTO interactions (provider,model,source,session_id,user_message,"
                "assistant_response,captured_at) VALUES ('c','m','s',?,?,?,?)",
                (sid, "u", "a", ts))
            rid = cur.lastrowid
            db.execute("INSERT INTO vec_interactions (rowid, interaction_embedding) VALUES (?,?)",
                       (rid, serialize_float32([0.1] * 768)))
            return rid

        ins("s1", "2026-06-02T00:00:00Z")
        ins("s1", "2026-06-02T00:00:30Z")
        ins("s2", "2026-06-02T00:01:00Z")

        by = {s["session_id"]: s for s in assemble_sessions(db)}
        assert len(by["s1"]["turns"]) == 2
        assert len(by["s2"]["turns"]) == 1
        assert by["s1"]["turns"][0]["emb"].shape == (768,)
