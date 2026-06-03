# tests/test_reembed.py
"""Re-embed the corpus with a (mock) embedding service — rebuilds vec_interactions."""
import numpy as np
from sqlite_vec import serialize_float32


class TestReembedAll:
    def test_rebuilds_vectors(self, db):
        from bespoke.capture.pipeline import reembed_all
        for i in range(3):
            cur = db.execute("INSERT INTO interactions (provider,model,source,user_message,"
                             "assistant_response) VALUES ('c','m','s',?,?)", (f"q{i}", f"a{i}"))
            db.execute("INSERT INTO vec_interactions (rowid,interaction_embedding) VALUES (?,?)",
                       (cur.lastrowid, serialize_float32([0.0] * 768)))  # stale ONNX-era vectors

        fake = type("S", (), {
            "embed_many": lambda self, texts, prefix="document": [np.full(768, 0.5, np.float32) for _ in texts]
        })()
        stats = reembed_all(conn=db, embedding_svc=fake)

        assert stats["reembedded"] == 3
        assert db.execute("SELECT COUNT(*) FROM vec_interactions").fetchone()[0] == 3
        blob = db.execute("SELECT interaction_embedding FROM vec_interactions LIMIT 1").fetchone()[0]
        v = np.frombuffer(blob, dtype=np.float32)
        assert abs(float(v[0]) - 0.5) < 1e-6  # replaced with the new embedding
