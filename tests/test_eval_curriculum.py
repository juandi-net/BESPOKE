# tests/test_eval_curriculum.py
"""Leiden-based curriculum weighting writes weights back to interactions."""
import numpy as np
from sqlite_vec import serialize_float32
from unittest.mock import patch, MagicMock


def _seed(db, n, base):
    ids = []
    for i in range(n):
        cur = db.execute("INSERT INTO interactions (provider, model, source, user_message, "
                         "assistant_response) VALUES ('c','m','s','u','a')")
        rid = cur.lastrowid
        vec = (np.full(768, base, np.float32) + np.random.RandomState(i).normal(0, 0.05, 768)).astype(np.float32)
        db.execute("INSERT INTO vec_interactions (rowid, interaction_embedding) VALUES (?, ?)",
                   (rid, serialize_float32(vec.tolist())))
        ids.append(rid)
    return ids


class TestApplyCurriculumWeights:
    def test_writes_weights(self, db):
        from bespoke.eval.curriculum import apply_curriculum_weights
        _seed(db, 20, 0.0)
        _seed(db, 4, 20.0)  # rare cluster

        proxy = MagicMock(wraps=db); proxy.close = MagicMock()
        with patch("bespoke.eval.curriculum.get_connection", return_value=proxy):
            updated = apply_curriculum_weights(n_neighbors=5)

        assert updated == 24
        weights = [r["curriculum_weight"] for r in db.execute(
            "SELECT curriculum_weight FROM interactions").fetchall()]
        assert max(weights) > min(weights)  # rarity differentiated the weights
