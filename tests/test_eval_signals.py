# tests/test_eval_signals.py
"""Tests for eval signal loading."""
import numpy as np
from sqlite_vec import serialize_float32


def _insert(db, fid, feedback_class, vec):
    cur = db.execute(
        "INSERT INTO interactions (provider, model, source, user_message, assistant_response, feedback_class) "
        "VALUES ('claude','m','s','u','a',?)", (feedback_class,))
    rid = cur.lastrowid
    db.execute("INSERT INTO vec_interactions (rowid, interaction_embedding) VALUES (?, ?)",
               (rid, serialize_float32(vec)))
    return rid


class TestGetLabeledEmbeddings:
    def test_maps_feedback_to_labels(self, db):
        from bespoke.eval.signals import get_labeled_embeddings
        _insert(db, 1, "accept", [0.1] * 768)
        _insert(db, 2, "strong_reject", [0.2] * 768)
        _insert(db, 3, "neutral", [0.3] * 768)
        _insert(db, 4, None, [0.4] * 768)

        ids, X, y = get_labeled_embeddings(db)

        assert X.shape == (4, 768)
        label_by_id = dict(zip(ids.tolist(), y.tolist()))
        assert label_by_id[1] == 1     # accept -> 1
        assert label_by_id[2] == 0     # strong_reject -> 0
        assert label_by_id[3] == -1    # neutral -> unlabeled
        assert label_by_id[4] == -1    # None -> unlabeled


class TestFeedbackConfidence:
    def test_confidence_column_loads(self, db):
        from bespoke.eval.signals import get_labeled_embeddings
        cur = db.execute(
            "INSERT INTO interactions (provider, model, source, user_message, assistant_response, "
            "feedback_class, feedback_confidence) VALUES ('c','m','s','u','a','accept',0.3)")
        rid = cur.lastrowid
        db.execute("INSERT INTO vec_interactions (rowid, interaction_embedding) VALUES (?, ?)",
                   (rid, serialize_float32([0.1] * 768)))
        ids, X, y, conf = get_labeled_embeddings(db, with_confidence=True)
        assert conf[list(ids).index(rid)] == 0.3
