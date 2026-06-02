"""Load labeled embeddings and accept/reject labels from the warehouse."""
import numpy as np

# feedback_class -> binary preference label. -1 = unlabeled (propagation target).
FEEDBACK_TO_LABEL = {
    "strong_accept": 1, "accept": 1,
    "reject": 0, "strong_reject": 0,
    "neutral": -1,
}


def get_labeled_embeddings(conn, with_confidence=False):
    """Return (ids, X, y[, conf]) over all interactions that have an embedding.

    X: (n, 768) float32 embeddings. y: (n,) in {1 accept, 0 reject, -1 unlabeled}.
    When with_confidence=True, also returns conf: (n,) feedback_confidence in [0, 1].
    """
    rows = conn.execute("""
        SELECT i.id AS id, i.feedback_class AS fc,
               COALESCE(i.feedback_confidence, 1.0) AS conf,
               v.interaction_embedding AS emb
        FROM interactions i
        JOIN vec_interactions v ON v.rowid = i.id
        ORDER BY i.id ASC
    """).fetchall()

    ids, X, y, conf = [], [], [], []
    for r in rows:
        ids.append(r["id"])
        X.append(np.frombuffer(r["emb"], dtype=np.float32))
        y.append(FEEDBACK_TO_LABEL.get(r["fc"], -1))
        conf.append(float(r["conf"]))

    if not ids:
        empty = (np.array([], dtype=int),
                 np.zeros((0, 768), dtype=np.float32),
                 np.array([], dtype=int))
        return empty + (np.array([], dtype=float),) if with_confidence else empty

    out = (np.array(ids, dtype=int), np.vstack(X).astype(np.float32), np.array(y, dtype=int))
    return out + (np.array(conf, dtype=float),) if with_confidence else out
