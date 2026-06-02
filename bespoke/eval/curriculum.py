"""Deterministic curriculum weighting (replaces Stage 2b's LLM weighting).

Reads all embeddings, runs Leiden, writes rarity-based weights back to
interactions.curriculum_weight. No LLM call.
"""
import numpy as np

from bespoke.db.init import get_connection
from bespoke.eval.cluster import leiden_curriculum_weights


def apply_curriculum_weights(n_neighbors=10, min_weight=1.0, max_weight=2.0):
    """Compute Leiden rarity weights over all embedded interactions and persist them.

    Returns the number of rows updated.
    """
    conn = get_connection()
    rows = conn.execute("""
        SELECT i.id AS id, v.interaction_embedding AS emb
        FROM interactions i JOIN vec_interactions v ON v.rowid = i.id
        ORDER BY i.id ASC
    """).fetchall()

    if not rows:
        conn.close()
        return 0

    ids = [r["id"] for r in rows]
    X = np.vstack([np.frombuffer(r["emb"], dtype=np.float32) for r in rows]).astype(np.float32)

    weights = leiden_curriculum_weights(X, ids, n_neighbors=n_neighbors,
                                        min_weight=min_weight, max_weight=max_weight)

    for iid, w in weights.items():
        conn.execute("UPDATE interactions SET curriculum_weight = ? WHERE id = ?", (w, iid))
    conn.commit()
    conn.close()
    return len(weights)
