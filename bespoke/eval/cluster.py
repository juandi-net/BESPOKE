"""Leiden community detection -> curriculum weights (rare cluster = upweight).

Deterministic replacement for Stage 2b's LLM curriculum weighting. We do NOT name the
clusters (that needed language); we only use cluster RARITY to weight training examples.
Leiden (not Louvain): Leiden guarantees connected communities; Louvain can produce broken ones.
"""
import numpy as np
from sklearn.neighbors import kneighbors_graph


def leiden_curriculum_weights(X, ids, n_neighbors=10, min_weight=1.0, max_weight=2.0, seed=0):
    """Return {interaction_id: weight}. Smaller (rarer) community -> weight nearer max."""
    import igraph as ig
    import leidenalg

    n = len(ids)
    if n == 0:
        return {}
    if n == 1:
        return {int(ids[0]): float(min_weight)}

    k = max(1, min(n_neighbors, n - 1))
    A = kneighbors_graph(X, k, mode="connectivity", include_self=False)
    sources, targets = A.nonzero()
    edges = list(zip(sources.tolist(), targets.tolist()))

    g = ig.Graph(n=n, edges=edges, directed=False)
    g.simplify()
    part = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, seed=seed)

    size_of = {}
    for comm in part:
        for v in comm:
            size_of[v] = len(comm)

    weights = {}
    for vi, iid in enumerate(ids):
        frac = size_of.get(vi, 1) / n           # rare -> small frac
        w = max_weight - (max_weight - min_weight) * frac
        weights[int(iid)] = float(round(w, 3))
    return weights
