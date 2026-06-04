"""Does a topic-invariant "approach" direction exist in juandi's latent space?

Falsifiable foundation for the meta-approach design (docs/bespoke-meta-approach-design.md).
Personality-as-geometry test: factor TOPIC out via within-topic accept-reject contrast (the topic
centroid cancels in accept_mean - reject_mean), then ask whether the CONSENSUS direction across
topics predicts accept/reject on a HELD-OUT topic.

    meta generalizes to unseen topics  <=>  a measurable, topic-independent "you" exists.

If leave-one-topic-out AUC >> 0.5 (and near the within-topic ceiling), the approach is largely
topic-invariant -> the personality-as-geometry premise holds. If it sits at chance, the single
direction premise is wrong (approach is topic-specific or needs a subspace, not a line) -- a real
finding, reported straight.

Run:  python -m bespoke.latent.approach
"""
import numpy as np


def _l2norm(M, axis=-1, eps=1e-8):
    return M / (np.linalg.norm(M, axis=axis, keepdims=True) + eps)


def leiden_clusters(X, n_neighbors=15, resolution=1.0, seed=0):
    """Leiden community labels over a kNN graph of X (topic territories). Returns int labels (n,)."""
    import igraph as ig
    import leidenalg
    from sklearn.neighbors import kneighbors_graph

    n = len(X)
    k = max(1, min(n_neighbors, n - 1))
    A = kneighbors_graph(X, k, mode="connectivity", include_self=False)
    sources, targets = A.nonzero()
    edges = list(zip(sources.tolist(), targets.tolist()))
    g = ig.Graph(n=n, edges=edges, directed=False)
    g.simplify()
    part = leidenalg.find_partition(
        g, leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution, seed=seed)
    labels = np.empty(n, dtype=int)
    for ci, comm in enumerate(part):
        for v in comm:
            labels[v] = ci
    return labels


def _direction(Xc, yc):
    """Within-topic accept-reject direction (unit): mean(accept) - mean(reject)."""
    d = Xc[yc == 1].mean(axis=0) - Xc[yc == 0].mean(axis=0)
    return d / (np.linalg.norm(d) + 1e-8)


def run(X, y, n_neighbors=15, resolution=1.0, min_accept=8, min_reject=8, seed=0, n_random=20):
    """Return a report dict. X: (n,768) float embeddings; y in {1 accept, 0 reject, -1 unlabeled}."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)

    # Labeled subset only; L2-normalize rows (cosine geometry, like the rest of the eval stack).
    mask = y != -1
    Xl = _l2norm(X[mask].astype(np.float64))
    yl = y[mask].astype(int)

    labels = leiden_clusters(Xl, n_neighbors=n_neighbors, resolution=resolution, seed=seed)

    # Qualifying clusters: enough of BOTH classes to define a stable direction.
    qual = {}
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        na = int((yl[idx] == 1).sum())
        nr = int((yl[idx] == 0).sum())
        if na >= min_accept and nr >= min_reject:
            qual[int(c)] = idx

    n_clusters = int(len(np.unique(labels)))
    report = {
        "n_labeled": int(mask.sum()),
        "n_clusters_total": n_clusters,
        "n_clusters_qualifying": len(qual),
        "accepts_covered": int(sum((yl[idx] == 1).sum() for idx in qual.values())),
        "rejects_covered": int(sum((yl[idx] == 0).sum() for idx in qual.values())),
        "accepts_total": int((yl == 1).sum()),
    }
    if len(qual) < 3:
        report["error"] = (f"Only {len(qual)} qualifying clusters (need >=3 for leave-one-out). "
                           f"Lower --resolution for coarser topics or lower min counts.")
        return report

    # Per-topic directions.
    dirs = {c: _direction(Xl[idx], yl[idx]) for c, idx in qual.items()}
    D = np.vstack([dirs[c] for c in qual])              # (k, 768) unit rows

    # How aligned are the per-topic directions? (one personality vs many)
    G = D @ D.T
    iu = np.triu_indices(len(D), k=1)
    report["mean_pairwise_cosine"] = float(G[iu].mean())
    # top principal direction of the stacked directions -> fraction of their variance on one axis
    svals = np.linalg.svd(D - D.mean(0, keepdims=True), compute_uv=False)
    report["top_pc_var_frac"] = float((svals[0] ** 2) / (svals ** 2).sum())

    # --- Headline: leave-one-TOPIC-out cross-topic generalization ---
    loto = []
    for h, idx in qual.items():
        others = [dirs[c] for c in qual if c != h]
        meta = np.mean(others, axis=0)
        meta /= (np.linalg.norm(meta) + 1e-8)
        scores = Xl[idx] @ meta
        loto.append(roc_auc_score(yl[idx], scores))
    report["loto_meta_auc_mean"] = float(np.mean(loto))
    report["loto_meta_auc_min"] = float(np.min(loto))
    report["loto_meta_auc_max"] = float(np.max(loto))

    # --- Ceiling: within-topic split-half (topic-specific approach, no transfer required) ---
    within = []
    for idx in qual.values():
        a = idx[yl[idx] == 1]; r = idx[yl[idx] == 0]
        rng.shuffle(a); rng.shuffle(r)
        ah, rh = a[: len(a) // 2], r[: len(r) // 2]      # train half
        at, rt = a[len(a) // 2:], r[len(r) // 2:]        # test half
        if len(ah) < 2 or len(rh) < 2 or len(at) < 1 or len(rt) < 1:
            continue
        d = Xl[ah].mean(0) - Xl[rh].mean(0); d /= (np.linalg.norm(d) + 1e-8)
        ti = np.concatenate([at, rt]); ty = np.concatenate([np.ones(len(at)), np.zeros(len(rt))])
        within.append(roc_auc_score(ty, Xl[ti] @ d))
    report["within_topic_auc_mean"] = float(np.mean(within)) if within else None

    # --- Baseline: random directions, same per-cluster scoring ---
    rand = []
    for _ in range(n_random):
        v = rng.standard_normal(Xl.shape[1]); v /= np.linalg.norm(v)
        rand.append(np.mean([roc_auc_score(yl[idx], Xl[idx] @ v) for idx in qual.values()]))
    report["random_dir_auc_mean"] = float(np.mean(rand))

    return report


def main():
    from bespoke.db.init import get_connection
    from bespoke.eval.signals import get_labeled_embeddings

    import argparse
    ap = argparse.ArgumentParser(description="Topic-invariant approach-direction test")
    ap.add_argument("--resolution", type=float, default=1.0, help="Leiden resolution (lower=coarser topics)")
    ap.add_argument("--n-neighbors", type=int, default=15)
    ap.add_argument("--min-accept", type=int, default=8)
    ap.add_argument("--min-reject", type=int, default=8)
    args = ap.parse_args()

    conn = get_connection()
    ids, X, y = get_labeled_embeddings(conn)
    conn.close()

    print(f"Loaded {len(ids)} embeddings ({(y==1).sum()} accept / {(y==0).sum()} reject / "
          f"{(y==-1).sum()} neutral), dim={X.shape[1]}")
    print(f"Leiden resolution={args.resolution}, min accept/reject per topic={args.min_accept}/{args.min_reject}\n")

    rep = run(X, y, n_neighbors=args.n_neighbors, resolution=args.resolution,
              min_accept=args.min_accept, min_reject=args.min_reject)

    print(f"topics: {rep['n_clusters_qualifying']}/{rep['n_clusters_total']} qualifying "
          f"(cover {rep['accepts_covered']}/{rep['accepts_total']} accepts, {rep['rejects_covered']} rejects)")
    if "error" in rep:
        print("\n  " + rep["error"])
        return
    print(f"per-topic directions: mean pairwise cosine={rep['mean_pairwise_cosine']:+.3f}, "
          f"top-PC var frac={rep['top_pc_var_frac']:.2f}  (1.0 => one shared axis)\n")
    print("AUC (accept>reject by projection):")
    print(f"  random direction      : {rep['random_dir_auc_mean']:.3f}   (chance ~0.50)")
    if rep.get("within_topic_auc_mean") is not None:
        print(f"  within-topic (ceiling): {rep['within_topic_auc_mean']:.3f}   (topic-specific approach)")
    print(f"  META, cross-topic LOTO: {rep['loto_meta_auc_mean']:.3f}   "
          f"(range {rep['loto_meta_auc_min']:.3f}-{rep['loto_meta_auc_max']:.3f})  <-- headline")
    print()
    auc = rep["loto_meta_auc_mean"]
    verdict = ("STRONG: a topic-invariant approach direction exists and generalizes." if auc >= 0.65
               else "WEAK but present: some shared approach signal." if auc >= 0.57
               else "NULL: no single topic-invariant approach direction (topic-specific or needs a subspace).")
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
