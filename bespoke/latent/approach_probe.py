"""GOAL RUN: can a trained probe / better representation beat the ~0.62 mean-diff baseline?

On junk-cleaned data, compare cross-topic leave-one-TOPIC-out AUC across estimators/representations to
isolate WHY the topic-invariant approach signal is weak:
  H1 estimator   — logistic probe vs crude mean(accept)-mean(reject)
  H2 representation — standardize / topic-residualize before the probe
  H4 data        — learning curve (AUC vs # training points)
  H5 generic     — length-only control (is "accept" just answer length?)

Run:  python -m bespoke.latent.approach_probe [--resolution R]
"""
import numpy as np
from bespoke.latent.approach import leiden_clusters, _l2norm


def _loto(X, y, qual, fit_predict, seed=0):
    """Leave-one-topic-out AUC. fit_predict(Xtr,ytr,Xte)->scores; averaged over held-out topics."""
    from sklearn.metrics import roc_auc_score
    aucs = []
    for h, idx in qual.items():
        tr = np.concatenate([qual[c] for c in qual if c != h])
        if len(set(y[tr].tolist())) < 2:
            continue
        scores = fit_predict(X[tr], y[tr], X[idx])
        aucs.append(roc_auc_score(y[idx], scores))
    return float(np.mean(aucs)) if aucs else float("nan")


def _mean_diff(Xtr, ytr, Xte):
    d = Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0)
    return Xte @ (d / (np.linalg.norm(d) + 1e-8))


def _logistic(Xtr, ytr, Xte):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(Xtr, ytr)
    return clf.decision_function(Xte)


def _standardized_logistic(Xtr, ytr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    return _logistic((Xtr - mu) / sd, ytr, (Xte - mu) / sd)


def _rbf_svm(Xtr, ytr, Xte):
    from sklearn.svm import SVC
    return SVC(kernel="rbf", class_weight="balanced", gamma="scale").fit(Xtr, ytr).decision_function(Xte)


def _mlp(Xtr, ytr, Xte):
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(hidden_layer_sizes=(128,), max_iter=400, early_stopping=True,
                        random_state=0).fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


def main():
    import argparse
    from bespoke.db.init import get_connection
    from bespoke.eval.signals import get_labeled_embeddings
    from sklearn.metrics import roc_auc_score

    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=float, default=1.0)
    ap.add_argument("--n-neighbors", type=int, default=15)
    ap.add_argument("--min-accept", type=int, default=8)
    ap.add_argument("--min-reject", type=int, default=8)
    args = ap.parse_args()

    conn = get_connection()
    ids, X, y = get_labeled_embeddings(conn)
    meta = {r["id"]: (r["content_type"], len(r["assistant_response"] or ""))
            for r in conn.execute("SELECT id, content_type, assistant_response FROM interactions").fetchall()}
    conn.close()

    # clean only (drop observer / tool_result_only)
    keep = np.array([meta.get(int(i), ("clean", 0))[0] in ("agentic", "clean") for i in ids])
    ids, X, y = ids[keep], X[keep], y[keep]
    lengths = np.array([meta.get(int(i), ("clean", 0))[1] for i in ids], dtype=float)
    mask = y != -1
    Xl = _l2norm(X[mask].astype(np.float64)); yl = y[mask].astype(int); Ll = lengths[mask]
    print(f"clean labeled: {len(yl)}  ({int((yl==1).sum())} accept / {int((yl==0).sum())} reject)")

    labels = leiden_clusters(Xl, n_neighbors=args.n_neighbors, resolution=args.resolution)
    qual = {}
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if (yl[idx] == 1).sum() >= args.min_accept and (yl[idx] == 0).sum() >= args.min_reject:
            qual[int(c)] = idx
    print(f"topics: {len(qual)} qualifying\n")

    # topic-residualized representation (subtract per-cluster centroid — label-agnostic, no leakage)
    Xres = Xl.copy()
    for idx in qual.values():
        Xres[idx] -= Xl[idx].mean(0)

    # length-only control (within held-out topic, score = response length)
    len_aucs = []
    for idx in qual.values():
        len_aucs.append(roc_auc_score(yl[idx], Ll[idx]))
    len_auc = float(np.mean(len_aucs))

    rng = np.random.default_rng(0)
    rand = np.mean([roc_auc_score(yl[idx], Xl[idx] @ (lambda v: v/np.linalg.norm(v))(rng.standard_normal(Xl.shape[1])))
                    for idx in qual.values() for _ in range(1)])

    print("cross-topic leave-one-TOPIC-out AUC (clean data):")
    print(f"  random direction         : {rand:.3f}")
    print(f"  length-only (control)     : {len_auc:.3f}   <- is 'accept' just answer length?")
    print(f"  mean-diff (RT-002 baseline): {_loto(Xl, yl, qual, _mean_diff):.3f}")
    print(f"  logistic probe (H1)       : {_loto(Xl, yl, qual, _logistic):.3f}")
    print(f"  standardized + logistic   : {_loto(Xl, yl, qual, _standardized_logistic):.3f}")
    print(f"  topic-residualized + logit (H2): {_loto(Xres, yl, qual, _logistic):.3f}")
    print(f"  RBF SVM (nonlinear)       : {_loto(Xl, yl, qual, _rbf_svm):.3f}")
    print(f"  MLP 128 (nonlinear)       : {_loto(Xl, yl, qual, _mlp):.3f}")

    # H4 learning curve — subsample training points (best estimator = logistic)
    print("\nlearning curve (logistic, LOTO, fraction of train points):")
    for frac in (0.25, 0.5, 0.75, 1.0):
        def fp(Xtr, ytr, Xte, frac=frac):
            k = max(4, int(len(Xtr) * frac))
            sel = rng.permutation(len(Xtr))[:k]
            return _logistic(Xtr[sel], ytr[sel], Xte)
        print(f"  {int(frac*100):3}% : {_loto(Xl, yl, qual, fp):.3f}")


if __name__ == "__main__":
    main()
