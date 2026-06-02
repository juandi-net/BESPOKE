"""Graph label propagation (sklearn LabelSpreading) + leave-one-out stability meter.

LabelSpreading uses soft clamping (alpha) and a normalized Laplacian — more robust to
NOISY labels than hard-clamping LabelPropagation. That robustness is deliberate: with no
LLM judge, accept/reject labels are the ground truth and they get noisy (tired sessions).
"""
import numpy as np
from sklearn.semi_supervised import LabelSpreading


def _safe_k(n_labeled_or_samples, n_neighbors):
    return max(1, min(n_neighbors, n_labeled_or_samples - 1))


def propagate_scores(X, y, n_neighbors=7, alpha=0.2):
    """Propagate accept(1)/reject(0) seed labels across a kNN graph of X.

    Returns (n,) array of P(accept) in [0, 1]. Rows with y == -1 are unlabeled targets.
    """
    n = X.shape[0]
    k = _safe_k(n, n_neighbors)
    model = LabelSpreading(kernel="knn", n_neighbors=k, alpha=alpha)
    model.fit(X, y)
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(n, dtype=float)
    return model.predict_proba(X)[:, classes.index(1)]


def leave_one_out_accuracy(X, y, n_neighbors=7, alpha=0.2):
    """STABILITY METER: hold out each labeled point, predict it from the rest via
    propagation, report accuracy. High = embedding proximity tracks accept/reject
    (the silver thread is real). Low = topic-vs-quality confound dominates.

    This is an INSTRUMENT to watch, not a go/no-go gate.
    """
    labeled = y != -1
    Xl, yl = X[labeled], y[labeled]
    n = len(yl)
    if n < 2 or len(set(yl.tolist())) < 2:
        return 0.0
    correct = 0
    for i in range(n):
        yy = yl.copy()
        yy[i] = -1  # hide this label
        k = _safe_k(n - 1, n_neighbors)
        m = LabelSpreading(kernel="knn", n_neighbors=k, alpha=alpha).fit(Xl, yy)
        correct += int(m.transduction_[i] == yl[i])
    return correct / n
