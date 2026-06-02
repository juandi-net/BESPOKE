# tests/test_eval_propagation.py
"""Tests for label propagation scoring and the stability meter."""
import numpy as np


def _two_clusters():
    # Cluster A near origin (accept=1), cluster B near (10,..) (reject=0)
    rng = np.random.RandomState(0)
    a = rng.normal(0.0, 0.1, size=(10, 8)).astype(np.float32)
    b = rng.normal(10.0, 0.1, size=(10, 8)).astype(np.float32)
    X = np.vstack([a, b])
    y = np.array([1] * 10 + [0] * 10)
    return X, y


class TestPropagation:
    def test_scores_track_seed_labels(self):
        from bespoke.eval.propagation import propagate_scores
        X, y_full = _two_clusters()
        # Seed only 2 labels per cluster, rest unlabeled
        y = np.full(20, -1)
        y[0] = 1; y[1] = 1; y[10] = 0; y[11] = 0
        scores = propagate_scores(X, y)
        assert scores.shape == (20,)
        # Cluster A should score high (accept), cluster B low
        assert scores[:10].mean() > 0.7
        assert scores[10:].mean() < 0.3

    def test_leave_one_out_separable_is_high(self):
        from bespoke.eval.propagation import leave_one_out_accuracy
        X, y = _two_clusters()
        acc = leave_one_out_accuracy(X, y)
        assert acc > 0.9  # cleanly separable -> geometry tracks the label
