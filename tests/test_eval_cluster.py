# tests/test_eval_cluster.py
"""Tests for Leiden curriculum weighting."""
import numpy as np


class TestLeidenWeights:
    def test_rare_cluster_weighted_higher(self):
        from bespoke.eval.cluster import leiden_curriculum_weights
        rng = np.random.RandomState(0)
        big = rng.normal(0.0, 0.1, size=(30, 8)).astype(np.float32)   # common
        small = rng.normal(20.0, 0.1, size=(4, 8)).astype(np.float32)  # rare
        X = np.vstack([big, small])
        ids = list(range(34))
        weights = leiden_curriculum_weights(X, ids, n_neighbors=5)
        rare_mean = np.mean([weights[i] for i in range(30, 34)])
        common_mean = np.mean([weights[i] for i in range(30)])
        assert rare_mean > common_mean

    def test_weights_within_bounds(self):
        from bespoke.eval.cluster import leiden_curriculum_weights
        X = np.random.RandomState(1).normal(size=(20, 8)).astype(np.float32)
        weights = leiden_curriculum_weights(X, list(range(20)), n_neighbors=5,
                                            min_weight=1.0, max_weight=2.0)
        assert all(1.0 <= w <= 2.0 for w in weights.values())
