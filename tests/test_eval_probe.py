# tests/test_eval_probe.py
"""Tests for the linear preference probe."""
import numpy as np


class TestLinearPreferenceProbe:
    def test_fits_and_scores_separable(self):
        from bespoke.eval.probe import LinearPreferenceProbe
        rng = np.random.RandomState(1)
        good = rng.normal(1.0, 0.1, size=(20, 8)).astype(np.float32)
        bad = rng.normal(-1.0, 0.1, size=(20, 8)).astype(np.float32)
        X = np.vstack([good, bad])
        y = np.array([1] * 20 + [0] * 20)

        probe = LinearPreferenceProbe().fit(X, y)
        s_good = probe.score(good)
        s_bad = probe.score(bad)
        assert s_good.mean() > 0.8
        assert s_bad.mean() < 0.2

    def test_score_shape(self):
        from bespoke.eval.probe import LinearPreferenceProbe
        X = np.random.RandomState(2).normal(size=(10, 8)).astype(np.float32)
        y = np.array([1, 0] * 5)
        probe = LinearPreferenceProbe().fit(X, y)
        assert probe.score(X).shape == (10,)
