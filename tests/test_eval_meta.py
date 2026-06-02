# tests/test_eval_meta.py
"""Tests for the meta-scorer fusing per-response signals."""


class TestMetaScorer:
    def test_gate_fail_is_zero(self):
        from bespoke.eval.meta import MetaScorer
        ms = MetaScorer()  # untrained -> rule fallback
        q = ms.quality({"gate_passed": 0, "propagation": 0.9, "probe": 0.9})
        assert q == 0.0

    def test_rule_fallback_is_mean_when_gate_ok(self):
        from bespoke.eval.meta import MetaScorer
        ms = MetaScorer()
        q = ms.quality({"gate_passed": 1, "propagation": 0.8, "probe": 0.6})
        assert abs(q - 0.7) < 1e-6

    def test_trained_model_used_when_present(self):
        from bespoke.eval.meta import MetaScorer
        import numpy as np
        # Features: [gate, propagation, probe]; label = high when probe high
        F = np.array([[1, 0.9, 0.9], [1, 0.1, 0.1], [1, 0.8, 0.85], [1, 0.2, 0.15]])
        y = np.array([1, 0, 1, 0])
        ms = MetaScorer().fit(F, y)
        good = ms.quality({"gate_passed": 1, "propagation": 0.85, "probe": 0.9})
        bad = ms.quality({"gate_passed": 1, "propagation": 0.15, "probe": 0.1})
        assert good > bad
