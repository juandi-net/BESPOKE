# tests/test_extract_mechanical.py
"""Track 2: mechanical pair extraction + tangled routing."""
import numpy as np


class TestExtractPairs:
    def test_copies_and_filters_rejects(self):
        from bespoke.extract.mechanical import extract_sft_pairs
        turns = [
            {"user_message": "q1", "assistant_response": "a1", "feedback_class": "accept", "emb": None},
            {"user_message": "q2", "assistant_response": "a2", "feedback_class": "strong_reject", "emb": None},
            {"user_message": "  ", "assistant_response": "a3", "feedback_class": "neutral", "emb": None},
        ]
        pairs = extract_sft_pairs(turns)
        assert len(pairs) == 1
        assert pairs[0]["instruction"] == "q1" and pairs[0]["response"] == "a1"
        assert pairs[0]["quality"] == 0.5

    def test_quality_from_probe(self):
        from bespoke.extract.mechanical import extract_sft_pairs
        from bespoke.eval.probe import LinearPreferenceProbe
        X = np.vstack([np.ones((6, 768), np.float32), -np.ones((6, 768), np.float32)])
        y = np.array([1] * 6 + [0] * 6)
        probe = LinearPreferenceProbe().fit(X, y)
        turns = [{"user_message": "q", "assistant_response": "a",
                  "feedback_class": "neutral", "emb": np.ones(768, np.float32)}]
        pairs = extract_sft_pairs(turns, probe=probe)
        assert pairs[0]["quality"] > 0.5


class TestTangled:
    def test_long_unresolved_is_tangled(self):
        from bespoke.extract.mechanical import is_tangled
        turns = [{"feedback_class": "neutral", "ts": None, "emb": None} for _ in range(9)]
        assert is_tangled(turns) is True

    def test_short_resolved_not_tangled(self):
        from bespoke.extract.mechanical import is_tangled
        turns = [{"feedback_class": "neutral", "ts": None, "emb": None},
                 {"feedback_class": "accept", "ts": None, "emb": None}]
        assert is_tangled(turns) is False

    def test_high_topic_churn_is_tangled(self):
        from bespoke.extract.mechanical import is_tangled
        a = np.array([1.0, 0.0] + [0.0] * 766, np.float32)
        b = np.array([0.0, 1.0] + [0.0] * 766, np.float32)
        turns = [{"feedback_class": "accept", "ts": None, "emb": e} for e in (a, b, a, b)]
        assert is_tangled(turns) is True  # orthogonal hops => high churn
