# tests/test_eval_anchor.py
"""Tests for benchmark-interview anchor seeds."""
import numpy as np


class TestAnchor:
    def test_parses_good_bad_examples(self):
        from bespoke.eval.anchor import extract_anchor_examples
        benchmark = {"benchmark": {"anchor_examples": {
            "good": ["a crisp decomposed answer", "clear tradeoffs"],
            "bad": ["a rambling vague answer"],
        }}}
        texts, labels = extract_anchor_examples(benchmark)
        assert texts == ["a crisp decomposed answer", "clear tradeoffs", "a rambling vague answer"]
        assert labels == [1, 1, 0]

    def test_empty_when_no_anchor_section(self):
        from bespoke.eval.anchor import extract_anchor_examples
        texts, labels = extract_anchor_examples({"benchmark": {}})
        assert texts == [] and labels == []

    def test_embed_anchor_uses_service(self):
        from bespoke.eval import anchor
        fake = type("S", (), {"embed_batch": lambda self, t, prefix="document": [np.ones(768, np.float32) for _ in t]})()
        X, y = anchor.embed_anchor(["g", "b"], [1, 0], embedding_svc=fake)
        assert X.shape == (2, 768)
        assert y.tolist() == [1, 0]
