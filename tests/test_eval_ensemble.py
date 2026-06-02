# tests/test_eval_ensemble.py
"""Tests for the geometric ensemble orchestrator and comparison."""
import numpy as np


class TestScoreItems:
    def test_scorecard_aggregates_quality(self):
        from bespoke.eval.ensemble import score_items
        from bespoke.eval.probe import LinearPreferenceProbe

        # Train a probe where 'high' direction = +1 vector
        Xtrain = np.vstack([np.ones((10, 768), np.float32), -np.ones((10, 768), np.float32)])
        ytrain = np.array([1] * 10 + [0] * 10)
        probe = LinearPreferenceProbe().fit(Xtrain, ytrain)

        items = [
            {"prompt": "p1", "output": "good answer", "domain": "strategy",
             "embedding": np.ones(768, np.float32)},
            {"prompt": "p2", "output": "", "domain": "strategy",        # gate fails (empty)
             "embedding": np.ones(768, np.float32)},
        ]
        # propagation scores passed in directly (computed upstream); probe from model
        card = score_items(items, probe=probe, propagation_scores=[0.9, 0.9])

        assert card["eval_set_size"] == 2
        assert 0.0 <= card["reward"] <= 1.0
        # item 2 gate-fails -> 0, item 1 high -> reward is the average
        assert card["per_item"][1]["quality"] == 0.0
        assert card["per_item"][0]["quality"] > 0.5


class TestCompareGeometric:
    def test_keep_when_reward_improves(self):
        from bespoke.eval.ensemble import compare_geometric
        d = compare_geometric({"reward": 0.8}, {"reward": 0.7})
        assert d["decision"] == "keep"

    def test_revert_on_meaningful_drop(self):
        from bespoke.eval.ensemble import compare_geometric
        d = compare_geometric({"reward": 0.6}, {"reward": 0.8})
        assert d["decision"] == "revert"

    def test_keep_within_noise_band(self):
        from bespoke.eval.ensemble import compare_geometric
        d = compare_geometric({"reward": 0.795}, {"reward": 0.80}, noise=0.02)
        assert d["decision"] == "keep"


class TestConfigFlag:
    def test_llm_judge_off_by_default(self):
        from bespoke.config import config
        assert config.pipeline.use_llm_judge is False
