# tests/test_evaluate_geometric.py
"""run_evaluation should use the geometric ensemble when use_llm_judge is False."""
from unittest.mock import patch


class TestRunEvaluationGeometric:
    def test_geometric_path_builds_scorecard_without_llm(self):
        from bespoke.train import evaluate

        eval_prompts = [{"user_message": "q1", "domain": "strategy"},
                        {"user_message": "q2", "domain": "strategy"}]
        outputs = ["a good answer", "another good answer"]

        with patch.object(evaluate, "get_eval_prompts", return_value=eval_prompts), \
             patch.object(evaluate, "generate_with_mlx", return_value=outputs), \
             patch.object(evaluate, "_geometric_score_outputs") as mock_geo, \
             patch.object(evaluate, "load_benchmark", return_value={"benchmark": {}}):
            mock_geo.return_value = {"eval_set_size": 2, "reward": 0.75, "gate_pass_rate": 1.0,
                                    "per_item": []}
            card = evaluate.run_evaluation(adapter_name="t", adapter_path="/tmp/x", num_prompts=2)

        assert card["reward"] == 0.75
        assert card["adapter_name"] == "t"
        mock_geo.assert_called_once()


class TestCompareUsesReward:
    def test_compare_scorecards_uses_geometric_reward(self):
        from bespoke.train.evaluate import compare_scorecards
        keep = compare_scorecards({"reward": 0.82}, {"reward": 0.80})
        revert = compare_scorecards({"reward": 0.60}, {"reward": 0.80})
        assert keep["decision"] == "keep"
        assert revert["decision"] == "revert"
