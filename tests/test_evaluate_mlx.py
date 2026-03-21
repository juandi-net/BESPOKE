# tests/test_evaluate_mlx.py
"""Tests for MLX generation and adapter_path evaluation support."""

from unittest.mock import patch, MagicMock

import pytest


class TestGenerateWithMlx:
    def test_returns_list_of_responses(self):
        """generate_with_mlx returns one response per prompt."""
        from bespoke.train.evaluate import generate_with_mlx

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        with patch("bespoke.train.evaluate.mlx_lm") as mock_mlx:
            mock_mlx.load.return_value = (mock_model, mock_tokenizer)
            mock_mlx.generate.side_effect = ["response 1", "response 2"]

            responses = generate_with_mlx(
                prompts=["prompt 1", "prompt 2"],
                model_path="/fake/model",
                adapter_path="/fake/adapter",
            )

        assert responses == ["response 1", "response 2"]
        assert mock_mlx.load.call_count == 1
        assert mock_mlx.generate.call_count == 2

    def test_frees_model_after_generation(self):
        """Model references are deleted and gc.collect called."""
        from bespoke.train.evaluate import generate_with_mlx

        with patch("bespoke.train.evaluate.mlx_lm") as mock_mlx, \
             patch("bespoke.train.evaluate.gc") as mock_gc:
            mock_mlx.load.return_value = (MagicMock(), MagicMock())
            mock_mlx.generate.return_value = "resp"

            generate_with_mlx(
                prompts=["test"],
                model_path="/fake/model",
                adapter_path="/fake/adapter",
            )

        mock_gc.collect.assert_called_once()


class TestRunEvaluationWithAdapterPath:
    def test_uses_mlx_when_adapter_path_provided(self):
        """When adapter_path is set, run_evaluation uses generate_with_mlx."""
        from bespoke.train.evaluate import run_evaluation

        mock_benchmark = {
            "benchmark": {
                "version": 1,
                "quality_dimensions": [{"id": "clarity", "checks": ["Is it clear?"]}],
                "correctness_gates": {"all_domains": ["Is it correct?"]},
            }
        }
        mock_prompts = [
            {"user_message": "test prompt 1", "domain": "code"},
            {"user_message": "test prompt 2", "domain": "code"},
        ]
        mock_judge_result = {
            "gates_passed": True,
            "reward_vector": [0.8],
            "dimension_scores": [],
        }

        with patch("bespoke.train.evaluate.load_benchmark", return_value=mock_benchmark), \
             patch("bespoke.train.evaluate.get_eval_prompts", return_value=mock_prompts), \
             patch("bespoke.train.evaluate.generate_with_mlx", return_value=["resp1", "resp2"]) as mock_gen, \
             patch("bespoke.train.evaluate.evaluate_output", return_value=mock_judge_result), \
             patch("bespoke.train.evaluate.config") as mock_config:
            mock_config.base_model.training_model_path = "/fake/model"

            scorecard = run_evaluation(
                adapter_name="test",
                adapter_path="/fake/adapter",
                num_prompts=2,
            )

        mock_gen.assert_called_once()
        assert scorecard["adapter_name"] == "test"
        assert scorecard["gate_pass_rate"] == 1.0
