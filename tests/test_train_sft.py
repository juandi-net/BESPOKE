# tests/test_train_sft.py
"""Tests for train_sft module."""

from unittest.mock import patch, MagicMock
from pathlib import Path

import pytest


class TestRunSftTrainingOverrides:
    def test_passes_domain_filter_to_export(self):
        """domain_filter is passed through to export_sft_data as domain."""
        from bespoke.train.train_sft import run_sft_training

        with patch("bespoke.train.train_sft.export_sft_data") as mock_export, \
             patch("bespoke.train.train_sft.subprocess") as mock_sub, \
             patch("bespoke.train.train_sft.config") as mock_config:
            mock_export.return_value = Path("/tmp/train.jsonl")
            mock_sub.run.return_value = MagicMock(returncode=0)
            mock_config.adapters_dir = Path("/tmp/adapters")
            mock_config.base_model.training_model_path = Path("/tmp/model")
            mock_config.training.use_dora = True

            run_sft_training(
                adapter_name="test",
                domain_filter="code",
                min_quality="high",
                recency_days=30,
            )

        mock_export.assert_called_once()
        _, kwargs = mock_export.call_args
        assert kwargs.get("domain") == "code"

    def test_uses_override_rank_lr_epochs(self):
        """Override rank/lr/epochs appear in the subprocess command."""
        from bespoke.train.train_sft import run_sft_training

        with patch("bespoke.train.train_sft.export_sft_data") as mock_export, \
             patch("bespoke.train.train_sft.subprocess") as mock_sub, \
             patch("bespoke.train.train_sft.config") as mock_config:
            mock_export.return_value = Path("/tmp/train.jsonl")
            mock_sub.run.return_value = MagicMock(returncode=0)
            mock_config.adapters_dir = Path("/tmp/adapters")
            mock_config.base_model.training_model_path = Path("/tmp/model")
            mock_config.training.use_dora = True

            run_sft_training(
                adapter_name="test",
                rank=8,
                lr=1e-4,
                epochs=1,
            )

        cmd = mock_sub.run.call_args[0][0]
        cmd_str = " ".join(str(c) for c in cmd)
        assert "--lora-rank 8" in cmd_str
        assert "--learning-rate 0.0001" in cmd_str
        assert "--epochs 1" in cmd_str

    def test_defaults_fall_through_to_config(self):
        """When no overrides, uses config values (backward compat)."""
        from bespoke.train.train_sft import run_sft_training

        with patch("bespoke.train.train_sft.export_sft_data") as mock_export, \
             patch("bespoke.train.train_sft.subprocess") as mock_sub, \
             patch("bespoke.train.train_sft.config") as mock_config:
            mock_export.return_value = Path("/tmp/train.jsonl")
            mock_sub.run.return_value = MagicMock(returncode=0)
            mock_config.adapters_dir = Path("/tmp/adapters")
            mock_config.base_model.training_model_path = Path("/tmp/model")
            mock_config.training.use_dora = True
            mock_config.training.sft_rank = 16
            mock_config.training.sft_learning_rate = 2e-4
            mock_config.training.sft_epochs = 2
            mock_config.training.sft_batch_size = 4

            run_sft_training(adapter_name="test")

        cmd = mock_sub.run.call_args[0][0]
        cmd_str = " ".join(str(c) for c in cmd)
        assert "--lora-rank 16" in cmd_str
        assert "--learning-rate 0.0002" in cmd_str
        assert "--epochs 2" in cmd_str
