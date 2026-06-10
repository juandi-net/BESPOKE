# tests/test_train_sft.py
"""Tests for train_sft module (current mlx-lm API: `python -m mlx_lm lora`, --iters,
rank via lora_config.yaml — NOT the old --lora-rank/--epochs flags)."""

from unittest.mock import patch, MagicMock
from pathlib import Path

import yaml


def _proc(lines=("Iter 1: Val loss 2.0, Val took 1s\n",), rc=0):
    """A fake Popen whose stdout streams the given lines."""
    p = MagicMock()
    p.stdout = iter(lines)
    p.wait.return_value = rc
    p.returncode = rc
    return p


def _run(tmp_path, popen, **overrides):
    from bespoke.train.train_sft import run_sft_training
    with patch("bespoke.train.train_sft.export_sft_data") as mock_export, \
         patch("bespoke.train.train_sft.subprocess") as mock_sub, \
         patch("bespoke.train.train_sft.config") as mock_config:
        mock_export.return_value = tmp_path / "train.jsonl"
        mock_sub.Popen.return_value = popen
        mock_config.adapters_dir = tmp_path / "adapters"
        mock_config.base_model.training_model_path = tmp_path / "model"
        mock_config.training.use_dora = False
        mock_config.training.sft_rank = 16
        mock_config.training.sft_learning_rate = 2e-4
        mock_config.training.sft_epochs = 2
        adapter_dir = run_sft_training(adapter_name="test", **overrides)
    return adapter_dir, mock_export, mock_sub


class TestRunSftTrainingOverrides:
    def test_passes_domain_filter_to_export(self, tmp_path):
        _, mock_export, _ = _run(tmp_path, _proc(), domain_filter="code",
                                 min_quality="high", recency_days=30)
        kwargs = mock_export.call_args.kwargs
        assert kwargs.get("domain") == "code"
        assert kwargs.get("min_quality") == "high"
        assert kwargs.get("recency_days") == 30

    def test_uses_override_rank_lr(self, tmp_path):
        adapter_dir, _, mock_sub = _run(tmp_path, _proc(), rank=8, lr=1e-4)
        cmd_str = " ".join(str(c) for c in mock_sub.Popen.call_args[0][0])
        assert "--learning-rate 0.0001" in cmd_str
        assert "--iters" in cmd_str and "--epochs" not in cmd_str
        # rank has no CLI flag in current mlx-lm — it goes in lora_config.yaml
        cfg = yaml.safe_load((adapter_dir / "lora_config.yaml").read_text())
        assert cfg["lora_parameters"]["rank"] == 8

    def test_defaults_fall_through_to_config(self, tmp_path):
        adapter_dir, _, mock_sub = _run(tmp_path, _proc())
        cmd_str = " ".join(str(c) for c in mock_sub.Popen.call_args[0][0])
        assert "--learning-rate 0.0002" in cmd_str
        cfg = yaml.safe_load((adapter_dir / "lora_config.yaml").read_text())
        assert cfg["lora_parameters"]["rank"] == 16


class TestBestCheckpointPromotion:
    def test_promotes_best_checkpoint_over_final(self, tmp_path):
        """When an earlier save-point has lower val loss than the final, it replaces
        adapters.safetensors (RT-002: final != best)."""
        adapter_dir = tmp_path / "adapters" / "test" / "sft"
        adapter_dir.mkdir(parents=True)
        (adapter_dir / "0000200_adapters.safetensors").write_bytes(b"BEST")
        (adapter_dir / "0000400_adapters.safetensors").write_bytes(b"WORSE")
        lines = ("Iter 200: Val loss 2.00, Val took 1s\n",
                 "Iter 400: Val loss 2.50, Val took 1s\n")

        def fake_popen(*a, **k):
            (adapter_dir / "adapters.safetensors").write_bytes(b"WORSE")  # mlx writes final=last
            return _proc(lines)

        from bespoke.train.train_sft import run_sft_training
        with patch("bespoke.train.train_sft.export_sft_data", return_value=tmp_path / "t.jsonl"), \
             patch("bespoke.train.train_sft.subprocess") as mock_sub, \
             patch("bespoke.train.train_sft.config") as mock_config:
            mock_sub.Popen.side_effect = fake_popen
            mock_config.adapters_dir = tmp_path / "adapters"
            mock_config.base_model.training_model_path = tmp_path / "model"
            mock_config.training.use_dora = False
            mock_config.training.sft_rank = 16
            mock_config.training.sft_learning_rate = 2e-4
            mock_config.training.sft_epochs = 2
            run_sft_training(adapter_name="test")

        assert (adapter_dir / "adapters.safetensors").read_bytes() == b"BEST"
        assert "Val loss" in (adapter_dir / "train.log").read_text()

    def test_keeps_final_when_it_is_best(self, tmp_path):
        adapter_dir = tmp_path / "adapters" / "test" / "sft"
        adapter_dir.mkdir(parents=True)
        (adapter_dir / "0000200_adapters.safetensors").write_bytes(b"OK")
        (adapter_dir / "0000400_adapters.safetensors").write_bytes(b"FINALBEST")
        lines = ("Iter 200: Val loss 2.50, Val took 1s\n",
                 "Iter 400: Val loss 2.00, Val took 1s\n")

        def fake_popen(*a, **k):
            (adapter_dir / "adapters.safetensors").write_bytes(b"FINALBEST")
            return _proc(lines)

        from bespoke.train.train_sft import run_sft_training
        with patch("bespoke.train.train_sft.export_sft_data", return_value=tmp_path / "t.jsonl"), \
             patch("bespoke.train.train_sft.subprocess") as mock_sub, \
             patch("bespoke.train.train_sft.config") as mock_config:
            mock_sub.Popen.side_effect = fake_popen
            mock_config.adapters_dir = tmp_path / "adapters"
            mock_config.base_model.training_model_path = tmp_path / "model"
            mock_config.training.use_dora = False
            mock_config.training.sft_rank = 16
            mock_config.training.sft_learning_rate = 2e-4
            mock_config.training.sft_epochs = 2
            run_sft_training(adapter_name="test")

        assert (adapter_dir / "adapters.safetensors").read_bytes() == b"FINALBEST"


class TestTrainingFailure:
    def test_raises_on_nonzero_exit(self, tmp_path):
        import pytest
        from bespoke.train.train_sft import run_sft_training
        with patch("bespoke.train.train_sft.export_sft_data", return_value=tmp_path / "t.jsonl"), \
             patch("bespoke.train.train_sft.subprocess") as mock_sub, \
             patch("bespoke.train.train_sft.config") as mock_config:
            mock_sub.Popen.return_value = _proc(lines=("boom\n",), rc=1)
            mock_config.adapters_dir = tmp_path / "adapters"
            mock_config.base_model.training_model_path = tmp_path / "model"
            mock_config.training.use_dora = False
            mock_config.training.sft_rank = 16
            mock_config.training.sft_learning_rate = 2e-4
            mock_config.training.sft_epochs = 2
            with pytest.raises(RuntimeError):
                run_sft_training(adapter_name="test")
