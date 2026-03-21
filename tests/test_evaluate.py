# tests/test_evaluate.py
"""Tests for evaluate module."""

import json
from unittest.mock import patch, MagicMock

import pytest


def _no_close_db(db):
    """Wrap a sqlite3.Connection so that close() is a no-op (for test reuse)."""
    proxy = MagicMock(wraps=db)
    proxy.close = MagicMock()  # swallow close calls
    return proxy


class TestLogExperiment:
    def test_log_experiment_stores_training_config(self, db):
        """training_config dict is JSON-serialized into the experiments table."""
        from bespoke.train.evaluate import log_experiment

        scorecard = {
            "adapter_name": "test-001",
            "benchmark_version": 1,
            "reward_vector": [0.8, 0.9],
            "gate_pass_rate": 0.85,
        }
        training_config = {"rank": 16, "lr": 2e-4, "epochs": 2}
        data_config = {"min_quality": "high", "domain_filter": "code", "recency_days": 30}

        proxy = _no_close_db(db)
        with patch("bespoke.train.evaluate.get_connection", return_value=proxy), \
             patch("bespoke.train.evaluate.config") as mock_config:
            mock_config.base_model.training_model_path = "test-model"
            exp_id = log_experiment(
                scorecard=scorecard,
                training_config=training_config,
                data_config=data_config,
            )

        row = db.execute("SELECT * FROM experiments WHERE id = ?", (exp_id,)).fetchone()
        assert json.loads(row["training_config"]) == training_config
        assert json.loads(row["data_config"]) == data_config

    def test_log_experiment_stores_previous_experiment_id(self, db):
        """previous_experiment_id FK is written to the experiments table."""
        from bespoke.train.evaluate import log_experiment

        scorecard = {
            "adapter_name": "test-001",
            "benchmark_version": 1,
            "reward_vector": [0.8],
            "gate_pass_rate": 0.85,
        }

        proxy = _no_close_db(db)
        with patch("bespoke.train.evaluate.get_connection", return_value=proxy), \
             patch("bespoke.train.evaluate.config") as mock_config:
            mock_config.base_model.training_model_path = "test-model"
            first_id = log_experiment(scorecard=scorecard)
            second_id = log_experiment(
                scorecard=scorecard,
                previous_experiment_id=first_id,
            )

        row = db.execute("SELECT * FROM experiments WHERE id = ?", (second_id,)).fetchone()
        assert row["previous_experiment_id"] == first_id

    def test_log_experiment_defaults_to_empty_configs(self, db):
        """When no configs passed, defaults to empty dicts (backward compat)."""
        from bespoke.train.evaluate import log_experiment

        scorecard = {
            "adapter_name": "test-001",
            "benchmark_version": 1,
            "reward_vector": [],
            "gate_pass_rate": 0.5,
        }

        proxy = _no_close_db(db)
        with patch("bespoke.train.evaluate.get_connection", return_value=proxy), \
             patch("bespoke.train.evaluate.config") as mock_config:
            mock_config.base_model.training_model_path = "test-model"
            exp_id = log_experiment(scorecard=scorecard)

        row = db.execute("SELECT * FROM experiments WHERE id = ?", (exp_id,)).fetchone()
        assert json.loads(row["training_config"]) == {}
        assert json.loads(row["data_config"]) == {}
        assert row["previous_experiment_id"] is None
