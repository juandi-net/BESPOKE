# tests/test_search.py
"""Tests for the autoresearch search loop."""

import json
from datetime import datetime
from unittest.mock import patch, MagicMock
from pathlib import Path

import pytest


class TestGetTopDomains:
    def test_returns_top_domains_by_count(self, db):
        """Returns domains ordered by interaction count, limited to n."""
        # Seed interactions with different domains (include all NOT NULL columns)
        for i in range(5):
            db.execute(
                "INSERT INTO interactions (provider, model, source, user_message, assistant_response, domain) VALUES (?, ?, ?, ?, ?, ?)",
                ("claude", "claude-sonnet-4-6", "claude-code", f"msg{i}", f"resp{i}", "code"),
            )
        for i in range(3):
            db.execute(
                "INSERT INTO interactions (provider, model, source, user_message, assistant_response, domain) VALUES (?, ?, ?, ?, ?, ?)",
                ("claude", "claude-sonnet-4-6", "claude-code", f"msg{i}", f"resp{i}", "planning"),
            )
        db.execute(
            "INSERT INTO interactions (provider, model, source, user_message, assistant_response, domain) VALUES (?, ?, ?, ?, ?, ?)",
            ("claude", "claude-sonnet-4-6", "claude-code", "msg", "resp", "organizing"),
        )
        db.commit()

        from bespoke.train.train_sft import _get_top_domains

        with patch("bespoke.train.train_sft.get_connection", return_value=db):
            domains = _get_top_domains(2)

        assert domains == ["code", "planning"]

    def test_returns_empty_when_no_domains(self, db):
        """Returns empty list when no interactions have domains."""
        from bespoke.train.train_sft import _get_top_domains

        with patch("bespoke.train.train_sft.get_connection", return_value=db):
            domains = _get_top_domains(2)

        assert domains == []


class TestGenerateConfigs:
    def test_generates_all_combinations(self):
        """Produces one config per combination of search space values."""
        from bespoke.train.train_sft import _generate_configs

        space = {
            "min_quality": ["high", "medium"],
            "rank": [8, 16],
        }
        configs = list(_generate_configs(space))
        assert len(configs) == 4

    def test_configs_have_name_key(self):
        """Each config has a sequential 'name' key."""
        from bespoke.train.train_sft import _generate_configs

        space = {"rank": [8, 16]}
        configs = list(_generate_configs(space))
        assert configs[0]["name"] == "search-000"
        assert configs[1]["name"] == "search-001"

    def test_configs_contain_all_keys(self):
        """Each config contains all search space keys plus 'name'."""
        from bespoke.train.train_sft import _generate_configs

        space = {"min_quality": ["high"], "rank": [16], "lr": [2e-4]}
        configs = list(_generate_configs(space))
        assert len(configs) == 1
        cfg = configs[0]
        assert cfg["min_quality"] == "high"
        assert cfg["rank"] == 16
        assert cfg["lr"] == 2e-4


class TestGetBaseline:
    def test_returns_none_when_no_experiments(self, db):
        """Returns (None, None) when experiments table is empty."""
        from bespoke.train.train_sft import _get_baseline

        with patch("bespoke.train.train_sft.get_connection", return_value=db):
            exp_id, scorecard = _get_baseline()

        assert exp_id is None
        assert scorecard is None

    def test_returns_most_recent_kept_experiment(self, db):
        """Returns the most recent experiment with decision='keep'."""
        scorecard_data = {"reward_vector": [0.8, 0.9], "gate_pass_rate": 0.85}
        db.execute("""
            INSERT INTO experiments (
                adapter_name, base_model, training_config, data_config,
                benchmark_version, scorecard, reward_vector,
                predicted_accept_rate, decision, decision_reasoning
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            "old", "model", "{}", "{}", 1,
            json.dumps(scorecard_data), json.dumps([0.8, 0.9]),
            0.85, "keep", "first",
        ))
        db.execute("""
            INSERT INTO experiments (
                adapter_name, base_model, training_config, data_config,
                benchmark_version, scorecard, reward_vector,
                predicted_accept_rate, decision, decision_reasoning
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            "reverted", "model", "{}", "{}", 1,
            json.dumps({}), "[]",
            0.5, "revert", "bad",
        ))
        db.commit()

        from bespoke.train.train_sft import _get_baseline

        with patch("bespoke.train.train_sft.get_connection", return_value=db):
            exp_id, scorecard = _get_baseline()

        assert exp_id is not None
        assert scorecard == scorecard_data


class TestCliSearchFlags:
    def test_train_with_deadline_calls_run_search(self):
        """bespoke train --deadline 06:00 calls run_search."""
        import argparse
        from bespoke.cli import cmd_train

        args = argparse.Namespace(
            deadline="06:00",
            max=None,
            domain=None,
            adapter_name="general-v1",
        )

        with patch("bespoke.train.train_sft.run_search") as mock_search:
            cmd_train(args)

        mock_search.assert_called_once_with(deadline="06:00", max_experiments=None)

    def test_train_without_flags_calls_run_sft_training(self):
        """bespoke train (no flags) calls run_sft_training as before."""
        import argparse
        from bespoke.cli import cmd_train

        args = argparse.Namespace(
            deadline=None,
            max=None,
            domain="code",
            adapter_name="general-v1",
        )

        with patch("bespoke.train.train_sft.run_sft_training") as mock_train:
            mock_train.return_value = Path("/tmp/adapter")
            cmd_train(args)

        mock_train.assert_called_once()
        call_kwargs = mock_train.call_args[1]
        assert call_kwargs["domain_filter"] == "code"
        assert call_kwargs["adapter_name"] == "general-v1"
