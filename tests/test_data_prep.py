# tests/test_data_prep.py
"""Tests for data_prep export functions."""

import json
from datetime import datetime, timedelta
from unittest.mock import patch
from pathlib import Path

import pytest


def _seed_training_pairs(db, pairs):
    """Insert training pairs into the test database."""
    for p in pairs:
        db.execute("""
            INSERT INTO training_pairs (
                interaction_id, pair_type, domain, instruction, response,
                quality_score, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            p.get("interaction_id", 1),
            p.get("pair_type", "sft"),
            p.get("domain", "code"),
            p.get("instruction", "test instruction"),
            p.get("response", "test response"),
            p.get("quality_score", "high"),
            p.get("created_at", datetime.now().isoformat()),
        ))
    db.commit()


def _seed_interaction(db, interaction_id, captured_at):
    """Insert a minimal interaction row for FK purposes."""
    db.execute("""
        INSERT INTO interactions (id, provider, model, source, user_message, assistant_response, captured_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (interaction_id, "claude", "claude-sonnet-4-6", "claude-code", "msg", "resp", captured_at))
    db.commit()


class TestExportSftDataRecency:
    def test_recency_days_filters_old_pairs(self, db, tmp_path):
        """Pairs linked to interactions older than recency_days are excluded."""
        now = datetime.now()
        old = (now - timedelta(days=90)).isoformat()
        recent = (now - timedelta(days=10)).isoformat()

        _seed_interaction(db, 1, old)
        _seed_interaction(db, 2, recent)

        _seed_training_pairs(db, [
            {"interaction_id": 1, "instruction": "old", "response": "old resp"},
            {"interaction_id": 2, "instruction": "recent", "response": "recent resp"},
        ])

        from bespoke.train.data_prep import export_sft_data

        with patch("bespoke.train.data_prep.get_connection", return_value=db):
            path = export_sft_data(
                recency_days=30,
                output_path=tmp_path / "train.jsonl",
            )

        lines = path.read_text().strip().split("\n")
        all_lines = lines
        # Also check valid.jsonl
        valid_path = path.with_name("valid.jsonl")
        if valid_path.exists():
            all_lines += valid_path.read_text().strip().split("\n")

        instructions = [json.loads(l)["messages"][0]["content"] for l in all_lines if l]
        assert "recent" in instructions
        assert "old" not in instructions

    def test_recency_days_none_includes_all(self, db, tmp_path):
        """When recency_days is None, all pairs are included."""
        now = datetime.now()
        old = (now - timedelta(days=365)).isoformat()

        _seed_interaction(db, 1, old)
        _seed_training_pairs(db, [
            {"interaction_id": 1, "instruction": "ancient", "response": "resp"},
        ])

        from bespoke.train.data_prep import export_sft_data

        with patch("bespoke.train.data_prep.get_connection", return_value=db):
            path = export_sft_data(
                recency_days=None,
                output_path=tmp_path / "train.jsonl",
            )

        all_text = path.read_text()
        valid_path = path.with_name("valid.jsonl")
        if valid_path.exists():
            all_text += valid_path.read_text()
        assert "ancient" in all_text


class TestExportSftDataMaxSamples:
    def test_max_samples_limits_output(self, db, tmp_path):
        """Only max_samples pairs are exported."""
        _seed_interaction(db, 1, datetime.now().isoformat())
        _seed_training_pairs(db, [
            {"interaction_id": 1, "instruction": f"q{i}", "response": f"a{i}"}
            for i in range(10)
        ])

        from bespoke.train.data_prep import export_sft_data

        with patch("bespoke.train.data_prep.get_connection", return_value=db):
            path = export_sft_data(
                max_samples=3,
                output_path=tmp_path / "train.jsonl",
            )

        lines = path.read_text().strip().split("\n")
        valid_path = path.with_name("valid.jsonl")
        if valid_path.exists():
            lines += valid_path.read_text().strip().split("\n")
        total = len([l for l in lines if l])
        assert total == 3
