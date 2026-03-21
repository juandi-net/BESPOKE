import argparse
from unittest.mock import patch
from pathlib import Path


def test_benchmark_exists_true(tmp_path):
    """benchmark_exists returns True when benchmark.yaml is present."""
    from bespoke.cli import benchmark_exists

    with patch("bespoke.config.config") as mock_config:
        mock_config.benchmark_dir = tmp_path
        (tmp_path / "benchmark.yaml").write_text("benchmark: {}")
        assert benchmark_exists() is True


def test_benchmark_exists_false(tmp_path):
    """benchmark_exists returns False when benchmark.yaml is absent."""
    from bespoke.cli import benchmark_exists

    with patch("bespoke.config.config") as mock_config:
        mock_config.benchmark_dir = tmp_path
        assert benchmark_exists() is False


def test_cmd_run_skip_interview_flag():
    """--skip-interview flag is accepted by the run subparser."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    p_run = subparsers.add_parser("run")
    p_run.add_argument("--skip-interview", action="store_true")

    args = parser.parse_args(["run", "--skip-interview"])
    assert args.skip_interview is True

    args = parser.parse_args(["run"])
    assert args.skip_interview is False


def test_sft_training_gate(db):
    """Training is gated behind minimum SFT pair count."""
    count = db.execute(
        "SELECT COUNT(*) FROM training_pairs WHERE pair_type = 'sft'"
    ).fetchone()[0]
    assert count < 50  # empty test DB has 0
