import yaml
from unittest.mock import patch

from bespoke.teach.prompts import get_benchmark_context, make_session_extraction_prompt


def test_benchmark_context_with_benchmark(tmp_path):
    """get_benchmark_context returns formatted context when benchmark exists."""
    benchmark = {
        "benchmark": {
            "version": 1,
            "quality_dimensions": [
                {"id": "correctness", "name": "Correctness", "checks": ["Is it correct?"]}
            ],
        }
    }
    benchmark_path = tmp_path / "benchmark.yaml"
    benchmark_path.write_text(yaml.dump(benchmark))

    with patch("bespoke.teach.prompts.config") as mock_config:
        mock_config.benchmark_dir = tmp_path
        context = get_benchmark_context()

    assert context is not None
    assert "correct" in context.lower()


def test_benchmark_context_without_benchmark(tmp_path):
    """get_benchmark_context returns fallback string when no benchmark exists."""
    with patch("bespoke.teach.prompts.config") as mock_config:
        mock_config.benchmark_dir = tmp_path
        context = get_benchmark_context()

    assert "generic" in context.lower() or "no benchmark" in context.lower()


def test_benchmark_context_with_new_schema(tmp_path):
    """get_benchmark_context handles the new interview schema with domains, gates, etc."""
    benchmark = {
        "benchmark": {
            "version": 2,
            "operating_altitude": "artifact",
            "domains": [
                {
                    "id": "backend_code",
                    "name": "Backend Code",
                    "gates": [{"id": "compiles", "check": "Does it compile?", "severity": "hard_fail"}],
                    "quality_checks": [{"id": "tests", "check": "Has tests?"}],
                }
            ],
            "universal_gates": [{"id": "addresses_prompt", "check": "Addresses what was asked"}],
            "quality_dimensions": [],
        }
    }
    benchmark_path = tmp_path / "benchmark.yaml"
    benchmark_path.write_text(yaml.dump(benchmark))

    with patch("bespoke.teach.prompts.config") as mock_config:
        mock_config.benchmark_dir = tmp_path
        context = get_benchmark_context()

    assert "artifact" in context.lower()
    assert "backend_code" in context.lower() or "Backend Code" in context


def test_session_prompt_includes_benchmark_context():
    """Session extraction prompt includes benchmark context when provided."""
    turns = [
        {"user_message": "Fix the bug", "assistant_response": "Fixed it."},
    ]

    benchmark_ctx = "This user operates at the artifact level.\nTheir domains are: Backend Code"
    messages = make_session_extraction_prompt(turns, benchmark_context=benchmark_ctx)

    system_content = messages[0]["content"]
    assert "artifact level" in system_content
    assert "Backend Code" in system_content


def test_session_prompt_without_benchmark():
    """Session extraction prompt works without benchmark context."""
    turns = [
        {"user_message": "Fix the bug", "assistant_response": "Fixed it."},
    ]

    messages = make_session_extraction_prompt(turns)
    system_content = messages[0]["content"]
    assert "BESPOKE" in system_content
    assert "BENCHMARK CONTEXT" not in system_content
