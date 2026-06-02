# tests/test_eval_gates.py
"""Tests for deterministic correctness gates."""


class TestGates:
    def test_empty_output_fails(self):
        from bespoke.eval.gates import run_gates
        result = run_gates(prompt="q", output="   ", domain=None)
        assert result["passed"] is False
        assert any(r["gate"] == "non_empty" and not r["passed"] for r in result["results"])

    def test_code_domain_requires_code_block(self):
        from bespoke.eval.gates import run_gates
        no_code = run_gates(prompt="write a fn", output="just prose", domain="code")
        assert no_code["passed"] is False
        with_code = run_gates(prompt="write a fn", output="here:\n```py\nx=1\n```", domain="code")
        assert with_code["passed"] is True

    def test_non_code_passes_on_nonempty(self):
        from bespoke.eval.gates import run_gates
        assert run_gates(prompt="q", output="a real answer", domain="strategy")["passed"] is True
