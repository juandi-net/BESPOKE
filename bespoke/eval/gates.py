"""Deterministic correctness gates — the un-foolable anchor. Correctness was never the
LLM's job; it's a check you RUN. Extend per-domain (e.g. execute tests for code) over time.
"""
import re


def run_gates(prompt, output, domain=None):
    """Return {'passed': bool, 'results': [{'gate','passed'}, ...]}. ANY fail => passed=False."""
    results = []

    results.append({"gate": "non_empty", "passed": bool(output and output.strip())})

    if domain == "code":
        results.append({"gate": "has_code_block", "passed": bool(re.search(r"```", output or ""))})

    return {"passed": all(r["passed"] for r in results), "results": results}
