"""V0 Benchmark Reconciliation -- merge stated + revealed preferences into benchmark YAML."""

import json
import yaml
from datetime import datetime
from pathlib import Path
from typing import Optional

from bespoke.teach.llm_client import chat_json
from bespoke.config import config


RECONCILE_SYSTEM = """You are reconciling two sets of quality dimensions for an AI evaluation benchmark.

SOURCE 1 (Interview -- stated preferences): Dimensions the user explicitly said they care about.
SOURCE 2 (Data analysis -- revealed preferences): Dimensions extracted from the user's actual accept/reject behavior.

For each dimension, categorize as:
- CONFIRMED: Both sources identify this dimension. High confidence.
- DATA_ONLY: Data shows this pattern but the user didn't mention it. Medium confidence.
- INTERVIEW_ONLY: User stated this preference but data doesn't show it. Low confidence -- monitor.
- DIVERGENT: Sources conflict. Flag for user resolution.

Merge into a unified set of 4-6 dimensions. For each dimension, produce:
- id, name, description
- source: confirmed / data_only / interview_only / divergent
- confidence: high / medium / low
- weight_direction: relative importance (0.0 to 1.0, should sum roughly to 1.0)
- checks: 3-4 binary yes/no evaluation questions

Output JSON:
{
    "dimensions": [...],
    "divergences": [{"description": "...", "interview_says": "...", "data_says": "..."}]
}"""


def reconcile(interview_result: dict, data_result: dict) -> dict:
    """Reconcile interview and data extraction results."""
    messages = [
        {"role": "system", "content": RECONCILE_SYSTEM},
        {"role": "user", "content": f"""Reconcile these two sources:

SOURCE 1 -- Interview (stated preferences):
{json.dumps(interview_result.get('dimensions', []), indent=2)}

SOURCE 2 -- Data analysis (revealed preferences):
{json.dumps(data_result.get('dimensions', []), indent=2)}

Produce the unified benchmark dimensions."""},
    ]

    return chat_json(
        messages=messages,
        model=config.llm.model_benchmark,
        max_tokens=4096,
    )


def generate_benchmark_yaml(reconciled: dict, version: int = 1) -> str:
    """Generate the benchmark YAML file from reconciled dimensions."""
    benchmark = {
        "benchmark": {
            "version": version,
            "created": datetime.now().strftime("%Y-%m-%d"),
            "last_updated": datetime.now().strftime("%Y-%m-%d"),
            "scope": "v0_basic",

            "correctness_gates": {
                "all_domains": [
                    {"id": "addresses_prompt", "check": "Output addresses what was asked, not a reinterpretation"},
                    {"id": "internally_consistent", "check": "No internal contradictions in the response"},
                    {"id": "factually_sound", "check": "No obviously false claims where verifiable"},
                ]
            },

            "quality_dimensions": [],

            "divergences": reconciled.get("divergences", []),

            "notes": "V0 benchmark -- basic interview + contrastive extraction. Full benchmark evolution (calibration layer, drift detection, per-dimension versioning) activates at V1+. See bespokerubric.pdf for the complete specification."
        }
    }

    for dim in reconciled.get("dimensions", []):
        benchmark["benchmark"]["quality_dimensions"].append({
            "id": dim.get("id"),
            "name": dim.get("name"),
            "description": dim.get("description"),
            "source": dim.get("source", "confirmed"),
            "confidence": dim.get("confidence", "medium"),
            "weight_direction": dim.get("weight_direction", 0.2),
            "checks": dim.get("checks", []),
        })

    return yaml.dump(benchmark, default_flow_style=False, sort_keys=False, width=100)


def run_benchmark_init(
    interview_result: Optional[dict] = None,
    data_result: Optional[dict] = None,
) -> Path:
    """Run the full V0 benchmark initialization pipeline.

    If interview_result or data_result are not provided, runs those steps first.
    """
    from bespoke.benchmark.interview import run_interview
    from bespoke.benchmark.extract import run_extraction

    print("\nBESPOKE V0 Benchmark Initialization")
    print("=" * 50)

    # Step 1: Interview (stated preferences)
    if interview_result is None:
        interview_result = run_interview()

    # Step 2: Contrastive extraction (revealed preferences)
    if data_result is None:
        data_result = run_extraction()

    # Step 3: Reconcile
    print("\nReconciling stated and revealed preferences...")
    reconciled = reconcile(interview_result, data_result)

    # Surface divergences to user
    if reconciled.get("divergences"):
        print("\nDIVERGENCES found between what you said and what your data shows:")
        for div in reconciled["divergences"]:
            print(f"  - {div.get('description', '')}")
            print(f"    Interview says: {div.get('interview_says', '')}")
            print(f"    Data says: {div.get('data_says', '')}")
        print("\nThese are recorded in the benchmark for future resolution.\n")

    # Step 4: Generate YAML
    benchmark_yaml = generate_benchmark_yaml(reconciled)

    output_path = config.benchmark_dir / "benchmark.yaml"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(benchmark_yaml)

    print(f"\nBenchmark v1 saved to {output_path}")
    print(f"Dimensions: {len(reconciled.get('dimensions', []))}")

    return output_path


def write_interview_benchmark(interview_result: dict) -> Path:
    """Write interview result directly to benchmark YAML (first-run path).

    Used when no Stage 2a data exists yet for contrastive extraction.
    The full reconciliation flow (interview + contrastive + reconcile)
    is for benchmark refresh after data accumulates.
    """
    benchmark = {
        "benchmark": {
            "version": 1,
            "created": datetime.now().strftime("%Y-%m-%d"),
            "last_updated": datetime.now().strftime("%Y-%m-%d"),
            "scope": "interview_only",
            "operating_altitude": interview_result.get("operating_altitude"),
            "altitude_evidence": interview_result.get("altitude_evidence"),
            "domains": interview_result.get("domains", []),
            "universal_gates": interview_result.get("universal_gates", []),
            "pragmatic_tradeoffs": interview_result.get("pragmatic_tradeoffs", []),
            "data_confrontation_notes": interview_result.get("data_confrontation_notes", []),
            "quality_dimensions": [],
            "divergences": [],
            "notes": "Initial benchmark from interview only. Will be reconciled with contrastive data after first extraction run.",
        }
    }

    output_path = config.benchmark_dir / "benchmark.yaml"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.dump(benchmark, default_flow_style=False, sort_keys=False, width=100)
    )

    print(f"\nBenchmark saved to {output_path}")
    return output_path


if __name__ == "__main__":
    run_benchmark_init()
