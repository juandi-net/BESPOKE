"""Stage 3 Evaluation: LLM-as-judge with benchmark-governed keep/revert.

Three-layer evaluation:
1. Correctness gates (pass/fail)
2. Quality dimensions (binary checklist scoring per benchmark)
3. Aspiration targets (directional, not scored)
"""

import gc
import json
import yaml
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

from openai import OpenAI

try:
    import mlx_lm
except ImportError:
    mlx_lm = None  # Not available in test environments

from bespoke.db.init import get_connection
from bespoke.teach.llm_client import chat_json
from bespoke.config import config


def load_benchmark() -> dict:
    """Load the current benchmark YAML file."""
    benchmark_path = config.benchmark_dir / "benchmark.yaml"
    if not benchmark_path.exists():
        return {"benchmark": {"quality_dimensions": [], "correctness_gates": {}}}
    with open(benchmark_path) as f:
        return yaml.safe_load(f)


def build_judge_prompt(benchmark: dict, domain: str = None) -> str:
    """Generate the LLM-as-judge system prompt from the benchmark."""
    r = benchmark.get("benchmark", {})

    prompt = """You are evaluating specialist model output against a specific user's quality standards.

INSTRUCTIONS:
- For each evaluation criterion, provide your reasoning BEFORE your judgment
- Score each binary check as YES or NO
- Be strict but fair — evaluate what's actually in the output

"""

    # Correctness gates
    gates = r.get("correctness_gates", {})
    domain_gates = gates.get(domain, gates.get("all_domains", []))
    if domain_gates:
        prompt += "CORRECTNESS GATES (all must pass):\n"
        for gate in domain_gates:
            check = gate.get("check", gate) if isinstance(gate, dict) else gate
            prompt += f"  - {check}\n"
        prompt += "\nIf ANY gate fails, the output scores ZERO. Do not evaluate quality dimensions.\n\n"

    # Quality dimensions
    dims = r.get("quality_dimensions", [])
    if domain:
        dims = [d for d in dims if domain in d.get("applies_to", [domain])]

    if dims:
        prompt += "QUALITY DIMENSIONS:\n"
        for dim in dims:
            prompt += f"\n{dim.get('name', dim.get('id', 'Unknown'))}:\n"
            for check in dim.get("checks", []):
                prompt += f"  - {check}\n"

    prompt += """
OUTPUT FORMAT (JSON):
{
    "gates_passed": true | false,
    "gate_results": [{"gate": "...", "passed": true|false, "reasoning": "..."}],
    "dimension_scores": [
        {
            "dimension_id": "...",
            "checks": [{"check": "...", "passed": true|false, "reasoning": "..."}],
            "score": 0.75
        }
    ],
    "reward_vector": [0.75, 0.87, ...],
    "overall_reasoning": "Brief summary of evaluation"
}

Output ONLY valid JSON."""

    return prompt


def evaluate_output(
    prompt: str,
    output: str,
    benchmark: dict,
    domain: str = None,
) -> dict:
    """Evaluate a single output against the benchmark using LLM-as-judge."""
    judge_system = build_judge_prompt(benchmark, domain)

    messages = [
        {"role": "system", "content": judge_system},
        {"role": "user", "content": f"""Evaluate this output:

PROMPT: {prompt}

OUTPUT: {output[:3000]}"""},
    ]

    return chat_json(
        messages=messages,
        model=config.llm.model_judge,
        max_tokens=4096,
    )


def get_eval_prompts(domain: str = None, limit: int = 20) -> List[dict]:
    """Get held-out evaluation prompts from the warehouse (prompts only)."""
    conn = get_connection()

    query = """
        SELECT user_message, domain
        FROM interactions
        WHERE quality_score = 'high'
        AND feedback_class IN ('accept', 'strong_accept')
        AND used_in_sft = 0
    """
    params = []

    if domain:
        query += " AND domain = ?"
        params.append(domain)

    query += " ORDER BY RANDOM() LIMIT ?"
    params.append(limit)

    rows = conn.execute(query, params).fetchall()
    conn.close()

    return [dict(row) for row in rows]


def generate_specialist_response(
    prompt: str,
    port: int = None,
) -> str:
    """Query the running llama-server (specialist adapter) for a response."""
    if port is None:
        port = config.base_model.llama_server_port
    client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="not-needed")
    response = client.chat.completions.create(
        model="specialist",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
        temperature=0.7,
    )
    return response.choices[0].message.content


def generate_with_mlx(
    prompts: List[str],
    model_path: str,
    adapter_path: str,
) -> List[str]:
    """Load model+adapter, generate responses for all prompts, free memory.

    Loads the model once, generates all responses, then unloads.
    One load/unload cycle per experiment — matches the 16GB memory model.
    """
    model, tokenizer = mlx_lm.load(str(model_path), adapter_path=str(adapter_path))

    responses = []
    for prompt in prompts:
        response = mlx_lm.generate(
            model, tokenizer, prompt=prompt,
            max_tokens=2048, temp=0.7,
        )
        responses.append(response)

    del model, tokenizer
    gc.collect()

    return responses


def run_evaluation(
    adapter_name: str,
    domain: str = None,
    num_prompts: int = 20,
    server_port: int = None,
    adapter_path: str = None,
) -> dict:
    """Run full evaluation of an adapter against the benchmark.

    If adapter_path is provided, uses mlx_lm.generate directly (no server needed).
    Otherwise requires llama-server to be running with the adapter loaded.
    Returns a scorecard dict.
    """
    benchmark = load_benchmark()
    eval_prompts = get_eval_prompts(domain=domain, limit=num_prompts)

    if not eval_prompts:
        print("No evaluation prompts available.")
        return {"error": "no_eval_prompts"}

    print(f"Evaluating adapter '{adapter_name}' on {len(eval_prompts)} prompts...")

    # Generate all responses
    if adapter_path:
        # Direct MLX inference — load model once, generate all, unload
        prompt_texts = [ep["user_message"] for ep in eval_prompts]
        specialist_outputs = generate_with_mlx(
            prompts=prompt_texts,
            model_path=str(config.base_model.training_model_path),
            adapter_path=str(adapter_path),
        )
    else:
        # HTTP-based inference via llama-server
        specialist_outputs = []
        for ep in eval_prompts:
            output = generate_specialist_response(ep["user_message"], port=server_port)
            specialist_outputs.append(output)

    # Judge each response
    results = []
    gate_pass_count = 0

    for i, (ep, specialist_output) in enumerate(zip(eval_prompts, specialist_outputs)):
        print(f"  Judging prompt {i+1}/{len(eval_prompts)}...")

        result = evaluate_output(
            prompt=ep["user_message"],
            output=specialist_output,
            benchmark=benchmark,
            domain=ep.get("domain", domain),
        )
        results.append(result)

        if result.get("gates_passed", False):
            gate_pass_count += 1

    # Aggregate scores (unchanged from current code)
    r = benchmark.get("benchmark", {})
    dim_ids = [d.get("id") for d in r.get("quality_dimensions", [])]

    reward_vectors = []
    for result in results:
        if result.get("gates_passed"):
            rv = result.get("reward_vector", [])
            if rv:
                reward_vectors.append(rv)

    avg_reward = []
    if reward_vectors:
        vec_len = len(reward_vectors[0])
        for j in range(vec_len):
            vals = [rv[j] for rv in reward_vectors if j < len(rv)]
            avg_reward.append(sum(vals) / len(vals) if vals else 0.0)

    scorecard = {
        "adapter_name": adapter_name,
        "benchmark_version": r.get("version", 0),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "eval_set_size": len(eval_prompts),
        "gate_pass_rate": gate_pass_count / len(eval_prompts) if eval_prompts else 0,
        "reward_vector": avg_reward,
        "dimension_ids": dim_ids,
        "individual_results": results,
    }

    return scorecard


def compare_scorecards(current: dict, previous: dict) -> dict:
    """Compare two scorecards and make a keep/revert decision."""
    curr_rv = current.get("reward_vector", [])
    prev_rv = previous.get("reward_vector", [])

    if not curr_rv or not prev_rv:
        return {"decision": "keep", "reasoning": "No previous baseline to compare against."}

    # Compute delta
    delta = [c - p for c, p in zip(curr_rv, prev_rv)]

    # Check direction alignment — did all dimensions improve?
    improved = sum(1 for d in delta if d > 0)
    regressed = sum(1 for d in delta if d < 0)
    unchanged = sum(1 for d in delta if d == 0)

    # Decision logic
    if all(d >= 0 for d in delta):
        decision = "keep"
        reasoning = f"All dimensions improved or held steady. Delta: {delta}"
    elif improved > regressed and min(delta) > -0.1:
        decision = "keep"
        reasoning = f"Net improvement ({improved} improved, {regressed} regressed). Max regression: {min(delta):.3f}"
    else:
        decision = "revert"
        reasoning = f"Significant regression detected ({regressed} dimensions). Delta: {delta}"

    return {
        "decision": decision,
        "reasoning": reasoning,
        "delta": delta,
        "improved": improved,
        "regressed": regressed,
        "unchanged": unchanged,
    }


def save_scorecard(scorecard: dict, comparison: dict = None) -> Path:
    """Save a scorecard to ~/.bespoke/scorecards/."""
    adapter = scorecard.get("adapter_name", "unknown")
    date = scorecard.get("date", datetime.now().strftime("%Y-%m-%d"))

    if comparison:
        scorecard["comparison"] = comparison

    output_path = config.scorecards_dir / f"{adapter}-{date}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(scorecard, f, indent=2)

    # Also write a human-readable markdown version
    md_path = output_path.with_suffix(".md")
    md = generate_scorecard_markdown(scorecard, comparison)
    md_path.write_text(md)

    print(f"Scorecard saved to {output_path}")
    return output_path


def generate_scorecard_markdown(scorecard: dict, comparison: dict = None) -> str:
    """Generate a human-readable markdown scorecard."""
    lines = [
        f"# Scorecard: {scorecard.get('adapter_name', 'unknown')}",
        f"",
        f"- **Date:** {scorecard.get('date', 'unknown')}",
        f"- **Benchmark version:** {scorecard.get('benchmark_version', '?')}",
        f"- **Eval set size:** {scorecard.get('eval_set_size', '?')}",
        f"- **Gate pass rate:** {scorecard.get('gate_pass_rate', 0):.1%}",
        f"",
        f"## Reward Vector",
        f"",
    ]

    dim_ids = scorecard.get("dimension_ids", [])
    rv = scorecard.get("reward_vector", [])
    for i, score in enumerate(rv):
        dim_name = dim_ids[i] if i < len(dim_ids) else f"dim_{i}"
        lines.append(f"- **{dim_name}:** {score:.3f}")

    if comparison:
        lines.extend([
            f"",
            f"## Comparison",
            f"",
            f"- **Decision:** {comparison.get('decision', '?').upper()}",
            f"- **Reasoning:** {comparison.get('reasoning', '')}",
            f"- **Delta:** {comparison.get('delta', [])}",
            f"- **Improved:** {comparison.get('improved', 0)} dimensions",
            f"- **Regressed:** {comparison.get('regressed', 0)} dimensions",
        ])

    return "\n".join(lines) + "\n"


def log_experiment(
    scorecard: dict,
    comparison: dict = None,
    training_config: dict = None,
    data_config: dict = None,
    previous_experiment_id: int = None,
) -> int:
    """Log the experiment to the database."""
    conn = get_connection()

    decision = "keep"
    decision_reasoning = ""
    if comparison:
        decision = comparison.get("decision", "keep")
        decision_reasoning = comparison.get("reasoning", "")

    previous_reward = None
    if previous_experiment_id:
        prev_row = conn.execute(
            "SELECT reward_vector FROM experiments WHERE id = ?",
            (previous_experiment_id,),
        ).fetchone()
        if prev_row:
            previous_reward = prev_row["reward_vector"]

    cursor = conn.execute("""
        INSERT INTO experiments (
            adapter_name, base_model, training_config, data_config,
            benchmark_version, scorecard, reward_vector,
            predicted_accept_rate, decision, decision_reasoning,
            previous_experiment_id, previous_reward_vector
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        scorecard.get("adapter_name", "unknown"),
        str(config.base_model.training_model_path),
        json.dumps(training_config or {}),
        json.dumps(data_config or {}),
        scorecard.get("benchmark_version", 0),
        json.dumps(scorecard),
        json.dumps(scorecard.get("reward_vector", [])),
        scorecard.get("gate_pass_rate", 0),
        decision,
        decision_reasoning,
        previous_experiment_id,
        previous_reward,
    ))

    conn.commit()
    experiment_id = cursor.lastrowid
    conn.close()

    return experiment_id


def main():
    """CLI entry point for evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="BESPOKE: Evaluate adapter")
    parser.add_argument("--adapter-name", type=str, default="general-v1", help="Adapter to evaluate")
    parser.add_argument("--domain", type=str, help="Filter eval prompts by domain")
    parser.add_argument("--num-prompts", type=int, default=20, help="Number of eval prompts")
    args = parser.parse_args()

    print("BESPOKE Evaluation")
    print("=" * 50)

    scorecard = run_evaluation(
        adapter_name=args.adapter_name,
        domain=args.domain,
        num_prompts=args.num_prompts,
    )

    if "error" in scorecard:
        print(f"Evaluation failed: {scorecard['error']}")
        return

    # Save scorecard
    save_scorecard(scorecard)

    # Log experiment
    exp_id = log_experiment(scorecard)
    print(f"Experiment logged as #{exp_id}")


if __name__ == "__main__":
    main()
