"""Stage 3: SFT training using MLX."""

import itertools
import json
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from bespoke.config import config
from bespoke.db.init import get_connection
from bespoke.train.data_prep import export_sft_data
from bespoke.train.evaluate import (
    run_evaluation, compare_scorecards, save_scorecard, log_experiment,
)


def run_sft_training(
    adapter_name: str = "general-v1",
    domain_filter: str = None,
    min_quality: str = None,
    recency_days: int = None,
    rank: int = None,
    lr: float = None,
    epochs: int = None,
) -> Path:
    """Run SFT training via mlx_lm.lora CLI.

    All override params default to None, falling through to config values.
    Returns the path to the trained adapter directory.
    """
    rank = rank or config.training.sft_rank
    lr = lr or config.training.sft_learning_rate
    epochs = epochs or config.training.sft_epochs
    min_quality = min_quality or "medium"

    # Export training data with filters
    train_path = export_sft_data(
        domain=domain_filter,
        min_quality=min_quality,
        recency_days=recency_days,
    )

    # Adapter output path
    adapter_dir = config.adapters_dir / adapter_name / "sft"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # Build MLX training command
    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model", str(config.base_model.training_model_path),
        "--train",
        "--data", str(train_path.parent),
        "--adapter-path", str(adapter_dir),
        "--fine-tune-type", "dora" if config.training.use_dora else "lora",
        "--num-layers", "-1",
        "--lora-rank", str(rank),
        "--learning-rate", str(lr),
        "--batch-size", str(config.training.sft_batch_size),
        "--epochs", str(epochs),
        "--steps-per-eval", "50",
        "--save-every", "100",
    ]

    print(f"Running SFT training...")
    print(f"  Model: {config.base_model.training_model_path}")
    print(f"  Adapter: {adapter_dir}")
    print(f"  DoRA: {config.training.use_dora}")
    print(f"  Rank: {rank}")
    print(f"  LR: {lr}")
    print(f"  Epochs: {epochs}")
    if domain_filter:
        print(f"  Domain: {domain_filter}")
    if recency_days:
        print(f"  Recency: {recency_days} days")
    print(f"  Quality: {min_quality}+")

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        raise RuntimeError(f"SFT training failed with return code {result.returncode}")

    print(f"SFT training complete. Adapter saved to {adapter_dir}")
    return adapter_dir


def convert_adapter_to_gguf(
    adapter_dir: Path,
    output_path: Path = None,
) -> Path:
    """Convert MLX adapter to GGUF format for llama.cpp serving.

    Pipeline: MLX adapter -> fuse -> PEFT format -> GGUF
    """
    if output_path is None:
        output_path = adapter_dir.parent / f"{adapter_dir.parent.name}.gguf"

    # Step 1: Fuse adapter with base model (MLX format)
    fused_dir = adapter_dir.parent / "fused"
    cmd_fuse = [
        sys.executable, "-m", "mlx_lm.fuse",
        "--model", str(config.base_model.training_model_path),
        "--adapter-path", str(adapter_dir),
        "--save-path", str(fused_dir),
    ]

    print(f"Fusing adapter...")
    subprocess.run(cmd_fuse, check=True)

    # Step 2: Convert to GGUF
    # Note: This requires the convert_lora_to_gguf.py script from llama.cpp
    # For V0, we serve from MLX format directly or use the fused model
    # TODO: implement GGUF conversion once the adapter workflow is validated

    print(f"Adapter fused to {fused_dir}")
    print(f"NOTE: GGUF conversion not yet implemented. Serve from MLX format for now.")
    return fused_dir


def _get_top_domains(n: int = 3) -> list:
    """Return the top N domains by interaction count from the warehouse."""
    conn = get_connection()
    rows = conn.execute("""
        SELECT domain, count(*) as cnt FROM interactions
        WHERE domain IS NOT NULL
        GROUP BY domain ORDER BY cnt DESC LIMIT ?
    """, (n,)).fetchall()
    conn.close()
    return [row["domain"] for row in rows]


def _generate_configs(search_space: dict):
    """All combinations of search_space as plain dicts with sequential names."""
    keys = list(search_space.keys())
    for i, combo in enumerate(itertools.product(*search_space.values())):
        yield {"name": f"search-{i:03d}", **dict(zip(keys, combo))}


def _get_baseline():
    """Most recent kept experiment's scorecard, or (None, None)."""
    conn = get_connection()
    row = conn.execute("""
        SELECT id, scorecard FROM experiments
        WHERE decision = 'keep'
        ORDER BY id DESC LIMIT 1
    """).fetchone()
    conn.close()
    if not row:
        return None, None
    return row["id"], json.loads(row["scorecard"])


def run_search(deadline: str = "06:00", max_experiments: int = None, num_eval_prompts: int = 20):
    """Iterate over a config grid, training and evaluating each adapter.

    Keeps the best adapter, cleans up the rest. GGUF-converts the winner.
    """
    # Parse deadline
    h, m = map(int, deadline.split(":"))
    stop_time = datetime.now().replace(hour=h, minute=m, second=0)
    if stop_time <= datetime.now():
        stop_time += timedelta(days=1)

    baseline_id, baseline_scorecard = _get_baseline()
    best_scorecard = baseline_scorecard
    best_config = None

    # Dynamic domain discovery
    domains = [None] + _get_top_domains(2)
    search_space = {
        "min_quality": ["high", "medium"],
        "domain_filter": domains,
        "recency_days": [None, 60, 30],
        "rank": [8, 16],
        "lr": [1e-4, 2e-4],
        "epochs": [1, 2],
    }

    configs = list(_generate_configs(search_space))

    # Shuffle with date-based seed for reproducible but varied exploration
    random.seed(int(datetime.now().strftime("%Y%m%d")))
    random.shuffle(configs)

    if max_experiments:
        configs = configs[:max_experiments]

    print(f"Search: {len(configs)} experiments, deadline {stop_time.strftime('%H:%M')}")
    print(f"Domains: {domains}")

    completed = 0
    kept = 0
    failed = 0

    for cfg in configs:
        if datetime.now() >= stop_time:
            print(f"Deadline reached after {completed} experiments.")
            break

        print(f"\n--- {cfg['name']}: quality={cfg['min_quality']} "
              f"domain={cfg['domain_filter'] or 'all'} "
              f"recency={cfg['recency_days'] or 'all'} "
              f"rank={cfg['rank']} lr={cfg['lr']} epochs={cfg['epochs']} ---")

        start = time.time()

        # Train (subprocess — loads/unloads its own model)
        try:
            adapter_dir = run_sft_training(
                adapter_name=cfg["name"],
                min_quality=cfg["min_quality"],
                domain_filter=cfg["domain_filter"],
                recency_days=cfg["recency_days"],
                rank=cfg["rank"],
                lr=cfg["lr"],
                epochs=cfg["epochs"],
            )
        except RuntimeError as e:
            print(f"  FAILED: {e}")
            failed += 1
            completed += 1
            continue

        # Eval (loads model+adapter, generates all responses, frees memory)
        scorecard = run_evaluation(
            adapter_name=cfg["name"],
            adapter_path=adapter_dir,
            domain=cfg["domain_filter"],
            num_prompts=num_eval_prompts,
        )

        # Guard: failed eval → skip, don't let it become "best"
        if "error" in scorecard:
            print(f"  EVAL FAILED: {scorecard['error']}")
            failed += 1
            completed += 1
            continue

        # Compare
        comparison = {"decision": "keep", "reasoning": "No baseline."}
        if best_scorecard:
            comparison = compare_scorecards(scorecard, best_scorecard)

        # Log with real configs
        training_config = {"rank": cfg["rank"], "lr": cfg["lr"], "epochs": cfg["epochs"]}
        data_config = {
            "min_quality": cfg["min_quality"],
            "domain_filter": cfg["domain_filter"],
            "recency_days": cfg["recency_days"],
        }
        exp_id = log_experiment(
            scorecard=scorecard,
            comparison=comparison,
            training_config=training_config,
            data_config=data_config,
            previous_experiment_id=baseline_id,
        )
        save_scorecard(scorecard, comparison)

        duration = time.time() - start
        print(f"  {comparison['decision'].upper()} ({duration:.0f}s)")

        if comparison["decision"] == "keep":
            best_scorecard = scorecard
            best_config = cfg
            baseline_id = exp_id
            kept += 1
            print(f"  >>> New best")

        completed += 1

    if not best_config:
        print(f"\nNo successful experiments out of {completed} runs ({failed} failed).")
        return

    # GGUF conversion on winner only
    winner_dir = config.adapters_dir / best_config["name"] / "sft"
    gguf_path = convert_adapter_to_gguf(winner_dir)
    print(f"Winner GGUF: {gguf_path}")

    # Cleanup non-best adapters
    for cfg in configs[:completed]:
        if cfg != best_config:
            d = config.adapters_dir / cfg["name"]
            if d.exists():
                shutil.rmtree(d)

    # Rename winner to canonical name so downstream consumers find it
    final_dir = config.adapters_dir / "general-v1"
    if final_dir.exists():
        shutil.rmtree(final_dir)
    (config.adapters_dir / best_config["name"]).rename(final_dir)
    print(f"Winner promoted to {final_dir}")

    print(f"\nDone: {completed} run, {kept} kept, {failed} failed")
    print(f"Best: {best_config}")


def main():
    """CLI entry point for Stage 3 SFT training."""
    import argparse

    parser = argparse.ArgumentParser(description="BESPOKE Stage 3: SFT Training")
    parser.add_argument("--domain", type=str, help="Filter training data by domain")
    parser.add_argument("--adapter-name", type=str, default="general-v1", help="Name for the adapter")
    args = parser.parse_args()

    print("BESPOKE Stage 3: SFT Training")
    print("=" * 50)

    adapter_dir = run_sft_training(
        domain_filter=args.domain,
        adapter_name=args.adapter_name,
    )

    print(f"\nAdapter saved to: {adapter_dir}")
    print(f"To test: mlx_lm.generate --model {config.base_model.training_model_path} --adapter-path {adapter_dir} --prompt 'test'")


if __name__ == "__main__":
    main()
