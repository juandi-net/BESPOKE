"""Prepare training data from the curated warehouse."""

import json
from pathlib import Path
from typing import List, Dict

from bespoke.db.init import get_connection
from bespoke.config import config


def export_sft_data(
    domain: str = None,
    min_quality: str = "medium",
    max_samples: int = None,
    recency_days: int = None,
    max_prompt_tokens: int = None,
    output_path: Path = None,
) -> Path:
    """Export SFT training pairs as JSONL in chat format for MLX.

    Returns the path to the exported file.
    """
    conn = get_connection()

    quality_levels = {"high": 3, "medium": 2, "low": 1, "exclude": 0}
    min_level = quality_levels.get(min_quality, 2)

    query = """
        SELECT tp.instruction, tp.response, tp.domain, tp.quality_score
        FROM training_pairs tp
        WHERE tp.pair_type = 'sft'
    """
    params = []

    if domain:
        query += " AND tp.domain = ?"
        params.append(domain)

    if recency_days:
        query += """ AND tp.interaction_id IN (
            SELECT id FROM interactions
            WHERE captured_at >= date('now', ? || ' days')
        )"""
        params.append(f"-{int(recency_days)}")

    # Quality ordering: high=3, medium=2, low=1
    query += """ ORDER BY CASE tp.quality_score
        WHEN 'high' THEN 3 WHEN 'medium' THEN 2 WHEN 'low' THEN 1 ELSE 0
    END DESC, tp.created_at DESC"""

    if max_samples:
        query += " LIMIT ?"
        params.append(int(max_samples))

    rows = conn.execute(query, params).fetchall()

    # Filter by quality level
    pairs = []
    for row in rows:
        level = quality_levels.get(row["quality_score"], 0)
        if level >= min_level:
            pairs.append({
                "messages": [
                    {"role": "user", "content": row["instruction"]},
                    {"role": "assistant", "content": row["response"]},
                ]
            })

    conn.close()

    # Drop pairs whose prompt alone fills the training window. With --mask-prompt and batch=1,
    # any pair whose prompt token-length >= max_seq has ZERO unmasked response tokens left after
    # truncation, so the loss divides by 0 -> NaN and kills the run. These over-long prompts are
    # also the observation-XML / tool-dump junk, not genuine reasoning, so this cleans data too.
    # mlx-lm computes the prompt offset as len(apply_chat_template(messages[:-1],
    # add_generation_prompt=True)) (mlx_lm/tuner/datasets.py ChatDataset); match it exactly.
    if max_prompt_tokens:
        from mlx_lm.utils import load_tokenizer
        tok = load_tokenizer(Path(config.base_model.training_model_path))
        kept = []
        for pair in pairs:
            offset = len(tok.apply_chat_template(
                pair["messages"][:-1], add_generation_prompt=True))
            if offset < max_prompt_tokens:
                kept.append(pair)
        dropped = len(pairs) - len(kept)
        if dropped:
            print(f"Filtered {dropped} pairs with prompt >= {max_prompt_tokens} tokens "
                  f"(NaN-safe + drops tool-dump junk); {len(kept)} remain")
        pairs = kept

    # Write JSONL
    if output_path is None:
        output_path = Path.home() / ".bespoke" / "training_data" / "train.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Split 90/10 train/eval
    split_idx = int(len(pairs) * 0.9)
    train_pairs = pairs[:split_idx]
    eval_pairs = pairs[split_idx:]

    train_path = output_path
    eval_path = output_path.with_name("valid.jsonl")

    for path, data in [(train_path, train_pairs), (eval_path, eval_pairs)]:
        with open(path, 'w') as f:
            for pair in data:
                f.write(json.dumps(pair) + "\n")

    print(f"Exported {len(train_pairs)} train + {len(eval_pairs)} eval SFT pairs")
    return train_path


def export_dpo_data(
    domain: str = None,
    output_path: Path = None,
) -> Path:
    """Export DPO preference pairs as JSONL for TRL/MLX DPO training."""
    conn = get_connection()

    query = """
        SELECT prompt, chosen, rejected, domain
        FROM training_pairs
        WHERE pair_type = 'dpo'
        AND chosen IS NOT NULL
        AND rejected IS NOT NULL
    """
    params = []

    if domain:
        query += " AND domain = ?"
        params.append(domain)

    rows = conn.execute(query, params).fetchall()

    pairs = []
    for row in rows:
        pairs.append({
            "prompt": row["prompt"],
            "chosen": row["chosen"],
            "rejected": row["rejected"],
        })

    conn.close()

    if output_path is None:
        output_path = Path.home() / ".bespoke" / "training_data" / "dpo_train.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"Exported {len(pairs)} DPO preference pairs")
    return output_path
