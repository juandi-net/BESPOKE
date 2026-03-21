"""Stage 2b: Cross-corpus pattern mining (weekly).

Reads across the full curated warehouse, identifies meta-patterns,
retroactively tags existing examples, and produces curriculum weighting
recommendations. Uses Opus-class LLM for deep analysis.
"""

import json
from datetime import datetime
from typing import List, Dict, Optional

from bespoke.db.init import get_connection
from bespoke.teach.llm_client import chat_json
from bespoke.config import config


PATTERN_MINING_SYSTEM = """You are an expert at identifying recurring reasoning patterns across a corpus of AI interactions. Your job is to find meta-patterns — recurring problem-solving strategies, decision heuristics, conceptual frameworks, and common failure modes that appear across multiple interactions.

You will receive a batch of classified interactions grouped by domain. For each batch:

1. Identify recurring reasoning patterns (minimum 3 occurrences to count)
2. Document each pattern: name, trigger conditions, reasoning steps
3. For each interaction, list which patterns it demonstrates
4. Identify gaps — reasoning types that are underrepresented

Output JSON:
{
    "patterns": [
        {
            "name": "pattern_snake_case_name",
            "description": "What this pattern is",
            "trigger_conditions": "When this pattern typically appears",
            "reasoning_steps": "The sequence of reasoning steps",
            "occurrence_count": 5,
            "domains": ["code", "strategy"]
        }
    ],
    "interaction_tags": [
        {
            "interaction_id": 123,
            "patterns": ["pattern_name_1", "pattern_name_2"]
        }
    ],
    "gaps": [
        {
            "domain": "code",
            "missing_pattern": "constraint_identification",
            "recommendation": "Seek more frontier interactions involving constraint-heavy problems"
        }
    ],
    "curriculum_weights": [
        {
            "interaction_id": 123,
            "weight": 1.5,
            "reason": "Demonstrates rare pattern combination"
        }
    ]
}

Output ONLY valid JSON."""


def get_corpus_batch(domain: Optional[str] = None, limit: int = 200) -> List[dict]:
    """Get a batch of classified interactions for pattern mining."""
    conn = get_connection()

    query = """
        SELECT id, user_message, assistant_response, domain,
               quality_score, reasoning_primitives, feedback_class
        FROM interactions
        WHERE processed_2a_at IS NOT NULL
        AND quality_score IN ('high', 'medium')
    """
    params = []

    if domain:
        query += " AND domain = ?"
        params.append(domain)

    query += " ORDER BY captured_at DESC LIMIT ?"
    params.append(limit)

    rows = conn.execute(query, params).fetchall()
    conn.close()

    return [dict(row) for row in rows]


def mine_patterns(interactions: List[dict]) -> dict:
    """Send a batch of interactions to Opus for pattern mining."""
    if not interactions:
        return {"patterns": [], "interaction_tags": [], "gaps": [], "curriculum_weights": []}

    # Format interactions for the prompt (truncate to fit context)
    batch_text = ""
    for ix in interactions[:50]:  # Cap at 50 to fit context window
        batch_text += f"""
--- Interaction {ix['id']} (domain: {ix['domain']}, quality: {ix['quality_score']}) ---
User: {ix['user_message'][:300]}
Assistant: {ix['assistant_response'][:300]}
Reasoning primitives: {ix.get('reasoning_primitives', '[]')}
Feedback: {ix.get('feedback_class', 'unknown')}
"""

    messages = [
        {"role": "system", "content": PATTERN_MINING_SYSTEM},
        {"role": "user", "content": f"Analyze these {len(interactions[:50])} interactions for meta-patterns:\n{batch_text}"},
    ]

    return chat_json(
        messages=messages,
        model=config.llm.model_stage_2b,
        max_tokens=8192,
    )


def update_meta_patterns(conn, patterns: List[dict]) -> int:
    """Insert or update meta-patterns in the registry."""
    count = 0
    for pattern in patterns:
        existing = conn.execute(
            "SELECT id, occurrence_count FROM meta_patterns WHERE name = ?",
            (pattern["name"],)
        ).fetchone()

        if existing:
            conn.execute("""
                UPDATE meta_patterns SET
                    description = ?,
                    trigger_conditions = ?,
                    reasoning_steps = ?,
                    occurrence_count = occurrence_count + ?,
                    domains = ?,
                    updated_at = ?
                WHERE name = ?
            """, (
                pattern.get("description", ""),
                pattern.get("trigger_conditions", ""),
                pattern.get("reasoning_steps", ""),
                pattern.get("occurrence_count", 0),
                json.dumps(pattern.get("domains", [])),
                datetime.now().isoformat(),
                pattern["name"],
            ))
        else:
            conn.execute("""
                INSERT INTO meta_patterns (
                    name, description, trigger_conditions,
                    reasoning_steps, occurrence_count, domains
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                pattern["name"],
                pattern.get("description", ""),
                pattern.get("trigger_conditions", ""),
                pattern.get("reasoning_steps", ""),
                pattern.get("occurrence_count", 0),
                json.dumps(pattern.get("domains", [])),
            ))
            count += 1

    return count


def apply_retroactive_tags(conn, interaction_tags: List[dict]) -> int:
    """Tag interactions with discovered meta-patterns."""
    count = 0
    for tag in interaction_tags:
        iid = tag.get("interaction_id")
        patterns = tag.get("patterns", [])
        if iid and patterns:
            conn.execute("""
                UPDATE interactions SET
                    meta_patterns = ?,
                    processed_2b_at = ?
                WHERE id = ?
            """, (
                json.dumps(patterns),
                datetime.now().isoformat(),
                iid,
            ))
            count += 1
    return count


def apply_curriculum_weights(conn, weights: List[dict]) -> int:
    """Update curriculum weights for interactions."""
    count = 0
    for w in weights:
        iid = w.get("interaction_id")
        weight = w.get("weight", 1.0)
        if iid:
            conn.execute(
                "UPDATE interactions SET curriculum_weight = ? WHERE id = ?",
                (weight, iid)
            )
            count += 1
    return count


def run_stage_2b() -> dict:
    """Run Stage 2b cross-corpus pattern mining."""
    conn = get_connection()

    stats = {
        "domains_processed": 0,
        "patterns_found": 0,
        "interactions_tagged": 0,
        "weights_updated": 0,
        "gaps_found": 0,
    }

    # Process each domain separately for focused analysis
    domains = conn.execute("""
        SELECT DISTINCT domain FROM interactions
        WHERE domain IS NOT NULL AND processed_2a_at IS NOT NULL
    """).fetchall()

    all_gaps = []

    for domain_row in domains:
        domain = domain_row["domain"]
        print(f"\nMining patterns for domain: {domain}")

        interactions = get_corpus_batch(domain=domain, limit=200)
        if len(interactions) < 5:
            print(f"  Skipping — only {len(interactions)} interactions")
            continue

        stats["domains_processed"] += 1

        result = mine_patterns(interactions)

        # Update pattern registry
        new_patterns = update_meta_patterns(conn, result.get("patterns", []))
        stats["patterns_found"] += len(result.get("patterns", []))

        # Apply retroactive tags
        tagged = apply_retroactive_tags(conn, result.get("interaction_tags", []))
        stats["interactions_tagged"] += tagged

        # Apply curriculum weights
        weighted = apply_curriculum_weights(conn, result.get("curriculum_weights", []))
        stats["weights_updated"] += weighted

        # Collect gaps
        gaps = result.get("gaps", [])
        all_gaps.extend(gaps)
        stats["gaps_found"] += len(gaps)

        conn.commit()

    # Update pipeline state
    conn.execute("""
        UPDATE pipeline_state
        SET last_successful_run = ?, metadata = ?
        WHERE stage = 'stage2b'
    """, (datetime.now().isoformat(), json.dumps(stats)))

    conn.commit()
    conn.close()

    # Print gap analysis
    if all_gaps:
        print("\n--- GAP ANALYSIS ---")
        for gap in all_gaps:
            print(f"  [{gap.get('domain', '?')}] Missing: {gap.get('missing_pattern', '?')}")
            print(f"    Recommendation: {gap.get('recommendation', '')}")

    return stats


def main():
    """CLI entry point for Stage 2b."""
    print("BESPOKE Stage 2b: Cross-Corpus Pattern Mining")
    print("=" * 50)

    stats = run_stage_2b()

    print(f"\nResults:")
    print(f"  Domains processed: {stats['domains_processed']}")
    print(f"  Patterns found: {stats['patterns_found']}")
    print(f"  Interactions tagged: {stats['interactions_tagged']}")
    print(f"  Curriculum weights updated: {stats['weights_updated']}")
    print(f"  Gaps identified: {stats['gaps_found']}")


if __name__ == "__main__":
    main()
