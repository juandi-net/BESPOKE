"""V0 Contrastive Benchmark Extraction -- analyze accept/reject patterns."""

import json
from bespoke.db.init import get_connection
from bespoke.capture.embeddings import EmbeddingService
from bespoke.teach.llm_client import chat_json
from bespoke.config import config
from sqlite_vec import serialize_float32


CONTRASTIVE_SYSTEM = """You are analyzing pairs of AI interactions where one was accepted by the user and one was rejected, on similar topics. Your job is to identify what differentiates accepted from rejected output.

You will receive several accept/reject pairs. For each pair, identify the specific quality that made the accepted output better.

After analyzing all pairs, synthesize your findings into 3-5 quality dimensions that consistently differentiate accepted from rejected output.

Output JSON:
{
    "pair_analyses": [
        {
            "pair_id": 1,
            "differentiator": "What made the accepted output better",
            "dimension": "Short name for the quality dimension"
        }
    ],
    "dimensions": [
        {
            "id": "short_snake_case",
            "name": "Human-readable name",
            "description": "What this dimension measures",
            "evidence_count": 5,
            "checks": [
                "Binary yes/no evaluation question",
                "Another binary check"
            ]
        }
    ]
}

Output ONLY valid JSON."""


def find_contrastive_pairs(limit: int = 20) -> list:
    """Find accepted/rejected interaction pairs on similar topics.

    Uses sqlite-vec embeddings to find pairs where:
    - One interaction was classified as accept/strong_accept
    - Another interaction on a similar topic was classified as reject/strong_reject
    """
    conn = get_connection()

    # Get accepted interactions with embeddings
    accepted = conn.execute("""
        SELECT id, user_message, assistant_response, domain, feedback_class
        FROM interactions
        WHERE feedback_class IN ('accept', 'strong_accept')
        AND quality_score IN ('high', 'medium')
        LIMIT 50
    """).fetchall()

    # Get rejected interactions
    rejected = conn.execute("""
        SELECT id, user_message, assistant_response, domain, feedback_class
        FROM interactions
        WHERE feedback_class IN ('reject', 'strong_reject')
        LIMIT 50
    """).fetchall()

    if not accepted or not rejected:
        print("Not enough classified accept/reject pairs for contrastive analysis.")
        print(f"  Accepted: {len(accepted)}, Rejected: {len(rejected)}")
        conn.close()
        return []

    # Use embeddings to find similar prompt pairs across accept/reject
    embedding_svc = EmbeddingService.get()
    pairs = []

    for rej in rejected[:limit]:
        rej_emb = embedding_svc.embed(rej["user_message"])

        # Find most similar accepted interaction
        results = conn.execute("""
            SELECT v.rowid, v.distance
            FROM vec_interactions v
            WHERE v.interaction_embedding MATCH ?
            ORDER BY v.distance
            LIMIT 10
        """, (serialize_float32(rej_emb.tolist()),)).fetchall()

        # Find the closest one that's in the accepted set
        accepted_ids = {a["id"] for a in accepted}
        for r in results:
            if r["rowid"] in accepted_ids:
                acc = next(a for a in accepted if a["id"] == r["rowid"])
                pairs.append({
                    "accepted": {
                        "user_message": acc["user_message"][:500],
                        "response": acc["assistant_response"][:500],
                        "domain": acc["domain"],
                    },
                    "rejected": {
                        "user_message": rej["user_message"][:500],
                        "response": rej["assistant_response"][:500],
                        "domain": rej["domain"],
                    },
                    "similarity": r["distance"],
                })
                break

    EmbeddingService.unload()
    conn.close()

    return pairs[:limit]


def extract_dimensions_from_pairs(pairs: list) -> dict:
    """Send contrastive pairs to Opus for dimension extraction."""
    if not pairs:
        return {"dimensions": []}

    # Format pairs for the prompt
    pairs_text = ""
    for i, pair in enumerate(pairs, 1):
        pairs_text += f"""
--- Pair {i} ---
ACCEPTED output:
  User asked: {pair['accepted']['user_message']}
  AI responded: {pair['accepted']['response']}

REJECTED output:
  User asked: {pair['rejected']['user_message']}
  AI responded: {pair['rejected']['response']}
"""

    messages = [
        {"role": "system", "content": CONTRASTIVE_SYSTEM},
        {"role": "user", "content": f"Analyze these {len(pairs)} accept/reject pairs:\n{pairs_text}"},
    ]

    return chat_json(
        messages=messages,
        model=config.llm.model_benchmark,
        max_tokens=4096,
    )


def run_extraction() -> dict:
    """Run the full contrastive extraction pipeline."""
    print("\nBESPOKE Contrastive Benchmark Extraction")
    print("=" * 50)

    pairs = find_contrastive_pairs(limit=15)
    print(f"Found {len(pairs)} contrastive pairs")

    if not pairs:
        return {"dimensions": []}

    result = extract_dimensions_from_pairs(pairs)
    print(f"Extracted {len(result.get('dimensions', []))} dimensions from data")

    return result


if __name__ == "__main__":
    result = run_extraction()
    print(json.dumps(result, indent=2))
