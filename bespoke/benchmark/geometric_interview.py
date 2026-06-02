"""Geometric benchmark interview — warm, deterministic, LOCAL (no LLM).

Static pre-recorded questions. The user answers in free text; we embed each answer and RETRIEVE
the nearest real interactions from their own warehouse — turning muddy stated preferences into
clean, commensurable anchors that already exist in their behavior. The gap between what they SAY
(stated) and what they ACCEPT (revealed) is itself a signal.

Output: anchor_examples + a stated-vs-revealed alignment score → benchmark.yaml (consumed by the
geometric eval's anchor seeds). First experience should feel like someone curious about your work,
not a form.
"""
import numpy as np
import yaml
from pathlib import Path

from sqlite_vec import serialize_float32

from bespoke.config import config

ACCEPT = {"accept", "strong_accept"}

QUESTIONS = [
    {"id": "domains", "type": "open",
     "prompt": "To start — what kind of work do you find yourself doing most? What problems light you up?"},
    {"id": "good", "type": "good_example",
     "prompt": "When a model gives you a *great* answer, what does that look like? Paste or describe one you loved."},
    {"id": "bad", "type": "bad_example",
     "prompt": "And one that fell flat — what frustrated you? Paste or describe it."},
    {"id": "values", "type": "open",
     "prompt": "What matters most in *how* an answer is delivered — speed, depth, directness, something else?"},
]


def retrieve_matches(query_emb, conn, k=5):
    """k nearest interactions to query_emb (np 768) via sqlite-vec KNN.

    Returns list of {id, distance, user_message, assistant_response, feedback_class}.
    """
    blob = serialize_float32(np.asarray(query_emb, np.float32).tolist())
    rows = conn.execute("""
        SELECT i.id AS id, m.distance AS distance,
               i.user_message AS um, i.assistant_response AS ar, i.feedback_class AS fc
        FROM (
            SELECT rowid, distance FROM vec_interactions
            WHERE interaction_embedding MATCH ? AND k = ?
            ORDER BY distance
        ) m
        JOIN interactions i ON i.id = m.rowid
        ORDER BY m.distance
    """, (blob, k)).fetchall()
    return [{"id": r["id"], "distance": r["distance"], "user_message": r["um"],
             "assistant_response": r["ar"], "feedback_class": r["fc"]} for r in rows]


def alignment_score(matches):
    """Stated-vs-revealed: fraction of retrieved matches the user actually ACCEPTED.

    High → the stated preference aligns with real behavior (strong anchor). Low → a gap (the
    user says one thing but accepts another) or noise — surfaced as a calibration signal.
    """
    if not matches:
        return 0.0
    return sum(1 for m in matches if m.get("feedback_class") in ACCEPT) / len(matches)


def build_anchor_examples(good_texts, bad_texts):
    """Shape collected good/bad example texts into the anchor_examples structure."""
    return {"good": [t.strip() for t in good_texts if t and t.strip()],
            "bad": [t.strip() for t in bad_texts if t and t.strip()]}


def write_geometric_benchmark(anchor_examples, alignment, path=None):
    """Write benchmark.yaml with anchor_examples + the stated-vs-revealed alignment score."""
    path = Path(path) if path else config.benchmark_dir / "benchmark.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"benchmark": {
        "version": 1,
        "source": "geometric_interview",
        "anchor_examples": anchor_examples,
        "stated_vs_revealed_alignment": round(float(alignment), 3),
    }}
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def run_geometric_interview(conn=None, input_fn=input, output_fn=print,
                            embedding_svc=None, benchmark_path=None):
    """Warm, deterministic, local interview. input_fn/output_fn/embedding_svc are injectable
    for testing. Returns the path to the written benchmark.yaml.
    """
    close = conn is None
    if conn is None:
        from bespoke.db.init import get_connection
        conn = get_connection()
    if embedding_svc is None:
        from bespoke.capture.embeddings import EmbeddingService
        embedding_svc = EmbeddingService.get()

    output_fn("\nA few questions so BESPOKE learns what *good* looks like for you.\n"
              "Answer however much you like — there are no wrong answers.\n")

    good_texts, bad_texts, alignments = [], [], []
    for q in QUESTIONS:
        answer = (input_fn(f"{q['prompt']}\n> ") or "").strip()
        if not answer:
            continue
        if q["type"] in ("good_example", "bad_example"):
            emb, _ = embedding_svc.embed(answer)
            matches = retrieve_matches(emb, conn, k=3)
            if matches:
                top = matches[0]
                output_fn(f"\n  …something like this, from your own history?\n"
                          f"  “{(top['user_message'] or '')[:120]}” → "
                          f"“{(top['assistant_response'] or '')[:120]}”\n")
                alignments.append(alignment_score(matches))
            (good_texts if q["type"] == "good_example" else bad_texts).append(answer)

    anchors = build_anchor_examples(good_texts, bad_texts)
    overall_alignment = float(np.mean(alignments)) if alignments else 0.0
    path = write_geometric_benchmark(anchors, overall_alignment, path=benchmark_path)

    output_fn(f"\nThanks. Saved your quality anchors → {path}")
    output_fn(f"Stated-vs-revealed alignment: {overall_alignment:.0%} "
              f"({'your words match your behavior' if overall_alignment >= 0.5 else 'some gap between what you said and what you accept — we will trust behavior'}).")
    if close:
        conn.close()
    return path
