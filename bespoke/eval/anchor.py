"""Benchmark-interview anchor: the user's STATED 'good/bad' examples become extra seed
labels in the same embedding space. This is the THIRD source of the silver thread
(stated preference), fused with revealed (accept/reject) and emergent (clusters). It
gives the geometry a north pole during COLD START, before enough behavioral labels exist.

Add an optional `anchor_examples: {good: [...], bad: [...]}` section to benchmark.yaml.
"""
import numpy as np


def extract_anchor_examples(benchmark):
    """Return (texts, labels) from benchmark['benchmark']['anchor_examples']."""
    section = benchmark.get("benchmark", {}).get("anchor_examples", {}) or {}
    good = list(section.get("good", []) or [])
    bad = list(section.get("bad", []) or [])
    texts = good + bad
    labels = [1] * len(good) + [0] * len(bad)
    return texts, labels


def embed_anchor(texts, labels, embedding_svc=None):
    """Embed anchor texts into (X, y) seed arrays. Returns ((0,768), (0,)) if empty."""
    if not texts:
        return np.zeros((0, 768), dtype=np.float32), np.array([], dtype=int)
    if embedding_svc is None:
        from bespoke.capture.embeddings import EmbeddingService
        embedding_svc = EmbeddingService.get()
    vecs = embedding_svc.embed_batch(texts, prefix="document")
    return np.vstack(vecs).astype(np.float32), np.array(labels, dtype=int)
