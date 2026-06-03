"""Embedding via MLX on the Apple Silicon GPU — EmbeddingGemma 300M.

Uses mlx-embeddings (Prince Canuma) with the MLX port of EmbeddingGemma, running on the M4 GPU.
~8x faster than the old ONNX-CPU path and truly batched, unifying the stack with MLX
training/inference. Long texts are truncated to the model's 2048-token context (the old manual
chunk-and-average was dropped — the first 2048 tokens carry the signal for retrieval/clustering).
"""
import numpy as np
from typing import List, Optional, Tuple

from bespoke.config import config

# Truncation cap: the user message + start of the answer carry the interaction's signal;
# trailing tool/code dumps are low-signal and blow up the forward-pass cost. 512 is ~4x cheaper
# than 2048 with little quality loss for retrieval/clustering/probe. Tunable.
CONTEXT = 512
BATCH = 64       # texts per MLX forward pass (length-bucketed so padding stays small)


class EmbeddingService:
    """Singleton MLX embedding service. Load once, embed many (batched on the GPU)."""

    _instance: Optional["EmbeddingService"] = None

    def __init__(self):
        from mlx_embeddings import load
        self.model, self.processor = load(config.embedding.model_id)
        self.dimension = config.embedding.dimension
        # EmbeddingGemma prompt prefixes (query vs document).
        self.prefixes = {
            "query": "task: search result | query: ",
            "document": "title: none | text: ",
        }

    @classmethod
    def get(cls) -> "EmbeddingService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def unload(cls):
        cls._instance = None

    def _embed_forward(self, texts: List[str], prefix: str) -> np.ndarray:
        """One batched MLX forward over texts (each truncated to context). Returns [N, 768]."""
        from mlx_embeddings.utils import prepare_inputs
        import mlx.core as mx

        prefixed = [self.prefixes.get(prefix, "") + (t or "") for t in texts]
        inputs = prepare_inputs(self.processor, None, prefixed, CONTEXT, True, True, None)
        out = self.model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
        mx.eval(out.text_embeds)
        return np.array(out.text_embeds, dtype=np.float32)

    def embed(self, text: str, prefix: str = "document") -> Tuple[np.ndarray, int]:
        """Embed one text. Returns (768-dim float32, num_chunks=1) — kept for interface parity."""
        emb = self._embed_forward([text], prefix)[0]
        return emb.astype(np.float32), 1

    def embed_many(self, texts: List[str], prefix: str = "document") -> List[np.ndarray]:
        """Embed many texts efficiently (length-bucketed batches on the GPU).

        Sorts by length so each batch pads only to its own max — a single long text no longer
        forces the whole batch to 2048. Returns list of 768-dim arrays in the original order.
        """
        n = len(texts)
        if n == 0:
            return []
        order = sorted(range(n), key=lambda i: len(texts[i] or ""))
        results: List[np.ndarray] = [None] * n
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            embs = self._embed_forward([texts[i] for i in idx], prefix)
            for j, i in enumerate(idx):
                results[i] = embs[j].astype(np.float32)
        return results

    def embed_batch(self, texts: List[str], prefix: str = "document") -> List[np.ndarray]:
        """Alias for embed_many (kept for backward compatibility)."""
        return self.embed_many(texts, prefix)
