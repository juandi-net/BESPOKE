"""Embedding via MLX on the Apple Silicon GPU — EmbeddingGemma 300M.

Uses mlx-embeddings (Prince Canuma) with the MLX port of EmbeddingGemma, running on the M4 GPU.
~8x faster than the old ONNX-CPU path and truly batched, unifying the stack with MLX
training/inference. Long texts are truncated to the model's 2048-token context (the old manual
chunk-and-average was dropped — the first 2048 tokens carry the signal for retrieval/clustering).
"""
import numpy as np
from typing import List, Optional, Tuple

from bespoke.config import config

CONTEXT = 2048   # EmbeddingGemma context window
BATCH = 32       # texts per MLX forward pass


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
        """Embed many texts efficiently (batched on the GPU). Returns list of 768-dim arrays."""
        results: List[np.ndarray] = []
        for s in range(0, len(texts), BATCH):
            results.extend(self._embed_forward(texts[s:s + BATCH], prefix))
        return [r.astype(np.float32) for r in results]

    def embed_batch(self, texts: List[str], prefix: str = "document") -> List[np.ndarray]:
        """Alias for embed_many (kept for backward compatibility)."""
        return self.embed_many(texts, prefix)
