"""Embedding computation using EmbeddingGemma 300M ONNX."""

import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np
from typing import List, Optional, Tuple

from bespoke.config import config

CHUNK_SIZE = 2048
CHUNK_OVERLAP = 200


class EmbeddingService:
    """Singleton embedding service. Load once, embed many."""

    _instance: Optional["EmbeddingService"] = None

    def __init__(self):
        model_dir = config.embedding.model_path
        onnx_path = config.embedding.onnx_path

        self.session = ort.InferenceSession(str(onnx_path))
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        # We handle chunking ourselves, so suppress length warnings from the tokenizer
        self.tokenizer.model_max_length = 1_000_000
        self.dimension = config.embedding.dimension

        # Prefix templates per EmbeddingGemma spec
        self.prefixes = {
            "query": "task: search result | query: ",
            "document": "title: none | text: ",
        }

    @classmethod
    def get(cls) -> "EmbeddingService":
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def unload(cls):
        """Release the model from memory."""
        if cls._instance is not None:
            del cls._instance.session
            cls._instance = None

    def _embed_single(self, text: str, prefix: str = "document") -> np.ndarray:
        """Embed a single text (must fit in context). Returns 768-dim float32 array."""
        prefixed = self.prefixes.get(prefix, "") + text
        inputs = self.tokenizer(
            prefixed,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=CHUNK_SIZE,
        )
        outputs = self.session.run(None, {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        })
        # outputs[1] is sentence_embedding (768-dim pooled), outputs[0] is last_hidden_state
        return outputs[1][0].astype(np.float32)

    def _chunk(self, text: str) -> List[str]:
        """Split text into overlapping token chunks, decoded back to text.

        If text fits in one chunk, returns it directly.
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= CHUNK_SIZE:
            return [text]

        chunks = []
        stride = CHUNK_SIZE - CHUNK_OVERLAP
        for start in range(0, len(tokens), stride):
            chunk_tokens = tokens[start:start + CHUNK_SIZE]
            if not chunk_tokens:
                break
            chunks.append(self.tokenizer.decode(chunk_tokens))
            if start + CHUNK_SIZE >= len(tokens):
                break

        return chunks

    def embed(self, text: str, prefix: str = "document") -> Tuple[np.ndarray, int]:
        """Embed text, chunking if needed.

        Returns (768-dim float32 array, num_chunks).
        num_chunks=1 means no chunking was needed.
        """
        chunks = self._chunk(text)

        if len(chunks) == 1:
            embedding = self._embed_single(chunks[0], prefix)
            return embedding, 1

        # Embed each chunk, average, L2-normalize
        chunk_embeddings = np.stack([self._embed_single(c, prefix) for c in chunks])
        avg = np.mean(chunk_embeddings, axis=0).astype(np.float32)
        avg = avg / np.linalg.norm(avg)
        return avg, len(chunks)

    def embed_batch(self, texts: List[str], prefix: str = "document") -> List[np.ndarray]:
        """Embed multiple texts. Returns list of 768-dim float32 arrays."""
        return [self._embed_single(t, prefix) for t in texts]
