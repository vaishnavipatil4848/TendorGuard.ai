"""
embedder.py
Agent 3 — Fact-Checker Agent

Dual-mode text embedder:
  Primary   — OpenAI text-embedding-3-large  (3072-dim)
  Fallback  — sentence-transformers BAAI/bge-large-en-v1.5  (1024-dim)

Responsibilities:
  - Embed query strings for dense retrieval scoring
  - Batch-embed chunk lists before indexing into ChromaDB
  - Provide a simple in-memory LRU cache to avoid re-embedding
    identical strings across multiple criteria in the same run

Design notes:
  - ChromaDB's DenseRetriever handles its own embedding internally
    via the OpenAIEmbeddingFunction — this module is used for
    *standalone* embedding needs (e.g. cross-encoder pre-filtering,
    similarity scoring, or future hybrid approaches).
  - Model choice is resolved at construction time; if OPENAI_API_KEY
    is absent the local model is used automatically.
"""

import logging
import os
from functools import lru_cache
from typing import List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# ── Model constants ───────────────────────────────────────────────────────────
OPENAI_EMBED_MODEL  = "text-embedding-3-large"
OPENAI_EMBED_DIM    = 3072

LOCAL_EMBED_MODEL   = "BAAI/bge-large-en-v1.5"
LOCAL_EMBED_DIM     = 1024

# Maximum tokens per OpenAI embedding request
OPENAI_MAX_TOKENS   = 8191

# Default batch size for local model inference
LOCAL_BATCH_SIZE    = 32


class TextEmbedder:
    """
    Unified embedding interface for Agent 3.

    Automatically selects:
      - OpenAI text-embedding-3-large  when OPENAI_API_KEY is set
      - BAAI/bge-large-en-v1.5 (local) otherwise

    Args:
        mode:       'openai' | 'local' | 'auto'  (default: 'auto')
        batch_size: batch size for local inference  (default: 32)
        cache_size: number of strings to keep in LRU embedding cache
    """

    def __init__(
        self,
        mode:       str = "auto",
        batch_size: int = LOCAL_BATCH_SIZE,
        cache_size: int = 512,
    ):
        self.batch_size = batch_size
        self._cache: dict = {}            # chunk_id → embedding
        self._cache_size  = cache_size
        self._cache_order: list = []      # LRU tracking

        self._mode = self._resolve_mode(mode)
        self._client = None               # OpenAI client
        self._local_model = None          # SentenceTransformer

        if self._mode == "openai":
            self._init_openai()
        else:
            self._init_local()

        logger.info(
            f"TextEmbedder initialised in '{self._mode}' mode "
            f"(dim={self.embedding_dim})"
        )

    # ── Initialisation ────────────────────────────────────────────────────────

    def _resolve_mode(self, mode: str) -> str:
        if mode == "openai":
            return "openai"
        if mode == "local":
            return "local"
        # auto: prefer OpenAI if key is present
        return "openai" if os.getenv("OPENAI_API_KEY") else "local"

    def _init_openai(self) -> None:
        try:
            from openai import OpenAI  # type: ignore
            self._client       = OpenAI()
            self.embedding_dim = OPENAI_EMBED_DIM
            logger.debug("OpenAI client ready for embeddings.")
        except ImportError as exc:
            raise ImportError(
                "openai package is required for OpenAI embedding mode. "
                "Install with: pip install openai"
            ) from exc

    def _init_local(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            logger.info(f"Loading local embedding model: {LOCAL_EMBED_MODEL}")
            self._local_model  = SentenceTransformer(LOCAL_EMBED_MODEL)
            self.embedding_dim = LOCAL_EMBED_DIM
            logger.debug("Local SentenceTransformer model loaded.")
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for local embedding mode. "
                "Install with: pip install sentence-transformers"
            ) from exc

    # ── Public API ────────────────────────────────────────────────────────────

    def embed_query(self, text: str) -> np.ndarray:
        """
        Embed a single query string.

        Args:
            text: criterion query string

        Returns:
            numpy array of shape (embedding_dim,), L2-normalised
        """
        text = text.strip()
        if not text:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        cached = self._cache_get(text)
        if cached is not None:
            return cached

        vec = self._embed_texts([text])[0]
        self._cache_set(text, vec)
        return vec

    def embed_chunks(
        self,
        chunks: List[dict],
        text_key: str = "text"
    ) -> List[np.ndarray]:
        """
        Embed a list of chunk dicts.

        Chunks are processed in batches for efficiency.
        Results are returned in the same order as the input.

        Args:
            chunks:   list of chunk dicts (from HierarchicalChunker)
            text_key: dict key containing the text to embed

        Returns:
            List of numpy arrays, one per chunk, each of shape (embedding_dim,)
        """
        if not chunks:
            return []

        texts       = [c.get(text_key, "").strip() for c in chunks]
        embeddings  = self._embed_texts_batched(texts)

        logger.info(
            f"TextEmbedder: embedded {len(chunks)} chunks "
            f"via '{self._mode}' mode"
        )
        return embeddings

    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """
        Embed a list of plain strings.

        Args:
            texts: list of text strings

        Returns:
            List of numpy arrays of shape (embedding_dim,)
        """
        return self._embed_texts_batched(texts)

    def similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Cosine similarity between two embedding vectors.

        Both vectors are expected to be L2-normalised (as returned
        by this class).  In that case this reduces to a dot product.

        Args:
            vec_a: embedding vector
            vec_b: embedding vector

        Returns:
            Cosine similarity score in [-1, 1]
        """
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    @property
    def mode(self) -> str:
        """Active embedding mode: 'openai' or 'local'."""
        return self._mode

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Embed a list of texts (no batching)."""
        if self._mode == "openai":
            return self._openai_embed(texts)
        return self._local_embed(texts)

    def _embed_texts_batched(self, texts: List[str]) -> List[np.ndarray]:
        """
        Embed texts in batches, checking cache per-item.
        Empty strings are replaced with zero vectors.
        """
        results   = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts:   List[str] = []

        for i, text in enumerate(texts):
            if not text:
                results[i] = np.zeros(self.embedding_dim, dtype=np.float32)
                continue
            cached = self._cache_get(text)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        # batch-process uncached texts
        if uncached_texts:
            batch_vecs = []
            for start in range(0, len(uncached_texts), self.batch_size):
                batch = uncached_texts[start: start + self.batch_size]
                batch_vecs.extend(self._embed_texts(batch))

            for idx, (text, vec) in zip(uncached_indices, zip(uncached_texts, batch_vecs)):
                self._cache_set(text, vec)
                results[idx] = vec

        return results  # type: ignore[return-value]

    def _openai_embed(self, texts: List[str]) -> List[np.ndarray]:
        """Call OpenAI Embeddings API and return L2-normalised vectors."""
        try:
            # Truncate texts that may exceed token limit
            safe_texts = [t[:OPENAI_MAX_TOKENS * 4] for t in texts]  # rough char cap

            response = self._client.embeddings.create(
                model=OPENAI_EMBED_MODEL,
                input=safe_texts
            )
            vecs = [
                self._normalise(np.array(item.embedding, dtype=np.float32))
                for item in response.data
            ]
            return vecs

        except Exception as exc:
            logger.error(f"OpenAI embedding call failed: {exc}")
            # Return zero vectors on failure to allow pipeline to continue
            return [np.zeros(self.embedding_dim, dtype=np.float32)] * len(texts)

    def _local_embed(self, texts: List[str]) -> List[np.ndarray]:
        """Embed using local SentenceTransformer model."""
        try:
            # bge models benefit from an instruction prefix for retrieval
            prefixed = [f"Represent this sentence for searching relevant passages: {t}"
                        for t in texts]
            vecs_raw = self._local_model.encode(
                prefixed,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            return [np.array(v, dtype=np.float32) for v in vecs_raw]

        except Exception as exc:
            logger.error(f"Local embedding failed: {exc}")
            return [np.zeros(self.embedding_dim, dtype=np.float32)] * len(texts)

    @staticmethod
    def _normalise(vec: np.ndarray) -> np.ndarray:
        """L2-normalise a vector; returns zero-vector if norm is zero."""
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    # ── LRU Cache helpers ─────────────────────────────────────────────────────

    def _cache_get(self, key: str) -> Optional[np.ndarray]:
        if key in self._cache:
            # move to end (most recently used)
            self._cache_order.remove(key)
            self._cache_order.append(key)
            return self._cache[key]
        return None

    def _cache_set(self, key: str, value: np.ndarray) -> None:
        if key in self._cache:
            self._cache_order.remove(key)
        elif len(self._cache) >= self._cache_size:
            # evict least recently used
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]
        self._cache[key]  = value
        self._cache_order.append(key)

    def clear_cache(self) -> None:
        """Clear the in-memory embedding cache."""
        self._cache.clear()
        self._cache_order.clear()
        logger.debug("TextEmbedder: cache cleared.")
