"""
reranker.py
Agent 3 — Fact-Checker Agent

bge-reranker-large cross-encoder reranking.
Runs on top-20 RRF-fused results → returns top-5.
Significant accuracy boost — kept despite latency cost.
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

RERANKER_MODEL = "BAAI/bge-reranker-large"
RERANK_TOP_K   = 5


class CrossEncoderReranker:
    """
    bge-reranker-large cross-encoder.
    Scores each (query, chunk) pair jointly — much more accurate
    than bi-encoder similarity for passage ranking.

    Kept at full accuracy (not flashrank) as per team decision.
    """

    def __init__(self, model_name: str = RERANKER_MODEL):
        logger.info(f"Loading reranker: {model_name}")
        try:
            from FlagEmbedding import FlagReranker
            self.reranker = FlagReranker(model_name, use_fp16=True)
            self._available = True
            logger.info("Reranker loaded successfully")
        except ImportError:
            logger.warning(
                "FlagEmbedding not installed — reranker disabled. "
                "Install with: pip install FlagEmbedding"
            )
            self.reranker   = None
            self._available = False

    def rerank(
        self,
        query:  str,
        chunks: List[Dict[str, Any]],
        top_k:  int = RERANK_TOP_K
    ) -> List[Dict[str, Any]]:
        """
        Rerank chunks using cross-encoder scoring.

        Args:
            query:  criterion name + description
            chunks: RRF-fused chunks (typically top-20)
            top_k:  number of final results to return

        Returns:
            Top-k chunks sorted by reranker_score descending,
            with 'reranker_score' added
        """
        if not chunks:
            return []

        if not self._available:
            logger.warning("Reranker unavailable — returning RRF order")
            for chunk in chunks:
                chunk["reranker_score"] = chunk.get("rrf_score", 0.0)
            return chunks[:top_k]

        pairs = [[query, c["text"]] for c in chunks]

        try:
            scores = self.reranker.compute_score(pairs, normalize=True)

            # compute_score returns a float for single pair
            if isinstance(scores, float):
                scores = [scores]

            for chunk, score in zip(chunks, scores):
                chunk["reranker_score"] = round(float(score), 4)

            reranked = sorted(
                chunks,
                key=lambda x: x.get("reranker_score", 0.0),
                reverse=True
            )

            top = reranked[:top_k]
            logger.debug(
                f"CrossEncoderReranker: top score = "
                f"{top[0].get('reranker_score', 0):.4f}" if top else
                "CrossEncoderReranker: no results"
            )
            return top

        except Exception as e:
            logger.error(f"Reranking failed: {e} — falling back to RRF order")
            for chunk in chunks:
                chunk["reranker_score"] = chunk.get("rrf_score", 0.0)
            return chunks[:top_k]