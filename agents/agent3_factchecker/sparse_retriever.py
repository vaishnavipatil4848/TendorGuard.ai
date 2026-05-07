"""
sparse_retriever.py
Agent 3 — Fact-Checker Agent

BM25-based sparse keyword retrieval using rank_bm25.
Critical for exact matching of numbers, registration codes,
rupee amounts, GSTIN, PAN, dates — things semantic search misses.
"""

import re
import logging
from typing import List, Dict, Any, Optional

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class SparseRetriever:
    """
    BM25 keyword retrieval over a bidder's chunks.

    Tokenizer is optimized for Indian government document patterns:
    - Preserves numeric tokens (amounts, years, codes)
    - Normalizes currency symbols (₹ → rs)
    - Handles Devanagari script tokens
    """

    def __init__(self):
        self.corpus: List[Dict[str, Any]] = []
        self.bm25:   Optional[BM25Okapi]  = None

    def index(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Build BM25 index from a list of chunks.
        Call this after MetadataFilter to index only relevant chunks.

        Args:
            chunks: filtered chunk list
        """
        self.corpus = chunks
        tokenized   = [self._tokenize(c["text"]) for c in chunks]
        self.bm25   = BM25Okapi(tokenized)
        logger.debug(f"SparseRetriever: BM25 index built on {len(chunks)} chunks")

    def retrieve(
        self,
        query: str,
        top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k chunks by BM25 score.

        Args:
            query: criterion name + description
            top_k: number of results to return

        Returns:
            List of chunk dicts with 'sparse_score' added,
            sorted descending by BM25 score
        """
        if not self.bm25 or not self.corpus:
            logger.warning("SparseRetriever: index is empty — skipping")
            return []

        tokenized_query = self._tokenize(query)
        scores          = self.bm25.get_scores(tokenized_query)

        scored = [
            {**chunk, "sparse_score": round(float(score), 4)}
            for chunk, score in zip(self.corpus, scores)
        ]

        scored.sort(key=lambda x: x["sparse_score"], reverse=True)
        top = scored[:top_k]

        logger.debug(
            f"SparseRetriever: top score = "
            f"{top[0]['sparse_score']:.4f}" if top else
            "SparseRetriever: no results"
        )
        return top

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25.

        Preserves:
        - Numbers and decimal values
        - Registration codes (GSTIN, PAN, CIN)
        - Currency amounts after normalization
        - Devanagari word tokens
        """
        text = text.lower()

        # normalize currency symbols
        text = re.sub(r'₹\s*', 'rs ', text)
        text = re.sub(r'rs\.?\s*', 'rs ', text)

        # normalize common abbreviations
        text = re.sub(r'\bcr(?:ore)?s?\b', 'crore', text)
        text = re.sub(r'\blakh\b|\blac\b', 'lakh', text)

        # extract all alphanumeric tokens (captures both Latin and numbers)
        latin_tokens = re.findall(r'[a-z0-9]+', text)

        # extract Devanagari word tokens separately
        devanagari_tokens = re.findall(r'[\u0900-\u097F]+', text)

        return latin_tokens + devanagari_tokens