"""
rrf_fusion.py
Agent 3 — Fact-Checker Agent

Reciprocal Rank Fusion — combines dense and sparse retrieval results.
Standard k=60 as per the original RRF paper (Cormack et al., 2009).
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class RRFFusion:
    """
    Fuses dense (ChromaDB) and sparse (BM25) retrieval results
    using Reciprocal Rank Fusion scoring.

    RRF score for document d:
        RRF(d) = Σ 1 / (k + rank(d))
    where rank(d) is the document's position in each ranked list.
    """

    def __init__(self, k: int = 60):
        self.k = k

    def fuse(
        self,
        dense_results:  List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Fuse two ranked lists into one using RRF.

        Args:
            dense_results:  ranked list from DenseRetriever
            sparse_results: ranked list from SparseRetriever

        Returns:
            Merged list sorted by RRF score descending,
            with 'rrf_score' added to each chunk
        """
        rrf_scores: Dict[str, float] = {}
        chunk_map:  Dict[str, Dict]  = {}

        # accumulate RRF scores from dense list
        for rank, chunk in enumerate(dense_results):
            cid = chunk["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + \
                              1.0 / (self.k + rank + 1)
            chunk_map[cid]  = chunk

        # accumulate RRF scores from sparse list
        for rank, chunk in enumerate(sparse_results):
            cid = chunk["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + \
                              1.0 / (self.k + rank + 1)
            if cid not in chunk_map:
                chunk_map[cid] = chunk

        # sort by combined RRF score
        sorted_ids = sorted(
            rrf_scores, key=lambda x: rrf_scores[x], reverse=True
        )

        fused = []
        for cid in sorted_ids:
            chunk = chunk_map[cid].copy()
            chunk["rrf_score"] = round(rrf_scores[cid], 6)
            fused.append(chunk)

        logger.debug(
            f"RRFFusion: {len(dense_results)} dense + "
            f"{len(sparse_results)} sparse → {len(fused)} unique chunks"
        )
        return fused