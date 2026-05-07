"""
dense_retriever.py
Agent 3 — Fact-Checker Agent

ChromaDB-based dense semantic retrieval.
Uses OpenAI text-embedding-3-large for embeddings.
"""

import logging
from typing import List, Dict, Any

import chromadb
from chromadb.utils import embedding_functions

from agents.agent3_factchecker.embedder import TextEmbedder

logger = logging.getLogger(__name__)

COLLECTION_PREFIX = "tendor_bidder_"


class TendorGuardEmbeddingFunction:
    """
    Adapter to use our TextEmbedder as a ChromaDB EmbeddingFunction.
    """
    def __init__(self, embedder: TextEmbedder):
        self.embedder = embedder

    def __call__(self, input: List[str]) -> List[List[float]]:
        # embed_texts returns a list of numpy arrays
        vecs = self.embedder.embed_texts(input)
        return [v.tolist() for v in vecs]


class DenseRetriever:
    """
    Indexes and retrieves chunks using ChromaDB.
    
    Uses TextEmbedder which automatically selects between 
    OpenAI and Local (SentenceTransformers) based on API key presence.
    """

    def __init__(self, persist_dir: str = "./database/chroma_store"):
        self.client   = chromadb.PersistentClient(path=persist_dir)
        self.embedder = TextEmbedder(mode="auto")
        self.embed_fn = TendorGuardEmbeddingFunction(self.embedder)

    def index(
        self,
        chunks:    List[Dict[str, Any]],
        bidder_id: str
    ) -> None:
        """
        Index a bidder's chunks into ChromaDB.
        Deletes existing collection for this bidder first
        to prevent duplicate entries.

        Args:
            chunks:    list of chunk dicts from HierarchicalChunker
            bidder_id: unique bidder identifier
        """
        collection_name = f"{COLLECTION_PREFIX}{bidder_id}"

        # clean slate for this bidder
        try:
            self.client.delete_collection(collection_name)
        except Exception:
            pass

        collection = self.client.create_collection(
            name=collection_name,
            embedding_function=self.embed_fn
        )

        if not chunks:
            logger.warning(f"No chunks to index for bidder '{bidder_id}'")
            return

        collection.add(
            ids=[c["chunk_id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[{
                "page_number": c.get("page_number", 1),
                "line_start":  c.get("line_start", 1),
                "line_end":    c.get("line_end", 1),
                "doc_type":    c.get("doc_type", "PDF"),
                "chunk_level": c.get("chunk_level", "field"),
                "source_file": c.get("source_file", "")
            } for c in chunks]
        )

        logger.info(
            f"DenseRetriever: indexed {len(chunks)} chunks "
            f"for bidder '{bidder_id}'"
        )

    def retrieve(
        self,
        query:     str,
        bidder_id: str,
        top_k:     int = 20
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k chunks by semantic similarity.

        Args:
            query:     criterion name + description
            bidder_id: which bidder collection to search
            top_k:     number of results to return

        Returns:
            List of chunk dicts with 'dense_score' added
        """
        collection_name = f"{COLLECTION_PREFIX}{bidder_id}"

        try:
            collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embed_fn
            )

            n = min(top_k, collection.count())
            if n == 0:
                return []

            results = collection.query(
                query_texts=[query],
                n_results=n
            )

            chunks = []
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i]
                dist = results["distances"][0][i]
                chunks.append({
                    "chunk_id":    results["ids"][0][i],
                    "text":        doc,
                    "dense_score": round(1.0 - dist, 4),
                    **meta
                })

            logger.debug(
                f"DenseRetriever: {len(chunks)} results for "
                f"bidder '{bidder_id}'"
            )
            return chunks

        except Exception as e:
            logger.error(
                f"DenseRetriever failed for '{bidder_id}': {e}"
            )
            return []