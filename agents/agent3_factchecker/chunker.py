"""
chunker.py
Agent 3 — Fact-Checker Agent

Hierarchical chunking strategy:
  Level 1 — document-level summary chunk
  Level 2 — field-level sliding window chunks (5 lines each)

Absorbed from uploaded code:
  - pages dict structure {page_num: [{line_no, text}]}
  - page/line metadata preserved for Evidence Overlay citation
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class HierarchicalChunker:
    """
    Two-level chunker that works directly with Agent 1's
    IngestionAgent output format.

    Level 1: One document-level summary chunk — used for broad retrieval
    Level 2: Field-level chunks (sliding window of 5 lines) — for precise evidence
    """

    def __init__(self, window_size: int = 5):
        self.window_size = window_size

    def chunk(self, bidder_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk a bidder document into retrievable units.

        Args:
            bidder_data: output from Agent 1 IngestionAgent
                         must contain 'pages', 'raw_text', 'filename'

        Returns:
            List of chunk dicts:
            {
                chunk_id, text, page_number, line_start,
                line_end, doc_type, chunk_level, source_file
            }
        """
        chunks = []
        filename = bidder_data.get("filename", "unknown")
        doc_type = bidder_data.get("doc_type", "PDF")
        pages    = bidder_data.get("pages", {})
        raw_text = bidder_data.get("raw_text", "")

        # ── Level 1: Document summary chunk ──────────────────────────────
        if raw_text.strip():
            chunks.append({
                "chunk_id":    f"{filename}_doc_summary",
                "text":        raw_text[:800],
                "page_number": 1,
                "line_start":  1,
                "line_end":    10,
                "doc_type":    doc_type,
                "chunk_level": "document",
                "source_file": filename
            })

        # ── Level 2: Field-level sliding window chunks ────────────────────
        for page_num, lines in pages.items():
            if not lines:
                continue

            for i in range(0, len(lines), self.window_size):
                window = lines[i: i + self.window_size]
                chunk_text = " ".join(
                    ln.get("text", "") for ln in window
                    if ln.get("text", "").strip()
                ).strip()

                if len(chunk_text) < 10:
                    continue

                line_start = window[0].get("line_no", i + 1)
                line_end   = window[-1].get("line_no", i + self.window_size)

                chunks.append({
                    "chunk_id":    f"{filename}_p{page_num}_l{line_start}",
                    "text":        chunk_text,
                    "page_number": int(page_num),
                    "line_start":  line_start,
                    "line_end":    line_end,
                    "doc_type":    doc_type,
                    "chunk_level": "field",
                    "source_file": filename
                })

        logger.info(
            f"Chunked '{filename}': {len(chunks)} total chunks "
            f"({sum(1 for c in chunks if c['chunk_level'] == 'field')} field-level)"
        )
        return chunks