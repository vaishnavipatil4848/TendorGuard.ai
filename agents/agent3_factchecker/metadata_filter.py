"""
metadata_filter.py
Agent 3 — Fact-Checker Agent

Narrows the retrieval search space by filtering chunks
to those relevant for the criterion's category.

Powered by CLIP document type classification from Agent 2.
This avoids searching CA certificates for GST numbers
or bank statements for experience letters.
"""

import logging
from typing import List, Dict, Any, Union

logger = logging.getLogger(__name__)

# Maps criterion category → allowed document types
# Document types come from CLIP classifier in Agent 2
CATEGORY_DOC_MAP = {
    "financial": [
        "turnover_certificate", "balance_sheet",
        "bank_statement", "ca_certificate",
        "PDF", "TXT"
    ],
    "technical": [
        "experience_letter", "work_order",
        "completion_certificate", "PDF", "TXT"
    ],
    "compliance": [
        "gst_certificate", "pan_card",
        "incorporation_certificate", "msme_certificate",
        "PDF", "TXT"
    ],
    "eligibility": [
        "PDF", "TXT"  # general — search all
    ],
    "mandatory": [
        "PDF", "TXT"
    ]
}


class MetadataFilter:
    """
    Filters chunks by criterion category and document type.
    Always includes document-level summary chunks regardless of type
    to preserve broad context.
    """

    def filter(
        self,
        chunks: List[Dict[str, Any]],
        criteria_context: Union[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Filter chunks to those relevant for the given criteria.

        Args:
            chunks: all chunks for a bidder
            criteria_context: either a single category string (e.g. "financial")
                            or a list of criterion dicts.

        Returns:
            Filtered list — always includes document summary chunks
        """
        if isinstance(criteria_context, list):
            # Extract unique categories from criteria list
            categories = {c.get("category", "").lower().strip() for c in criteria_context}
            allowed_types = set()
            for cat in categories:
                allowed_types.update(CATEGORY_DOC_MAP.get(cat, ["PDF", "TXT"]))
            if not allowed_types:
                allowed_types = {"PDF", "TXT"}
        else:
            category_key = criteria_context.lower().strip()
            allowed_types = set(CATEGORY_DOC_MAP.get(
                category_key,
                ["PDF", "TXT"]
            ))

        # always include document-level summary chunks
        summaries = [
            c for c in chunks
            if c.get("chunk_level") == "document"
        ]

        # filter field-level chunks by doc type
        field_chunks = [
            c for c in chunks
            if c.get("chunk_level") == "field"
            and c.get("doc_type", "PDF") in allowed_types
        ]

        # deduplicate by chunk_id
        seen = set()
        filtered = []
        for c in summaries + field_chunks:
            if c["chunk_id"] not in seen:
                seen.add(c["chunk_id"])
                filtered.append(c)

        logger.debug(
            f"MetadataFilter [{criteria_context}]: "
            f"{len(chunks)} → {len(filtered)} chunks "
            f"(allowed types: {allowed_types})"
        )
        return filtered