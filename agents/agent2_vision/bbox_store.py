"""
bbox_store.py
Agent 2 — Vision Specialist Agent
Stores and retrieves bounding box references for every extracted
text region. These bbox references power the Source Citation Mapper
in the Evidence Overlay UI.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class BBoxStore:
    """
    Maintains a persistent store of bounding box references
    for every text region extracted from bidder documents.

    Each entry links:
    - bidder_id + document_type + page_number + bbox coordinates
    - to the extracted text and confidence score

    This enables the UI Evidence Overlay to highlight the exact
    source location of any piece of extracted evidence.
    """

    def __init__(self, store_path: str = "storage/bbox_store.json"):
        self.store_path = Path(store_path)
        self.store: Dict[str, List[Dict[str, Any]]] = {}
        self._load()

    def add(
        self,
        bidder_id: str,
        doc_type: str,
        page_number: int,
        bbox: List[float],
        text: str,
        confidence: float,
        field_label: Optional[str] = None,
        source_file: Optional[str] = None
    ) -> str:
        """
        Add a bbox reference to the store.

        Returns:
            bbox_id: unique identifier for this bbox entry
        """
        import uuid
        bbox_id = str(uuid.uuid4())

        entry = {
            "bbox_id": bbox_id,
            "bidder_id": bidder_id,
            "doc_type": doc_type,
            "page_number": page_number,
            "bbox": bbox,         # [x1, y1, x2, y2]
            "text": text,
            "confidence": confidence,
            "field_label": field_label,
            "source_file": source_file
        }

        key = f"{bidder_id}:{doc_type}"
        if key not in self.store:
            self.store[key] = []
        self.store[key].append(entry)

        logger.debug(
            f"BBox stored: {bbox_id} | {bidder_id} | {doc_type} | "
            f"page {page_number} | confidence {confidence:.2f}"
        )
        return bbox_id

    def add_batch(
        self,
        detections: List[Dict[str, Any]],
        bidder_id: str,
        source_file: Optional[str] = None
    ) -> List[str]:
        """
        Add a batch of detections from TextDetector output.

        Returns:
            List of bbox_ids
        """
        bbox_ids = []
        for det in detections:
            bbox_id = self.add(
                bidder_id=bidder_id,
                doc_type=det.get("doc_type", "other"),
                page_number=det.get("page_number", 0),
                bbox=det.get("bbox", [0, 0, 0, 0]),
                text=det.get("final_text", det.get("text", "")),
                confidence=det.get("final_confidence", det.get("confidence", 0.0)),
                field_label=det.get("field_label"),
                source_file=source_file
            )
            det["bbox_id"] = bbox_id
            bbox_ids.append(bbox_id)

        self.save()
        return bbox_ids

    def get_by_bidder(
        self, bidder_id: str
    ) -> List[Dict[str, Any]]:
        """Get all bbox entries for a bidder."""
        results = []
        for key, entries in self.store.items():
            if key.startswith(f"{bidder_id}:"):
                results.extend(entries)
        return results

    def get_by_bbox_id(self, bbox_id: str) -> Optional[Dict[str, Any]]:
        """Look up a specific bbox entry by its ID."""
        for entries in self.store.values():
            for entry in entries:
                if entry["bbox_id"] == bbox_id:
                    return entry
        return None

    def get_by_field(
        self, bidder_id: str, field_label: str
    ) -> List[Dict[str, Any]]:
        """Find all bbox entries for a specific field label."""
        results = []
        for key, entries in self.store.items():
            if key.startswith(f"{bidder_id}:"):
                for entry in entries:
                    if entry.get("field_label") == field_label:
                        results.append(entry)
        return results

    def save(self) -> None:
        """Persist the store to disk."""
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump(self.store, f, indent=2, ensure_ascii=False)

    def _load(self) -> None:
        """Load existing store from disk if present."""
        if self.store_path.exists():
            try:
                with open(self.store_path, "r", encoding="utf-8") as f:
                    self.store = json.load(f)
                logger.info(
                    f"BBox store loaded: {sum(len(v) for v in self.store.values())} entries"
                )
            except Exception as e:
                logger.warning(f"Failed to load bbox store: {e}")
                self.store = {}
        else:
            self.store = {}