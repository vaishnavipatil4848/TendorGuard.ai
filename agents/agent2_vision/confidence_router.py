"""
confidence_router.py
Agent 2 — Vision Specialist Agent
Routes detections between direct extraction (high confidence)
and Claude Vision fallback (low confidence).
Also logs low confidence cases for future RT-DETR fine-tuning.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

LOW_CONFIDENCE_LOG_PATH = "storage/low_confidence_log.jsonl"


class ConfidenceRouter:
    """
    Routes RT-DETR detections based on confidence score.
    High confidence → direct extraction path (fast)
    Low confidence → Claude Vision fallback path (accurate)

    Also maintains a log of all low confidence cases as a
    future fine-tuning dataset for RT-DETR.
    """

    def __init__(self, log_path: str = LOW_CONFIDENCE_LOG_PATH):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def route(
        self,
        detections: List[Dict[str, Any]],
        bidder_id: str,
        doc_type: str
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Split detections into high and low confidence groups.

        Args:
            detections: all RT-DETR detections for a page
            bidder_id: for logging purposes
            doc_type: document type from CLIP

        Returns:
            (high_confidence, low_confidence)
        """
        high_conf = []
        low_conf = []

        for det in detections:
            if det.get("is_high_confidence", False):
                det["extraction_path"] = "direct"
                high_conf.append(det)
            else:
                det["extraction_path"] = "claude_vision"
                low_conf.append(det)

        logger.info(
            f"Router: {len(high_conf)} direct, {len(low_conf)} → Claude Vision "
            f"(bidder={bidder_id}, doc_type={doc_type})"
        )

        # log low confidence cases for future fine-tuning
        if low_conf:
            self._log_low_confidence(low_conf, bidder_id, doc_type)

        return high_conf, low_conf

    def merge_results(
        self,
        high_conf: List[Dict[str, Any]],
        low_conf_processed: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Merge high confidence direct extractions with
        Claude Vision processed low confidence extractions.
        Sorts by page number then bbox position (top to bottom).

        Returns:
            Combined and sorted list of all extractions
        """
        all_results = []

        # high confidence — use RT-DETR text directly
        for det in high_conf:
            all_results.append({
                "text": det.get("text", ""),
                "bbox": det.get("bbox", []),
                "confidence": det.get("confidence", 0.0),
                "field_label": det.get("field_label"),
                "doc_type": det.get("doc_type", "other"),
                "extraction_path": "direct",
                "bbox_id": det.get("bbox_id"),
                "extracted_fields": det.get("extracted_fields", {}),
                "page_number": det.get("page_number", 0)
            })

        # low confidence — use Claude Vision corrected text
        for det in low_conf_processed:
            claude_result = det.get("claude_extraction", {})
            all_results.append({
                "text": det.get("final_text", det.get("text", "")),
                "bbox": det.get("bbox", []),
                "confidence": det.get("final_confidence", 0.0),
                "field_label": det.get("field_label"),
                "doc_type": det.get("doc_type", "other"),
                "extraction_path": "claude_vision",
                "bbox_id": det.get("bbox_id"),
                "extracted_fields": det.get("extracted_fields", {}),
                "claude_notes": claude_result.get("notes", ""),
                "page_number": det.get("page_number", 0)
            })

        # sort by page then vertical position
        all_results.sort(
            key=lambda x: (
                x.get("page_number", 0),
                x.get("bbox", [0, 0, 0, 0])[1]  # y1 coordinate
            )
        )

        return all_results

    def _log_low_confidence(
        self,
        detections: List[Dict[str, Any]],
        bidder_id: str,
        doc_type: str
    ) -> None:
        """
        Append low confidence cases to JSONL log for future RT-DETR fine-tuning.
        Each line is one detection with metadata.
        """
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                for det in detections:
                    log_entry = {
                        "timestamp": datetime.utcnow().isoformat(),
                        "bidder_id": bidder_id,
                        "doc_type": doc_type,
                        "bbox": det.get("bbox", []),
                        "ocr_attempt": det.get("text", ""),
                        "confidence": det.get("confidence", 0.0),
                        "threshold_used": det.get("threshold_used", 0.85),
                        "page_number": det.get("page_number", 0)
                    }
                    f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        except Exception as e:
            logger.warning(f"Failed to log low confidence case: {e}")

    def get_routing_summary(
        self,
        high_conf: List[Dict],
        low_conf: List[Dict]
    ) -> Dict[str, Any]:
        """
        Return a summary of routing decisions for the pipeline message bus.
        """
        total = len(high_conf) + len(low_conf)
        return {
            "total_detections": total,
            "direct_extractions": len(high_conf),
            "claude_vision_fallbacks": len(low_conf),
            "fallback_rate": len(low_conf) / total if total > 0 else 0.0
        }