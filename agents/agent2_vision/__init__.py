"""
Agent 2 — Vision Specialist Agent
Orchestrates the full vision pipeline:
OpenCV Preprocessing → CLIP Classification → RT-DETR Detection
→ Confidence Routing → Claude Vision Fallback → BBox Store
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import fitz  # PyMuPDF
from PIL import Image

from .preprocessor import Preprocessor
from .document_classifier import DocumentClassifier
from .text_detector import TextDetector
from .vision_fallback import VisionFallback
from .confidence_router import ConfidenceRouter
from .bbox_store import BBoxStore

logger = logging.getLogger(__name__)


class VisionAgent:
    """
    Main entry point for Agent 2.
    Processes all bidder documents and returns structured
    extractions with bbox references for every detected field.
    """

    def __init__(self):
        logger.info("Initializing Vision Agent")
        self.preprocessor = Preprocessor()
        self.classifier = DocumentClassifier()
        self.detector = TextDetector()
        self.fallback = VisionFallback()
        self.router = ConfidenceRouter()
        self.bbox_store = BBoxStore()

    def run(
        self,
        bidder_id: str,
        document_paths: List[str]
    ) -> Dict[str, Any]:
        """
        Process all documents for a bidder.

        Args:
            bidder_id: unique bidder identifier
            document_paths: list of paths to bidder document files (PDF or image)

        Returns:
            {
                bidder_id: str,
                documents: list of processed document results,
                total_extractions: int,
                routing_summary: dict
            }
        """
        logger.info(
            f"Vision Agent starting for bidder: {bidder_id}, "
            f"{len(document_paths)} documents"
        )

        all_documents = []
        total_high = 0
        total_low = 0

        for doc_path in document_paths:
            doc_result = self._process_document(bidder_id, doc_path)
            all_documents.append(doc_result)
            total_high += doc_result.get("routing_summary", {}).get("direct_extractions", 0)
            total_low += doc_result.get("routing_summary", {}).get("claude_vision_fallbacks", 0)

        total = total_high + total_low
        result = {
            "bidder_id": bidder_id,
            "documents": all_documents,
            "total_extractions": total,
            "routing_summary": {
                "total_detections": total,
                "direct_extractions": total_high,
                "claude_vision_fallbacks": total_low,
                "fallback_rate": total_low / total if total > 0 else 0.0
            }
        }

        logger.info(
            f"Vision Agent complete for {bidder_id}: "
            f"{total} extractions, {total_low} via Claude Vision"
        )
        return result

    def _process_document(
        self, bidder_id: str, doc_path: str
    ) -> Dict[str, Any]:
        """
        Process a single document file (PDF or image).
        Handles multi-page PDFs by processing each page separately.
        """
        path = Path(doc_path)
        if not path.exists():
            logger.error(f"Document not found: {doc_path}")
            return {"error": f"File not found: {doc_path}"}

        logger.info(f"Processing document: {path.name}")

        # extract pages as PIL images
        pages = self._extract_pages(doc_path)
        if not pages:
            return {"error": f"No pages extracted from: {path.name}"}

        all_extractions = []
        doc_type = "other"
        routing_summary = {}

        for page_num, page_image in enumerate(pages, start=1):
            page_result = self._process_page(
                page_image=page_image,
                bidder_id=bidder_id,
                page_number=page_num,
                source_file=path.name
            )

            # use doc_type from first page classification
            if page_num == 1:
                doc_type = page_result.get("doc_type", "other")

            all_extractions.extend(page_result.get("extractions", []))
            routing_summary = page_result.get("routing_summary", {})

        return {
            "file_name": path.name,
            "doc_type": doc_type,
            "page_count": len(pages),
            "extractions": all_extractions,
            "routing_summary": routing_summary
        }

    def _process_page(
        self,
        page_image: Image.Image,
        bidder_id: str,
        page_number: int,
        source_file: str
    ) -> Dict[str, Any]:
        """
        Full pipeline for a single page image.
        """
        # step 1 — OpenCV preprocessing
        preprocessed_np = self.preprocessor.process(page_image)
        preprocessed_pil = self.preprocessor.to_pil(preprocessed_np)

        # step 2 — CLIP document classification
        doc_type, clip_confidence = self.classifier.classify(preprocessed_pil)
        template = self.classifier.get_extraction_template(doc_type)
        key_fields = template.get("key_fields", [])

        logger.info(
            f"Page {page_number}: classified as '{doc_type}' "
            f"(CLIP confidence: {clip_confidence:.2f})"
        )

        # step 3 — RT-DETR text detection
        detections = self.detector.detect(
            preprocessed_pil, doc_type=doc_type, template=template
        )

        # add page number to each detection
        for det in detections:
            det["page_number"] = page_number

        # step 4 — confidence routing
        high_conf, low_conf = self.router.route(
            detections, bidder_id, doc_type
        )

        # step 5 — Claude Vision for low confidence detections
        low_conf_processed = []
        if low_conf:
            low_conf_processed = self.fallback.extract_batch(
                low_confidence_detections=low_conf,
                full_image=preprocessed_pil,
                doc_type=doc_type,
                key_fields=key_fields
            )

        # step 6 — merge and sort all extractions
        all_extractions = self.router.merge_results(high_conf, low_conf_processed)

        # step 7 — store bbox references
        self.bbox_store.add_batch(
            all_extractions, bidder_id, source_file=source_file
        )

        routing_summary = self.router.get_routing_summary(high_conf, low_conf)

        return {
            "page_number": page_number,
            "doc_type": doc_type,
            "clip_confidence": clip_confidence,
            "extractions": all_extractions,
            "routing_summary": routing_summary
        }

    def _extract_pages(self, doc_path: str) -> List[Image.Image]:
        """
        Extract all pages from a PDF or image file as PIL Images.
        """
        path = Path(doc_path)
        ext = path.suffix.lower()

        if ext == ".pdf":
            return self._extract_pdf_pages(doc_path)
        elif ext in (".jpg", ".jpeg", ".png", ".tiff", ".bmp"):
            img = Image.open(doc_path).convert("RGB")
            return [img]
        else:
            logger.warning(f"Unsupported file type: {ext}")
            return []

    def _extract_pdf_pages(self, pdf_path: str) -> List[Image.Image]:
        """Render each PDF page to a PIL Image at 200 DPI."""
        pages = []
        try:
            doc = fitz.open(pdf_path)
            mat = fitz.Matrix(200 / 72, 200 / 72)
            for page in doc:
                pix = page.get_pixmap(matrix=mat)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                pages.append(img)
            doc.close()
        except Exception as e:
            logger.error(f"PDF page extraction failed: {e}")
        return pages