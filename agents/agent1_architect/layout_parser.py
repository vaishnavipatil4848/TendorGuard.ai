"""
layout_parser.py
Agent 1 — Architect Agent
Uses LayoutLMv3 / DocLayNet to segment the tender PDF into
logical sections before LLM extraction.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any

import fitz  # PyMuPDF
from transformers import LayoutLMv3Processor, LayoutLMv3ForSequenceClassification
from PIL import Image
import torch
from pipeline.utils import extract_text_from_docx

logger = logging.getLogger(__name__)

# Section labels DocLayNet / LayoutLMv3 maps to
SECTION_LABELS = [
    "Caption", "Footnote", "Formula", "List-item",
    "Page-footer", "Page-header", "Picture",
    "Section-header", "Table", "Text", "Title"
]

# Which labels we care about for downstream processing
RELEVANT_LABELS = {"Section-header", "Text", "Table", "List-item", "Title"}


class LayoutParser:
    """
    Parses a tender PDF using LayoutLMv3 to detect and extract
    document regions with their structural labels and positions.
    """

    def __init__(self, model_name: str = "microsoft/layoutlmv3-base"):
        logger.info(f"Loading LayoutLMv3 model: {model_name}")
        self.processor = LayoutLMv3Processor.from_pretrained(
            model_name, apply_ocr=True
        )
        self.model = LayoutLMv3ForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(SECTION_LABELS)
        )
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        logger.info(f"LayoutLMv3 loaded on {self.device}")

    def parse_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Parse a full PDF and return a list of detected regions
        across all pages.

        Returns:
            List of region dicts:
            {
                page_number: int,
                label: str,
                text: str,
                bbox: [x0, y0, x1, y1],
                confidence: float
            }
        """
        pdf_path = Path(pdf_path)
        logger.info(f"Parsing document: {pdf_path.name}")
        ext = pdf_path.suffix.lower()

        if ext in (".docx", ".doc"):
            return self._parse_docx(str(pdf_path))

        # fitz handles PDF and images (jpg, png, etc.)
        doc = fitz.open(str(pdf_path))
        all_regions = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            regions = self._parse_page(page, page_num + 1)
            all_regions.extend(regions)
            logger.debug(f"Page {page_num + 1}: found {len(regions)} regions")

        doc.close()
        logger.info(f"Total regions detected: {len(all_regions)}")
        return all_regions

    def _parse_docx(self, docx_path: str) -> List[Dict[str, Any]]:
        """Extract text from DOCX and create a dummy 'Text' region."""
        text = extract_text_from_docx(docx_path)
        if not text:
            return []
        
        # Split into paragraphs to simulate regions
        paragraphs = text.split("\n")
        regions = []
        for i, p in enumerate(paragraphs):
            if p.strip():
                regions.append({
                    "page_number": 1,
                    "label": "Text",
                    "text": p.strip(),
                    "bbox": [0, i*20, 500, (i+1)*20],  # dummy bbox
                    "confidence": 1.0
                })
        return regions

    def _parse_page(self, page: fitz.Page, page_number: int) -> List[Dict[str, Any]]:
        """
        Process a single PDF page through LayoutLMv3.
        Converts page to image, runs processor and model inference.
        """
        # render page to image at 150 DPI for reasonable speed
        mat = fitz.Matrix(150 / 72, 150 / 72)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        try:
            encoding = self.processor(
                img,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            encoding = {k: v.to(self.device) for k, v in encoding.items()}

            with torch.no_grad():
                outputs = self.model(**encoding)

            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            confidence, pred_idx = torch.max(probs, dim=-1)
            label = SECTION_LABELS[pred_idx.item()]

            # extract raw text blocks from PyMuPDF for text content
            text_blocks = page.get_text("blocks")
            regions = []

            for block in text_blocks:
                x0, y0, x1, y1, text, *_ = block
                if not text.strip():
                    continue
                regions.append({
                    "page_number": page_number,
                    "label": label,
                    "text": text.strip(),
                    "bbox": [x0, y0, x1, y1],
                    "confidence": float(confidence.item())
                })

            return regions

        except Exception as e:
            logger.warning(f"LayoutLMv3 failed on page {page_number}: {e}")
            # fallback — return raw text blocks without structural label
            return self._fallback_text_extraction(page, page_number)

    def _fallback_text_extraction(
        self, page: fitz.Page, page_number: int
    ) -> List[Dict[str, Any]]:
        """
        Fallback: plain PyMuPDF text extraction if LayoutLMv3 fails.
        Labels everything as Text with zero confidence.
        """
        blocks = page.get_text("blocks")
        regions = []
        for block in blocks:
            x0, y0, x1, y1, text, *_ = block
            if text.strip():
                regions.append({
                    "page_number": page_number,
                    "label": "Text",
                    "text": text.strip(),
                    "bbox": [x0, y0, x1, y1],
                    "confidence": 0.0
                })
        return regions

    def filter_relevant_regions(
        self, regions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Filter out page headers, footers, figures etc.
        Keep only regions relevant for criteria extraction.
        """
        return [r for r in regions if r["label"] in RELEVANT_LABELS]