"""
utils.py
pipeline/ — TendorGuard.ai

Utility functions for document conversion and multi-format support.
Handles conversion of DOCX to PDF (where possible) or direct text extraction.
"""

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    logger.warning("python-docx not installed. DOCX support will be limited.")

try:
    from docx2pdf import convert as docx_to_pdf
    CONVERTER_AVAILABLE = True
except ImportError:
    CONVERTER_AVAILABLE = False
    # logger.warning("docx2pdf not installed. Direct DOCX->PDF conversion disabled.")


def ensure_pdf(file_path: str) -> str:
    """
    Ensures the given file is in PDF format for LayoutParser.
    If it's an image, PyMuPDF can open it directly.
    If it's a DOCX, tries to convert to PDF or extract text.

    Returns:
        Path to a PDF file (original or converted)
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        return file_path

    # PyMuPDF (fitz) can open images as 1-page PDFs
    if ext in (".jpg", ".jpeg", ".png", ".tiff", ".bmp"):
        return file_path

    if ext in (".docx", ".doc"):
        return _handle_docx(path)

    return file_path


def _handle_docx(path: Path) -> str:
    """Handle DOCX conversion."""
    if not DOCX_AVAILABLE:
        logger.error("Cannot process DOCX: python-docx not installed.")
        return str(path)

    pdf_path = path.with_suffix(".pdf")
    
    # Try docx2pdf if available (needs Word on Windows/macOS)
    if CONVERTER_AVAILABLE:
        try:
            logger.info(f"Converting DOCX to PDF: {path.name}")
            docx_to_pdf(str(path), str(pdf_path))
            if pdf_path.exists():
                return str(pdf_path)
        except Exception as e:
            logger.warning(f"docx2pdf conversion failed: {e}. Falling back to text extraction.")

    # Fallback: We don't actually need to return a PDF path if we 
    # handle DOCX separately in the orchestrator.
    # For now, we'll return the original path and let downstream agents handle it.
    return str(path)


def extract_text_from_docx(file_path: str) -> str:
    """Extract plain text from a DOCX file."""
    if not DOCX_AVAILABLE:
        return ""
    
    try:
        doc = docx.Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return "\n".join(full_text)
    except Exception as e:
        logger.error(f"Failed to extract text from DOCX: {e}")
        return ""
