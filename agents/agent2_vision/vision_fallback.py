"""
vision_fallback.py
Agent 2 — Vision Specialist Agent
Claude Vision fallback for low-confidence RT-DETR detections.
Receives the crop image + surrounding context + document type
and returns structured field extraction.
"""

import base64
import json
import logging
from io import BytesIO
from typing import Dict, Any, List, Optional

import anthropic
from PIL import Image

logger = logging.getLogger(__name__)


class VisionFallback:
    """
    Uses Claude Vision to handle low-confidence crops that
    RT-DETR could not reliably extract.
    Receives crop + context + doc type for accurate extraction.
    """

    def __init__(self, model: str = "claude-opus-4-6"):
        self.client = anthropic.Anthropic()
        self.model = model

    def extract(
        self,
        crop_image: Image.Image,
        doc_type: str,
        key_fields: List[str],
        ocr_attempt: str = "",
        page_number: int = 0,
        context_text: str = ""
    ) -> Dict[str, Any]:
        """
        Extract structured fields from a low-confidence crop.

        Args:
            crop_image: cropped PIL Image of the problematic region
            doc_type: document type from CLIP
            key_fields: fields to extract (from extraction template)
            ocr_attempt: what RT-DETR tried to extract (may be garbled)
            page_number: source page for audit trail
            context_text: surrounding text for additional context

        Returns:
            {
                extracted_fields: dict of field → value,
                corrected_text: cleaned full text,
                confidence: 0.0 - 1.0,
                notes: any observations about the image quality
            }
        """
        logger.info(
            f"Claude Vision fallback: doc_type={doc_type}, "
            f"page={page_number}, fields={key_fields}"
        )

        image_b64 = self._encode_image(crop_image)
        prompt = self._build_prompt(
            doc_type, key_fields, ocr_attempt, context_text
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_b64
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }]
            )

            raw = response.content[0].text
            result = self._parse_response(raw)
            result["page_number"] = page_number
            result["doc_type"] = doc_type

            logger.info(
                f"Claude Vision extracted {len(result.get('extracted_fields', {}))} fields "
                f"with confidence {result.get('confidence', 0):.2f}"
            )
            return result

        except Exception as e:
            logger.error(f"Claude Vision fallback failed: {e}")
            return {
                "extracted_fields": {},
                "corrected_text": ocr_attempt,
                "confidence": 0.0,
                "notes": f"Claude Vision failed: {str(e)}",
                "page_number": page_number,
                "doc_type": doc_type
            }

    def extract_batch(
        self,
        low_confidence_detections: List[Dict[str, Any]],
        full_image: Image.Image,
        doc_type: str,
        key_fields: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of low-confidence detections.

        Args:
            low_confidence_detections: list from TextDetector.split_by_confidence
            full_image: the full page image for context
            doc_type: document type from CLIP
            key_fields: fields to extract

        Returns:
            List of extraction results, one per detection
        """
        results = []
        for detection in low_confidence_detections:
            # crop the region from the full image
            bbox = detection["bbox"]
            crop = full_image.crop((
                int(bbox[0]), int(bbox[1]),
                int(bbox[2]), int(bbox[3])
            ))

            result = self.extract(
                crop_image=crop,
                doc_type=doc_type,
                key_fields=key_fields,
                ocr_attempt=detection.get("text", ""),
                page_number=detection.get("page_number", 0)
            )

            # merge back with original detection
            detection.update({
                "claude_extraction": result,
                "final_text": result.get("corrected_text", detection.get("text", "")),
                "final_confidence": result.get("confidence", 0.0),
                "extracted_fields": result.get("extracted_fields", {})
            })
            results.append(detection)

        return results

    def _build_prompt(
        self,
        doc_type: str,
        key_fields: List[str],
        ocr_attempt: str,
        context_text: str
    ) -> str:
        fields_str = ", ".join(key_fields) if key_fields else "all visible fields"
        ocr_section = (
            f"\nThe OCR attempted to read this region but may have errors:\n"
            f"OCR attempt: {ocr_attempt}\n"
            if ocr_attempt else ""
        )
        context_section = (
            f"\nSurrounding context from the document:\n{context_text}\n"
            if context_text else ""
        )

        return f"""You are an expert document extraction system for Indian government procurement.

This image is a crop from a {doc_type.replace('_', ' ')} document.
{ocr_section}{context_section}
Your task:
1. Read ALL text visible in this image carefully — including stamps, handwriting, and Hindi text
2. Extract the following specific fields: {fields_str}
3. Correct any OCR errors using visual context
4. Assess your confidence in the extraction

Return ONLY a valid JSON object. No preamble, no markdown.

{{
  "extracted_fields": {{
    "field_name": "extracted_value"
  }},
  "corrected_text": "complete text content of the image, corrected",
  "confidence": 0.0_to_1.0,
  "notes": "any observations about image quality, stamps, Hindi text etc"
}}"""

    def _encode_image(self, image: Image.Image) -> str:
        """Encode PIL Image to base64 JPEG string."""
        buffer = BytesIO()
        # convert to RGB before saving as JPEG
        image.convert("RGB").save(buffer, format="JPEG", quality=95)
        return base64.standard_b64encode(buffer.getvalue()).decode("utf-8")

    def _parse_response(self, raw: str) -> Dict[str, Any]:
        """Parse Claude Vision JSON response."""
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1])

        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error(f"Claude Vision response parse error: {e}")
            return {
                "extracted_fields": {},
                "corrected_text": raw[:500],
                "confidence": 0.1,
                "notes": "JSON parse failed — raw text stored"
            }