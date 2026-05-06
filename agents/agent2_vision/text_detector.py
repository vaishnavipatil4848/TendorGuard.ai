"""
text_detector.py
Agent 2 — Vision Specialist Agent
RT-DETR based text region detection and recognition.
Uses template from CLIP classification to guide field extraction.
"""

import logging
from typing import List, Dict, Any, Union, Optional

import numpy as np
import torch
from PIL import Image
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

logger = logging.getLogger(__name__)

# Confidence threshold — below this routes to Claude Vision
DEFAULT_CONFIDENCE_THRESHOLD = 0.85

# Per document type confidence thresholds
# Stamps and mixed-script docs need higher threshold
DOCUMENT_TYPE_THRESHOLDS = {
    "gst_certificate":        0.85,
    "turnover_certificate":   0.82,
    "experience_letter":      0.80,
    "bank_statement":         0.82,
    "pan_card":               0.88,
    "incorporation_certificate": 0.85,
    "msme_certificate":       0.85,
    "other":                  0.90   # unknown type — be conservative
}


class TextDetector:
    """
    Runs RT-DETR on preprocessed document images to detect
    and localize text regions with bounding boxes and confidence scores.
    """

    def __init__(
        self,
        model_path: str = "PekingU/rtdetr_r50vd",
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    ):
        """
        Args:
            model_path: path to fine-tuned RT-DETR model or HF hub model name
            confidence_threshold: global fallback threshold
        """
        logger.info(f"Loading RT-DETR model: {model_path}")
        self.processor = RTDetrImageProcessor.from_pretrained(model_path)
        self.model = RTDetrForObjectDetection.from_pretrained(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        self.global_threshold = confidence_threshold
        logger.info(f"RT-DETR loaded on {self.device}")

    def detect(
        self,
        image: Union[np.ndarray, Image.Image],
        doc_type: str = "other",
        template: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect text regions in a document image.

        Args:
            image: preprocessed document image
            doc_type: document type from CLIP classifier
            template: extraction template from DocumentClassifier

        Returns:
            List of detection dicts:
            {
                text: str,
                bbox: [x1, y1, x2, y2],
                confidence: float,
                field_label: str or None,
                is_high_confidence: bool,
                doc_type: str
            }
        """
        pil_image = self._ensure_pil(image)
        threshold = DOCUMENT_TYPE_THRESHOLDS.get(doc_type, self.global_threshold)
        template = template or {}
        key_fields = template.get("key_fields", [])

        try:
            inputs = self.processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            # post-process detections
            target_sizes = torch.tensor([pil_image.size[::-1]])
            results = self.processor.post_process_object_detection(
                outputs,
                target_sizes=target_sizes,
                threshold=0.3   # low threshold — we filter by confidence later
            )[0]

            detections = []
            for score, label_id, box in zip(
                results["scores"], results["labels"], results["boxes"]
            ):
                confidence = float(score.item())
                bbox = [float(x) for x in box.tolist()]

                # extract text from the detected region using OCR crop
                text = self._extract_text_from_region(pil_image, bbox)
                if not text.strip():
                    continue

                # try to match to a known field from the template
                field_label = self._match_field_label(text, key_fields)

                detections.append({
                    "text": text,
                    "bbox": bbox,
                    "confidence": confidence,
                    "field_label": field_label,
                    "is_high_confidence": confidence >= threshold,
                    "doc_type": doc_type,
                    "threshold_used": threshold
                })

            logger.debug(
                f"RT-DETR: {len(detections)} detections, "
                f"{sum(1 for d in detections if d['is_high_confidence'])} high confidence"
            )
            return detections

        except Exception as e:
            logger.error(f"RT-DETR detection failed: {e}")
            return []

    def split_by_confidence(
        self, detections: List[Dict[str, Any]]
    ) -> tuple[List[Dict], List[Dict]]:
        """
        Split detections into high and low confidence lists.

        Returns:
            (high_confidence, low_confidence)
        """
        high = [d for d in detections if d["is_high_confidence"]]
        low = [d for d in detections if not d["is_high_confidence"]]
        logger.info(
            f"Confidence split: {len(high)} high, {len(low)} low confidence"
        )
        return high, low

    def crop_region(
        self,
        image: Union[np.ndarray, Image.Image],
        bbox: List[float],
        padding: int = 10
    ) -> Image.Image:
        """
        Crop a detected region from the image with padding.
        Used to send low-confidence crops to Claude Vision.

        Args:
            image: source image
            bbox: [x1, y1, x2, y2] bounding box
            padding: pixels to add around the crop

        Returns:
            Cropped PIL Image
        """
        pil = self._ensure_pil(image)
        w, h = pil.size

        x1 = max(0, int(bbox[0]) - padding)
        y1 = max(0, int(bbox[1]) - padding)
        x2 = min(w, int(bbox[2]) + padding)
        y2 = min(h, int(bbox[3]) + padding)

        return pil.crop((x1, y1, x2, y2))

    def _extract_text_from_region(
        self, image: Image.Image, bbox: List[float]
    ) -> str:
        """
        Extract text from a detected bounding box region.
        Uses pytesseract as the text recognition backend.
        In production, replace with your fine-tuned RT-DETR OCR head.
        """
        try:
            import pytesseract
            crop = image.crop((
                int(bbox[0]), int(bbox[1]),
                int(bbox[2]), int(bbox[3])
            ))
            # use both Hindi and English
            text = pytesseract.image_to_string(
                crop,
                lang="eng+hin",
                config="--psm 6"
            )
            return text.strip()
        except Exception as e:
            logger.warning(f"Text extraction from region failed: {e}")
            return ""

    def _match_field_label(
        self, text: str, key_fields: List[str]
    ) -> Optional[str]:
        """
        Try to match detected text to a known field label from the template.
        Simple case-insensitive keyword match.
        """
        text_lower = text.lower()
        for field in key_fields:
            if field.lower().replace("_", " ") in text_lower:
                return field
        return None

    def _ensure_pil(
        self, image: Union[np.ndarray, Image.Image]
    ) -> Image.Image:
        """Convert to PIL Image if needed."""
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, np.ndarray):
            import cv2
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb)
        raise TypeError(f"Unsupported image type: {type(image)}")