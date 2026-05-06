"""
document_classifier.py
Agent 2 — Vision Specialist Agent
Uses CLIP to classify the document type of each scanned page.
Classification informs RT-DETR's extraction template selection.
"""

import logging
from typing import Union, List, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

logger = logging.getLogger(__name__)

# Document type labels and their descriptive prompts for CLIP
DOCUMENT_TYPES = {
    "gst_certificate": [
        "GST registration certificate",
        "Goods and Services Tax certificate India",
        "GSTIN certificate"
    ],
    "turnover_certificate": [
        "annual turnover certificate chartered accountant",
        "CA certificate turnover financial statement",
        "audited balance sheet turnover"
    ],
    "experience_letter": [
        "experience certificate work completion letter",
        "project completion certificate government",
        "past experience work order letter"
    ],
    "bank_statement": [
        "bank statement account statement",
        "bank solvency certificate",
        "bank guarantee letter"
    ],
    "pan_card": [
        "PAN card permanent account number India",
        "income tax PAN certificate"
    ],
    "incorporation_certificate": [
        "certificate of incorporation company registration",
        "ministry of corporate affairs registration"
    ],
    "msme_certificate": [
        "MSME certificate Udyam registration",
        "micro small medium enterprise certificate"
    ],
    "tender_document": [
        "tender document notice inviting tender",
        "government procurement tender RFP"
    ],
    "other": [
        "document certificate form"
    ]
}


class DocumentClassifier:
    """
    Classifies document type using CLIP zero-shot image classification.
    Returns document type label and confidence score.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        logger.info(f"Loading CLIP model: {model_name}")
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

        # build flat list of all text prompts and their type mapping
        self.all_prompts = []
        self.prompt_to_type = {}

        for doc_type, prompts in DOCUMENT_TYPES.items():
            for prompt in prompts:
                self.all_prompts.append(prompt)
                self.prompt_to_type[prompt] = doc_type

        logger.info(
            f"CLIP loaded with {len(DOCUMENT_TYPES)} document types, "
            f"{len(self.all_prompts)} prompts"
        )

    def classify(
        self,
        image: Union[np.ndarray, Image.Image]
    ) -> Tuple[str, float]:
        """
        Classify a single document image.

        Args:
            image: PIL Image or numpy array

        Returns:
            (document_type, confidence_score)
            e.g. ("gst_certificate", 0.87)
        """
        pil_image = self._ensure_pil(image)

        try:
            inputs = self.processor(
                text=self.all_prompts,
                images=pil_image,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            # image-text similarity scores
            logits = outputs.logits_per_image  # shape: [1, num_prompts]
            probs = logits.softmax(dim=1)      # normalize to probabilities

            best_idx = probs.argmax(dim=1).item()
            best_prompt = self.all_prompts[best_idx]
            confidence = probs[0, best_idx].item()
            doc_type = self.prompt_to_type[best_prompt]

            logger.debug(
                f"Document classified as '{doc_type}' "
                f"(confidence: {confidence:.3f})"
            )
            return doc_type, confidence

        except Exception as e:
            logger.error(f"CLIP classification failed: {e}")
            return "other", 0.0

    def classify_batch(
        self,
        images: List[Union[np.ndarray, Image.Image]]
    ) -> List[Tuple[str, float]]:
        """
        Classify a batch of document images.

        Returns:
            List of (document_type, confidence) tuples
        """
        return [self.classify(img) for img in images]

    def get_extraction_template(self, doc_type: str) -> dict:
        """
        Return the extraction template for a given document type.
        Templates define which fields RT-DETR should prioritize.

        Returns:
            Template dict with field definitions
        """
        templates = {
            "gst_certificate": {
                "key_fields": ["GSTIN", "legal_name", "date_of_registration",
                                "business_type", "state"],
                "table_expected": False
            },
            "turnover_certificate": {
                "key_fields": ["company_name", "financial_year", "turnover_amount",
                                "ca_name", "ca_registration", "date"],
                "table_expected": True
            },
            "experience_letter": {
                "key_fields": ["project_name", "client_name", "contract_value",
                                "start_date", "completion_date", "work_description"],
                "table_expected": False
            },
            "bank_statement": {
                "key_fields": ["account_number", "account_holder", "bank_name",
                                "balance", "statement_period"],
                "table_expected": True
            },
            "pan_card": {
                "key_fields": ["pan_number", "name", "date_of_birth", "father_name"],
                "table_expected": False
            },
            "incorporation_certificate": {
                "key_fields": ["company_name", "cin_number", "date_of_incorporation",
                                "registered_office"],
                "table_expected": False
            },
            "msme_certificate": {
                "key_fields": ["udyam_number", "enterprise_name", "date_of_registration",
                                "enterprise_type"],
                "table_expected": False
            }
        }
        return templates.get(doc_type, {
            "key_fields": [],
            "table_expected": False
        })

    def _ensure_pil(
        self, image: Union[np.ndarray, Image.Image]
    ) -> Image.Image:
        """Convert numpy array to PIL Image if needed."""
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, np.ndarray):
            import cv2
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return Image.fromarray(image)
        raise TypeError(f"Unsupported image type: {type(image)}")