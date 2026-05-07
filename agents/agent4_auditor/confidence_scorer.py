"""
confidence_scorer.py
Agent 4 — Auditor Agent
Computes the composite confidence score for each criterion verdict.

Formula (per brainstorm doc):
  composite = w_ocr * ocr_conf
             + w_retrieval * retrieval_conf
             + w_llm * llm_conf
             + w_agreement * agreement_bonus
             + w_extraction * extraction_bonus

Weights are designed so model agreement is the highest contributor.
Extraction type (programmatic vs LLM) also modulates final score.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Weight table — must sum to 1.0
WEIGHTS = {
    "ocr":          0.15,   # confidence from Agent 2 vision pipeline
    "retrieval":    0.20,   # confidence from Agent 3 retrieval/reranking
    "llm":          0.25,   # average of both LLM confidence scores
    "agreement":    0.30,   # agreement bonus: STRONG > WEAK > DISAGREEMENT
    "extraction":   0.10,   # extraction type bonus: programmatic > LLM
}

# Agreement status → score contribution
AGREEMENT_SCORES = {
    "AGREEMENT_STRONG": 1.0,
    "AGREEMENT_WEAK":   0.6,
    "DISAGREEMENT":     0.2,
    "PROGRAMMATIC":     0.95,  # no LLM comparison — high trust if programmatic
}

# Extraction type → score contribution
EXTRACTION_SCORES = {
    "programmatic":          1.0,
    "regex":                 0.95,
    "spacy":                 0.85,
    "llm_qualitative":       0.65,
    "unknown":               0.50,
}


class ConfidenceScorer:
    """
    Computes a weighted composite confidence score for a criterion verdict.
    Takes inputs from Agent 2 (OCR confidence), Agent 3 (retrieval confidence),
    and Agent 4's own evaluation outputs.
    """

    def score(
        self,
        evidence: Dict[str, Any],
        agreement_report: Dict[str, Any],
        programmatic: bool = False
    ) -> Dict[str, Any]:
        """
        Compute composite confidence score.

        Args:
            evidence:          evidence dict from Agent 3
            agreement_report:  dict from AgreementChecker.check()
                               OR None if programmatic check was used
            programmatic:      True if NumericalChecker / LogicalChecker ran
                               (no dual-LLM involved)

        Returns:
            scoring_report dict:
            {
                composite_confidence,   # final 0.0–1.0 score
                component_scores,       # breakdown per component
                confidence_band,        # HIGH | MEDIUM | LOW
                route_to_hitl,          # bool — should this go to human review?
            }
        """
        ocr_conf = float(evidence.get("ocr_confidence", 0.7))
        retrieval_conf = float(evidence.get("confidence", 0.6))
        extraction_type = evidence.get("extraction_type", "unknown")

        extraction_score = EXTRACTION_SCORES.get(
            extraction_type, EXTRACTION_SCORES["unknown"]
        )

        if programmatic:
            llm_score = 0.0
            agreement_score = AGREEMENT_SCORES["PROGRAMMATIC"]
            # for programmatic, agreement weight goes to extraction weight
            composite = (
                WEIGHTS["ocr"] * ocr_conf
                + WEIGHTS["retrieval"] * retrieval_conf
                + (WEIGHTS["llm"] + WEIGHTS["agreement"]) * agreement_score
                + WEIGHTS["extraction"] * extraction_score
            )
        else:
            # LLM path
            claude_conf = float(
                agreement_report.get("claude_confidence") or
                agreement_report.get("weight_adjusted_confidence", 0.5)
            )
            gpt4o_conf = float(
                agreement_report.get("gpt4o_confidence") or
                agreement_report.get("weight_adjusted_confidence", 0.5)
            )
            llm_score = (claude_conf + gpt4o_conf) / 2.0

            agreement_status = agreement_report.get("agreement_status", "DISAGREEMENT")
            agreement_score = AGREEMENT_SCORES.get(agreement_status, 0.2)

            composite = (
                WEIGHTS["ocr"] * ocr_conf
                + WEIGHTS["retrieval"] * retrieval_conf
                + WEIGHTS["llm"] * llm_score
                + WEIGHTS["agreement"] * agreement_score
                + WEIGHTS["extraction"] * extraction_score
            )

        composite = round(max(0.0, min(1.0, composite)), 3)

        # classify into confidence band
        if composite >= 0.75:
            band = "HIGH"
        elif composite >= 0.50:
            band = "MEDIUM"
        else:
            band = "LOW"

        # route to HITL if low confidence OR model disagreement
        route_to_hitl = (
            band == "LOW" or
            (not programmatic and
             agreement_report.get("agreement_status") == "DISAGREEMENT")
        )

        component_scores = {
            "ocr_confidence":        round(ocr_conf, 3),
            "retrieval_confidence":  round(retrieval_conf, 3),
            "llm_confidence":        round(llm_score, 3) if not programmatic else "N/A",
            "agreement_score":       round(agreement_score, 3),
            "extraction_score":      round(extraction_score, 3),
        }

        cid = evidence.get("criterion_id", "?")
        logger.info(
            f"[{cid}] Composite confidence: {composite} ({band}) | "
            f"HITL={route_to_hitl}"
        )

        return {
            "composite_confidence": composite,
            "component_scores": component_scores,
            "confidence_band": band,
            "route_to_hitl": route_to_hitl,
        }