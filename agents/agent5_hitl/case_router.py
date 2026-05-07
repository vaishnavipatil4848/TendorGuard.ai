"""
case_router.py
Agent 5 — HITL Agent
Routes Auditor verdicts into one of four review case types,
or auto-approves them if confidence is high and models agree.

Case types (priority order in the review queue):
  MISSING_EVIDENCE   — Agent 3 returned no evidence for the criterion
  MODEL_DISAGREEMENT — GPT-4o and Claude returned different verdicts
  LOW_CONFIDENCE     — composite confidence below HITL threshold
  AMBIGUOUS_CRITERION— criterion itself was flagged uncertain by Agent 1

Context-aware: each case type gets a different review interface
in the Streamlit UI (different fields pre-filled, different instructions).
"""

import logging
from typing import Dict, Any, Literal

logger = logging.getLogger(__name__)

# Matches HITL_CONFIDENCE_THRESHOLD in .env
DEFAULT_HITL_THRESHOLD = 0.70

CaseType = Literal[
    "MISSING_EVIDENCE",
    "MODEL_DISAGREEMENT",
    "LOW_CONFIDENCE",
    "AMBIGUOUS_CRITERION",
    "AUTO_APPROVED"
]

# Human-readable context shown in the Streamlit review panel per case type
CASE_CONTEXT = {
    "MISSING_EVIDENCE": (
        "No supporting evidence was found in the bidder's documents for this criterion. "
        "Please check if the document was submitted under a different name or category, "
        "or mark as FAIL if genuinely absent."
    ),
    "MODEL_DISAGREEMENT": (
        "GPT-4o and Claude reached different verdicts on this criterion. "
        "Both reasoning chains are shown below. "
        "Review the evidence and select the correct verdict."
    ),
    "LOW_CONFIDENCE": (
        "The automated system has low confidence in this verdict. "
        "This is often caused by blurry scans, handwritten text, or ambiguous phrasing. "
        "Please review the highlighted evidence and confirm or override."
    ),
    "AMBIGUOUS_CRITERION": (
        "The criterion itself was flagged as ambiguous or incomplete during tender parsing. "
        "Please interpret the requirement and apply your judgment to the evidence shown."
    ),
}


def route_case(
    verdict: Dict[str, Any],
    hitl_threshold: float = DEFAULT_HITL_THRESHOLD
) -> CaseType:
    """
    Determine the review case type for a single criterion verdict.

    Args:
        verdict:         verdict dict from AuditorAgent._evaluate_criterion()
                         Expected fields:
                           route_to_hitl, composite_confidence,
                           agreement_status, verdict (PASS/FAIL/UNCERTAIN),
                           programmatic, ambiguities
        hitl_threshold:  confidence below which a case is routed to HITL

    Returns:
        CaseType string
    """
    # Already flagged by confidence scorer
    if not verdict.get("route_to_hitl", False):
        return "AUTO_APPROVED"

    composite_conf = float(
        verdict.get("composite_confidence", verdict.get("confidence", 0.0))
    )
    agreement_status = verdict.get("agreement_status", "")
    ambiguities = verdict.get("ambiguities", [])
    extracted_value = verdict.get("extracted_value", None)

    # Priority 1 — no evidence found
    if extracted_value is None or str(extracted_value).strip() == "":
        logger.debug(
            f"[{verdict.get('criterion_id')}] → MISSING_EVIDENCE"
        )
        return "MISSING_EVIDENCE"

    # Priority 2 — models disagreed
    if agreement_status == "DISAGREEMENT":
        logger.debug(
            f"[{verdict.get('criterion_id')}] → MODEL_DISAGREEMENT"
        )
        return "MODEL_DISAGREEMENT"

    # Priority 3 — ambiguous criterion (flagged by Agent 1 or LLM ambiguities)
    if ambiguities and composite_conf < hitl_threshold:
        logger.debug(
            f"[{verdict.get('criterion_id')}] → AMBIGUOUS_CRITERION"
        )
        return "AMBIGUOUS_CRITERION"

    # Priority 4 — low confidence (catch-all)
    logger.debug(
        f"[{verdict.get('criterion_id')}] → LOW_CONFIDENCE (conf={composite_conf:.2f})"
    )
    return "LOW_CONFIDENCE"


def get_case_context(case_type: CaseType) -> str:
    """Return the human-readable reviewer context string for a case type."""
    return CASE_CONTEXT.get(case_type, "")