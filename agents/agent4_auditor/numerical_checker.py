"""
numerical_checker.py
Agent 4 — Auditor Agent
Programmatic evaluation of Numerical and Logical criteria.
No LLM call — pure Python comparison against extracted evidence values.

Why programmatic?
  - Eliminates LLM hallucination on objective comparisons
  - Deterministic and auditable
  - Faster and cheaper
  - LLM is reserved for Semantic criteria and disambiguation
"""

import re
import logging
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

# Regex to pull the first numeric value (handles 1,00,00,000 Indian formatting)
_VALUE_RE = re.compile(r"[\d,]+\.?\d*")


class NumericalChecker:
    """
    Programmatically evaluates Numerical and Logical criteria
    against evidence extracted by Agent 3.

    Returns a verdict dict compatible with the Auditor's output schema.
    """

    def evaluate(
        self,
        criterion: Dict[str, Any],
        evidence: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate a single criterion against its evidence.

        Args:
            criterion: criterion dict (with criterion_type attached)
            evidence: evidence dict from Agent 3 FactChecker
                      Expected fields:
                        extracted_value, extracted_unit,
                        field_name, confidence, raw_text

        Returns:
            verdict dict:
            {
                criterion_id, criterion_type,
                verdict,           # PASS | FAIL | UNCERTAIN
                confidence,        # 0.0–1.0
                reasoning,
                key_factor,
                programmatic,      # True — signals no LLM was used
            }
        """
        c_type = criterion.get("criterion_type", "Numerical")
        cid = criterion.get("criterion_id", "?")

        if c_type == "Logical":
            return self._evaluate_logical(criterion, evidence)
        else:
            return self._evaluate_numerical(criterion, evidence)

    def _evaluate_numerical(
        self,
        criterion: Dict[str, Any],
        evidence: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare extracted numeric value against criterion threshold."""
        cid = criterion.get("criterion_id", "?")
        threshold_raw = str(criterion.get("threshold") or "")
        unit = str(criterion.get("threshold_unit") or "")
        req_text = criterion.get("requirement_text", "")

        extracted_value = evidence.get("extracted_value")
        evidence_confidence = float(evidence.get("confidence", 0.5))

        # parse threshold number
        threshold_val = self._parse_number(threshold_raw)
        if threshold_val is None:
            logger.warning(
                f"[{cid}] Could not parse threshold from: {threshold_raw!r}"
            )
            return self._uncertain(
                criterion,
                reason=f"Threshold could not be parsed: {threshold_raw!r}",
                key_factor="unparseable_threshold"
            )

        # parse extracted value
        extracted_val = self._parse_number(str(extracted_value or ""))
        if extracted_val is None:
            logger.warning(
                f"[{cid}] No extractable numeric value in evidence: "
                f"{extracted_value!r}"
            )
            return self._uncertain(
                criterion,
                reason=f"Extracted value could not be parsed: {extracted_value!r}",
                key_factor="missing_extracted_value"
            )

        # normalise units (Cr / Lakh → base INR if both are financial)
        extracted_val, threshold_val = self._normalise_units(
            extracted_val, threshold_val, unit
        )

        # comparison: extracted must be >= threshold
        passed = extracted_val >= threshold_val
        verdict = "PASS" if passed else "FAIL"

        reasoning = (
            f"Criterion requires {req_text}. "
            f"Extracted value: {extracted_val:,.2f} {unit}. "
            f"Required threshold: {threshold_val:,.2f} {unit}. "
            f"{'Meets' if passed else 'Does not meet'} the requirement."
        )

        # composite confidence: penalise low evidence confidence
        confidence = round(0.95 * evidence_confidence, 3)

        logger.info(
            f"[{cid}] Numerical check: {extracted_val} >= {threshold_val} "
            f"→ {verdict} (conf={confidence:.2f})"
        )

        return {
            "criterion_id": cid,
            "criterion_type": "Numerical",
            "verdict": verdict,
            "confidence": confidence,
            "reasoning": reasoning,
            "key_factor": f"extracted={extracted_val}, threshold={threshold_val}",
            "ambiguities": [],
            "programmatic": True,
        }

    def _evaluate_logical(
        self,
        criterion: Dict[str, Any],
        evidence: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check boolean presence/absence of a document or registration."""
        cid = criterion.get("criterion_id", "?")
        req_text = criterion.get("requirement_text", "")
        doc_required = criterion.get("document_required", "")
        evidence_confidence = float(evidence.get("confidence", 0.5))

        extracted_value = evidence.get("extracted_value", "")
        raw_text = evidence.get("raw_text", "")

        # evidence present at all?
        has_evidence = bool(
            (extracted_value and str(extracted_value).strip()) or
            (raw_text and str(raw_text).strip())
        )

        if not has_evidence:
            return self._fail(
                criterion,
                reason=f"No evidence found for required document: {doc_required!r}",
                key_factor="missing_document",
                evidence_confidence=evidence_confidence
            )

        # look for explicit negative signals (blacklisted, expired, cancelled)
        negative_signals = ["blacklisted", "debarred", "expired", "cancelled",
                            "revoked", "invalid", "not registered"]
        combined_text = f"{extracted_value} {raw_text}".lower()
        for signal in negative_signals:
            if signal in combined_text:
                return self._fail(
                    criterion,
                    reason=f"Negative signal detected in evidence: '{signal}'",
                    key_factor=f"negative_signal:{signal}",
                    evidence_confidence=evidence_confidence
                )

        verdict = "PASS"
        confidence = round(0.90 * evidence_confidence, 3)
        reasoning = (
            f"Criterion requires: {req_text}. "
            f"Evidence for '{doc_required}' found and no negative signals detected."
        )

        logger.info(f"[{cid}] Logical check → {verdict} (conf={confidence:.2f})")

        return {
            "criterion_id": cid,
            "criterion_type": "Logical",
            "verdict": verdict,
            "confidence": confidence,
            "reasoning": reasoning,
            "key_factor": f"document_present:{doc_required}",
            "ambiguities": [],
            "programmatic": True,
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _parse_number(self, text: str) -> Optional[float]:
        """Extract the first numeric value from a string."""
        if not text:
            return None
        # remove commas used as thousand separators (Indian style)
        cleaned = text.replace(",", "")
        match = _VALUE_RE.search(cleaned)
        if match:
            try:
                return float(match.group())
            except ValueError:
                pass
        return None

    def _normalise_units(
        self,
        extracted: float,
        threshold: float,
        unit: str
    ) -> Tuple[float, float]:
        """
        Attempt to normalise financial units so comparison is fair.
        E.g. if threshold is in Cr but extracted is raw INR, convert.
        Returns (extracted_normalised, threshold_normalised).
        Simple heuristic: if extracted >> threshold by factor 1e7, assume raw INR → Cr.
        """
        unit_lower = unit.lower()
        if "cr" in unit_lower or "crore" in unit_lower:
            # if extracted looks like it's in raw rupees (>= 1e7 times threshold)
            if threshold > 0 and extracted / threshold >= 1e5:
                extracted = extracted / 1e7  # convert paise or raw INR to Cr
        return extracted, threshold

    def _uncertain(
        self, criterion: Dict[str, Any], reason: str, key_factor: str
    ) -> Dict[str, Any]:
        return {
            "criterion_id": criterion.get("criterion_id", "?"),
            "criterion_type": criterion.get("criterion_type", "Numerical"),
            "verdict": "UNCERTAIN",
            "confidence": 0.3,
            "reasoning": reason,
            "key_factor": key_factor,
            "ambiguities": [reason],
            "programmatic": True,
        }

    def _fail(
        self,
        criterion: Dict[str, Any],
        reason: str,
        key_factor: str,
        evidence_confidence: float = 0.5
    ) -> Dict[str, Any]:
        return {
            "criterion_id": criterion.get("criterion_id", "?"),
            "criterion_type": criterion.get("criterion_type", "Logical"),
            "verdict": "FAIL",
            "confidence": round(0.85 * evidence_confidence, 3),
            "reasoning": reason,
            "key_factor": key_factor,
            "ambiguities": [],
            "programmatic": True,
        }