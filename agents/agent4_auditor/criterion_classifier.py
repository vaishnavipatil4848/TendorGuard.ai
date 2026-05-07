"""
criterion_classifier.py
Agent 4 — Auditor Agent
Classifies each criterion into one of three evaluation types:
  - Numerical  : has a numeric threshold (turnover >= 5 Cr, experience >= 3 years)
  - Logical    : boolean / document presence check (GST registered? = yes/no)
  - Semantic   : qualitative judgment required (past work "similar in nature")

The classification drives which evaluation path the Auditor uses:
  - Numerical  → programmatic comparison first, LLM only for ambiguous cases
  - Logical    → programmatic presence/absence check
  - Semantic   → always routed to dual-LLM chain-of-thought
"""

import re
import logging
from typing import Dict, Any, Literal

logger = logging.getLogger(__name__)

CriterionType = Literal["Numerical", "Logical", "Semantic"]

# Regex patterns that signal a numeric threshold
_NUMERIC_PATTERNS = [
    r"\d+\s*(?:crore|cr|lakh|lakhs|INR|₹)",      # financial values
    r"\d+\s*(?:years?|yrs?)",                      # experience years
    r"\d+\s*(?:projects?|works?|contracts?)",       # project counts
    r"(?:minimum|atleast|at least|>=|>|≥)\s*\d+",  # explicit minimum
    r"\d+\s*%",                                    # percentages
    r"(?:net worth|turnover|revenue|profit).{0,40}\d+",
]

# Keywords that signal a logical / document presence check
_LOGICAL_KEYWORDS = [
    "registered", "registration", "certificate", "certified",
    "license", "licensed", "gst", "pan", "msme", "dpiit",
    "must have", "should have", "shall have",
    "blacklisted", "debarred", "insolvent", "convicted",
    "valid", "active", "in force",
]

_NUMERIC_RE = re.compile("|".join(_NUMERIC_PATTERNS), re.IGNORECASE)
_LOGICAL_RE = re.compile(
    "|".join(re.escape(kw) for kw in _LOGICAL_KEYWORDS), re.IGNORECASE
)


class CriterionClassifier:
    """
    Classifies a criterion dict into Numerical / Logical / Semantic.

    Classification logic (priority order):
      1. If requirement_text or threshold contains a numeric pattern → Numerical
      2. If requirement_text matches logical/document keywords → Logical
      3. Otherwise → Semantic
    """

    def classify(self, criterion: Dict[str, Any]) -> CriterionType:
        """
        Classify a single criterion.

        Args:
            criterion: criterion dict from Agent 1 ruleset
                       Expected fields: requirement_text, threshold, category

        Returns:
            "Numerical" | "Logical" | "Semantic"
        """
        req_text = criterion.get("requirement_text", "")
        threshold = str(criterion.get("threshold") or "")
        combined = f"{req_text} {threshold}"

        if self._is_numerical(combined):
            criterion_type = "Numerical"
        elif self._is_logical(combined):
            criterion_type = "Logical"
        else:
            criterion_type = "Semantic"

        logger.debug(
            f"[{criterion.get('criterion_id', '?')}] "
            f"classified as {criterion_type} | "
            f"text: {req_text[:60]!r}"
        )
        return criterion_type

    def classify_batch(
        self, criteria: list[Dict[str, Any]]
    ) -> list[Dict[str, Any]]:
        """
        Classify all criteria and attach 'criterion_type' field in place.

        Returns:
            Same list with 'criterion_type' added to each criterion dict
        """
        for criterion in criteria:
            criterion["criterion_type"] = self.classify(criterion)

        counts = {"Numerical": 0, "Logical": 0, "Semantic": 0}
        for c in criteria:
            counts[c["criterion_type"]] += 1

        logger.info(
            f"Criterion classification complete: "
            f"Numerical={counts['Numerical']}, "
            f"Logical={counts['Logical']}, "
            f"Semantic={counts['Semantic']}"
        )
        return criteria

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _is_numerical(self, text: str) -> bool:
        return bool(_NUMERIC_RE.search(text))

    def _is_logical(self, text: str) -> bool:
        return bool(_LOGICAL_RE.search(text))