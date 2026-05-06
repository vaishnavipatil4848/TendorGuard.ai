"""
two_pass_validator.py
Agent 1 — Architect Agent
Handles the self-critique pass logic separately —
checks for cross-referenced, implicit, or conditional criteria
that the first extraction pass may have missed.
"""

import json
import logging
from typing import List, Dict, Any

import anthropic

logger = logging.getLogger(__name__)


class TwoPassValidator:
    """
    Runs a second LLM pass over the tender text and initial criteria
    to catch missed cross-references, conditional criteria,
    and implicit requirements.
    """

    def __init__(self, model: str = "claude-opus-4-6"):
        self.client = anthropic.Anthropic()
        self.model = model

    def validate_and_enrich(
        self,
        tender_text: str,
        initial_criteria: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Run self-critique pass to catch missed criteria.

        Checks for:
        - Cross-referenced criteria ("as per clause 4.2")
        - Conditional criteria ("if bidder is MSME, then...")
        - Implicit criteria buried in general terms
        - Duplicate criteria that should be merged

        Returns:
            Enriched and deduplicated criteria list
        """
        logger.info("Running two-pass self-critique validation")

        prompt = self._build_critique_prompt(tender_text, initial_criteria)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}]
            )
            raw = response.content[0].text
            enriched = self._parse_response(raw)

            # merge with initial — enriched set takes precedence
            final = self._merge_criteria(initial_criteria, enriched)
            logger.info(
                f"Two-pass complete: {len(initial_criteria)} → {len(final)} criteria"
            )
            return final

        except Exception as e:
            logger.error(f"Two-pass validation failed: {e}")
            return initial_criteria

    def _build_critique_prompt(
        self,
        tender_text: str,
        initial_criteria: List[Dict[str, Any]]
    ) -> str:
        return f"""You are auditing the extraction of eligibility criteria from a government tender document.

Below is the tender text and the initial list of criteria that was extracted from it.

Your task:
1. Review the tender text carefully for any criteria that were MISSED in the initial extraction
2. Look specifically for:
   - Cross-referenced criteria (e.g. "as per clause X", "refer section Y")
   - Conditional criteria (e.g. "if the bidder is MSME...")
   - Implicit criteria buried in general terms or footnotes
   - Criteria stated as negative requirements (e.g. "bidder must NOT have...")
3. Check for duplicate criteria in the initial list that should be merged
4. Return the COMPLETE final list of criteria — including both the correct ones from the initial extraction and any new ones you found

Return ONLY a valid JSON array. No preamble, no explanation, no markdown.

Each criterion must follow this structure:
{{
  "criterion_id": "placeholder",
  "category": "eligibility|financial|technical|compliance",
  "requirement_text": "exact requirement as stated",
  "threshold": "specific value or null",
  "threshold_unit": "Cr|years|number|null",
  "document_required": "document name or null",
  "logical_operator": "AND|OR|NOT|null",
  "page_reference": page_number_or_null,
  "is_mandatory": true|false
}}

TENDER TEXT:
{tender_text[:6000]}

INITIAL CRITERIA EXTRACTED:
{json.dumps(initial_criteria, indent=2)}

Return the complete final JSON array:"""

    def _merge_criteria(
        self,
        initial: List[Dict[str, Any]],
        enriched: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Merge initial and enriched criteria lists.
        Uses requirement_text similarity to detect duplicates.
        Enriched list takes precedence on conflicts.
        """
        if not enriched:
            return initial

        # build lookup from enriched by requirement text (first 80 chars)
        enriched_keys = {
            c.get("requirement_text", "")[:80].lower(): c
            for c in enriched
        }

        # keep initial criteria not superseded by enriched
        merged = list(enriched)
        enriched_texts = set(enriched_keys.keys())

        for criterion in initial:
            key = criterion.get("requirement_text", "")[:80].lower()
            if key not in enriched_texts:
                merged.append(criterion)

        return merged

    def _parse_response(self, raw: str) -> List[Dict[str, Any]]:
        """Parse JSON array from LLM response."""
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1])

        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict) and "criteria" in parsed:
                return parsed["criteria"]
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Two-pass JSON parse error: {e}")
            return []