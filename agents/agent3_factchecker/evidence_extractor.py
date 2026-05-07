"""
evidence_extractor.py
Agent 3 — Fact-Checker Agent

Extracts specific evidence values from top reranked chunks.

Strategy:
  Programmatic (regex + spaCy) for structured fields → fast, reliable
  LLM (Claude) fallback for qualitative / ambiguous fields

Absorbed from uploaded code:
  - Structured evidence output schema (page, line, confidence, context_lines)
  - Defensive numeric parsing (strip ₹, commas, Cr/Lakh conversion)
  - ambiguity_reason field
  - context_lines field
  - extracted_numeric as clean float
"""

import json
import re
import logging
from typing import List, Dict, Any, Optional

import anthropic

logger = logging.getLogger(__name__)

QUALITATIVE_EXTRACTION_PROMPT_PATH = (
    "agents/agent3_factchecker/prompts/qualitative_extraction_prompt.txt"
)

# Regex patterns for common structured fields in Indian govt documents
PATTERNS = {
    "turnover_cr":     r'([\d,]+(?:\.\d+)?)\s*(?:cr(?:ore)?s?)',
    "turnover_lakh":   r'([\d,]+(?:\.\d+)?)\s*(?:lakh|lac)',
    "amount_rs":       r'(?:₹|rs\.?|inr)\s*([\d,]+(?:\.\d+)?)',
    "gstin":           r'\b\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}[Z]{1}[A-Z\d]{1}\b',
    "pan":             r'\b[A-Z]{5}\d{4}[A-Z]{1}\b',
    "cin":             r'\b[LU]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6}\b',
    "udyam":           r'\bUDYAM-[A-Z]{2}-\d{2}-\d{7}\b',
    "date":            r'\b\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}\b',
    "project_count":   r'\b(\d+)\s+(?:similar\s+)?(?:projects?|works?)\b',
    "years_experience": r'\b(\d+)\s+years?\b',
}

# Which patterns to try per criterion category
CATEGORY_PATTERNS = {
    "financial":   ["turnover_cr", "turnover_lakh", "amount_rs"],
    "compliance":  ["gstin", "pan", "cin", "udyam"],
    "technical":   ["project_count", "years_experience", "date"],
    "eligibility": list(PATTERNS.keys()),
}


class EvidenceExtractor:
    """
    Extracts specific evidence from top reranked chunks.
    Tries programmatic extraction first; falls back to Claude for
    qualitative or unstructured criteria.
    """

    def __init__(self):
        self.client = anthropic.Anthropic()
        self.model  = "claude-sonnet-4-6"
        self._load_prompt()

    def extract(
        self,
        criterion:  Dict[str, Any],
        top_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Extract evidence from top reranked chunks.

        Args:
            criterion:  criterion dict from Agent 2
            top_chunks: top-5 chunks from CrossEncoderReranker

        Returns:
            Evidence dict with page, line, bbox, confidence etc.
            Compatible with Agent 4 VerdictAgent input schema.
        """
        if not top_chunks:
            return self._empty(criterion, "No relevant chunks retrieved after reranking")

        category      = criterion.get("criterion_type", "eligibility").lower()
        threshold_raw = criterion.get("threshold_value")

        # Step 1 — programmatic extraction
        prog = self._programmatic(criterion, top_chunks, category, threshold_raw)
        if prog["found"] and prog["confidence"] >= 0.80:
            logger.debug(
                f"Programmatic extraction succeeded: "
                f"{criterion.get('id')} → {prog['extracted_value']}"
            )
            return prog

        # Step 2 — LLM extraction fallback
        logger.debug(
            f"LLM extraction fallback for: {criterion.get('id')}"
        )
        return self._llm_extract(criterion, top_chunks)

    # ── Programmatic extraction ───────────────────────────────────────────────

    def _programmatic(
        self,
        criterion:     Dict[str, Any],
        chunks:        List[Dict[str, Any]],
        category:      str,
        threshold_raw: Optional[str]
    ) -> Dict[str, Any]:
        """
        Regex-based extraction across top chunks.
        Returns on first successful match.
        """
        threshold_numeric = self._parse_numeric(threshold_raw)
        patterns_to_try   = CATEGORY_PATTERNS.get(category, list(PATTERNS.keys()))

        for chunk in chunks:
            text = chunk.get("text", "")

            for pattern_name in patterns_to_try:
                pattern = PATTERNS.get(pattern_name)
                if not pattern:
                    continue

                matches = re.findall(pattern, text, re.IGNORECASE)
                if not matches:
                    continue

                raw_val   = matches[0] if isinstance(matches[0], str) else matches[0]
                extracted = self._parse_numeric(str(raw_val))

                if extracted is None:
                    # non-numeric match (GSTIN, PAN etc.) — still valid evidence
                    return {
                        "found":             True,
                        "page_number":       chunk.get("page_number", 1),
                        "line_number":       chunk.get("line_start", 1),
                        "matching_line_text": text[:200],
                        "context_lines":     [c.get("text", "")[:150] for c in chunks[:3]],
                        "extracted_value":   str(raw_val),
                        "extracted_numeric": None,
                        "threshold_numeric": threshold_numeric,
                        "meets_threshold":   True,  # registration codes just need presence
                        "confidence":        float(chunk.get("reranker_score", 0.8)),
                        "ambiguity_reason":  None,
                        "source_document":   chunk.get("source_file", ""),
                        "criterion_id":      criterion.get("id"),
                        "extraction_method": "programmatic"
                    }

                # normalize units to absolute value
                extracted = self._normalize_units(extracted, text)
                thr_norm  = self._normalize_threshold(threshold_numeric, text)

                meets = None
                if thr_norm is not None:
                    meets = extracted >= thr_norm

                return {
                    "found":             True,
                    "page_number":       chunk.get("page_number", 1),
                    "line_number":       chunk.get("line_start", 1),
                    "matching_line_text": text[:200],
                    "context_lines":     [c.get("text", "")[:150] for c in chunks[:3]],
                    "extracted_value":   str(raw_val),
                    "extracted_numeric": extracted,
                    "threshold_numeric": thr_norm,
                    "meets_threshold":   meets,
                    "confidence":        float(chunk.get("reranker_score", 0.8)),
                    "ambiguity_reason":  None,
                    "source_document":   chunk.get("source_file", ""),
                    "criterion_id":      criterion.get("id"),
                    "extraction_method": "programmatic"
                }

        return self._empty(criterion, "No programmatic patterns matched")

    # ── LLM extraction ────────────────────────────────────────────────────────

    def _llm_extract(
        self,
        criterion: Dict[str, Any],
        chunks:    List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Claude-based extraction for qualitative criteria.
        Uses top-5 chunks as focused context.
        """
        focused_context = "\n\n".join([
            f"[Page {c.get('page_number', 1)}, "
            f"Line {c.get('line_start', 1)}]\n{c.get('text', '')}"
            for c in chunks[:5]
        ])

        prompt = self.qualitative_prompt.format(
            criterion_id   = criterion.get("id", ""),
            criterion_name = criterion.get("name", ""),
            description    = criterion.get("description", ""),
            threshold      = criterion.get("threshold_value", "N/A"),
            unit           = criterion.get("unit", ""),
            context        = focused_context
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            raw = response.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

            evidence = json.loads(raw)
            evidence["source_document"]   = chunks[0].get("source_file", "") if chunks else ""
            evidence["criterion_id"]      = criterion.get("id")
            evidence["extraction_method"] = "llm"

            # defensive numeric parsing — absorbed from uploaded code
            for field in ("extracted_numeric", "threshold_numeric"):
                val = evidence.get(field)
                if val is not None:
                    evidence[field] = self._parse_numeric(str(val))

            return evidence

        except Exception as e:
            logger.error(
                f"LLM extraction failed for {criterion.get('id')}: {e}"
            )
            return self._empty(criterion, f"LLM extraction error: {str(e)}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _parse_numeric(self, value: Optional[str]) -> Optional[float]:
        """
        Robustly parse a numeric value from a string.
        Handles: ₹, Rs., commas, Cr, Lakh suffixes.
        Absorbed from uploaded code's defensive parsing pattern.
        """
        if value is None:
            return None
        clean = str(value)
        clean = re.sub(r'[₹,\s]', '', clean)
        clean = re.sub(r'(?i)rs\.?', '', clean)
        clean = re.sub(r'(?i)(crore|cr)', '', clean)
        clean = re.sub(r'(?i)(lakh|lac)', '', clean)
        try:
            return float(clean)
        except ValueError:
            return None

    def _normalize_units(self, value: float, text: str) -> float:
        """Convert Cr/Lakh amounts to absolute rupee values."""
        text_lower = text.lower()
        if "crore" in text_lower or " cr" in text_lower:
            return value * 10_000_000
        elif "lakh" in text_lower or "lac" in text_lower:
            return value * 100_000
        return value

    def _normalize_threshold(
        self,
        threshold: Optional[float],
        text:      str
    ) -> Optional[float]:
        """
        Normalize the threshold to the same unit as the extracted value.
        If threshold is given in Cr (e.g. 5.0) and text shows absolute values,
        convert accordingly.
        """
        if threshold is None:
            return None
        # if threshold looks like it's in Cr (< 10000), normalize to absolute
        if threshold < 10_000:
            text_lower = text.lower()
            if "crore" in text_lower or " cr" in text_lower:
                return threshold * 10_000_000
            elif "lakh" in text_lower or "lac" in text_lower:
                return threshold * 100_000
        return threshold

    def _empty(self, criterion: Dict[str, Any], reason: str) -> Dict[str, Any]:
        """Standardized empty evidence dict."""
        return {
            "found":             False,
            "page_number":       None,
            "line_number":       None,
            "matching_line_text": None,
            "context_lines":     [],
            "extracted_value":   None,
            "extracted_numeric": None,
            "threshold_numeric": None,
            "meets_threshold":   None,
            "confidence":        0.0,
            "ambiguity_reason":  reason,
            "source_document":   "",
            "criterion_id":      criterion.get("id"),
            "extraction_method": "none"
        }

    def _load_prompt(self) -> None:
        """Load the qualitative extraction prompt template."""
        try:
            with open(QUALITATIVE_EXTRACTION_PROMPT_PATH, "r", encoding="utf-8") as f:
                self.qualitative_prompt = f.read()
        except FileNotFoundError:
            # inline fallback prompt
            self.qualitative_prompt = (
                "You are a Lead Procurement Auditor.\n"
                "CRITERION: {criterion_name} (ID: {criterion_id})\n"
                "REQUIREMENT: {description}\n"
                "THRESHOLD: {threshold} {unit}\n\n"
                "DOCUMENT CONTENT:\n{context}\n\n"
                "Extract evidence. Return ONLY valid JSON:\n"
                '{{"found": bool, "page_number": int, "line_number": int, '
                '"matching_line_text": str, "context_lines": [str], '
                '"extracted_value": str, "extracted_numeric": float or null, '
                '"threshold_numeric": float or null, "meets_threshold": bool or null, '
                '"confidence": float, "ambiguity_reason": str or null}}'
            )