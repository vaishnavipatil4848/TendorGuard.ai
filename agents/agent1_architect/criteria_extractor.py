"""
criteria_extractor.py
Agent 1 — Architect Agent
Uses Claude with schema-driven structured output to extract
eligibility criteria from classified tender sections into JSON ruleset.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any

import anthropic

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"

# Strict JSON schema for each criterion
CRITERION_SCHEMA = {
    "type": "object",
    "properties": {
        "criterion_id":       {"type": "string"},
        "category":           {"type": "string",
                               "enum": ["eligibility", "financial",
                                        "technical", "compliance"]},
        "requirement_text":   {"type": "string"},
        "threshold":          {"type": ["string", "null"]},
        "threshold_unit":     {"type": ["string", "null"]},
        "document_required":  {"type": ["string", "null"]},
        "logical_operator":   {"type": ["string", "null"],
                               "enum": ["AND", "OR", "NOT", None]},
        "page_reference":     {"type": ["integer", "null"]},
        "is_mandatory":       {"type": "boolean"}
    },
    "required": [
        "criterion_id", "category", "requirement_text",
        "threshold", "document_required", "is_mandatory"
    ]
}


class CriteriaExtractor:
    """
    Extracts structured eligibility criteria from tender text sections
    using Claude with schema-driven prompting.
    Runs a two-pass extraction — initial pass + self-critique.
    """

    def __init__(self, model: str = "claude-opus-4-6"):
        self.client = anthropic.Anthropic()
        self.model = model
        self.extraction_prompt = self._load_prompt("extraction_prompt.txt")
        self.self_critique_prompt = self._load_prompt("self_critique_prompt.txt")

    def extract(
        self, sections: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Full two-pass extraction pipeline.

        Args:
            sections: grouped sections from SectionClassifier
                      keys: eligibility / financial / technical / compliance

        Returns:
            List of criterion dicts matching CRITERION_SCHEMA
        """
        # combine relevant sections into text
        relevant_text = self._build_context(sections)

        if not relevant_text.strip():
            logger.warning("No relevant section text found for extraction")
            return []

        logger.info("Pass 1: Initial criteria extraction")
        initial_criteria = self._run_extraction(relevant_text)

        logger.info(f"Pass 1 complete: {len(initial_criteria)} criteria extracted")

        logger.info("Pass 2: Self-critique for missed criteria")
        final_criteria = self._run_self_critique(relevant_text, initial_criteria)

        logger.info(f"Pass 2 complete: {len(final_criteria)} criteria after critique")

        # assign sequential IDs
        for i, criterion in enumerate(final_criteria):
            criterion["criterion_id"] = f"C{i+1:03d}"

        return final_criteria

    def _run_extraction(self, context_text: str) -> List[Dict[str, Any]]:
        """
        Pass 1: Initial LLM extraction of criteria.
        """
        prompt = self.extraction_prompt.format(
            tender_text=context_text,
            schema=json.dumps(CRITERION_SCHEMA, indent=2)
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}]
            )
            raw = response.content[0].text
            return self._parse_json_response(raw)

        except Exception as e:
            logger.error(f"Pass 1 extraction failed: {e}")
            return []

    def _run_self_critique(
        self,
        context_text: str,
        initial_criteria: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Pass 2: Self-critique — ask Claude to review the first pass
        and identify any missed or cross-referenced criteria.
        """
        prompt = self.self_critique_prompt.format(
            tender_text=context_text,
            initial_criteria=json.dumps(initial_criteria, indent=2)
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}]
            )
            raw = response.content[0].text
            return self._parse_json_response(raw)

        except Exception as e:
            logger.error(f"Pass 2 self-critique failed: {e}")
            logger.warning("Falling back to Pass 1 results")
            return initial_criteria

    def _build_context(
        self, sections: Dict[str, List[Dict[str, Any]]]
    ) -> str:
        """
        Build a single context string from relevant sections.
        Excludes boilerplate. Adds section headers for clarity.
        """
        relevant_categories = ["eligibility", "financial", "technical", "compliance"]
        parts = []

        for category in relevant_categories:
            regions = sections.get(category, [])
            if not regions:
                continue

            parts.append(f"\n=== {category.upper()} SECTION ===\n")
            for region in regions:
                text = region.get("text", "").strip()
                page = region.get("page_number", "?")
                if text:
                    parts.append(f"[Page {page}] {text}")

        return "\n".join(parts)

    def _parse_json_response(self, raw: str) -> List[Dict[str, Any]]:
        """
        Parse the LLM response — handles markdown code blocks
        and direct JSON arrays.
        """
        # strip markdown code fences if present
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
            logger.warning("Unexpected JSON structure from LLM")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            logger.debug(f"Raw response: {raw[:500]}")
            return []

    def _load_prompt(self, filename: str) -> str:
        """Load a prompt template from the prompts directory."""
        path = PROMPTS_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        return path.read_text(encoding="utf-8")