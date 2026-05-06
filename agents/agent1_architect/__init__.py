"""
Agent 1 — Architect Agent
Orchestrates the full tender parsing pipeline:
LayoutLMv3 → Section Classification → LLM Extraction → Validation
"""

import logging
from typing import List, Dict, Any, Tuple

from .layout_parser import LayoutParser
from .section_classifier import SectionClassifier
from .criteria_extractor import CriteriaExtractor
from .two_pass_validator import TwoPassValidator
from .ruleset_validator import RulesetValidator

logger = logging.getLogger(__name__)


class ArchitectAgent:
    """
    Main entry point for Agent 1.
    Parses a tender PDF and returns a validated ruleset JSON.
    """

    def __init__(self):
        logger.info("Initializing Architect Agent")
        self.layout_parser = LayoutParser()
        self.section_classifier = SectionClassifier()
        self.criteria_extractor = CriteriaExtractor()
        self.two_pass_validator = TwoPassValidator()
        self.ruleset_validator = RulesetValidator()

    def run(
        self,
        pdf_path: str,
        tender_id: str,
        ruleset_output_dir: str = "storage/rulesets"
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Full pipeline: PDF → validated ruleset JSON.

        Args:
            pdf_path: path to tender PDF
            tender_id: unique identifier for this tender
            ruleset_output_dir: where to save the ruleset JSON

        Returns:
            (criteria, validation_report)
        """
        logger.info(f"Architect Agent starting for tender: {tender_id}")

        # Step 1 — LayoutLMv3 parsing
        logger.info("Step 1: Layout parsing")
        regions = self.layout_parser.parse_pdf(pdf_path)
        regions = self.layout_parser.filter_relevant_regions(regions)

        # Step 2 — Section classification
        logger.info("Step 2: Section classification")
        classified = self.section_classifier.classify_regions(regions)
        sections = self.section_classifier.group_into_sections(classified)

        # Step 3 — LLM criteria extraction (two-pass)
        logger.info("Step 3: Criteria extraction")
        criteria = self.criteria_extractor.extract(sections)

        # Step 4 — Validate ruleset
        logger.info("Step 4: Ruleset validation")
        is_valid, report = self.ruleset_validator.validate(criteria)

        if not is_valid:
            logger.error(
                "Ruleset has failed criteria — human sign-off required before proceeding"
            )

        # Step 5 — Save ruleset
        saved_path = self.ruleset_validator.save_ruleset(
            criteria, ruleset_output_dir, tender_id
        )
        logger.info(f"Ruleset saved to: {saved_path}")
        logger.info(f"Architect Agent complete: {len(criteria)} criteria extracted")

        return criteria, report