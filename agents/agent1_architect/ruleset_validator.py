"""
ruleset_validator.py
Agent 1 — Architect Agent
Validates the extracted ruleset for completeness before
the pipeline runs. Flags incomplete criteria for human sign-off.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# Fields that must be present and non-null for a criterion to be complete
MANDATORY_FIELDS = [
    "criterion_id",
    "category",
    "requirement_text",
    "is_mandatory"
]

# Fields that are important but can be flagged rather than blocking
IMPORTANT_FIELDS = [
    "threshold",
    "document_required"
]


class RulesetValidator:
    """
    Validates the extracted ruleset JSON for completeness and consistency.
    Returns a validation report with passed, flagged, and failed criteria.
    """

    def validate(
        self, criteria: List[Dict[str, Any]]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate all extracted criteria.

        Returns:
            (is_valid, report)
            is_valid: True if all mandatory fields are present
            report: detailed validation report
        """
        if not criteria:
            logger.error("Empty ruleset — no criteria extracted")
            return False, {"error": "Empty ruleset", "criteria": []}

        passed = []
        flagged = []
        failed = []

        for criterion in criteria:
            status, issues = self._validate_criterion(criterion)
            entry = {
                "criterion_id": criterion.get("criterion_id", "UNKNOWN"),
                "requirement_text": criterion.get("requirement_text", "")[:100],
                "issues": issues
            }

            if status == "passed":
                passed.append(entry)
            elif status == "flagged":
                flagged.append(entry)
            else:
                failed.append(entry)

        is_valid = len(failed) == 0
        report = {
            "total": len(criteria),
            "passed": len(passed),
            "flagged": len(flagged),
            "failed": len(failed),
            "is_valid": is_valid,
            "passed_criteria": passed,
            "flagged_criteria": flagged,
            "failed_criteria": failed
        }

        self._log_report(report)
        return is_valid, report

    def _validate_criterion(
        self, criterion: Dict[str, Any]
    ) -> Tuple[str, List[str]]:
        """
        Validate a single criterion.

        Returns:
            (status, issues)
            status: 'passed' | 'flagged' | 'failed'
            issues: list of issue descriptions
        """
        issues = []

        # check mandatory fields — failure if missing
        for field in MANDATORY_FIELDS:
            if field not in criterion or criterion[field] is None:
                issues.append(f"Missing mandatory field: {field}")

        if issues:
            return "failed", issues

        # check important fields — flagged if missing
        for field in IMPORTANT_FIELDS:
            if field not in criterion or criterion[field] is None:
                issues.append(f"Missing important field: {field}")

        # check requirement text is meaningful
        req_text = criterion.get("requirement_text", "")
        if len(req_text.strip()) < 10:
            issues.append("requirement_text too short — may be incomplete")

        # check category is valid
        valid_categories = ["eligibility", "financial", "technical", "compliance"]
        if criterion.get("category") not in valid_categories:
            issues.append(f"Invalid category: {criterion.get('category')}")

        if issues:
            return "flagged", issues

        return "passed", []

    def save_ruleset(
        self,
        criteria: List[Dict[str, Any]],
        output_path: str,
        tender_id: str,
        version: str = "v1.0"
    ) -> str:
        """
        Save the validated ruleset to a versioned JSON file.

        Returns:
            Path to saved ruleset file
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        filename = f"ruleset_{tender_id}_{version}.json"
        filepath = output_path / filename

        ruleset = {
            "tender_id": tender_id,
            "version": version,
            "total_criteria": len(criteria),
            "criteria": criteria
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(ruleset, f, indent=2, ensure_ascii=False)

        logger.info(f"Ruleset saved: {filepath}")
        return str(filepath)

    def load_ruleset(self, ruleset_path: str) -> List[Dict[str, Any]]:
        """Load a previously saved ruleset JSON."""
        with open(ruleset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("criteria", [])

    def _log_report(self, report: Dict[str, Any]) -> None:
        """Log a summary of the validation report."""
        logger.info(f"Ruleset validation complete:")
        logger.info(f"  Total criteria : {report['total']}")
        logger.info(f"  Passed         : {report['passed']}")
        logger.info(f"  Flagged        : {report['flagged']}")
        logger.info(f"  Failed         : {report['failed']}")
        logger.info(f"  Valid          : {report['is_valid']}")

        if report["flagged_criteria"]:
            logger.warning("Flagged criteria (need human review):")
            for c in report["flagged_criteria"]:
                logger.warning(f"  [{c['criterion_id']}] {c['issues']}")

        if report["failed_criteria"]:
            logger.error("Failed criteria (blocking):")
            for c in report["failed_criteria"]:
                logger.error(f"  [{c['criterion_id']}] {c['issues']}")