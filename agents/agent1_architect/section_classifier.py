"""
section_classifier.py
Agent 1 — Architect Agent
Classifies detected regions into tender section categories:
eligibility / financial / technical / compliance / boilerplate
"""

import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Keywords that signal each section category
SECTION_KEYWORDS = {
    "eligibility": [
        "eligibility", "eligible", "qualification", "qualifying",
        "criteria", "criterion", "pre-qualification", "prequalification",
        "bidder must", "applicant must", "vendor must", "पात्रता", "अनिवार्यता",
        "eligibility criteria", "minimum eligibility", "eligibility condition"
    ],
    "financial": [
        "turnover", "annual turnover", "revenue", "financial",
        "net worth", "profit", "loss", "balance sheet", "ca certificate",
        "chartered accountant", "वित्तीय", "कारोबार", "वित्तीय स्थिति",
        "average annual turnover", "financial capability"
    ],
    "technical": [
        "technical", "experience", "similar work", "past project",
        "completion certificate", "work order", "performance",
        "specification", "तकनीकी", "अनुभव", "technical experience",
        "work experience", "similar nature"
    ],
    "compliance": [
        "gst", "pan", "registration", "license", "certificate",
        "compliance", "statutory", "msme", "dpiit", "iso",
        "पंजीकरण", "प्रमाण पत्र", "statutory documents", "compliance documents"
    ],
    "boilerplate": [
        "general terms", "definitions", "disclaimer", "preamble",
        "introduction", "background", "about", "contact",
        "page", "index", "table of contents", "annexure", "appendix"
    ]
}


class SectionClassifier:
    """
    Classifies text regions from LayoutLMv3 output into
    tender section categories using keyword matching.
    Groups consecutive regions into logical sections.
    """

    def __init__(self):
        # compile regex patterns for each category
        self.patterns = {
            category: re.compile(
                "|".join(re.escape(kw) for kw in keywords),
                re.IGNORECASE
            )
            for category, keywords in SECTION_KEYWORDS.items()
        }

    def classify_regions(
        self, regions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Classify each region into a section category.
        Adds 'section_category' field to each region dict.

        Args:
            regions: list of region dicts from LayoutParser

        Returns:
            Same list with 'section_category' added to each region
        """
        classified = []
        current_category = "boilerplate"

        for region in regions:
            text = region.get("text", "")
            label = region.get("label", "Text")

            # section headers update the current running category
            if label in ("Section-header", "Title"):
                detected = self._detect_category(text)
                if detected:
                    current_category = detected
                    logger.debug(
                        f"Section header detected: '{text[:60]}' → {current_category}"
                    )

            # classify the region
            region_category = self._detect_category(text) or current_category
            region["section_category"] = region_category
            classified.append(region)

        logger.info(f"Classified {len(classified)} regions")
        self._log_category_summary(classified)
        return classified

    def filter_by_categories(
        self,
        regions: List[Dict[str, Any]],
        categories: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Return only regions belonging to specified categories.
        Used to feed only relevant sections to the LLM.

        Args:
            regions: classified regions
            categories: list of categories to keep
                e.g. ["eligibility", "financial", "technical", "compliance"]

        Returns:
            Filtered list of regions
        """
        filtered = [r for r in regions if r.get("section_category") in categories]
        logger.info(
            f"Filtered to {len(filtered)} regions "
            f"from categories: {categories}"
        )
        return filtered

    def group_into_sections(
        self, regions: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group classified regions by their section category.

        Returns:
            Dict mapping category → list of regions
        """
        sections: Dict[str, List[Dict[str, Any]]] = {
            "eligibility": [],
            "financial": [],
            "technical": [],
            "compliance": [],
            "boilerplate": []
        }
        for region in regions:
            cat = region.get("section_category", "boilerplate")
            sections[cat].append(region)

        return sections

    def _detect_category(self, text: str) -> str | None:
        """
        Detect the category of a text string using keyword patterns.
        Returns the matched category or None if no match.
        """
        # skip very short strings
        if len(text.strip()) < 5:
            return None

        # score each category by number of keyword matches
        scores = {}
        for category, pattern in self.patterns.items():
            matches = pattern.findall(text)
            if matches:
                scores[category] = len(matches)

        if not scores:
            return None

        # return highest scoring category
        return max(scores, key=scores.get)

    def _log_category_summary(self, regions: List[Dict[str, Any]]) -> None:
        """Log a count summary of regions per category."""
        from collections import Counter
        counts = Counter(r.get("section_category") for r in regions)
        for cat, count in counts.most_common():
            logger.info(f"  {cat}: {count} regions")