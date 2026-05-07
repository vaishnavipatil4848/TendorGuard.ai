"""
feedback_loop.py
Agent 5 — HITL Agent
Tracks which model was correct on every human override and accumulates
accuracy metrics per model family, criterion type, and case type.

Purpose (per brainstorm doc):
  - Identify which LLM family (GPT-4o vs Claude) is more accurate
    for which criterion category over time
  - Surface systematic failures (e.g. "Numerical criteria are overridden
    30% of the time → numerical_checker threshold needs tuning")
  - Provide data for future fine-tuning / threshold calibration

This is described as future scope in the brainstorm doc but the
data collection layer is implemented here so every override is
captured from day one.
"""

import logging
from collections import defaultdict
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class FeedbackLoop:
    """
    Accumulates human override statistics across all reviewed cases.
    Persists in-memory during a pipeline run; caller is responsible
    for persisting the summary to the audit database if needed.
    """

    def __init__(self):
        # Overall stats
        self._total = 0
        self._model_correct = 0
        self._human_overrides = 0

        # Breakdown: criterion_type → {correct, override}
        self._by_criterion_type: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"correct": 0, "override": 0}
        )

        # Breakdown: case_type → {correct, override}
        self._by_case_type: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"correct": 0, "override": 0}
        )

        # Breakdown: model_family → {correct, override}
        # Tracks which model family's verdict the human agreed with
        self._by_model: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"correct": 0, "override": 0}
        )

        # Raw override records for detailed export
        self._override_records: List[Dict[str, Any]] = []

    def record(
        self,
        case: Dict[str, Any],
        system_verdict: str,
        human_verdict: str,
        claude_verdict: Optional[str] = None,
        gpt4o_verdict: Optional[str] = None
    ) -> None:
        """
        Record one human review decision.

        Args:
            case:            case dict from ReviewQueue
            system_verdict:  the automated system's recommended verdict
            human_verdict:   what the reviewer decided
            claude_verdict:  Claude's individual verdict (if dual-LLM path)
            gpt4o_verdict:   GPT-4o's individual verdict (if dual-LLM path)
        """
        self._total += 1
        was_override = system_verdict != human_verdict
        criterion_type = case.get("criterion", {}).get("criterion_type", "Unknown")
        case_type = case.get("case_type", "Unknown")

        if was_override:
            self._human_overrides += 1
        else:
            self._model_correct += 1

        # by criterion type
        key = "override" if was_override else "correct"
        self._by_criterion_type[criterion_type][key] += 1
        self._by_case_type[case_type][key] += 1

        # per-model tracking: which model agreed with the human?
        if claude_verdict and gpt4o_verdict:
            if claude_verdict == human_verdict:
                self._by_model["claude"]["correct"] += 1
            else:
                self._by_model["claude"]["override"] += 1

            if gpt4o_verdict == human_verdict:
                self._by_model["gpt4o"]["correct"] += 1
            else:
                self._by_model["gpt4o"]["override"] += 1

        if was_override:
            self._override_records.append({
                "criterion_id": case.get("criterion_id"),
                "bidder_id": case.get("bidder_id"),
                "case_type": case_type,
                "criterion_type": criterion_type,
                "system_verdict": system_verdict,
                "human_verdict": human_verdict,
                "claude_verdict": claude_verdict,
                "gpt4o_verdict": gpt4o_verdict,
            })

        logger.debug(
            f"Feedback recorded: [{case.get('criterion_id')}] "
            f"system={system_verdict} human={human_verdict} "
            f"override={was_override}"
        )

    def get_metrics(self) -> Dict[str, Any]:
        """
        Return aggregated accuracy metrics.

        Returns:
            Dict with overall accuracy, per-type breakdowns,
            and per-model accuracy
        """
        overall_accuracy = (
            round(self._model_correct / self._total, 3)
            if self._total else 0.0
        )

        def _accuracy(d: Dict) -> float:
            total = d["correct"] + d["override"]
            return round(d["correct"] / total, 3) if total else 0.0

        return {
            "total_reviewed": self._total,
            "model_correct": self._model_correct,
            "human_overrides": self._human_overrides,
            "overall_accuracy": overall_accuracy,
            "by_criterion_type": {
                ct: {**stats, "accuracy": _accuracy(stats)}
                for ct, stats in self._by_criterion_type.items()
            },
            "by_case_type": {
                ct: {**stats, "accuracy": _accuracy(stats)}
                for ct, stats in self._by_case_type.items()
            },
            "by_model": {
                m: {**stats, "accuracy": _accuracy(stats)}
                for m, stats in self._by_model.items()
            },
        }

    def get_override_records(self) -> List[Dict[str, Any]]:
        """Return the raw list of override records for export / analysis."""
        return list(self._override_records)

    def reset(self) -> None:
        """Reset all counters. Called between pipeline runs if needed."""
        self.__init__()