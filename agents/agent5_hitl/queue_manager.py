"""
queue_manager.py
Agent 5 — HITL Agent
Priority-ordered review queue for human reviewers.

Priority order (per brainstorm doc — missing evidence cases first):
  1  MISSING_EVIDENCE    — hardest to auto-resolve, highest stakes
  2  MODEL_DISAGREEMENT  — genuine ambiguity between model families
  3  AMBIGUOUS_CRITERION — criterion itself needs human interpretation
  4  LOW_CONFIDENCE      — weaker signal, often resolvable quickly

The queue is backed by a min-heap so get_next() is always O(log n).
Cases at the same priority are served FIFO via an insertion counter.
"""

import heapq
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Literal

from .case_router import CaseType

logger = logging.getLogger(__name__)

PRIORITY: Dict[str, int] = {
    "MISSING_EVIDENCE":    1,
    "MODEL_DISAGREEMENT":  2,
    "AMBIGUOUS_CRITERION": 3,
    "LOW_CONFIDENCE":      4,
}


class ReviewQueue:
    """
    Priority min-heap queue of HITL review cases.

    Each item in the heap is a tuple:
        (priority, insertion_index, case_dict)
    insertion_index breaks ties at the same priority level (FIFO).
    """

    def __init__(self):
        self._heap: list = []
        self._counter: int = 0          # tie-breaker for same-priority cases
        self._case_index: Dict[str, dict] = {}  # criterion_id → case for O(1) lookup

    def add_case(
        self,
        verdict: Dict[str, Any],
        case_type: CaseType,
        bidder_id: str,
        tender_id: str,
        criterion: Dict[str, Any],
        evidence: Dict[str, Any]
    ) -> None:
        """
        Add a case to the review queue.

        Args:
            verdict:      full verdict dict from AuditorAgent
            case_type:    from case_router.route_case()
            bidder_id:    bidder being evaluated
            tender_id:    tender this evaluation belongs to
            criterion:    criterion dict from Agent 1 ruleset
            evidence:     evidence dict from Agent 3
        """
        priority = PRIORITY.get(case_type, 5)
        cid = verdict.get("criterion_id", "unknown")

        case = {
            "criterion_id": cid,
            "case_type": case_type,
            "bidder_id": bidder_id,
            "tender_id": tender_id,
            "verdict": verdict,
            "criterion": criterion,
            "evidence": evidence,
            "queued_at": datetime.now(timezone.utc).isoformat(),
            "status": "PENDING",
            "system_suggested_verdict": verdict.get("verdict", "UNCERTAIN"),
            "composite_confidence": verdict.get("composite_confidence", 0.0),
        }

        heapq.heappush(self._heap, (priority, self._counter, case))
        self._case_index[cid] = case
        self._counter += 1

        logger.info(
            f"Case queued: [{cid}] type={case_type} priority={priority} "
            f"bidder={bidder_id} conf={case['composite_confidence']:.2f}"
        )

    def get_next(self) -> Optional[Dict[str, Any]]:
        """
        Pop and return the highest-priority pending case.
        Returns None if the queue is empty.
        """
        while self._heap:
            _, _, case = heapq.heappop(self._heap)
            if case["status"] == "PENDING":
                case["status"] = "IN_REVIEW"
                logger.debug(
                    f"Dequeued case [{case['criterion_id']}] "
                    f"type={case['case_type']}"
                )
                return case
        return None

    def peek_next(self) -> Optional[Dict[str, Any]]:
        """
        Return the highest-priority pending case without removing it.
        """
        for _, _, case in sorted(self._heap):
            if case["status"] == "PENDING":
                return case
        return None

    def get_case(self, criterion_id: str) -> Optional[Dict[str, Any]]:
        """Look up a specific case by criterion_id."""
        return self._case_index.get(criterion_id)

    def is_empty(self) -> bool:
        """Return True if no PENDING cases remain."""
        return all(c["status"] != "PENDING" for _, _, c in self._heap)

    def pending_count(self) -> int:
        """Return the number of PENDING cases in the queue."""
        return sum(1 for _, _, c in self._heap if c["status"] == "PENDING")

    def all_cases(self) -> List[Dict[str, Any]]:
        """Return all cases (any status), sorted by priority."""
        return [
            case for _, _, case in sorted(self._heap)
        ]

    def get_summary(self) -> Dict[str, Any]:
        """Return a count breakdown by case type and status."""
        summary: Dict[str, int] = {}
        status_counts: Dict[str, int] = {"PENDING": 0, "IN_REVIEW": 0, "RESOLVED": 0}

        for _, _, case in self._heap:
            ct = case["case_type"]
            summary[ct] = summary.get(ct, 0) + 1
            st = case.get("status", "PENDING")
            status_counts[st] = status_counts.get(st, 0) + 1

        return {
            "total": len(self._heap),
            "by_type": summary,
            "by_status": status_counts,
        }