"""
Agent 5 — Human-in-the-Loop (HITL) Agent
Routes low-confidence / disagreement cases from the Auditor to a human
reviewer, captures decisions with mandatory justifications, writes an
immutable audit trail, and generates a signed PDF audit report.

Pipeline:
  1. Receive all per-criterion verdicts from Agent 4
  2. Route each verdict via case_router → case type or AUTO_APPROVED
  3. Enqueue non-auto cases into ReviewQueue (priority-ordered)
  4. Expose queue to Streamlit UI for human review
  5. Log every decision (system + human) to AuditLogger
  6. Update FeedbackLoop model accuracy metrics per decision
  7. After all cases resolved → generate signed PDF via ReportGenerator
"""

import logging
from typing import Dict, Any, List, Tuple

from .case_router import route_case, get_case_context, CaseType
from .queue_manager import ReviewQueue
from .audit_logger import AuditLogger
from .feedback_loop import FeedbackLoop
from .report_generator import ReportGenerator

logger = logging.getLogger(__name__)


class HITLAgent:
    """
    Main entry point for Agent 5.
    Accepts Agent 4 outputs, routes cases, manages the review queue,
    and produces the final audit report.
    """

    def __init__(
        self,
        postgres_url: str = None,
        hitl_threshold: float = 0.70
    ):
        logger.info("Initialising HITL Agent")
        self.hitl_threshold = hitl_threshold
        self.queue = ReviewQueue()
        self.audit_logger = AuditLogger(postgres_url=postgres_url)
        self.feedback = FeedbackLoop()
        self.report_generator = ReportGenerator()

    def ingest_verdicts(
        self,
        tender_id: str,
        bidder_id: str,
        aggregated_report: Dict[str, Any],
        criteria_meta: List[Dict[str, Any]],
        evidence_map: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Process all per-criterion verdicts for one bidder.
        Routes each to AUTO_APPROVED or the review queue.

        Args:
            tender_id:          tender identifier
            bidder_id:          bidder identifier
            aggregated_report:  output from VerdictAggregator.aggregate()
            criteria_meta:      original criteria list from Agent 1 ruleset
            evidence_map:       criterion_id → evidence dict from Agent 3

        Returns:
            ingestion_summary dict
        """
        criterion_verdicts = aggregated_report.get("criterion_verdicts", [])
        criteria_lookup = {c["criterion_id"]: c for c in criteria_meta}

        auto_approved = 0
        queued = 0

        for verdict in criterion_verdicts:
            cid = verdict.get("criterion_id", "?")
            case_type: CaseType = route_case(verdict, self.hitl_threshold)

            if case_type == "AUTO_APPROVED":
                auto_approved += 1
                logger.debug(f"[{cid}] AUTO_APPROVED")
                continue

            self.queue.add_case(
                verdict=verdict,
                case_type=case_type,
                bidder_id=bidder_id,
                tender_id=tender_id,
                criterion=criteria_lookup.get(cid, {}),
                evidence=evidence_map.get(cid, {})
            )
            queued += 1

        logger.info(
            f"HITL ingestion for {bidder_id}: "
            f"{auto_approved} auto-approved, {queued} queued for review"
        )
        return {
            "bidder_id": bidder_id,
            "auto_approved": auto_approved,
            "queued_for_review": queued,
            "queue_summary": self.queue.get_summary(),
        }

    def submit_review(
        self,
        tender_id: str,
        criterion_id: str,
        human_verdict: str,
        reviewer_id: str,
        comment: str,
        rejection_reason: str = None
    ) -> str:
        """
        Accept a human reviewer's decision for one queued case.
        Logs to the audit trail and updates feedback metrics.

        Args:
            tender_id:        tender identifier
            criterion_id:     which criterion was reviewed
            human_verdict:    PASS | FAIL | UNCERTAIN
            reviewer_id:      reviewer's user ID
            comment:          mandatory free-text justification
            rejection_reason: optional structured taxonomy tag

        Returns:
            log_id from AuditLogger
        """
        case = self.queue.get_case(criterion_id)
        if not case:
            raise ValueError(
                f"No queued case found for criterion_id={criterion_id}"
            )

        system_verdict = case["system_suggested_verdict"]
        bidder_id = case["bidder_id"]
        case_type = case["case_type"]
        composite_conf = case["composite_confidence"]

        # mark case resolved in queue
        case["status"] = "RESOLVED"
        case["human_verdict"] = human_verdict
        case["reviewer_id"] = reviewer_id

        # write immutable audit record
        log_id = self.audit_logger.log(
            tender_id=tender_id,
            bidder_id=bidder_id,
            criterion_id=criterion_id,
            case_type=case_type,
            system_verdict=system_verdict,
            human_verdict=human_verdict,
            reviewer_id=reviewer_id,
            comment=comment,
            composite_confidence=composite_conf,
            rejection_reason=rejection_reason,
        )

        # update feedback loop
        verdict_dict = case.get("verdict", {})
        self.feedback.record(
            case=case,
            system_verdict=system_verdict,
            human_verdict=human_verdict,
            claude_verdict=verdict_dict.get("claude_verdict"),
            gpt4o_verdict=verdict_dict.get("gpt4o_verdict"),
        )

        logger.info(
            f"Review submitted: [{criterion_id}] "
            f"system={system_verdict} → human={human_verdict} "
            f"reviewer={reviewer_id} log_id={log_id}"
        )
        return log_id

    def generate_report(
        self,
        tender_id: str,
        bidder_summaries: List[Dict[str, Any]],
        output_dir: str = "storage/audit_reports"
    ) -> str:
        """
        Generate the signed PDF audit report for the completed evaluation.

        Args:
            tender_id:        tender identifier
            bidder_summaries: list of aggregated_report dicts (one per bidder)
            output_dir:       where to write the PDF

        Returns:
            Path to the generated PDF
        """
        audit_records = self.audit_logger.get_log_for_bidder.__func__  # use raw query
        # Fetch all audit records for this tender across all bidders
        all_records = []
        for b in bidder_summaries:
            all_records.extend(
                self.audit_logger.get_log_for_bidder(
                    tender_id, b.get("bidder_id", "")
                )
            )

        feedback_metrics = self.feedback.get_metrics()

        path = self.report_generator.generate(
            tender_id=tender_id,
            bidder_summaries=bidder_summaries,
            audit_log_records=all_records,
            feedback_metrics=feedback_metrics,
            output_dir=output_dir
        )
        return path

    def get_next_case(self) -> Dict[str, Any]:
        """
        Pop the next highest-priority case from the review queue.
        Includes the reviewer context string for the UI.
        """
        case = self.queue.get_next()
        if case:
            case["reviewer_context"] = get_case_context(case["case_type"])
        return case

    def get_queue_summary(self) -> Dict[str, Any]:
        """Return current queue state summary."""
        return self.queue.get_summary()

    def get_feedback_metrics(self) -> Dict[str, Any]:
        """Return current model accuracy metrics."""
        return self.feedback.get_metrics()