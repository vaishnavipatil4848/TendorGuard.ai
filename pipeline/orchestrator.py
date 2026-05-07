"""
orchestrator.py
pipeline/ — TendorGuard.ai

End-to-end pipeline coordinator.
Wires Agent 1 → Agent 2 → Agent 3 → Agent 4 → Agent 5
using each agent's actual interface as defined in their __init__.py.

Agent interfaces (read from agent source):
  Agent 1 — ArchitectAgent.run(pdf_path, tender_id) → (criteria, validation_report)
  Agent 2 — Vision agent; criteria extraction is handled by Agent 1's CriteriaExtractor
  Agent 3 — EvidenceExtractor.extract(criterion, top_chunks) + HierarchicalChunker + retrievers
  Agent 4 — AuditorAgent.run(bidder_id, criteria, evidence_map) → aggregated_report
  Agent 5 — HITLAgent.ingest_verdicts(...) + HITLAgent.generate_report(...)

Storage layout (storage/):
  tender_docs/    — uploaded tender PDFs
  bidder_docs/    — uploaded bidder PDFs
  rulesets/       — Agent 1 ruleset JSONs
  audit_reports/  — Agent 5 PDF reports
"""

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from pipeline.message_bus import MessageBus
from pipeline.error_handler import ErrorHandler

logger = logging.getLogger(__name__)

STORAGE_RULESETS      = "storage/rulesets"
STORAGE_AUDIT_REPORTS = "storage/audit_reports"


class TenderEvalOrchestrator:
    """
    Main pipeline coordinator for TendorGuard.ai.

    Each public method corresponds to one stage of the pipeline
    so the Streamlit UI can call them incrementally and show
    per-step progress feedback.
    """

    def __init__(
        self,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ):
        """
        Args:
            progress_callback: fn(step, total_steps, message) called at each stage.
                               Designed for Streamlit st.progress().
        """
        self.bus      = MessageBus()
        self.errors   = ErrorHandler()
        self._cb      = progress_callback
        self.all_logs: List[str] = []

        # ── Lazy agent refs (initialised on first use) ────────────────────────
        self._agent1: Any = None
        self._agent3: Any = None
        self._agent4: Any = None
        self._agent5: Any = None

        # ── State shared across pipeline stages ───────────────────────────────
        self.tender_id:   str               = ""
        self.criteria:    List[Dict]        = []
        self.bidder_reports: List[Dict]     = []

    # ── Agent accessors (lazy init) ───────────────────────────────────────────

    @property
    def agent1(self):
        if self._agent1 is None:
            from agents.agent1_architect import ArchitectAgent
            self._agent1 = ArchitectAgent()
        return self._agent1

    @property
    def agent3(self):
        if self._agent3 is None:
            from agents.agent3_factchecker.evidence_extractor import EvidenceExtractor
            from agents.agent3_factchecker.chunker          import HierarchicalChunker
            from agents.agent3_factchecker.dense_retriever  import DenseRetriever
            from agents.agent3_factchecker.sparse_retriever import SparseRetriever
            from agents.agent3_factchecker.rrf_fusion       import RRFFusion
            from agents.agent3_factchecker.reranker         import CrossEncoderReranker
            from agents.agent3_factchecker.metadata_filter  import MetadataFilter
            self._agent3 = {
                "extractor":  EvidenceExtractor(),
                "chunker":    HierarchicalChunker(),
                "dense":      DenseRetriever(),
                "sparse":     SparseRetriever(),
                "fusion":     RRFFusion(),
                "reranker":   CrossEncoderReranker(),
                "filter":     MetadataFilter(),
            }
        return self._agent3

    @property
    def agent4(self):
        if self._agent4 is None:
            from agents.agent4_auditor import AuditorAgent
            self._agent4 = AuditorAgent()
        return self._agent4

    @property
    def agent5(self):
        if self._agent5 is None:
            from agents.agent5_hitl import HITLAgent
            self._agent5 = HITLAgent()
        return self._agent5

    # ── Progress helper ───────────────────────────────────────────────────────

    def _step(self, current: float, total: float, msg: str) -> None:
        """Helper to report progress. Supports floats for sub-step granularity."""
        log_line = f"[{current:.1f}/{total}] {msg}"
        logger.info(log_line)
        self.all_logs.append(log_line)
        if self._cb:
            # streamlit progress expects float 0.0 to 1.0
            try:
                self._cb(float(current), float(total), msg)
            except Exception as e:
                logger.error(f"Progress callback failed: {e}")

        self.bus.publish(
            sender="Orchestrator",
            receiver="UI",
            msg_type="PROGRESS",
            payload={"step": float(current), "total": float(total), "message": msg}
        )

    # ── Stage 1: Parse tender → extract criteria (Agent 1) ───────────────────

    def run_stage1_tender(
        self,
        tender_pdf_path: str,
        tender_id:       str,
    ) -> List[Dict[str, Any]]:
        """
        Parse the tender PDF and extract eligibility criteria.

        Args:
            tender_pdf_path: absolute path to tender PDF in storage/tender_docs/
            tender_id:       unique tender identifier

        Returns:
            List of criterion dicts (Agent 1 ruleset)
        """
        self.tender_id = tender_id
        self._step(1, 5, f"Agent 1: Parsing tender '{Path(tender_pdf_path).name}'...")

        try:
            criteria, validation_report = self.agent1.run(
                pdf_path=tender_pdf_path,
                tender_id=tender_id,
                ruleset_output_dir=STORAGE_RULESETS,
            )
            self.criteria = criteria

            self.bus.publish(
                sender="Agent1_Architect",
                receiver="Orchestrator",
                msg_type="CRITERIA_EXTRACTED",
                payload={
                    "tender_id":         tender_id,
                    "criteria_count":    len(criteria),
                    "validation_report": validation_report,
                }
            )
            self._step(1, 5, f"Agent 1 complete: {len(criteria)} criteria extracted.")
            return criteria

        except Exception as exc:
            fallback = self.errors.handle(exc, agent="Agent1_Architect",
                                          context={"tender_id": tender_id})
            logger.error(f"Agent 1 failed — pipeline cannot continue: {exc}")
            raise

    # ── Stage 2: Index + retrieve evidence for one bidder (Agent 3) ──────────

    def run_stage3_evidence(
        self,
        bidder_data: Dict[str, Any],
        bidder_id:   str,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run the full Agent 3 RAG pipeline for one bidder.

        Pipeline:
          chunk → index (dense + sparse) → retrieve per criterion
          → metadata filter → RRF fusion → rerank → extract

        Args:
            bidder_data: parsed document dict (must have 'pages', 'raw_text', 'filename')
            bidder_id:   unique bidder identifier

        Returns:
            evidence_map: {criterion_id → evidence dict}
        """
        a3 = self.agent3

        # 1. Chunk
        chunks = a3["chunker"].chunk(bidder_data)
        logger.info(f"Agent 3: {len(chunks)} chunks for '{bidder_id}'")

        # 2. Index dense
        a3["dense"].index(chunks, bidder_id)

        # 3. Index sparse
        filtered_chunks = a3["filter"].filter(chunks, self.criteria)
        a3["sparse"].index(filtered_chunks)

        evidence_map: Dict[str, Dict[str, Any]] = {}

        for idx, criterion in enumerate(self.criteria, 1):
            cid   = criterion.get("criterion_id") or criterion.get("id")
            name  = criterion.get("name", "")
            query = f"{name} {criterion.get('description', '')}"

            self._step(2, 5, f"Agent 3: [{bidder_id}] Extracting '{name}' ({idx}/{len(self.criteria)})...")

            try:
                # Retrieve
                dense_hits  = a3["dense"].retrieve(query, bidder_id, top_k=20)
                sparse_hits = a3["sparse"].retrieve(query, top_k=20)

                # Fuse
                fused = a3["fusion"].fuse(dense_hits, sparse_hits)

                # Rerank
                top_chunks = a3["reranker"].rerank(query, fused, top_k=5)

                # Extract evidence
                evidence = a3["extractor"].extract(criterion, top_chunks)

            except Exception as exc:
                evidence = self.errors.handle(
                    exc, agent="Agent3_FactChecker",
                    context={"criterion_id": cid, "bidder_id": bidder_id}
                )

            evidence_map[cid] = evidence

        self.bus.publish(
            sender="Agent3_FactChecker",
            receiver="Agent4_Auditor",
            msg_type="EVIDENCE_READY",
            payload={"bidder_id": bidder_id, "criterion_count": len(evidence_map)}
        )
        return evidence_map

    # ── Stage 3: Generate verdicts for one bidder (Agent 4) ──────────────────

    def run_stage4_verdicts(
        self,
        bidder_id:    str,
        evidence_map: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Run Agent 4 to produce per-criterion verdicts for one bidder.

        Args:
            bidder_id:    unique bidder identifier
            evidence_map: output from run_stage3_evidence()

        Returns:
            aggregated_report from VerdictAggregator
        """
        try:
            report = self.agent4.run(
                bidder_id=bidder_id,
                criteria=self.criteria,
                evidence_map=evidence_map,
            )
        except Exception as exc:
            self.errors.handle(exc, agent="Agent4_Auditor",
                               context={"bidder_id": bidder_id})
            report = {
                "bidder_id":        bidder_id,
                "overall_verdict":  "NEEDS_MANUAL_REVIEW",
                "overall_confidence": 0.0,
                "criterion_verdicts": [],
                "error":            str(exc),
            }

        self.bus.publish(
            sender="Agent4_Auditor",
            receiver="Agent5_HITL",
            msg_type="VERDICTS_READY",
            payload={"bidder_id": bidder_id, "overall": report.get("overall_verdict")}
        )
        return report

    # ── Stage 4: Route to HITL queue (Agent 5) ───────────────────────────────

    def run_stage5_hitl(
        self,
        bidder_id:        str,
        aggregated_report: Dict[str, Any],
        evidence_map:      Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Ingest one bidder's verdicts into the HITL queue.

        Args:
            bidder_id:         unique bidder identifier
            aggregated_report: output from run_stage4_verdicts()
            evidence_map:      output from run_stage3_evidence()

        Returns:
            ingestion_summary with auto_approved / queued counts
        """
        try:
            summary = self.agent5.ingest_verdicts(
                tender_id=self.tender_id,
                bidder_id=bidder_id,
                aggregated_report=aggregated_report,
                criteria_meta=self.criteria,
                evidence_map=evidence_map,
            )
        except Exception as exc:
            self.errors.handle(exc, agent="Agent5_HITL",
                               context={"bidder_id": bidder_id})
            summary = {
                "bidder_id":        bidder_id,
                "auto_approved":    0,
                "queued_for_review": 0,
                "error":            str(exc),
            }

        self.bidder_reports.append({
            "bidder_id":        bidder_id,
            "aggregated_report": aggregated_report,
            "hitl_summary":     summary,
        })
        return summary

    # ── Final: Generate audit report (Agent 5) ────────────────────────────────

    def generate_final_report(
        self,
        output_dir: str = STORAGE_AUDIT_REPORTS,
    ) -> str:
        """
        Generate the signed PDF audit report once all bidders have been processed.

        Args:
            output_dir: directory to write the PDF

        Returns:
            Path to generated PDF report
        """
        self._step(5, 5, "Agent 5: Generating final audit report...")

        bidder_summaries = [r["aggregated_report"] for r in self.bidder_reports]

        try:
            report_path = self.agent5.generate_report(
                tender_id=self.tender_id,
                bidder_summaries=bidder_summaries,
                output_dir=output_dir,
            )
        except Exception as exc:
            self.errors.handle(exc, agent="Agent5_HITL_Report")
            report_path = ""

        self.bus.publish(
            sender="Agent5_HITL",
            receiver="UI",
            msg_type="REPORT_COMPLETE",
            payload={"tender_id": self.tender_id, "report_path": report_path}
        )
        self._step(5, 5, f"Pipeline complete. Report: {report_path}")
        return report_path

    # ── Convenience: run all bidders end-to-end ───────────────────────────────

    def run_bidders(
        self,
        bidder_data_list: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Run Stages 2–4 for all bidders.

        Args:
            bidder_data_list: list of parsed bidder data dicts,
                              each must have 'bidder_id' set.

        Returns:
            List of HITL ingestion summaries
        """
        total_bidders = len(bidder_data_list)
        summaries = []
        for i, b_data in enumerate(bidder_data_list, 1):
            bid = b_data.get("bidder_id", f"bidder_{i}")
            
            # Sub-steps for each bidder
            self._step(2, 5, f"Agent 3: [{i}/{total_bidders}] Retrieving evidence for {bid}...")
            evidence_map = self.run_stage3_evidence(b_data, bid)
            
            self._step(3, 5, f"Agent 4: [{i}/{total_bidders}] Generating verdicts for {bid}...")
            aggregated_report = self.run_stage4_verdicts(bid, evidence_map)
            
            self._step(4, 5, f"Agent 5: [{i}/{total_bidders}] Routing {bid} to review queue...")
            hitl_summary = self.run_stage5_hitl(bid, aggregated_report, evidence_map)
            
            summaries.append(hitl_summary)

        return summaries
