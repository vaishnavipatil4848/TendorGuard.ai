"""
pipeline_runner.py
pipeline/ — TendorGuard.ai

CLI-friendly runner that wraps TenderEvalOrchestrator.
Used by scripts/run_pipeline.py for batch / headless execution.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline.orchestrator import TenderEvalOrchestrator

logger = logging.getLogger(__name__)

STORAGE_TENDER_DOCS  = Path("storage/tender_docs")
STORAGE_BIDDER_DOCS  = Path("storage/bidder_docs")
STORAGE_AUDIT_REPORTS = Path("storage/audit_reports")


def _console_progress(step: int, total: int, message: str) -> None:
    """Simple console progress reporter for headless runs."""
    bar_width = 30
    filled    = int(bar_width * step / total)
    bar       = "█" * filled + "─" * (bar_width - filled)
    print(f"\r[{bar}] {step}/{total}  {message}", end="", flush=True)
    if step == total:
        print()  # newline at end


class PipelineRunner:
    """
    Wraps TenderEvalOrchestrator for script-driven / batch execution.

    Handles:
      - reading tender and bidder files from storage/
      - parsing bidder_data with the document parser Agent 1 uses
        (via ArchitectAgent internals — layout_parser.py)
      - running the full pipeline
      - writing a summary JSON alongside the PDF report
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        cb = _console_progress if verbose else None
        self.orch = TenderEvalOrchestrator(progress_callback=cb)

    def run(
        self,
        tender_id:       str,
        tender_pdf_path: str,
        bidder_pdf_paths: List[str],
        output_dir:       str = str(STORAGE_AUDIT_REPORTS),
    ) -> Dict[str, Any]:
        """
        Full end-to-end run.

        Args:
            tender_id:        unique tender identifier (e.g. "T-4821")
            tender_pdf_path:  path to tender PDF
            bidder_pdf_paths: list of bidder PDF paths
            output_dir:       where to write the PDF report and summary JSON

        Returns:
            run_summary dict
        """
        logger.info(
            f"PipelineRunner: tender={tender_id}, "
            f"{len(bidder_pdf_paths)} bidders"
        )

        # Stage 1 — tender parsing + criteria extraction
        criteria = self.orch.run_stage1_tender(tender_pdf_path, tender_id)
        if not criteria:
            logger.warning("No criteria extracted — aborting pipeline.")
            return {"status": "aborted", "reason": "no_criteria"}

        # Build bidder data list (parse each PDF path into a data dict)
        bidder_data_list = self._prepare_bidder_data(bidder_pdf_paths, criteria)

        # Stages 2–4 — evidence + verdicts + HITL per bidder
        hitl_summaries = self.orch.run_bidders(bidder_data_list)

        # Stage 5 — generate final PDF report
        report_path = self.orch.generate_final_report(output_dir=output_dir)

        # Write summary JSON
        summary = self._build_summary(
            tender_id, criteria, hitl_summaries, report_path
        )
        summary_path = Path(output_dir) / f"{tender_id}_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        if self.verbose:
            print(f"\n✅ Pipeline complete for tender '{tender_id}'")
            print(f"   Report  : {report_path}")
            print(f"   Summary : {summary_path}")

        return summary

    def _prepare_bidder_data(
        self,
        pdf_paths: List[str],
        criteria:  List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        For each bidder PDF, build the bidder_data dict expected by Agent 3.
        Agent 1's LayoutParser is used for consistent parsing.
        """
        from agents.agent1_architect.layout_parser import LayoutParser
        parser = LayoutParser()
        result = []

        for pdf_path in pdf_paths:
            p         = Path(pdf_path)
            bidder_id = p.stem  # filename without extension as bidder ID

            try:
                # layout_parser returns regions; we need the page/text dict
                regions = parser.parse_pdf(str(p))
                # Assemble pages dict compatible with Agent 3 chunker
                pages: Dict[int, list] = {}
                raw_text = ""
                for r in regions:
                    pn = r.get("page", 1)
                    pages.setdefault(pn, []).append({
                        "line_no": r.get("line_no", 1),
                        "text":    r.get("text", ""),
                    })
                    raw_text += r.get("text", "") + " "

                bidder_data = {
                    "filename":  p.name,
                    "bidder_id": bidder_id,
                    "pages":     pages,
                    "raw_text":  raw_text.strip(),
                    "doc_type":  "PDF",
                }
                result.append(bidder_data)
                logger.info(f"Parsed bidder: {bidder_id} ({len(pages)} pages)")

            except Exception as exc:
                logger.error(f"Failed to parse bidder '{bidder_id}': {exc}")
                result.append({
                    "filename":  p.name,
                    "bidder_id": bidder_id,
                    "pages":     {},
                    "raw_text":  "",
                    "doc_type":  "PDF",
                    "parse_error": str(exc),
                })

        return result

    @staticmethod
    def _build_summary(
        tender_id:      str,
        criteria:       List[Dict[str, Any]],
        hitl_summaries: List[Dict[str, Any]],
        report_path:    str,
    ) -> Dict[str, Any]:
        """Build the JSON run-summary written alongside the PDF report."""
        total    = len(hitl_summaries)
        auto_ok  = sum(s.get("auto_approved", 0) for s in hitl_summaries)
        queued   = sum(s.get("queued_for_review", 0) for s in hitl_summaries)

        return {
            "tender_id":       tender_id,
            "criteria_count":  len(criteria),
            "bidder_count":    total,
            "auto_approved":   auto_ok,
            "queued_for_review": queued,
            "report_path":     report_path,
            "bidder_summaries": hitl_summaries,
        }
