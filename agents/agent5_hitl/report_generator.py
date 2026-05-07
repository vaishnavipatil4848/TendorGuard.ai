"""
report_generator.py
Agent 5 — HITL Agent
Generates a signed PDF audit report for a completed tender evaluation.

Contents:
  - Cover page: tender ID, evaluation date, summary verdict counts
  - Per-bidder verdict table: overall verdict + confidence
  - Per-criterion breakdown: verdict, case type, human override flag
  - Audit trail section: all human decisions with reviewer, timestamp, comment
  - Model accuracy summary: override rates by criterion type and model family

Uses ReportLab for PDF generation (no external dependencies beyond pip).
The "signed" aspect is a SHA-256 hash of the report content appended
to the final page — sufficient for basic tamper detection in a demo context.
"""

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, PageBreak
    )
    REPORTLAB_AVAILABLE = True
    
    # Colour palette matching TendorGuard brand (navy + amber)
    NAVY  = colors.HexColor("#1B2A4A")
    AMBER = colors.HexColor("#F59E0B")
    LIGHT = colors.HexColor("#F3F4F6")
    RED   = colors.HexColor("#DC2626")
    GREEN = colors.HexColor("#16A34A")
    GREY  = colors.HexColor("#6B7280")
except ImportError:
    REPORTLAB_AVAILABLE = False
    NAVY = AMBER = LIGHT = RED = GREEN = GREY = None
    logger.warning(
        "ReportLab not installed — PDF generation disabled. "
        "Install with: pip install reportlab"
    )


class ReportGenerator:
    """
    Generates a signed PDF audit report from the HITL agent's collected data.
    """

    def generate(
        self,
        tender_id: str,
        bidder_summaries: List[Dict[str, Any]],
        audit_log_records: List[Dict[str, Any]],
        feedback_metrics: Dict[str, Any],
        output_dir: str = "storage/audit_reports"
    ) -> str:
        """
        Generate the full PDF audit report.

        Args:
            tender_id:         tender identifier
            bidder_summaries:  list of aggregated_report dicts from VerdictAggregator
            audit_log_records: all human review decisions from AuditLogger
            feedback_metrics:  model accuracy stats from FeedbackLoop
            output_dir:        directory to write the PDF to

        Returns:
            Path to the generated PDF file
        """
        if not REPORTLAB_AVAILABLE:
            raise RuntimeError(
                "ReportLab is not installed. Run: pip install reportlab"
            )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"audit_report_{tender_id}_{timestamp}.pdf"
        filepath = output_dir / filename

        doc = SimpleDocTemplate(
            str(filepath),
            pagesize=A4,
            leftMargin=2 * cm,
            rightMargin=2 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
        )

        styles = getSampleStyleSheet()
        story = []

        story += self._cover_page(tender_id, bidder_summaries, styles)
        story.append(PageBreak())
        story += self._bidder_summary_table(bidder_summaries, styles)
        story.append(PageBreak())
        story += self._audit_trail_section(audit_log_records, styles)
        story.append(PageBreak())
        story += self._model_accuracy_section(feedback_metrics, styles)
        story += self._signature_block(tender_id, styles)

        doc.build(story)

        # append SHA-256 content hash for tamper detection
        self._append_content_hash(filepath)

        logger.info(f"Audit report generated: {filepath}")
        return str(filepath)

    # ------------------------------------------------------------------ #
    # Section builders
    # ------------------------------------------------------------------ #

    def _cover_page(
        self,
        tender_id: str,
        bidder_summaries: List[Dict[str, Any]],
        styles
    ) -> list:
        elements = []

        title_style = ParagraphStyle(
            "ReportTitle",
            parent=styles["Title"],
            textColor=NAVY,
            fontSize=24,
            spaceAfter=12,
        )
        sub_style = ParagraphStyle(
            "ReportSub",
            parent=styles["Normal"],
            textColor=GREY,
            fontSize=11,
            spaceAfter=6,
        )

        elements.append(Spacer(1, 2 * cm))
        elements.append(Paragraph("TendorGuard AI", title_style))
        elements.append(Paragraph("Automated Eligibility Evaluation — Audit Report", sub_style))
        elements.append(HRFlowable(width="100%", thickness=2, color=AMBER))
        elements.append(Spacer(1, 0.5 * cm))

        elements.append(Paragraph(f"<b>Tender ID:</b> {tender_id}", styles["Normal"]))
        elements.append(Paragraph(
            f"<b>Generated:</b> {datetime.now(timezone.utc).strftime('%d %B %Y, %H:%M UTC')}",
            styles["Normal"]
        ))
        elements.append(Spacer(1, 1 * cm))

        # summary counts
        total = len(bidder_summaries)
        passed = sum(1 for b in bidder_summaries if b.get("overall_verdict") == "PASS")
        failed = sum(1 for b in bidder_summaries if b.get("overall_verdict") == "FAIL")
        review = sum(1 for b in bidder_summaries if b.get("overall_verdict") == "PENDING_REVIEW")
        uncertain = total - passed - failed - review

        summary_data = [
            ["Total Bidders", "Eligible", "Ineligible", "Pending Review", "Uncertain"],
            [str(total), str(passed), str(failed), str(review), str(uncertain)],
        ]
        summary_table = Table(summary_data, colWidths=[3.5 * cm] * 5)
        summary_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT, colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("FONTSIZE", (0, 1), (-1, -1), 14),
        ]))
        elements.append(summary_table)
        return elements

    def _bidder_summary_table(
        self,
        bidder_summaries: List[Dict[str, Any]],
        styles
    ) -> list:
        elements = []
        elements.append(Paragraph("Bidder Evaluation Summary", styles["Heading1"]))
        elements.append(Spacer(1, 0.3 * cm))

        headers = ["Bidder ID", "Overall Verdict", "Confidence", "Pass", "Fail", "Uncertain", "HITL"]
        data = [headers]

        for b in bidder_summaries:
            cs = b.get("criteria_summary", {})
            verdict = b.get("overall_verdict", "?")
            data.append([
                b.get("bidder_id", "?"),
                verdict,
                f"{b.get('overall_confidence', 0):.0%}",
                str(cs.get("pass", 0)),
                str(cs.get("fail", 0)),
                str(cs.get("uncertain", 0)),
                str(cs.get("pending_hitl", 0)),
            ])

        col_widths = [3.5 * cm, 3.5 * cm, 2.5 * cm, 1.5 * cm, 1.5 * cm, 2 * cm, 1.5 * cm]
        table = Table(data, colWidths=col_widths)
        table.setStyle(self._default_table_style(verdict_col=1))
        elements.append(table)
        return elements

    def _audit_trail_section(
        self,
        audit_records: List[Dict[str, Any]],
        styles
    ) -> list:
        elements = []
        elements.append(Paragraph("Human Review Audit Trail", styles["Heading1"]))
        elements.append(Spacer(1, 0.3 * cm))

        if not audit_records:
            elements.append(Paragraph(
                "No human review decisions recorded for this tender.",
                styles["Normal"]
            ))
            return elements

        headers = ["Criterion", "Bidder", "Case Type", "System", "Human", "Override", "Reviewer", "Timestamp"]
        data = [headers]

        for rec in audit_records:
            override = "YES" if rec.get("was_override") else "no"
            data.append([
                rec.get("criterion_id", "?"),
                rec.get("bidder_id", "?"),
                rec.get("case_type", "?")[:18],
                rec.get("system_verdict", "?"),
                rec.get("human_verdict", "?"),
                override,
                rec.get("reviewer_id", "?")[:10],
                str(rec.get("created_at", ""))[:16],
            ])

        col_widths = [1.8*cm, 2*cm, 3.5*cm, 1.5*cm, 1.5*cm, 1.8*cm, 2.5*cm, 3.5*cm]
        table = Table(data, colWidths=col_widths)
        table.setStyle(self._default_table_style())
        elements.append(table)

        # comments sub-table
        elements.append(Spacer(1, 0.5 * cm))
        elements.append(Paragraph("Reviewer Comments", styles["Heading2"]))
        for rec in audit_records:
            comment = rec.get("comment", "")
            if comment:
                elements.append(Paragraph(
                    f"<b>[{rec.get('criterion_id')} / {rec.get('bidder_id')}]</b> "
                    f"{rec.get('reviewer_id')}: {comment}",
                    styles["Normal"]
                ))
                elements.append(Spacer(1, 0.2 * cm))

        return elements

    def _model_accuracy_section(
        self,
        metrics: Dict[str, Any],
        styles
    ) -> list:
        elements = []
        elements.append(Paragraph("Model Accuracy Summary", styles["Heading1"]))
        elements.append(Spacer(1, 0.3 * cm))

        overall = [
            ["Total Reviewed", "Model Correct", "Human Overrides", "Overall Accuracy"],
            [
                str(metrics.get("total_reviewed", 0)),
                str(metrics.get("model_correct", 0)),
                str(metrics.get("human_overrides", 0)),
                f"{metrics.get('overall_accuracy', 0):.1%}",
            ]
        ]
        t = Table(overall, colWidths=[4 * cm] * 4)
        t.setStyle(self._default_table_style())
        elements.append(t)
        elements.append(Spacer(1, 0.4 * cm))

        # by model family
        by_model = metrics.get("by_model", {})
        if by_model:
            elements.append(Paragraph("By Model Family", styles["Heading2"]))
            model_data = [["Model", "Correct", "Overrides", "Accuracy"]]
            for model, stats in by_model.items():
                model_data.append([
                    model.upper(),
                    str(stats.get("correct", 0)),
                    str(stats.get("override", 0)),
                    f"{stats.get('accuracy', 0):.1%}",
                ])
            mt = Table(model_data, colWidths=[4 * cm] * 4)
            mt.setStyle(self._default_table_style())
            elements.append(mt)

        return elements

    def _signature_block(self, tender_id: str, styles) -> list:
        elements = [Spacer(1, 1 * cm), HRFlowable(width="100%", thickness=1, color=GREY)]
        elements.append(Paragraph(
            f"This report was generated automatically by TendorGuard AI for tender <b>{tender_id}</b>. "
            "All automated decisions are subject to human review. "
            "A SHA-256 content hash is appended to this file for tamper detection.",
            ParagraphStyle("Footer", parent=styles["Normal"], textColor=GREY, fontSize=8)
        ))
        return elements

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _default_table_style(self, verdict_col: Optional[int] = None) -> TableStyle:
        style = [
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("FONTSIZE", (0, 1), (-1, -1), 8),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT, colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]
        return TableStyle(style)

    def _append_content_hash(self, filepath: Path) -> None:
        """Append a SHA-256 hash of the PDF content for tamper detection."""
        try:
            content = filepath.read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            with open(filepath, "ab") as f:
                f.write(f"\n%% SHA256: {digest}\n".encode())
        except Exception as e:
            logger.warning(f"Could not append content hash: {e}")