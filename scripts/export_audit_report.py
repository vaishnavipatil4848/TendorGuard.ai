"""
export_audit_report.py
scripts/ — TendorGuard.ai

CLI script: export / regenerate a signed PDF audit report
from a completed evaluation that is already in the HITL queue.

Useful for:
  - Re-generating reports after manual reviews are submitted
  - Exporting a specific tender's report to a different directory

Usage:
    python scripts/export_audit_report.py \
        --tender T-4821 \
        --out storage/audit_reports/
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Export a PDF audit report via Agent 5 ReportGenerator"
    )
    parser.add_argument("--tender",   required=True, dest="tender_id",
                        help="Tender identifier to export report for")
    parser.add_argument("--out",      default="storage/audit_reports",
                        help="Output directory (default: storage/audit_reports)")
    parser.add_argument("--verbose",  action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info(f"Exporting audit report for tender '{args.tender_id}'...")

    try:
        from agents.agent5_hitl import HITLAgent
        from agents.agent5_hitl.audit_logger import AuditLogger

        hitl   = HITLAgent()
        a_log  = AuditLogger()

        # Fetch bidder summaries that have been processed for this tender
        # (In production these would come from the DB / queue state)
        bidder_ids = a_log.get_bidder_ids_for_tender(args.tender_id)
        if not bidder_ids:
            logger.warning(
                f"No bidder records found for tender '{args.tender_id}'. "
                "Run the pipeline first or check the AuditLogger storage."
            )
            sys.exit(1)

        # Build minimal bidder_summaries from audit log
        bidder_summaries = [
            {"bidder_id": bid}
            for bid in bidder_ids
        ]

        report_path = hitl.generate_report(
            tender_id=args.tender_id,
            bidder_summaries=bidder_summaries,
            output_dir=args.out,
        )

        print(f"\n✅ Report exported: {report_path}")

    except Exception as exc:
        logger.exception(f"Report export failed: {exc}")
        sys.exit(1)

    return 0


if __name__ == "__main__":
    sys.exit(main())
