"""
run_pipeline.py
scripts/ — TendorGuard.ai

CLI script: run the full end-to-end evaluation pipeline for a tender
against one or more bidder PDFs.

Usage:
    python scripts/run_pipeline.py \
        --tender  storage/tender_docs/T4821.pdf \
        --id      T-4821 \
        --bidders storage/bidder_docs/

    python scripts/run_pipeline.py \
        --tender  storage/tender_docs/T4821.pdf \
        --id      T-4821 \
        --bidders bidder1.pdf bidder2.pdf bidder3.pdf
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def collect_bidder_pdfs(sources: list) -> list:
    """
    Resolve bidder PDF paths from a mix of files and directories.
    Returns a sorted list of absolute PDF path strings.
    """
    paths = []
    for src in sources:
        p = Path(src)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.pdf")))
        elif p.is_file() and p.suffix.lower() == ".pdf":
            paths.append(p)
        else:
            logger.warning(f"Skipping non-PDF path: {p}")
    return [str(p.resolve()) for p in paths]


def main():
    parser = argparse.ArgumentParser(
        description="Run the full TendorGuard.ai evaluation pipeline"
    )
    parser.add_argument("--tender",   required=True,
                        help="Path to tender PDF")
    parser.add_argument("--id",       required=True, dest="tender_id",
                        help="Unique tender identifier (e.g. T-4821)")
    parser.add_argument("--bidders",  required=True, nargs="+",
                        help="Bidder PDF files or a directory containing them")
    parser.add_argument("--out",      default="storage/audit_reports",
                        help="Output directory for PDF report (default: storage/audit_reports)")
    parser.add_argument("--verbose",  action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    tender_path = Path(args.tender)
    if not tender_path.exists():
        logger.error(f"Tender PDF not found: {tender_path}")
        sys.exit(1)

    bidder_paths = collect_bidder_pdfs(args.bidders)
    if not bidder_paths:
        logger.error("No bidder PDFs found.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  TendorGuard.ai — Tender Evaluation Pipeline")
    print(f"{'='*60}")
    print(f"  Tender ID : {args.tender_id}")
    print(f"  Tender PDF: {tender_path.name}")
    print(f"  Bidders   : {len(bidder_paths)}")
    for bp in bidder_paths:
        print(f"    • {Path(bp).name}")
    print(f"  Output    : {args.out}")
    print(f"{'='*60}\n")

    try:
        from pipeline.pipeline_runner import PipelineRunner
        runner  = PipelineRunner(verbose=True)
        summary = runner.run(
            tender_id=args.tender_id,
            tender_pdf_path=str(tender_path.resolve()),
            bidder_pdf_paths=bidder_paths,
            output_dir=args.out,
        )
    except Exception as exc:
        logger.exception(f"Pipeline failed: {exc}")
        sys.exit(1)

    # Final summary
    print(f"\n{'='*60}")
    print(f"  Run Summary")
    print(f"{'='*60}")
    print(f"  Tender ID      : {summary.get('tender_id')}")
    print(f"  Criteria       : {summary.get('criteria_count')}")
    print(f"  Bidders        : {summary.get('bidder_count')}")
    print(f"  Auto-approved  : {summary.get('auto_approved')}")
    print(f"  Queued (HITL)  : {summary.get('queued_for_review')}")
    print(f"  Report         : {summary.get('report_path')}")
    print(f"{'='*60}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
