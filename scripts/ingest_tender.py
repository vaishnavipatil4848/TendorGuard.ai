"""
ingest_tender.py
scripts/ — TendorGuard.ai

CLI script: parse a tender PDF through Agent 1 (ArchitectAgent)
and save the extracted ruleset JSON to storage/rulesets/.

Usage:
    python scripts/ingest_tender.py --pdf path/to/tender.pdf --id T-4821
    python scripts/ingest_tender.py --pdf path/to/tender.pdf --id T-4821 --out storage/rulesets
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Ingest a tender PDF via Agent 1")
    parser.add_argument("--pdf",  required=True, help="Path to tender PDF file")
    parser.add_argument("--id",   required=True, dest="tender_id",
                        help="Unique tender identifier (e.g. T-4821)")
    parser.add_argument("--out",  default="storage/rulesets",
                        help="Output directory for ruleset JSON (default: storage/rulesets)")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Ingesting tender '{args.tender_id}' from {pdf_path}")

    try:
        from agents.agent1_architect import ArchitectAgent
        agent1 = ArchitectAgent()
        criteria, validation_report = agent1.run(
            pdf_path=str(pdf_path),
            tender_id=args.tender_id,
            ruleset_output_dir=str(out_dir),
        )
    except Exception as exc:
        logger.exception(f"Agent 1 failed: {exc}")
        sys.exit(1)

    # Print summary
    print(f"\n✅ Tender ingested: '{args.tender_id}'")
    print(f"   Criteria extracted  : {len(criteria)}")
    print(f"   Validation passed   : {validation_report.get('is_valid', 'unknown')}")
    print(f"   Ruleset saved to    : {out_dir}/{args.tender_id}_ruleset.json")

    # Print criteria table
    print("\n── Criteria Summary ────────────────────────────────")
    for c in criteria:
        cid   = c.get("criterion_id", c.get("id", "?"))
        name  = c.get("name", "")
        ctype = c.get("criterion_type", "")
        thr   = c.get("threshold_value", "")
        thr_str = f" (≥ {thr} {c.get('unit','')})" if thr else ""
        print(f"  [{ctype:9s}] {cid}: {name}{thr_str}")

    if validation_report.get("errors"):
        print("\n── Validation Warnings ─────────────────────────────")
        for err in validation_report["errors"]:
            print(f"  ⚠  {err}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
