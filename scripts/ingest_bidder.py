"""
ingest_bidder.py
scripts/ — TendorGuard.ai

CLI script: parse one bidder's PDF, run Agent 3 evidence extraction
for every criterion in a saved ruleset, and print evidence findings.

Usage:
    python scripts/ingest_bidder.py \
        --pdf storage/bidder_docs/acme_corp.pdf \
        --bidder ACME_Corp \
        --ruleset storage/rulesets/T-4821_ruleset.json
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
    parser = argparse.ArgumentParser(description="Ingest a bidder PDF and run evidence extraction")
    parser.add_argument("--pdf",      required=True, help="Path to bidder PDF")
    parser.add_argument("--bidder",   required=True, help="Unique bidder identifier")
    parser.add_argument("--ruleset",  required=True, help="Path to ruleset JSON from ingest_tender.py")
    parser.add_argument("--verbose",  action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    pdf_path     = Path(args.pdf)
    ruleset_path = Path(args.ruleset)

    for p in (pdf_path, ruleset_path):
        if not p.exists():
            logger.error(f"File not found: {p}")
            sys.exit(1)

    # Load criteria from ruleset
    with open(ruleset_path, encoding="utf-8") as f:
        criteria = json.load(f)
    logger.info(f"Loaded {len(criteria)} criteria from {ruleset_path.name}")

    # Parse bidder PDF via Agent 1's LayoutParser
    logger.info(f"Parsing bidder PDF: {pdf_path.name}")
    try:
        from agents.agent1_architect.layout_parser import LayoutParser
        lp      = LayoutParser()
        regions = lp.parse_pdf(str(pdf_path))

        pages: dict = {}
        raw_text = ""
        for r in regions:
            pn = r.get("page", 1)
            pages.setdefault(pn, []).append({
                "line_no": r.get("line_no", 1),
                "text":    r.get("text", ""),
            })
            raw_text += r.get("text", "") + " "

        bidder_data = {
            "filename":  pdf_path.name,
            "bidder_id": args.bidder,
            "pages":     pages,
            "raw_text":  raw_text.strip(),
            "doc_type":  "PDF",
        }
        logger.info(f"Parsed {len(pages)} pages for '{args.bidder}'")

    except Exception as exc:
        logger.exception(f"PDF parsing failed: {exc}")
        sys.exit(1)

    # Agent 3: chunk → index → retrieve → extract
    logger.info("Running Agent 3 evidence extraction...")
    try:
        from agents.agent3_factchecker.chunker          import HierarchicalChunker
        from agents.agent3_factchecker.dense_retriever  import DenseRetriever
        from agents.agent3_factchecker.sparse_retriever import SparseRetriever
        from agents.agent3_factchecker.metadata_filter  import MetadataFilter
        from agents.agent3_factchecker.rrf_fusion       import RRFFusion
        from agents.agent3_factchecker.reranker         import CrossEncoderReranker
        from agents.agent3_factchecker.evidence_extractor import EvidenceExtractor

        chunks = HierarchicalChunker().chunk(bidder_data)
        logger.info(f"Chunked into {len(chunks)} pieces")

        dense   = DenseRetriever()
        sparse  = SparseRetriever()
        dense.index(chunks, args.bidder)
        filtered = MetadataFilter().filter(chunks, criteria)
        sparse.index(filtered)

        fusion   = RRFFusion()
        reranker = CrossEncoderReranker()
        extractor = EvidenceExtractor()

    except Exception as exc:
        logger.exception(f"Agent 3 setup failed: {exc}")
        sys.exit(1)

    # Run per criterion
    print(f"\n── Evidence Report: {args.bidder} ─────────────────────────")
    for criterion in criteria:
        cid   = criterion.get("criterion_id") or criterion.get("id")
        query = f"{criterion.get('name', '')} {criterion.get('description', '')}"

        try:
            dense_hits  = dense.retrieve(query, args.bidder, top_k=20)
            sparse_hits = sparse.retrieve(query, top_k=20)
            fused       = fusion.fuse(dense_hits, sparse_hits)
            top_chunks  = reranker.rerank(query, fused, top_k=5)
            evidence    = extractor.extract(criterion, top_chunks)
        except Exception as exc:
            evidence = {"found": False, "ambiguity_reason": str(exc)}

        found  = evidence.get("found", False)
        conf   = evidence.get("confidence", 0.0)
        val    = evidence.get("extracted_value", "—")
        pg     = evidence.get("page_number", "?")
        ln     = evidence.get("line_number", "?")
        status = "✅" if found and conf >= 0.75 else ("⚠️" if found else "❌")

        print(
            f"  {status} [{cid}] {criterion.get('name', '')[:40]:<40} "
            f"| found={found}  conf={conf:.2f}  val={val}  "
            f"pg={pg} ln={ln}"
        )
        if evidence.get("ambiguity_reason"):
            print(f"       ↳ {evidence['ambiguity_reason']}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
