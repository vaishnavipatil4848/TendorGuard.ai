"""
leaderboard.py
ui/pages/ — TendorGuard.ai

Streamlit page: Bidder Leaderboard.
Shows all bidders ranked by overall verdict + confidence score
with per-row verdict badges and a drill-down expander.

Reads from st.session_state["bidder_reports"] — a list of
VerdictAggregator output dicts (one per bidder).
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Any

from ui.components.verdict_badge import verdict_badge, overall_verdict_banner
from ui.components.confidence_heatmap import confidence_heatmap


VERDICT_ORDER = {"PASS": 0, "ELIGIBLE": 0, "FAIL": 1, "INELIGIBLE": 1,
                 "UNCERTAIN": 2, "NEEDS_MANUAL_REVIEW": 2}


def render(bidder_reports: List[Dict[str, Any]]) -> None:
    """
    Render the leaderboard page.

    Args:
        bidder_reports: list of aggregated_report dicts from Agent 4.
                        Each must have: bidder_id, overall_verdict,
                        overall_confidence, criterion_verdicts.
    """
    st.markdown("## 🏆 Bidder Leaderboard")

    if not bidder_reports:
        st.info("No evaluation results yet. Run the pipeline first.")
        return

    # ── Summary KPI row ───────────────────────────────────────────────────────
    total     = len(bidder_reports)
    passed    = sum(1 for r in bidder_reports if r.get("overall_verdict") in ("PASS", "ELIGIBLE"))
    failed    = sum(1 for r in bidder_reports if r.get("overall_verdict") in ("FAIL", "INELIGIBLE"))
    uncertain = total - passed - failed

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Bidders",   total)
    c2.metric("✅ Eligible",     passed)
    c3.metric("❌ Ineligible",   failed)
    c4.metric("🔍 Needs Review", uncertain)

    st.markdown("---")

    # ── Sort by verdict priority, then confidence desc ────────────────────────
    sorted_reports = sorted(
        bidder_reports,
        key=lambda r: (
            VERDICT_ORDER.get(r.get("overall_verdict", "UNCERTAIN"), 2),
            -r.get("overall_confidence", 0.0)
        )
    )

    # ── Summary table ─────────────────────────────────────────────────────────
    table_rows = []
    for r in sorted_reports:
        cv = r.get("criterion_verdicts", [])
        passed_c  = sum(1 for c in cv if c.get("verdict") in ("PASS", "ELIGIBLE"))
        failed_c  = sum(1 for c in cv if c.get("verdict") in ("FAIL", "INELIGIBLE"))
        table_rows.append({
            "Bidder":      r.get("bidder_id", "?"),
            "Verdict":     r.get("overall_verdict", "?"),
            "Confidence":  f"{r.get('overall_confidence', 0):.0%}",
            "✅ Criteria": passed_c,
            "❌ Criteria": failed_c,
        })

    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    # ── Confidence heatmap ────────────────────────────────────────────────────
    st.markdown("---")
    heatmap_rows = []
    for r in bidder_reports:
        for cv in r.get("criterion_verdicts", []):
            heatmap_rows.append({
                "bidder_id":    r.get("bidder_id"),
                "criterion_id": cv.get("criterion_id"),
                "confidence":   cv.get("confidence", 0.0),
                "verdict":      cv.get("verdict", ""),
            })
    confidence_heatmap(heatmap_rows)

    # ── Per-bidder drill-down ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔍 Bidder Detail")

    for r in sorted_reports:
        bid     = r.get("bidder_id", "?")
        verdict = r.get("overall_verdict", "UNCERTAIN")
        conf    = r.get("overall_confidence", 0.0)

        overall_verdict_banner(verdict, bid)

        with st.expander(f"Criteria breakdown — {bid} ({conf:.0%} confidence)"):
            cv_list = r.get("criterion_verdicts", [])
            if not cv_list:
                st.caption("No criterion verdicts recorded.")
                continue

            for cv in cv_list:
                cid    = cv.get("criterion_id", "?")
                cv_v   = cv.get("verdict", "UNCERTAIN")
                cv_c   = cv.get("confidence", 0.0)
                reason = cv.get("reasoning", "")[:120]

                col_a, col_b, col_c = st.columns([1, 1, 4])
                with col_a:
                    st.caption(cid)
                with col_b:
                    verdict_badge(cv_v, compact=True)
                with col_c:
                    st.caption(f"{cv_c:.0%}  {reason}")
