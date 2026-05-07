"""
evidence_viewer.py
ui/pages/ — TendorGuard.ai

Streamlit page: Evidence Viewer.
Lets the reviewer select a bidder + criterion and see the full
evidence detail from Agent 3 — source text, page/line citation,
extracted value, and confidence.

Reads from:
  st.session_state["bidder_reports"]  — Agent 4 reports (contain evidence refs)
  st.session_state["evidence_maps"]   — dict {bidder_id: {criterion_id: evidence}}
"""

import streamlit as st
from typing import Any, Dict, List, Optional

from ui.components.verdict_badge import verdict_badge
from ui.components.reasoning_panel import reasoning_panel


def render(
    bidder_reports: List[Dict[str, Any]],
    evidence_maps:  Dict[str, Dict[str, Any]],
    criteria:       List[Dict[str, Any]],
) -> None:
    """
    Render the evidence viewer page.

    Args:
        bidder_reports: list of Agent 4 aggregated_report dicts
        evidence_maps:  {bidder_id: {criterion_id: evidence_dict}}
        criteria:       list of criterion dicts from Agent 1
    """
    st.markdown("## 🔎 Evidence Viewer")

    if not bidder_reports:
        st.info("No evaluation results yet.")
        return

    # ── Selection controls ────────────────────────────────────────────────────
    bidder_ids  = [r.get("bidder_id", "?") for r in bidder_reports]
    criteria_ids = [
        c.get("criterion_id", c.get("id", "?")) for c in criteria
    ]

    col1, col2 = st.columns(2)
    with col1:
        selected_bidder = st.selectbox("Select Bidder", bidder_ids)
    with col2:
        selected_cid = st.selectbox("Select Criterion", criteria_ids)

    # ── Resolve evidence + verdict ────────────────────────────────────────────
    evidence = (
        evidence_maps
        .get(selected_bidder, {})
        .get(selected_cid)
    )

    selected_report = next(
        (r for r in bidder_reports if r.get("bidder_id") == selected_bidder),
        None
    )
    criterion_verdict = None
    if selected_report:
        criterion_verdict = next(
            (cv for cv in selected_report.get("criterion_verdicts", [])
             if cv.get("criterion_id") == selected_cid),
            None
        )

    # ── Criterion metadata ────────────────────────────────────────────────────
    criterion_meta = next(
        (c for c in criteria
         if c.get("criterion_id", c.get("id")) == selected_cid),
        {}
    )

    if criterion_meta:
        st.markdown("---")
        col_a, col_b, col_c = st.columns(3)
        col_a.markdown(f"**Criterion:** `{selected_cid}`")
        col_b.markdown(f"**Type:** `{criterion_meta.get('criterion_type', '—')}`")
        col_c.markdown(
            f"**Threshold:** {criterion_meta.get('threshold_value', '—')} "
            f"{criterion_meta.get('unit', '')}"
        )
        st.caption(criterion_meta.get("description", ""))

    # ── Evidence display ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📄 Extracted Evidence")

    if evidence is None:
        st.warning("No evidence record found for this bidder / criterion combination.")
    elif not evidence.get("found"):
        st.error(
            f"**Evidence not found.**  \n"
            f"Reason: {evidence.get('ambiguity_reason', 'No evidence retrieved.')}"
        )
        st.caption(
            f"Extraction method: `{evidence.get('extraction_method', '—')}`  |  "
            f"Confidence: {evidence.get('confidence', 0):.0%}"
        )
    else:
        # Found evidence
        pg  = evidence.get("page_number", "?")
        ln  = evidence.get("line_number", "?")
        val = evidence.get("extracted_value", "")
        ext = evidence.get("extracted_numeric")
        thr = evidence.get("threshold_numeric")
        src = evidence.get("source_document", "")
        met = evidence.get("meets_threshold")
        method = evidence.get("extraction_method", "—")

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Page",       pg)
        col_m2.metric("Line",       ln)
        col_m3.metric("Confidence", f"{evidence.get('confidence', 0):.0%}")
        col_m4.metric("Meets Threshold", "✅ Yes" if met else ("❌ No" if met is False else "—"))

        st.markdown(f"**Source Document:** `{src}`")
        st.caption(f"Extraction method: `{method}`")

        # Matching text
        st.markdown("**Matching Line:**")
        st.code(evidence.get("matching_line_text", ""), language=None)

        # Context
        if evidence.get("context_lines"):
            with st.expander("Surrounding context lines"):
                for cl in evidence["context_lines"]:
                    st.text(cl)

        # Numeric comparison
        if ext is not None and thr is not None:
            st.markdown("---")
            st.markdown(
                f"**Numeric Comparison:** "
                f"Extracted `{ext:,.0f}` vs Required `{thr:,.0f}`"
            )

        # Ambiguity note
        if evidence.get("ambiguity_reason"):
            st.warning(f"⚠️ Note: {evidence['ambiguity_reason']}")

    # ── Verdict + reasoning ───────────────────────────────────────────────────
    if criterion_verdict:
        st.markdown("---")
        st.markdown("### ⚖️ Verdict & Reasoning")
        reasoning_panel(criterion_verdict, evidence=evidence, expanded=True)
