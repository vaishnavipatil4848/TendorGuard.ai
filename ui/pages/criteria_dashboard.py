"""
criteria_dashboard.py
ui/pages/ — TendorGuard.ai

Streamlit page: Criteria Dashboard.
Shows all criteria extracted by Agent 1 and the per-criterion
verdict distribution across all bidders with full reasoning.

Reads from:
  st.session_state["criteria"]        — Agent 1 criterion list
  st.session_state["bidder_reports"]  — Agent 4 aggregated reports
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Any

from ui.components.verdict_badge import verdict_badge
from ui.components.reasoning_panel import reasoning_panel


def render(
    criteria: List[Dict[str, Any]],
    bidder_reports: List[Dict[str, Any]],
) -> None:
    """
    Render the criteria dashboard.

    Args:
        criteria:       list of criterion dicts from Agent 1
        bidder_reports: list of aggregated_report dicts from Agent 4
    """
    st.markdown("## 📋 Criteria Dashboard")

    if not criteria:
        st.info("No criteria extracted yet. Run Stage 1 (Agent 1) first.")
        return

    # ── Build index: criterion_id → list of per-bidder verdicts ──────────────
    cv_index: Dict[str, List[Dict]] = {
        c.get("criterion_id", c.get("id", "?")): []
        for c in criteria
    }
    ev_index: Dict[str, Dict[str, Dict]] = {
        k: {} for k in cv_index
    }

    for report in bidder_reports:
        bid = report.get("bidder_id", "?")
        for cv in report.get("criterion_verdicts", []):
            cid = cv.get("criterion_id", "?")
            if cid in cv_index:
                cv_index[cid].append({**cv, "bidder_id": bid})
            if cid in ev_index:
                ev_index[cid][bid] = cv  # store full cv as proxy for evidence

    # ── Filters ───────────────────────────────────────────────────────────────
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        type_filter = st.selectbox(
            "Criterion type",
            options=["All", "MANDATORY", "OPTIONAL"],
            index=0,
        )
    with col_f2:
        verdict_filter = st.selectbox(
            "Show criteria with at least one",
            options=["All", "FAIL", "INELIGIBLE", "NEEDS_MANUAL_REVIEW", "UNCERTAIN"],
            index=0,
        )

    # ── Criteria list ─────────────────────────────────────────────────────────
    for criterion in criteria:
        cid   = criterion.get("criterion_id", criterion.get("id", "?"))
        name  = criterion.get("name", "")
        ctype = criterion.get("criterion_type", "")
        thr   = criterion.get("threshold_value", "")
        unit  = criterion.get("unit", "")
        desc  = criterion.get("description", "")

        verdicts_for_cid = cv_index.get(cid, [])

        # Apply type filter
        if type_filter != "All" and ctype.upper() != type_filter:
            continue

        # Apply verdict filter
        if verdict_filter != "All":
            has_match = any(
                v.get("verdict", "").upper() == verdict_filter.upper()
                for v in verdicts_for_cid
            )
            if not has_match:
                continue

        with st.expander(
            f"{'🔴' if ctype == 'MANDATORY' else '🔵'} [{cid}] {name}",
            expanded=False
        ):
            # Criterion metadata
            col_a, col_b, col_c = st.columns(3)
            col_a.markdown(f"**Type:** `{ctype}`")
            col_b.markdown(f"**Threshold:** {thr} {unit}" if thr else "**Threshold:** —")
            col_c.markdown(f"**Clause:** {criterion.get('legal_clause_reference', '—')}")

            st.caption(desc)

            if criterion.get("document_required"):
                st.markdown(f"📎 Required document: **{criterion['document_required']}**")

            # Per-bidder verdicts for this criterion
            if not bidder_reports:
                st.info("No bidder results yet.")
                continue

            st.markdown("---")
            st.markdown("**Bidder Results**")

            rows = []
            for cv in verdicts_for_cid:
                rows.append({
                    "Bidder":     cv.get("bidder_id", "?"),
                    "Verdict":    cv.get("verdict", "?"),
                    "Confidence": f"{cv.get('confidence', 0):.0%}",
                    "Agreement":  cv.get("agreement_status", "—"),
                    "Programmatic": "✔" if cv.get("programmatic") else "—",
                })

            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # Detailed reasoning for each bidder
            for cv in verdicts_for_cid:
                reasoning_panel(cv, expanded=False)
