"""
review_queue.py
ui/pages/ — TendorGuard.ai

Streamlit page: HITL Manual Review Queue.
Displays the Agent 5 ReviewQueue — cases flagged for human review —
and lets the reviewer submit decisions directly from the UI.

Calls:
  HITLAgent.get_next_case()
  HITLAgent.get_queue_summary()
  HITLAgent.submit_review(...)
"""

import streamlit as st
from typing import Any, Dict, Optional

from ui.components.verdict_badge import verdict_badge, overall_verdict_banner


def render(hitl_agent) -> None:
    """
    Render the HITL review queue page.

    Args:
        hitl_agent: live HITLAgent instance from st.session_state["hitl_agent"]
    """
    st.markdown("## 👤 Manual Review Queue")

    if hitl_agent is None:
        st.info("Run the pipeline first to populate the review queue.")
        return

    # ── Queue summary ─────────────────────────────────────────────────────────
    summary = hitl_agent.get_queue_summary()

    total_q    = summary.get("total", 0)
    pending    = summary.get("pending", 0)
    resolved   = summary.get("resolved", 0)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Cases",  total_q)
    c2.metric("⏳ Pending",   pending)
    c3.metric("✅ Resolved",  resolved)

    if pending == 0:
        st.success("🎉 All cases have been reviewed!")
        return

    st.markdown("---")

    # ── Reviewer identity ─────────────────────────────────────────────────────
    reviewer_id = st.text_input(
        "Reviewer ID",
        value=st.session_state.get("reviewer_id", ""),
        placeholder="e.g. officer_123",
        help="Your unique reviewer identifier — logged in the immutable audit trail."
    )
    if reviewer_id:
        st.session_state["reviewer_id"] = reviewer_id

    st.markdown("---")

    # ── Next case ─────────────────────────────────────────────────────────────
    case = hitl_agent.get_next_case()
    if case is None:
        st.info("No pending cases in the queue.")
        return

    cid      = case.get("criterion_id", "?")
    bid      = case.get("bidder_id", "?")
    ctype    = case.get("case_type", "?")
    sys_v    = case.get("system_suggested_verdict", "UNCERTAIN")
    conf     = case.get("composite_confidence", 0.0)
    ctx_note = case.get("reviewer_context", "")

    st.markdown(f"### Case: `{cid}` for Bidder `{bid}`")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**Case Type:** `{ctype}`")
        st.markdown("**System Verdict:**")
        verdict_badge(sys_v)
        st.markdown(f"**System Confidence:** {conf:.0%}")
    with col_b:
        if ctx_note:
            st.info(f"ℹ️ {ctx_note}")

    # Criterion metadata
    criterion = case.get("criterion", {})
    evidence  = case.get("evidence", {})

    if criterion:
        st.markdown("---")
        st.markdown("**Criterion Details**")
        st.markdown(
            f"**{criterion.get('name', cid)}** — "
            f"`{criterion.get('criterion_type', '')}` · "
            f"Threshold: {criterion.get('threshold_value', '—')} {criterion.get('unit', '')}"
        )
        st.caption(criterion.get("description", ""))

    # Evidence snippet
    if evidence:
        st.markdown("---")
        st.markdown("**Evidence**")
        if evidence.get("found"):
            pg  = evidence.get("page_number", "?")
            ln  = evidence.get("line_number", "?")
            val = evidence.get("extracted_value", "—")
            st.caption(f"Page {pg} · Line {ln} · Value: `{val}`")
            st.code(evidence.get("matching_line_text", ""), language=None)
        else:
            st.warning(
                f"Evidence not found. "
                f"{evidence.get('ambiguity_reason', '')}"
            )

    # ── Review form ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### ✍️ Submit Review Decision")

    with st.form(key=f"review_form_{cid}_{bid}"):
        human_verdict = st.radio(
            "Your Verdict",
            options=["PASS", "FAIL", "UNCERTAIN"],
            horizontal=True,
            help="UNCERTAIN will keep this case flagged for further review."
        )
        comment = st.text_area(
            "Mandatory Justification",
            placeholder="Describe your reasoning for this decision...",
            height=100,
        )
        rejection_reason = st.selectbox(
            "Rejection Category (optional)",
            options=["", "Insufficient Evidence", "Document Expired",
                     "OCR Error", "Threshold Not Met", "Other"],
            index=0,
        )
        submitted = st.form_submit_button("📨 Submit Review")

    if submitted:
        if not reviewer_id:
            st.error("Please enter your Reviewer ID before submitting.")
        elif not comment.strip():
            st.error("A justification comment is required.")
        else:
            try:
                log_id = hitl_agent.submit_review(
                    tender_id=st.session_state.get("tender_id", "unknown"),
                    criterion_id=cid,
                    human_verdict=human_verdict,
                    reviewer_id=reviewer_id,
                    comment=comment.strip(),
                    rejection_reason=rejection_reason or None,
                )
                st.success(
                    f"✅ Review submitted. "
                    f"Audit log ID: `{log_id}`"
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Submission failed: {exc}")

    # ── Feedback metrics ──────────────────────────────────────────────────────
    with st.expander("📈 Model Accuracy Metrics"):
        metrics = hitl_agent.get_feedback_metrics()
        if metrics:
            col1, col2 = st.columns(2)
            col1.metric("Agreement Rate",    f"{metrics.get('agreement_rate', 0):.1%}")
            col2.metric("Reviews Logged",    metrics.get("total_reviews", 0))
        else:
            st.caption("No feedback metrics yet.")
