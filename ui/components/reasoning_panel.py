"""
reasoning_panel.py
ui/components/ — TendorGuard.ai

Streamlit component: renders the full reasoning panel for one
criterion verdict — shows evidence snippet, dual-LLM reasoning
from Agent 4, and the HITL flag reason if present.

Expects a criterion_verdict dict (output of VerdictAggregator).
"""

import streamlit as st
from typing import Any, Dict, Optional

from ui.components.verdict_badge import verdict_badge


def reasoning_panel(
    criterion_verdict: Dict[str, Any],
    evidence:          Optional[Dict[str, Any]] = None,
    expanded:          bool = False,
) -> None:
    """
    Render an expandable reasoning panel for one criterion.

    Args:
        criterion_verdict: dict from Agent 4 VerdictAggregator
                           expected keys: criterion_id, verdict, confidence,
                           reasoning, key_factor, ambiguities,
                           agreement_status, divergence_details
        evidence:          Agent 3 evidence dict (optional)
        expanded:          whether the expander starts open
    """
    cid      = criterion_verdict.get("criterion_id", "?")
    verdict  = criterion_verdict.get("verdict", "UNCERTAIN")
    conf     = criterion_verdict.get("confidence", 0.0)
    agree    = criterion_verdict.get("agreement_status", "")
    prog     = criterion_verdict.get("programmatic", False)

    label = f"[{cid}]  conf={conf:.0%}  agreement={agree or '—'}"

    with st.expander(label, expanded=expanded):
        col1, col2 = st.columns([1, 3])
        with col1:
            verdict_badge(verdict)
            if prog:
                st.caption("🔢 Programmatic check")
            else:
                st.caption("🤖 Dual-LLM evaluation")

        with col2:
            st.markdown(f"**Confidence:** {conf:.0%}")

        # Evidence snippet
        if evidence and evidence.get("found"):
            st.markdown("---")
            st.markdown("**📄 Evidence Found**")

            pg  = evidence.get("page_number", "?")
            ln  = evidence.get("line_number", "?")
            val = evidence.get("extracted_value", "")
            txt = evidence.get("matching_line_text", "")

            st.caption(f"Page {pg} · Line {ln} · Extracted: `{val}`")
            if txt:
                st.code(txt, language=None)
            if evidence.get("context_lines"):
                with st.expander("Context lines"):
                    for cl in evidence["context_lines"]:
                        st.text(cl)

        elif evidence:
            st.warning(
                f"❌ Evidence not found. "
                f"{evidence.get('ambiguity_reason', '')}"
            )

        # Reasoning
        reasoning = criterion_verdict.get("reasoning", "")
        if reasoning:
            st.markdown("---")
            st.markdown("**💬 Reasoning**")
            # Split Claude / GPT-4o sections if present
            if "Claude reasoning:" in reasoning and "GPT-4o reasoning:" in reasoning:
                parts = reasoning.split("GPT-4o reasoning:")
                st.markdown("*Claude:* " + parts[0].replace("Claude reasoning:", "").strip())
                st.markdown("*GPT-4o:* " + parts[1].strip())
            else:
                st.markdown(reasoning)

        # Key factor
        key_factor = criterion_verdict.get("key_factor", "")
        if key_factor:
            st.info(f"🔑 Key Factor: {key_factor}")

        # Ambiguities
        ambiguities = criterion_verdict.get("ambiguities", [])
        if ambiguities:
            st.warning("⚠️ Ambiguities flagged by one or both models:")
            for a in ambiguities:
                st.markdown(f"  • {a}")

        # Divergence details
        divergence = criterion_verdict.get("divergence_details", [])
        if divergence:
            st.markdown("---")
            st.markdown("**⚡ Model Disagreement Details**")
            for d in divergence:
                st.markdown(f"  • {d}")
