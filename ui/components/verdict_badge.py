"""
verdict_badge.py
ui/components/ — TendorGuard.ai

Streamlit component: renders a colour-coded verdict badge.
Used by leaderboard.py, criteria_dashboard.py, and review_queue.py.
"""

import streamlit as st


# Verdict colour map — matches Agent 4 VerdictAggregator output values
VERDICT_STYLES: dict = {
    "PASS":               ("🟢", "#1f883d", "#dafbe1"),
    "FAIL":               ("🔴", "#cf222e", "#ffebe9"),
    "UNCERTAIN":          ("🟡", "#9a6700", "#fff8c5"),
    "NEEDS_MANUAL_REVIEW": ("🟠", "#bc4c00", "#fff1e5"),
    "ELIGIBLE":           ("🟢", "#1f883d", "#dafbe1"),
    "INELIGIBLE":         ("🔴", "#cf222e", "#ffebe9"),
}

DEFAULT_STYLE = ("⚪", "#57606a", "#f6f8fa")


def verdict_badge(verdict: str, label: str = "", compact: bool = False) -> None:
    """
    Render a verdict badge with icon and background colour.

    Args:
        verdict: verdict string (PASS / FAIL / UNCERTAIN / NEEDS_MANUAL_REVIEW)
        label:   optional text label to display beside the badge
        compact: if True, render as an inline HTML span (no block padding)
    """
    icon, fg, bg = VERDICT_STYLES.get(verdict.upper(), DEFAULT_STYLE)
    display_text = f"{icon} {label or verdict}"

    if compact:
        st.markdown(
            f'<span style="'
            f'background:{bg}; color:{fg}; '
            f'padding:2px 8px; border-radius:12px; '
            f'font-size:0.85em; font-weight:600;">'
            f'{display_text}</span>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div style="'
            f'background:{bg}; color:{fg}; '
            f'padding:6px 14px; border-radius:8px; '
            f'display:inline-block; font-weight:700; '
            f'font-size:1.05em; margin:4px 0;">'
            f'{display_text}</div>',
            unsafe_allow_html=True
        )


def overall_verdict_banner(verdict: str, bidder_name: str) -> None:
    """
    Full-width banner for a bidder's overall verdict.

    Args:
        verdict:     overall verdict string
        bidder_name: bidder display name
    """
    icon, fg, bg = VERDICT_STYLES.get(verdict.upper(), DEFAULT_STYLE)
    st.markdown(
        f'<div style="'
        f'background:{bg}; color:{fg}; '
        f'padding:14px 20px; border-radius:10px; '
        f'border-left:6px solid {fg}; '
        f'font-weight:700; font-size:1.15em; margin:8px 0;">'
        f'{icon}&nbsp; {bidder_name} — {verdict}'
        f'</div>',
        unsafe_allow_html=True
    )
