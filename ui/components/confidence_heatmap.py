"""
confidence_heatmap.py
ui/components/ — TendorGuard.ai

Streamlit component: renders a colour-coded confidence heatmap
for all criteria across one or more bidders.

Data expected:
    [
      {"bidder_id": "ACME", "criterion_id": "C-001", "confidence": 0.92, "verdict": "PASS"},
      ...
    ]
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Any


def _conf_color(conf: float) -> str:
    """Map a confidence float [0,1] to a hex background colour."""
    if conf >= 0.90:
        return "#dafbe1"   # green
    elif conf >= 0.75:
        return "#fff8c5"   # yellow
    elif conf >= 0.50:
        return "#fff1e5"   # orange
    else:
        return "#ffebe9"   # red


def confidence_heatmap(rows: List[Dict[str, Any]]) -> None:
    """
    Render a pivot-table heatmap:
        rows    → bidders
        columns → criterion IDs
        cells   → confidence score, colour-coded

    Args:
        rows: list of dicts with bidder_id, criterion_id, confidence, verdict
    """
    if not rows:
        st.info("No data to display in heatmap.")
        return

    df = pd.DataFrame(rows)

    if "bidder_id" not in df.columns or "criterion_id" not in df.columns:
        st.warning("Heatmap: expected columns 'bidder_id' and 'criterion_id'.")
        return

    pivot = df.pivot_table(
        index="bidder_id",
        columns="criterion_id",
        values="confidence",
        aggfunc="mean"
    ).round(2)

    def _style_cell(val):
        if pd.isna(val):
            return "background-color: #f6f8fa; color: #57606a;"
        return f"background-color: {_conf_color(val)}; font-weight: 600;"

    styled = pivot.style.applymap(_style_cell).format("{:.2f}", na_rep="N/A")

    st.markdown("#### 🌡️ Confidence Heatmap")
    st.dataframe(styled, use_container_width=True)

    # Legend
    st.markdown(
        '<div style="font-size:0.8em; color:#57606a; margin-top:4px;">'
        '<span style="background:#dafbe1; padding:2px 8px; border-radius:4px;">≥ 0.90</span>&nbsp;'
        '<span style="background:#fff8c5; padding:2px 8px; border-radius:4px;">≥ 0.75</span>&nbsp;'
        '<span style="background:#fff1e5; padding:2px 8px; border-radius:4px;">≥ 0.50</span>&nbsp;'
        '<span style="background:#ffebe9; padding:2px 8px; border-radius:4px;">&lt; 0.50</span>'
        '</div>',
        unsafe_allow_html=True
    )


def mini_confidence_bar(confidence: float, label: str = "") -> None:
    """
    Render a compact horizontal confidence bar for inline use
    (e.g. inside a criteria table row).

    Args:
        confidence: float in [0, 1]
        label:      optional label shown to the right
    """
    pct   = int(confidence * 100)
    color = _conf_color(confidence)
    label_html = f'<span style="font-size:0.8em; color:#57606a;">{label}</span>' if label else ""
    st.markdown(
        f'<div style="display:flex; align-items:center; gap:8px;">'
        f'<div style="flex:1; background:#f6f8fa; border-radius:4px; height:10px;">'
        f'<div style="width:{pct}%; background:{color.replace("1e5","4a00").replace("be1","1f883d").replace("c5","9a6700").replace("e9","cf222e")}; '
        f'height:10px; border-radius:4px;"></div>'
        f'</div>'
        f'<span style="font-size:0.8em; font-weight:600; color:#24292f;">{pct}%</span>'
        f'{label_html}'
        f'</div>',
        unsafe_allow_html=True
    )
