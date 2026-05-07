"""
app.py
ui/ — TendorGuard.ai

Main entry point for the TendorGuard.ai Streamlit dashboard.
Coordinates navigation between pages and manages the global
TenderEvalOrchestrator instance.
"""

import streamlit as st
import os
from pathlib import Path
from typing import List, Dict, Any

# Import Pipeline components
from pipeline.orchestrator import TenderEvalOrchestrator
from agents.agent1_architect.layout_parser import LayoutParser

# Import UI Pages
from ui.pages import leaderboard, criteria_dashboard, evidence_viewer, review_queue

# Configuration
st.set_page_config(
    page_title="TendorGuard.ai | Intelligent Tender Evaluation",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background-color: #f8fafc;
    }

    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
        padding: 0.5rem 1rem;
    }

    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    }

    .sidebar .sidebar-content {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }

    .stProgress > div > div > div > div {
        background-color: #3b82f6;
    }

    /* Glassmorphism card effect */
    .css-1r6slb0, .stExpander {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05);
    }
</style>
""", unsafe_allow_html=True)

# ── Session State Initialization ──────────────────────────────────────────────

if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = TenderEvalOrchestrator()

if "criteria" not in st.session_state:
    st.session_state.criteria = []

if "bidder_reports" not in st.session_state:
    st.session_state.bidder_reports = []

if "evidence_maps" not in st.session_state:
    st.session_state.evidence_maps = {}

if "tender_id" not in st.session_state:
    st.session_state.tender_id = ""

if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False

if "layout_parser" not in st.session_state:
    with st.spinner("Initializing AI Layout Engine..."):
        from agents.agent1_architect.layout_parser import LayoutParser
        st.session_state.layout_parser = LayoutParser()

# ── Storage Directories ───────────────────────────────────────────────────────

STORAGE_TENDER = Path("storage/tender_docs")
STORAGE_BIDDER = Path("storage/bidder_docs")
STORAGE_TENDER.mkdir(parents=True, exist_ok=True)
STORAGE_BIDDER.mkdir(parents=True, exist_ok=True)

# ── Sidebar Navigation ───────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=80)
    st.title("TendorGuard.ai")
    st.caption("v1.0.0 | Enterprise Edition")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠 Dashboard & Upload", "🏆 Bidder Leaderboard", "📋 Criteria Matrix", "🔎 Evidence Deep-dive", "👤 Manual Review Queue"]
    )

    st.markdown("---")
    if st.session_state.tender_id:
        st.info(f"**Active Tender:** {st.session_state.tender_id}")

    if st.button("🔄 Clear All State"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ── Page Routing ─────────────────────────────────────────────────────────────

if page == "🏠 Dashboard & Upload":
    st.title("🚀 Tender Processing Dashboard")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        Welcome to **TendorGuard.ai**. Upload your tender documents and bidder proposals to begin the automated compliance evaluation.
        """)

        # 1. Tender Upload
        st.subheader("Step 1: Ingest Tender")
        t_id = st.text_input("Tender ID", value=st.session_state.tender_id, placeholder="e.g. T-2024-001")
        tender_file = st.file_uploader("Upload Tender PDF", type="pdf")

        if tender_file and t_id:
            st.session_state.tender_id = t_id
            save_path = STORAGE_TENDER / tender_file.name
            with open(save_path, "wb") as f:
                f.write(tender_file.getbuffer())

            if st.button("Analyze Tender & Extract Criteria"):
                with st.status("Agent 1: Extracting Criteria...", expanded=True) as status:
                    try:
                        criteria = st.session_state.orchestrator.run_stage1_tender(str(save_path), t_id)
                        st.session_state.criteria = criteria
                        st.session_state.step1_done = True
                        status.update(label=f"✅ {len(criteria)} Criteria Extracted", state="complete")
                        st.success(f"Tender analysis complete. Found {len(criteria)} eligibility criteria.")
                        st.rerun()
                    except Exception as e:
                        status.update(label="❌ Analysis Failed", state="error")
                        st.error(f"Error: {e}")

        st.markdown("---")

        # 2. Bidder Upload
        st.subheader("Step 2: Evaluate Bidders")
        
        # Check if Step 1 has been attempted (criteria is initialized as [] but we set it after run)
        if "criteria" not in st.session_state or (not st.session_state.criteria and not st.session_state.get("step1_done", False)):
            st.warning("Please complete Step 1 (Tender Ingestion) first.")
        else:
            if not st.session_state.criteria:
                st.warning("⚠️ No criteria were extracted from the tender. You can still upload bidders, but compliance checks will be limited.")
            
            bidder_files = st.file_uploader("Upload Bidder Proposals (Multiple)", type="pdf", accept_multiple_files=True)

            if bidder_files:
                if st.button("Run Full Evaluation Pipeline"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    def update_progress(step, total, msg):
                        progress_bar.progress(step / total)
                        status_text.markdown(f"**Current Task:** {msg}")

                    st.session_state.orchestrator._cb = update_progress

                    all_bidder_data = []
                    
                    # 2.1 Parse Bidder PDFs with Progress
                    for idx, bf in enumerate(bidder_files, 1):
                        status_text.markdown(f"**Step 2/5:** Parsing Bidder PDF ({idx}/{len(bidder_files)}): `{bf.name}`...")
                        progress_bar.progress((idx / len(bidder_files)) * 0.1) # First 10% for parsing
                        
                        b_path = STORAGE_BIDDER / bf.name
                        with open(b_path, "wb") as f:
                            f.write(bf.getbuffer())

                        # Use cached parser
                        regions = st.session_state.layout_parser.parse_pdf(str(b_path))
                        pages = {}
                        raw_text = ""
                        for r in regions:
                            pn = r.get("page_number", 1)
                            pages.setdefault(pn, []).append({
                                "text": r.get("text", ""),
                                "bbox": r.get("bbox", [])
                            })
                            raw_text += r.get("text", "") + " "

                        all_bidder_data.append({
                            "filename": bf.name,
                            "bidder_id": b_path.stem,
                            "pages": pages,
                            "raw_text": raw_text,
                            "doc_type": "PDF"
                        })

                        # Run Pipeline
                        st.session_state.bidder_reports = []
                        st.session_state.evidence_maps = {}

                        # Stages 2-4 are now orchestrated with granular progress
                        # We use the internal run_bidders logic for consistency
                        st.session_state.orchestrator.run_bidders(all_bidder_data)

                        # Finalize
                        st.session_state.orchestrator.generate_final_report()
                        st.session_state.processing_complete = True
                        st.success("🎉 Evaluation complete! Head over to the Leaderboard to see results.")

    with col2:
        st.subheader("System Status")
        if st.session_state.criteria:
            st.metric("Criteria Extracted", len(st.session_state.criteria))
        if st.session_state.bidder_reports:
            st.metric("Bidders Evaluated", len(st.session_state.bidder_reports))

        st.markdown("---")
        st.markdown("**Agent Activity Log**")
        for log in st.session_state.orchestrator.all_logs[-10:]:
            st.caption(log)

elif page == "🏆 Bidder Leaderboard":
    leaderboard.render(st.session_state.bidder_reports)

elif page == "📋 Criteria Matrix":
    criteria_dashboard.render(st.session_state.criteria, st.session_state.bidder_reports)

elif page == "🔎 Evidence Deep-dive":
    evidence_viewer.render(
        st.session_state.bidder_reports,
        st.session_state.evidence_maps,
        st.session_state.criteria
    )

elif page == "👤 Manual Review Queue":
    # Pass the HITL agent instance from the orchestrator
    review_queue.render(st.session_state.orchestrator.agent5)

# ── Footer ───────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #64748b; font-size: 0.8rem;">'
    '© 2026 TendorGuard.ai - Powered by Multi-Agent RAG Architecture'
    '</div>',
    unsafe_allow_html=True
)
