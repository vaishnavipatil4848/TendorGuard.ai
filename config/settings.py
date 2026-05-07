"""
settings.py
config/ — TendorGuard.ai

Shared types, dataclasses, enums and runtime configuration
for all agents in the TendorGuard pipeline.

Merged from: Tendor_Eval_Project_AI_Bharat/tender_eval/config.py
"""

import os
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()

# ── LLM Provider Keys ─────────────────────────────────────────────────────────
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY", "")   # used by Agent 1 Vision + Agent 2

# ── Model Selections ──────────────────────────────────────────────────────────
GEMINI_FLASH_MODEL  = "gemini-2.0-flash"       # Agent 1 OCR + Agent 2 criteria
OPENAI_CHAT_MODEL   = "gpt-4o"                 # Agent 4 primary LLM
ANTHROPIC_MODEL     = "claude-3-5-sonnet-20241022"  # Agent 4 secondary LLM
OPENAI_EMBED_MODEL  = "text-embedding-3-large"  # Agent 3 dense retrieval

# ── Confidence Thresholds ─────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD_LOW  = 0.75   # below → NEEDS_MANUAL_REVIEW
CONFIDENCE_THRESHOLD_HIGH = 0.90   # above → trusted auto-verdict

# ── Storage Paths ─────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = "./database/chroma_store"
SAMPLE_DATA_DIR    = "./storage/sample"

# ═══════════════════════════════════════════════════════════════════════════════
# Shared Enums and Dataclasses
# ═══════════════════════════════════════════════════════════════════════════════

class Verdict(Enum):
    ELIGIBLE            = "ELIGIBLE"
    INELIGIBLE          = "INELIGIBLE"
    NEEDS_MANUAL_REVIEW = "NEEDS_MANUAL_REVIEW"


@dataclass
class Criterion:
    """Structured representation of a single eligibility criterion."""
    id:                    str
    name:                  str
    description:           str
    criterion_type:        str                   # "MANDATORY" | "OPTIONAL"
    threshold_value:       Optional[str] = None
    unit:                  Optional[str] = None
    time_period:           Optional[str] = None
    document_required:     Optional[str] = None
    original_text:         Optional[str] = None
    legal_clause_reference: Optional[str] = None


@dataclass
class Evidence:
    """Evidence extracted by Agent 3 for a specific criterion."""
    found:               bool
    page_number:         Optional[int]       = None
    line_number:         Optional[int]       = None
    matching_line_text:  Optional[str]       = None
    context_lines:       Optional[List[str]] = None
    extracted_value:     Optional[str]       = None
    extracted_numeric:   Optional[float]     = None
    threshold_numeric:   Optional[float]     = None
    meets_threshold:     Optional[bool]      = None
    source_document:     Optional[str]       = None
    page_reference:      Optional[str]       = None
    raw_text_quote:      Optional[str]       = None
    confidence:          float               = 0.0
    ambiguity_reason:    Optional[str]       = None


@dataclass
class CriterionResult:
    """Outcome of evaluating one criterion for one bidder."""
    criterion_id: str
    verdict:      str            # stored as string for JSON/Streamlit compatibility
    evidence:     Evidence
    reasoning:    str
    flag_reason:  Optional[str] = None


@dataclass
class BidderResult:
    """Aggregate result for a single bidder across all criteria."""
    bidder_name:      str
    overall_verdict:  str
    criteria_results: List[CriterionResult]
    summary:          Optional[str] = None


@dataclass
class AgentMessage:
    """Inter-agent message envelope used by the message bus."""
    sender:   str
    receiver: str
    msg_type: str
    payload:  Dict[str, Any]
