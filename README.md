# TendorGuard AI 🛡️
### High-Fidelity Automated Eligibility & Audit System for Government Tenders

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Theme](https://img.shields.io/badge/Theme-AI%20Tender%20Evaluation-orange.svg)]()

---

## 👥 Team Zeta

**Hackathon Theme:** Theme 3 — AI-Based Tender Evaluation and Eligibility Analysis
<br>
**Members:**
* Vaishnavi Patil
* Harshita Agrawal
* Jivika Dixit
---

## 📌 Problem Statement

Manual evaluation of government tenders (like those issued by CRPF) is a high-stakes bottleneck:

- **Complexity** — Tenders contain dense technical, financial, and compliance criteria spread across hundreds of pages
- **Format Variance** — Bidders submit heterogeneous documents: clean PDFs, scanned blurry certificates, and photos of physical documents
- **Audit Risk** — Manual rejection is often subjective; without a clear evidence trail, the procurement process faces legal delays and lack of transparency

---

## 💡 Solution

TendorGuard AI is a **Collaborative 5-Agent AI System** that automates government tender eligibility evaluation end-to-end. Each agent is a specialist, communicating via a typed **Agent Message Bus** to ensure reliability, explainability, and full auditability.

### Key Differentiators

- **Source Citation Mapper** — Every decision links back to a specific page and bounding box in the bidder's document. Zero guesswork, full traceability
- **Confidence Heatmap** — Every piece of evidence has a 0–100% confidence score, ensuring human officers spend 90% of their time on the 10% of documents that actually need attention
- **Dual-LLM Cross-Check** — GPT-4o and Claude 3.5 Sonnet independently verify each verdict. If they disagree, it goes to human review — no single model can wrongly eliminate a bidder
- **Multilingual OCR** — Native support for Hindi + English documents including stamps, certificates, and handwritten notes
- **Immutable Audit Trail** — Every decision, override, and human sign-off is permanently logged for legal defensibility

---

## 🏗️ System Architecture

```
Tender PDF + Bidder Documents
          ↓
┌─────────────────────────────────────────────────────┐
│                  Agent Message Bus                   │
│         (typed AgentMessage objects)                 │
└─────────────────────────────────────────────────────┘
          ↓
┌─────────────────┐
│  Agent 1        │  LayoutLMv3 → LLM (Claude)
│  Architect      │  Extracts eligibility ruleset JSON
└────────┬────────┘
         ↓
┌─────────────────┐
│  Agent 2        │  OpenCV → CLIP → RT-DETR → Claude Vision
│  Vision         │  Processes scanned/bilingual documents
└────────┬────────┘
         ↓
┌─────────────────┐
│  Agent 3        │  ChromaDB + BM25 → RRF → bge-reranker-large
│  Fact-Checker   │  Retrieves evidence per criterion
└────────┬────────┘
         ↓
┌─────────────────┐
│  Agent 4        │  GPT-4o + Claude 3.5 Sonnet (async parallel)
│  Auditor        │  Dual-LLM verdict + composite confidence score
└────────┬────────┘
         ↓
┌─────────────────┐
│  Agent 5        │  Streamlit UI + PostgreSQL audit log
│  HITL           │  Human review queue + signed PDF report
└─────────────────┘
          ↓
    Leaderboard + Audit Report
```

---

## 🤖 Agent Details

### Agent 1 — Architect Agent
Parses the tender document and extracts a structured eligibility ruleset.

| Component | Tool |
|---|---|
| Document layout parsing | LayoutLMv3 / DocLayNet |
| Section classification | Keyword-based classifier (eligibility / financial / technical / compliance) |
| Criteria extraction | Claude (schema-driven structured output) |
| Self-critique pass | Two-pass LLM extraction for missed cross-references |
| Validation | Completeness check + human sign-off before pipeline runs |

**Output:** Versioned `ruleset_{tender_id}.json` — the ground truth for all downstream agents

---

### Agent 2 — Vision Specialist Agent
Handles scanned, blurry, and bilingual documents.

| Component | Tool |
|---|---|
| Preprocessing | OpenCV (deskew, CLAHE, DPI normalization) |
| Document classification | CLIP zero-shot (GST cert / turnover cert / experience letter etc.) |
| Text detection | RT-DETR (template-guided, per-document confidence thresholds) |
| Low confidence fallback | Claude Vision (crop + context + document type) |
| Bbox storage | Persistent store powering the Source Citation Mapper |

**Output:** Structured extractions with bbox references for every detected field

---

### Agent 3 — Fact-Checker Agent
Retrieves supporting evidence from bidder documents for each criterion.

| Component | Tool |
|---|---|
| Chunking | Hierarchical (document-level + field-level) |
| Dense retrieval | ChromaDB (`text-embedding-3-large`) |
| Sparse retrieval | BM25 (`rank_bm25`) |
| Fusion | Reciprocal Rank Fusion (RRF) |
| Reranking | `bge-reranker-large` cross-encoder |
| Structured extraction | spaCy + regex for numbers/dates; LLM for qualitative fields |
| Metadata filtering | Powered by CLIP document type from Agent 2 |

**Output:** Per-criterion evidence with `{text, value, page, bbox, confidence}`

---

### Agent 4 — Auditor Agent
Compares evidence against ruleset and generates verdicts.

| Component | Tool |
|---|---|
| Criterion classification | Numerical / Logical / Semantic |
| Numerical checks | Programmatic (no LLM arithmetic) |
| Verdict generation | GPT-4o + Claude 3.5 Sonnet (async parallel, CoT prompting) |
| Cross-check | Dual-LLM agreement check — disagreement → HITL |
| Confidence scoring | Composite: OCR + retrieval + LLM + model agreement + extraction type |
| Aggregation | Any UNCERTAIN → REVIEW, any FAIL → INELIGIBLE |

**Output:** Per-criterion verdicts with reasoning chains + overall bidder verdict

---

### Agent 5 — HITL Agent
Routes low-confidence cases to human reviewers with full context.

| Component | Tool |
|---|---|
| Case routing | By type: low OCR / model disagreement / missing evidence / ambiguous |
| Review UI | Streamlit Evidence Overlay with bbox-highlighted source document |
| Audit log | PostgreSQL immutable records (no edits, only superseding entries) |
| Report generation | ReportLab signed PDF audit report |
| Feedback loop | Human decisions → model accuracy tracking → confidence weight adjustment |

**Output:** Final verified verdicts + immutable audit trail + signed PDF report

---

## 🗂️ Project Structure

```
tendorguard-ai/
│
├── README.md
├── requirements.txt
├── .env.example
├── docker-compose.yml
│
├── config/
│   ├── settings.py
│   ├── confidence_thresholds.json
│   └── ruleset_schema.json
│
├── agents/
│   ├── agent1_architect/
│   │   ├── layout_parser.py
│   │   ├── section_classifier.py
│   │   ├── criteria_extractor.py
│   │   ├── two_pass_validator.py
│   │   ├── ruleset_validator.py
│   │   └── prompts/
│   │       ├── extraction_prompt.txt
│   │       └── self_critique_prompt.txt
│   │
│   ├── agent2_vision/
│   │   ├── preprocessor.py
│   │   ├── document_classifier.py
│   │   ├── text_detector.py
│   │   ├── vision_fallback.py
│   │   ├── confidence_router.py
│   │   ├── bbox_store.py
│   │   └── templates/
│   │
│   ├── agent3_factchecker/
│   │   ├── chunker.py
│   │   ├── embedder.py
│   │   ├── dense_retriever.py
│   │   ├── sparse_retriever.py
│   │   ├── rrf_fusion.py
│   │   ├── reranker.py
│   │   ├── metadata_filter.py
│   │   └── evidence_extractor.py
│   │
│   ├── agent4_auditor/
│   │   ├── criterion_classifier.py
│   │   ├── numerical_checker.py
│   │   ├── llm_evaluator.py
│   │   ├── dual_llm_runner.py
│   │   ├── agreement_checker.py
│   │   ├── confidence_scorer.py
│   │   └── verdict_aggregator.py
│   │
│   └── agent5_hitl/
│       ├── case_router.py
│       ├── queue_manager.py
│       ├── audit_logger.py
│       ├── feedback_loop.py
│       └── report_generator.py
│
├── pipeline/
│   ├── orchestrator.py
│   ├── message_bus.py
│   ├── pipeline_runner.py
│   └── error_handler.py
│
├── ui/
│   ├── app.py
│   ├── pages/
│   │   ├── leaderboard.py
│   │   ├── evidence_viewer.py
│   │   ├── review_queue.py
│   │   └── criteria_dashboard.py
│   └── components/
│       ├── verdict_badge.py
│       ├── confidence_heatmap.py
│       └── reasoning_panel.py
│
├── database/
│   ├── models.py
│   └── migrations/
│
├── storage/
│   ├── tender_docs/
│   ├── bidder_docs/
│   ├── audit_reports/
│   └── rulesets/
│
├── tests/
│   ├── test_agent1_architect.py
│   ├── test_agent2_vision.py
│   ├── test_agent3_factchecker.py
│   ├── test_agent4_auditor.py
│   ├── test_agent5_hitl.py
│   ├── test_pipeline.py
│   └── gold_standard/
│
└── scripts/
    ├── ingest_tender.py
    ├── ingest_bidder.py
    ├── run_pipeline.py
    └── export_audit_report.py
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10+
- Git
- Tesseract OCR installed on system ([Windows guide](https://github.com/UB-Mannheim/tesseract/wiki))
- PostgreSQL (optional for local dev — SQLite fallback available)

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/tendorguard-ai.git
cd tendorguard-ai
```

### 2. Create Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4. Configure Environment Variables
```bash
# copy the example env file
copy .env.example .env       # Windows
cp .env.example .env         # macOS/Linux

# fill in your API keys in .env
```

`.env` file:
```env
ANTHROPIC_API_KEY=your_anthropic_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
POSTGRES_URL=postgresql://localhost:5432/tendorguard
CHROMA_PERSIST_DIR=./database/chroma_store
DEFAULT_CONFIDENCE_THRESHOLD=0.85
HITL_CONFIDENCE_THRESHOLD=0.70
PIPELINE_VERSION=v1.0
```

### 5. Run the App
```bash
streamlit run ui/app.py
```

---

## 🚀 Usage

### Ingest a Tender Document
```bash
python scripts/ingest_tender.py --file storage/tender_docs/crpf_tender.pdf --tender_id T001
```

### Ingest Bidder Documents
```bash
python scripts/ingest_bidder.py --bidder_id B001 --folder storage/bidder_docs/B001/
```

### Run Full Evaluation Pipeline
```bash
python scripts/run_pipeline.py --tender_id T001
```

### Export Signed Audit Report
```bash
python scripts/export_audit_report.py --tender_id T001 --output storage/audit_reports/
```

### Run Tests
```bash
pytest tests/
```

---

## 🔄 Pipeline Flow

```
1. Tender PDF ingested
        ↓
2. Agent 1 parses tender → Ruleset JSON
   (human sign-off required before evaluation starts)
        ↓
3. Bidder documents ingested
        ↓
4. Agent 2 processes each document
   OpenCV → CLIP → RT-DETR → Claude Vision (low confidence only)
        ↓
5. Agent 3 retrieves evidence per criterion
   ChromaDB + BM25 → RRF → bge-reranker-large
        ↓
6. Agent 4 generates verdicts
   Programmatic (numerical) + GPT-4o || Claude (async parallel)
        ↓
7. Low confidence / disagreement → Agent 5 HITL queue
        ↓
8. Final verdicts aggregated → Leaderboard updated
        ↓
9. Signed PDF audit report exported
```

---

## 📊 Result Matrix

| View | Description |
|---|---|
| **Leaderboard** | All bidders with status: Eligible / Ineligible / Review |
| **Evidence Overlay** | Extracted text side-by-side with highlighted source document |
| **Criteria Dashboard** | Per-criterion pass/fail breakdown across all bidders |
| **Review Queue** | HITL cases prioritized by type and urgency |
| **Audit Report** | Signed PDF with full decision trail for compliance teams |

---

## 🎯 Target Users

| User | Benefit |
|---|---|
| **Procurement Officers** | Reduce review time from days to minutes |
| **Compliance / Audit Teams** | Verify every automated decision followed exact tender rules |
| **Department Heads** | Bird's-eye view of all bidders via the Bidder Leaderboard |

---


## 🛠️ Tech Stack

| Layer | Technologies |
|---|---|
| Document Parsing | LayoutLMv3, DocLayNet, PyMuPDF |
| Vision & OCR | OpenCV, CLIP, RT-DETR, Claude Vision |
| Retrieval | ChromaDB, BM25 (rank_bm25), bge-reranker-large |
| AI Evaluation | GPT-4o, Claude 3.5 Sonnet, spaCy |
| Persistence | PostgreSQL, ChromaDB, S3/MinIO |
| Pipeline | FastAPI, Celery, Redis |
| UI | Streamlit |
| Reporting | ReportLab |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## ⚠️ Disclaimer

TendorGuard AI is designed to **assist** procurement officers — not replace them. All automated decisions are subject to human review. The system is built with a Human-in-the-Loop architecture to ensure no bidder is disqualified without human oversight on low-confidence cases.
