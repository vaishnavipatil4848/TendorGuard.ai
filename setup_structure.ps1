# run this from your repo root
# Right click setup_structure.ps1 -> Run with PowerShell
# OR run in PowerShell: .\setup_structure.ps1

# create all folders
$folders = @(
    "config",
    "agents/agent1_architect/prompts",
    "agents/agent2_vision/templates",
    "agents/agent3_factchecker/prompts",
    "agents/agent4_auditor/prompts",
    "agents/agent5_hitl",
    "pipeline",
    "ui/pages",
    "ui/components",
    "database/migrations",
    "database/chroma_store",
    "storage/tender_docs",
    "storage/bidder_docs",
    "storage/audit_reports",
    "storage/rulesets",
    "tests/gold_standard/bidder_samples",
    "scripts"
)

foreach ($folder in $folders) {
    New-Item -ItemType Directory -Force -Path $folder | Out-Null
}

# create all files
$files = @(
    # config
    "config/settings.py",
    "config/confidence_thresholds.json",
    "config/ruleset_schema.json",

    # agent 1
    "agents/agent1_architect/__init__.py",
    "agents/agent1_architect/layout_parser.py",
    "agents/agent1_architect/section_classifier.py",
    "agents/agent1_architect/criteria_extractor.py",
    "agents/agent1_architect/two_pass_validator.py",
    "agents/agent1_architect/ruleset_validator.py",
    "agents/agent1_architect/prompts/extraction_prompt.txt",
    "agents/agent1_architect/prompts/self_critique_prompt.txt",

    # agent 2
    "agents/agent2_vision/__init__.py",
    "agents/agent2_vision/preprocessor.py",
    "agents/agent2_vision/document_classifier.py",
    "agents/agent2_vision/text_detector.py",
    "agents/agent2_vision/vision_fallback.py",
    "agents/agent2_vision/confidence_router.py",
    "agents/agent2_vision/bbox_store.py",
    "agents/agent2_vision/templates/gst_certificate.json",
    "agents/agent2_vision/templates/turnover_certificate.json",
    "agents/agent2_vision/templates/experience_letter.json",
    "agents/agent2_vision/templates/bank_statement.json",

    # agent 3
    "agents/agent3_factchecker/__init__.py",
    "agents/agent3_factchecker/chunker.py",
    "agents/agent3_factchecker/embedder.py",
    "agents/agent3_factchecker/dense_retriever.py",
    "agents/agent3_factchecker/sparse_retriever.py",
    "agents/agent3_factchecker/rrf_fusion.py",
    "agents/agent3_factchecker/reranker.py",
    "agents/agent3_factchecker/metadata_filter.py",
    "agents/agent3_factchecker/evidence_extractor.py",
    "agents/agent3_factchecker/prompts/qualitative_extraction_prompt.txt",

    # agent 4
    "agents/agent4_auditor/__init__.py",
    "agents/agent4_auditor/criterion_classifier.py",
    "agents/agent4_auditor/numerical_checker.py",
    "agents/agent4_auditor/llm_evaluator.py",
    "agents/agent4_auditor/dual_llm_runner.py",
    "agents/agent4_auditor/agreement_checker.py",
    "agents/agent4_auditor/confidence_scorer.py",
    "agents/agent4_auditor/verdict_aggregator.py",
    "agents/agent4_auditor/prompts/gpt4o_cot_prompt.txt",
    "agents/agent4_auditor/prompts/claude_cot_prompt.txt",

    # agent 5
    "agents/agent5_hitl/__init__.py",
    "agents/agent5_hitl/case_router.py",
    "agents/agent5_hitl/queue_manager.py",
    "agents/agent5_hitl/audit_logger.py",
    "agents/agent5_hitl/feedback_loop.py",
    "agents/agent5_hitl/report_generator.py",

    # pipeline
    "pipeline/__init__.py",
    "pipeline/orchestrator.py",
    "pipeline/message_bus.py",
    "pipeline/pipeline_runner.py",
    "pipeline/error_handler.py",

    # ui
    "ui/app.py",
    "ui/pages/leaderboard.py",
    "ui/pages/evidence_viewer.py",
    "ui/pages/review_queue.py",
    "ui/pages/criteria_dashboard.py",
    "ui/components/verdict_badge.py",
    "ui/components/confidence_heatmap.py",
    "ui/components/reasoning_panel.py",

    # database
    "database/models.py",
    "database/migrations/001_initial.sql",

    # tests
    "tests/test_agent1_architect.py",
    "tests/test_agent2_vision.py",
    "tests/test_agent3_factchecker.py",
    "tests/test_agent4_auditor.py",
    "tests/test_agent5_hitl.py",
    "tests/test_pipeline.py",
    "tests/gold_standard/expected_verdicts.json",

    # scripts
    "scripts/ingest_tender.py",
    "scripts/ingest_bidder.py",
    "scripts/run_pipeline.py",
    "scripts/export_audit_report.py",

    # gitkeep for empty folders
    "storage/tender_docs/.gitkeep",
    "storage/bidder_docs/.gitkeep",
    "storage/audit_reports/.gitkeep",
    "storage/rulesets/.gitkeep",
    "database/chroma_store/.gitkeep",
    "tests/gold_standard/bidder_samples/.gitkeep",

    # root files
    "requirements.txt",
    "docker-compose.yml",
    ".env",
    ".env.example",
    "README.md"
)

foreach ($file in $files) {
    New-Item -ItemType File -Force -Path $file | Out-Null
}

Write-Host "TendorGuard AI folder structure created successfully" -ForegroundColor Green