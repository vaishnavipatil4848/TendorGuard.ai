from .case_router import route_case
from .queue_manager import ReviewQueue
from .auditor_logger import AuditLogger
from .feedback_loop import FeedbackLoop
from .report_generator import ReportGenerator

from agent5_hitl import (
    route_case,
    ReviewQueue,
    AuditLogger,
    FeedbackLoop,
    ReportGenerator
)

# Initialize components
queue = ReviewQueue()
logger = AuditLogger()
feedback = FeedbackLoop()

# Sample cases
cases = [
    {"id": "C1", "confidence": 0.4, "agreement": True, "evidence_present": True},
    {"id": "C2", "confidence": 0.9, "agreement": False, "evidence_present": True},
    {"id": "C3", "confidence": 0.8, "agreement": True, "evidence_present": False},
]

# Routing
for case in cases:
    case_type = route_case(case)
    if case_type != "AUTO_APPROVED":
        queue.add_case(case, case_type)

# Human review simulation
while not queue.is_empty():
    case = queue.get_next()
    
    # Simulated human decision
    human_decision = "PASS" # wait for human loop
    model_decision = "FAIL" # fetch model decision

    logger.log(case["id"], human_decision, "reviewer_1", "Checked manually")
    feedback.update(model_decision, human_decision)

# Report
report = ReportGenerator()
report.print_report()

print("\nFeedback Metrics:", feedback.get_metrics())