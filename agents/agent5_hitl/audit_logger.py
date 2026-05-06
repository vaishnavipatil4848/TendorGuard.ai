import json
from datetime import datetime

class AuditLogger:
    def __init__(self, filepath="audit_log.jsonl"):
        self.filepath = filepath

    def log(self, case_id, decision, reviewer, comment):
        record = {
            "case_id": case_id,
            "decision": decision,
            "reviewer": reviewer,
            "comment": comment,
            "timestamp": datetime.utcnow().isoformat()
        }

        with open(self.filepath, "a") as f:
            f.write(json.dumps(record) + "\n")