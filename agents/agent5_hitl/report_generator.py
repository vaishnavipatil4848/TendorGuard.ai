import json

class ReportGenerator:
    def __init__(self, log_file="audit_log.jsonl"):
        self.log_file = log_file

    def generate_summary(self):
        decisions = {}

        with open(self.log_file, "r") as f:
            for line in f:
                record = json.loads(line)
                decision = record["decision"]
                decisions[decision] = decisions.get(decision, 0) + 1

        return decisions

    def print_report(self):
        summary = self.generate_summary()
        print("=== AUDIT REPORT ===")
        for k, v in summary.items():
            print(f"{k}: {v}")