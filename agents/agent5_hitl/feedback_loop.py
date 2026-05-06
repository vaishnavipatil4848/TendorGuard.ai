class FeedbackLoop:
    def __init__(self):
        self.stats = {
            "model_correct": 0,
            "human_override": 0
        }

    def update(self, model_decision, human_decision):
        if model_decision == human_decision:
            self.stats["model_correct"] += 1
        else:
            self.stats["human_override"] += 1

    def get_metrics(self):
        total = sum(self.stats.values())
        if total == 0:
            return self.stats

        return {
            "accuracy": self.stats["model_correct"] / total,
            "overrides": self.stats["human_override"]
        }