import heapq

PRIORITY = {
    "MISSING_EVIDENCE": 1,
    "MODEL_DISAGREEMENT": 2,
    "LOW_CONFIDENCE": 3
}

class ReviewQueue:
    def __init__(self):
        self.queue = []

    def add_case(self, case, case_type):
        priority = PRIORITY.get(case_type, 4)
        heapq.heappush(self.queue, (priority, case))

    def get_next(self):
        if self.queue:
            return heapq.heappop(self.queue)[1]
        return None

    def is_empty(self):
        return len(self.queue) == 0