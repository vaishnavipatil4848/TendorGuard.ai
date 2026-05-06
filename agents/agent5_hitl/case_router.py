def route_case(case):
    """
    case = {
        "id": str,
        "confidence": float,
        "agreement": bool,
        "evidence_present": bool
    }
    """
    if not case["evidence_present"]:
        return "MISSING_EVIDENCE"

    if case["confidence"] < 0.6:
        return "LOW_CONFIDENCE"

    if not case["agreement"]:
        return "MODEL_DISAGREEMENT"

    return "AUTO_APPROVED"