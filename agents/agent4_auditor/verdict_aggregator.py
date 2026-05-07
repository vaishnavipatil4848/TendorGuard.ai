"""
verdict_aggregator.py
Agent 4 — Auditor Agent
Aggregates per-criterion verdicts into a final bidder-level evaluation report.

Rules:
  - Any FAIL on a mandatory criterion → bidder overall = FAIL
  - All PASSes on mandatory criteria, all non-mandatory at PASS/UNCERTAIN → PASS
  - Majority UNCERTAIN with no mandatory FAILs → UNCERTAIN
  - HITL-routed criteria are excluded from auto-aggregation and marked pending
"""

import logging
from typing import Dict, Any, List, Literal

logger = logging.getLogger(__name__)

OverallVerdict = Literal["PASS", "FAIL", "UNCERTAIN", "PENDING_REVIEW"]


class VerdictAggregator:
    """
    Aggregates per-criterion verdict reports into a final bidder-level report.
    """

    def aggregate(
        self,
        bidder_id: str,
        criterion_verdicts: List[Dict[str, Any]],
        criteria_meta: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregate per-criterion verdicts.

        Args:
            bidder_id:          unique bidder identifier
            criterion_verdicts: list of final verdict dicts from the Auditor pipeline
                                Each dict contains:
                                  criterion_id, verdict, confidence,
                                  route_to_hitl, criterion_type, ...
            criteria_meta:      original criteria list from Agent 1 ruleset
                                (used for is_mandatory flag lookup)

        Returns:
            aggregated_report dict:
            {
                bidder_id,
                overall_verdict,       # PASS | FAIL | UNCERTAIN | PENDING_REVIEW
                overall_confidence,    # mean composite confidence across auto-decided criteria
                criteria_summary,      # per-verdict counts
                mandatory_fails,       # list of mandatory criteria that failed
                pending_hitl,          # list of criterion_ids routed to HITL
                criterion_verdicts,    # full per-criterion list
            }
        """
        # build is_mandatory lookup
        mandatory_lookup: Dict[str, bool] = {
            c["criterion_id"]: c.get("is_mandatory", True)
            for c in criteria_meta
        }

        mandatory_fails = []
        pending_hitl = []
        auto_verdicts = []

        pass_count = fail_count = uncertain_count = 0

        for v in criterion_verdicts:
            cid = v.get("criterion_id", "?")
            verdict = v.get("verdict", "UNCERTAIN")
            is_mandatory = mandatory_lookup.get(cid, True)
            route_to_hitl = v.get("route_to_hitl", False)

            if route_to_hitl:
                pending_hitl.append(cid)
                continue  # exclude from auto-aggregation

            auto_verdicts.append(v)

            if verdict == "PASS":
                pass_count += 1
            elif verdict == "FAIL":
                fail_count += 1
                if is_mandatory:
                    mandatory_fails.append({
                        "criterion_id": cid,
                        "requirement_text": v.get("reasoning", "")[:120],
                        "confidence": v.get("composite_confidence",
                                            v.get("confidence", 0.0))
                    })
            else:  # UNCERTAIN
                uncertain_count += 1

        # determine overall verdict
        overall_verdict = self._determine_overall(
            mandatory_fails, fail_count, uncertain_count,
            len(auto_verdicts), pending_hitl
        )

        # mean confidence over auto-decided criteria
        if auto_verdicts:
            overall_confidence = round(
                sum(
                    v.get("composite_confidence", v.get("confidence", 0.5))
                    for v in auto_verdicts
                ) / len(auto_verdicts),
                3
            )
        else:
            overall_confidence = 0.0

        logger.info(
            f"[{bidder_id}] Aggregated verdict: {overall_verdict} "
            f"(conf={overall_confidence}) | "
            f"PASS={pass_count}, FAIL={fail_count}, "
            f"UNCERTAIN={uncertain_count}, HITL={len(pending_hitl)}"
        )

        return {
            "bidder_id": bidder_id,
            "overall_verdict": overall_verdict,
            "overall_confidence": overall_confidence,
            "criteria_summary": {
                "total": len(criterion_verdicts),
                "auto_decided": len(auto_verdicts),
                "pass": pass_count,
                "fail": fail_count,
                "uncertain": uncertain_count,
                "pending_hitl": len(pending_hitl),
            },
            "mandatory_fails": mandatory_fails,
            "pending_hitl": pending_hitl,
            "criterion_verdicts": criterion_verdicts,
        }

    # ------------------------------------------------------------------ #
    # Private
    # ------------------------------------------------------------------ #

    def _determine_overall(
        self,
        mandatory_fails: List,
        total_fails: int,
        uncertain_count: int,
        total_auto: int,
        pending_hitl: List
    ) -> OverallVerdict:
        # any mandatory fail → hard FAIL
        if mandatory_fails:
            return "FAIL"

        # pending human review on any criterion → defer
        if pending_hitl:
            return "PENDING_REVIEW"

        if total_auto == 0:
            return "UNCERTAIN"

        # majority uncertain → uncertain
        if uncertain_count > total_auto / 2:
            return "UNCERTAIN"

        # non-mandatory fails only → still PASS (flag them but don't block)
        if total_fails == 0 or (total_fails > 0 and not mandatory_fails):
            if uncertain_count == 0:
                return "PASS"
            return "UNCERTAIN"

        return "FAIL"