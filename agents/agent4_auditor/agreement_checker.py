"""
agreement_checker.py
Agent 4 — Auditor Agent
Compares the two LLM verdict dicts from dual_llm_runner.py and
produces a unified agreement result.

Three cases:
  1. Verdicts agree AND reasoning chains align      → HIGH confidence agreement
  2. Verdicts agree BUT reasoning chains diverge   → MODERATE confidence, flag for review
  3. Verdicts disagree                              → DISAGREEMENT, route to HITL

Reasoning chain divergence is detected by comparing key_factors and ambiguities,
not by deep NLP — keeps it fast and auditable.
"""

import logging
from typing import Dict, Any, Tuple, Literal

logger = logging.getLogger(__name__)

AgreementStatus = Literal[
    "AGREEMENT_STRONG",
    "AGREEMENT_WEAK",   # same verdict but divergent reasoning
    "DISAGREEMENT"
]


class AgreementChecker:
    """
    Compares two LLM verdict dicts and determines the agreement status.
    """

    def check(
        self,
        claude_verdict: Dict[str, Any],
        gpt4o_verdict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare two verdict dicts and return an agreement report.

        Args:
            claude_verdict: verdict dict from Claude evaluator
            gpt4o_verdict:  verdict dict from GPT-4o evaluator

        Returns:
            agreement_report dict:
            {
                agreement_status,      # AGREEMENT_STRONG | AGREEMENT_WEAK | DISAGREEMENT
                agreed_verdict,        # the consensus verdict (or None on disagreement)
                reasoning_divergence,  # True if reasoning chains differ
                divergence_details,    # list of specific divergence notes
                weight_adjusted_confidence,  # float
            }
        """
        cid = claude_verdict.get("criterion_id", "?")
        v_claude = claude_verdict.get("verdict", "UNCERTAIN")
        v_gpt4o = gpt4o_verdict.get("verdict", "UNCERTAIN")

        if v_claude != v_gpt4o:
            return self._disagreement_report(
                cid, claude_verdict, gpt4o_verdict
            )

        # same verdict — check reasoning alignment
        divergence, details = self._check_reasoning_divergence(
            claude_verdict, gpt4o_verdict
        )

        status: AgreementStatus = (
            "AGREEMENT_WEAK" if divergence else "AGREEMENT_STRONG"
        )

        # weight model agreement highest in composite score
        # AGREEMENT_STRONG: full average confidence
        # AGREEMENT_WEAK:   penalise by 15%
        avg_conf = (
            claude_verdict.get("confidence", 0.5) +
            gpt4o_verdict.get("confidence", 0.5)
        ) / 2.0
        weight_adjusted = avg_conf * (0.85 if divergence else 1.0)

        logger.info(
            f"[{cid}] Agreement: {status} | verdict={v_claude} | "
            f"conf={weight_adjusted:.2f} | divergence={divergence}"
        )

        return {
            "criterion_id": cid,
            "agreement_status": status,
            "agreed_verdict": v_claude,
            "reasoning_divergence": divergence,
            "divergence_details": details,
            "weight_adjusted_confidence": round(weight_adjusted, 3),
            "claude_confidence": claude_verdict.get("confidence"),
            "gpt4o_confidence": gpt4o_verdict.get("confidence"),
        }

    # ------------------------------------------------------------------ #
    # Private
    # ------------------------------------------------------------------ #

    def _disagreement_report(
        self,
        cid: str,
        claude_verdict: Dict[str, Any],
        gpt4o_verdict: Dict[str, Any]
    ) -> Dict[str, Any]:
        v_claude = claude_verdict.get("verdict")
        v_gpt4o = gpt4o_verdict.get("verdict")

        details = [
            f"Claude: {v_claude} (key_factor={claude_verdict.get('key_factor')!r})",
            f"GPT-4o: {v_gpt4o} (key_factor={gpt4o_verdict.get('key_factor')!r})",
        ]

        # confidence on disagreement is low — average the individual values but cap at 0.5
        avg_conf = (
            claude_verdict.get("confidence", 0.5) +
            gpt4o_verdict.get("confidence", 0.5)
        ) / 2.0
        disagreement_conf = min(avg_conf, 0.50)

        logger.warning(
            f"[{cid}] Model DISAGREEMENT: Claude={v_claude}, GPT-4o={v_gpt4o}"
        )

        return {
            "criterion_id": cid,
            "agreement_status": "DISAGREEMENT",
            "agreed_verdict": None,
            "reasoning_divergence": True,
            "divergence_details": details,
            "weight_adjusted_confidence": round(disagreement_conf, 3),
            "claude_confidence": claude_verdict.get("confidence"),
            "gpt4o_confidence": gpt4o_verdict.get("confidence"),
        }

    def _check_reasoning_divergence(
        self,
        claude_verdict: Dict[str, Any],
        gpt4o_verdict: Dict[str, Any]
    ) -> Tuple[bool, list[str]]:
        """
        Detect if reasoning chains diverge even when verdicts agree.

        Heuristics:
          - key_factors mention different evidence fields
          - ambiguities list is non-empty in one but not the other
          - confidence delta > 0.25 between models
        """
        details = []

        kf_claude = (claude_verdict.get("key_factor") or "").lower()
        kf_gpt4o = (gpt4o_verdict.get("key_factor") or "").lower()

        conf_delta = abs(
            claude_verdict.get("confidence", 0.5) -
            gpt4o_verdict.get("confidence", 0.5)
        )

        # key factors significantly different (no common words > 3 chars)
        def _tokens(s: str) -> set:
            return {w for w in s.split() if len(w) > 3}

        kf_overlap = _tokens(kf_claude) & _tokens(kf_gpt4o)
        if not kf_overlap and kf_claude and kf_gpt4o:
            details.append(
                f"Key factors diverge: Claude={kf_claude!r} vs GPT-4o={kf_gpt4o!r}"
            )

        # ambiguities asymmetry
        amb_claude = set(claude_verdict.get("ambiguities", []))
        amb_gpt4o = set(gpt4o_verdict.get("ambiguities", []))
        if bool(amb_claude) != bool(amb_gpt4o):
            details.append(
                f"Ambiguity asymmetry: Claude={list(amb_claude)}, GPT-4o={list(amb_gpt4o)}"
            )

        # large confidence gap
        if conf_delta > 0.25:
            details.append(
                f"Confidence delta {conf_delta:.2f} exceeds 0.25 threshold"
            )

        diverges = len(details) > 0
        return diverges, details