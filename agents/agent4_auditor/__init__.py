"""
Agent 4 — Auditor Agent
Compares Fact-Checker evidence against the ruleset and produces
per-criterion verdicts (PASS / FAIL / UNCERTAIN) with composite
confidence scores.

Pipeline per criterion:
  1. Classify criterion type (Numerical / Logical / Semantic)
  2. Numerical / Logical → NumericalChecker (programmatic)
  3. Semantic → DualLLMRunner (GPT-4o + Claude async parallel)
  4. AgreementChecker → resolve or flag disagreement
  5. ConfidenceScorer → composite confidence
  6. VerdictAggregator → bidder-level report
"""

import logging
from typing import List, Dict, Any

from .criterion_classifier import CriterionClassifier
from .numerical_checker import NumericalChecker
from .dual_llm_runner import DualLLMRunner
from .agreement_checker import AgreementChecker
from .confidence_scorer import ConfidenceScorer
from .verdict_aggregator import VerdictAggregator

logger = logging.getLogger(__name__)


class AuditorAgent:
    """
    Main entry point for Agent 4.
    Accepts Agent 1 ruleset and Agent 3 evidence, returns final verdict report.
    """

    def __init__(self):
        logger.info("Initialising Auditor Agent")
        self.classifier = CriterionClassifier()
        self.numerical_checker = NumericalChecker()
        self.dual_llm = DualLLMRunner()
        self.agreement_checker = AgreementChecker()
        self.confidence_scorer = ConfidenceScorer()
        self.aggregator = VerdictAggregator()

    def run(
        self,
        bidder_id: str,
        criteria: List[Dict[str, Any]],
        evidence_map: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Full audit pipeline for one bidder.

        Args:
            bidder_id:    unique bidder identifier
            criteria:     list of criterion dicts from Agent 1 ruleset
            evidence_map: dict mapping criterion_id → evidence dict from Agent 3

        Returns:
            aggregated_report (see VerdictAggregator.aggregate)
        """
        logger.info(
            f"Auditor Agent starting for bidder={bidder_id}, "
            f"{len(criteria)} criteria"
        )

        # Step 1 — classify all criteria
        criteria = self.classifier.classify_batch(criteria)

        # Step 2–5 — evaluate each criterion
        criterion_verdicts = []
        for criterion in criteria:
            cid = criterion["criterion_id"]
            evidence = evidence_map.get(cid, {})

            verdict = self._evaluate_criterion(criterion, evidence)
            criterion_verdicts.append(verdict)

        # Step 6 — aggregate
        report = self.aggregator.aggregate(
            bidder_id=bidder_id,
            criterion_verdicts=criterion_verdicts,
            criteria_meta=criteria
        )

        logger.info(
            f"Auditor Agent complete for {bidder_id}: "
            f"overall={report['overall_verdict']} "
            f"(conf={report['overall_confidence']})"
        )
        return report

    def _evaluate_criterion(
        self,
        criterion: Dict[str, Any],
        evidence: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Route a single criterion through the appropriate evaluation path.
        """
        cid = criterion["criterion_id"]
        c_type = criterion["criterion_type"]

        logger.info(f"[{cid}] Evaluating ({c_type})")

        if c_type in ("Numerical", "Logical"):
            # programmatic path
            verdict = self.numerical_checker.evaluate(criterion, evidence)
            scoring = self.confidence_scorer.score(
                evidence=evidence,
                agreement_report={},
                programmatic=True
            )
            verdict.update(scoring)

        else:
            # Semantic path — dual LLM
            claude_v, gpt4o_v = self.dual_llm.run(criterion, evidence)
            agreement = self.agreement_checker.check(claude_v, gpt4o_v)

            # use agreed verdict or fall back to Claude's if disagreement
            final_verdict_str = agreement.get("agreed_verdict") or claude_v.get("verdict", "UNCERTAIN")
            combined_reasoning = self._merge_reasoning(claude_v, gpt4o_v)

            scoring = self.confidence_scorer.score(
                evidence=evidence,
                agreement_report=agreement,
                programmatic=False
            )

            verdict = {
                "criterion_id": cid,
                "criterion_type": c_type,
                "verdict": final_verdict_str,
                "confidence": agreement.get("weight_adjusted_confidence", 0.5),
                "reasoning": combined_reasoning,
                "key_factor": claude_v.get("key_factor", ""),
                "ambiguities": list(set(
                    claude_v.get("ambiguities", []) +
                    gpt4o_v.get("ambiguities", [])
                )),
                "agreement_status": agreement["agreement_status"],
                "divergence_details": agreement.get("divergence_details", []),
                "programmatic": False,
            }
            verdict.update(scoring)

        return verdict

    def _merge_reasoning(
        self,
        claude_v: Dict[str, Any],
        gpt4o_v: Dict[str, Any]
    ) -> str:
        """
        Build a combined reasoning string from both models' chain-of-thought steps.
        """
        claude_steps = claude_v.get("reasoning_steps", [])
        gpt4o_steps = gpt4o_v.get("reasoning_steps", [])

        parts = []
        if claude_steps:
            parts.append(
                "Claude reasoning: " + " | ".join(str(s) for s in claude_steps)
            )
        if gpt4o_steps:
            parts.append(
                "GPT-4o reasoning: " + " | ".join(str(s) for s in gpt4o_steps)
            )
        return "\n".join(parts) if parts else ""