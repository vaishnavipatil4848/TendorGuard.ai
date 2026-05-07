"""
dual_llm_runner.py
Agent 4 — Auditor Agent
Runs GPT-4o and Claude 3.5 Sonnet in async parallel for Semantic criteria.
Passes both verdict dicts to agreement_checker.py.

Key design decisions (per brainstorm doc):
  - Async parallel: latency = max(GPT-4o, Claude), not sum
  - Two independent model families — genuine signal diversity
  - Divergent reasoning chains are flagged even when verdicts agree
"""

import asyncio
import logging
from typing import Dict, Any, Tuple

from .llm_evaluator import LLMEvaluator

logger = logging.getLogger(__name__)


class DualLLMRunner:
    """
    Asynchronously calls both GPT-4o and Claude 3.5 Sonnet for a single
    criterion+evidence pair and returns both verdict dicts.
    """

    def __init__(self):
        self.claude_evaluator = LLMEvaluator(model_family="claude")
        self.gpt4o_evaluator = LLMEvaluator(model_family="gpt4o")
        logger.info("DualLLMRunner initialised (GPT-4o + Claude)")

    def run(
        self,
        criterion: Dict[str, Any],
        evidence: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Synchronous entry point — runs both models via asyncio.

        Returns:
            (claude_verdict, gpt4o_verdict)
        """
        return asyncio.run(self._run_parallel(criterion, evidence))

    async def _run_parallel(
        self,
        criterion: Dict[str, Any],
        evidence: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Kick off both LLM calls concurrently in the event loop.
        asyncio.gather ensures they run in parallel, not sequentially.
        """
        cid = criterion.get("criterion_id", "?")
        logger.info(f"[{cid}] Dispatching to GPT-4o + Claude in parallel")

        claude_task = asyncio.to_thread(
            self.claude_evaluator.evaluate, criterion, evidence
        )
        gpt4o_task = asyncio.to_thread(
            self.gpt4o_evaluator.evaluate, criterion, evidence
        )

        results = await asyncio.gather(
            claude_task, gpt4o_task,
            return_exceptions=True
        )

        claude_result, gpt4o_result = results

        # handle exceptions from individual models gracefully
        if isinstance(claude_result, Exception):
            logger.error(f"[{cid}] Claude task raised: {claude_result}")
            claude_result = self._fallback_uncertain(criterion, "claude", str(claude_result))

        if isinstance(gpt4o_result, Exception):
            logger.error(f"[{cid}] GPT-4o task raised: {gpt4o_result}")
            gpt4o_result = self._fallback_uncertain(criterion, "gpt-4o", str(gpt4o_result))

        logger.info(
            f"[{cid}] Parallel calls complete — "
            f"Claude: {claude_result.get('verdict')} ({claude_result.get('confidence'):.2f}), "
            f"GPT-4o: {gpt4o_result.get('verdict')} ({gpt4o_result.get('confidence'):.2f})"
        )

        return claude_result, gpt4o_result

    def _fallback_uncertain(
        self,
        criterion: Dict[str, Any],
        model_label: str,
        error_msg: str
    ) -> Dict[str, Any]:
        """Return an UNCERTAIN verdict when one model's call fails."""
        return {
            "criterion_id": criterion.get("criterion_id", "?"),
            "criterion_type": criterion.get("criterion_type", "Semantic"),
            "verdict": "UNCERTAIN",
            "confidence": 0.2,
            "reasoning_steps": [f"Model call failed: {error_msg}"],
            "key_factor": "model_call_failed",
            "ambiguities": [error_msg],
            "model": model_label,
            "programmatic": False,
        }