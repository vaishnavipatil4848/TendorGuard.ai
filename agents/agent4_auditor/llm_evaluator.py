"""
llm_evaluator.py
Agent 4 — Auditor Agent
Single-model LLM evaluator using chain-of-thought structured prompting.
Used for Semantic criteria and as a fallback for ambiguous Numerical/Logical cases.

Each model (GPT-4o, Claude 3.5 Sonnet) has its own evaluator instance.
The dual_llm_runner.py calls both in async parallel and passes results
to agreement_checker.py.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional, Literal

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"

ModelFamily = Literal["claude", "gpt4o"]

# Expected JSON structure from LLM
_VERDICT_SCHEMA = {
    "reasoning_steps": ["step 1 ...", "step 2 ..."],
    "verdict": "PASS | FAIL | UNCERTAIN",
    "confidence": 0.85,
    "key_factor": "the single most decisive evidence point",
    "ambiguities": ["any unresolved points"]
}


class LLMEvaluator:
    """
    Evaluates a single criterion+evidence pair using one LLM
    via structured chain-of-thought prompting.

    Instantiate one per model family.
    """

    def __init__(self, model_family: ModelFamily):
        self.model_family = model_family
        self.prompt_template = self._load_prompt()

        if model_family == "claude":
            import anthropic
            self.client = anthropic.Anthropic()
            self.model_id = "claude-sonnet-4-6"
        else:
            import openai
            self.client = openai.OpenAI()
            self.model_id = "gpt-4o"

        logger.info(f"LLMEvaluator initialised: {self.model_id}")

    def evaluate(
        self,
        criterion: Dict[str, Any],
        evidence: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run chain-of-thought evaluation for one criterion+evidence pair.

        Args:
            criterion: criterion dict from Agent 1 ruleset
            evidence:  evidence dict from Agent 3 FactChecker

        Returns:
            verdict dict:
            {
                criterion_id, criterion_type,
                verdict,           # PASS | FAIL | UNCERTAIN
                confidence,        # 0.0–1.0
                reasoning_steps,   # list of CoT steps
                key_factor,
                ambiguities,
                model,             # which model produced this
                programmatic,      # False
            }
        """
        cid = criterion.get("criterion_id", "?")
        prompt = self._build_prompt(criterion, evidence)

        try:
            raw = self._call_llm(prompt)
            parsed = self._parse_response(raw)
            parsed["criterion_id"] = cid
            parsed["criterion_type"] = criterion.get("criterion_type", "Semantic")
            parsed["model"] = self.model_id
            parsed["programmatic"] = False

            logger.info(
                f"[{cid}] {self.model_family} verdict: "
                f"{parsed.get('verdict')} (conf={parsed.get('confidence')})"
            )
            return parsed

        except Exception as e:
            logger.error(
                f"[{cid}] LLM evaluation failed ({self.model_family}): {e}"
            )
            return self._uncertain(criterion, reason=str(e))

    # ------------------------------------------------------------------ #
    # Private
    # ------------------------------------------------------------------ #

    def _build_prompt(
        self,
        criterion: Dict[str, Any],
        evidence: Dict[str, Any]
    ) -> str:
        return self.prompt_template.format(
            criterion_id=criterion.get("criterion_id", "?"),
            criterion_type=criterion.get("criterion_type", "Semantic"),
            category=criterion.get("category", ""),
            requirement_text=criterion.get("requirement_text", ""),
            threshold=criterion.get("threshold") or "N/A",
            threshold_unit=criterion.get("threshold_unit") or "N/A",
            document_required=criterion.get("document_required") or "N/A",
            is_mandatory=criterion.get("is_mandatory", True),
            evidence_field=evidence.get("field_name", ""),
            extracted_value=evidence.get("extracted_value", ""),
            extraction_type=evidence.get("extraction_type", ""),
            evidence_confidence=evidence.get("confidence", 0.5),
            raw_text=str(evidence.get("raw_text", ""))[:1500],
            bbox_ref=evidence.get("bbox_ref", "N/A"),
            schema=json.dumps(_VERDICT_SCHEMA, indent=2)
        )

    def _call_llm(self, prompt: str) -> str:
        if self.model_family == "claude":
            response = self.client.messages.create(
                model=self.model_id,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text

        else:  # gpt4o
            response = self.client.chat.completions.create(
                model=self.model_id,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content

    def _parse_response(self, raw: str) -> Dict[str, Any]:
        """Parse structured JSON from LLM response."""
        raw = raw.strip()
        # strip markdown fences
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1])

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # try to extract JSON block
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
            else:
                raise ValueError(f"No valid JSON in LLM response: {raw[:300]}")

        # validate required fields
        verdict = parsed.get("verdict", "UNCERTAIN").upper()
        if verdict not in ("PASS", "FAIL", "UNCERTAIN"):
            verdict = "UNCERTAIN"

        confidence = float(parsed.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        return {
            "verdict": verdict,
            "confidence": confidence,
            "reasoning_steps": parsed.get("reasoning_steps", []),
            "key_factor": parsed.get("key_factor", ""),
            "ambiguities": parsed.get("ambiguities", []),
        }

    def _load_prompt(self) -> str:
        prompt_file = PROMPTS_DIR / (
            "claude_cot_prompt.txt"
            if self.model_family == "claude"
            else "gpt4o_cot_prompt.txt"
        )
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
        return prompt_file.read_text(encoding="utf-8")

    def _uncertain(
        self, criterion: Dict[str, Any], reason: str
    ) -> Dict[str, Any]:
        return {
            "criterion_id": criterion.get("criterion_id", "?"),
            "criterion_type": criterion.get("criterion_type", "Semantic"),
            "verdict": "UNCERTAIN",
            "confidence": 0.2,
            "reasoning_steps": [reason],
            "key_factor": "llm_call_failed",
            "ambiguities": [reason],
            "model": self.model_id,
            "programmatic": False,
        }