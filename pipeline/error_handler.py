"""
error_handler.py
pipeline/ — TendorGuard.ai

Centralised error classification and recovery logic for the pipeline.

Distinguishes between:
  - Recoverable errors  → log + continue with a safe default value
  - Agent failures      → log + mark criterion as NEEDS_MANUAL_REVIEW
  - Fatal errors        → log + re-raise to stop the pipeline
"""

import logging
import traceback
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    RECOVERABLE = "RECOVERABLE"   # pipeline continues
    AGENT_FAULT = "AGENT_FAULT"   # criterion flagged for manual review
    FATAL       = "FATAL"         # pipeline must stop


class PipelineError(Exception):
    """Wrapper exception carrying structured error context."""

    def __init__(
        self,
        message:    str,
        severity:   ErrorSeverity = ErrorSeverity.AGENT_FAULT,
        agent:      str = "unknown",
        context:    Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.severity = severity
        self.agent    = agent
        self.context  = context or {}

    def __str__(self) -> str:
        return (
            f"[{self.severity.value}] [{self.agent}] "
            f"{super().__str__()} | context={self.context}"
        )


class ErrorHandler:
    """
    Centralised handler used by the orchestrator.

    Usage:
        handler = ErrorHandler()
        try:
            result = agent.run(...)
        except Exception as exc:
            safe_result = handler.handle(exc, agent="Agent3", context={...})
    """

    def __init__(self):
        self.errors: list = []   # accumulated error log for this run

    # ── Classification ────────────────────────────────────────────────────────

    @staticmethod
    def classify(exc: Exception) -> ErrorSeverity:
        """
        Classify an exception into a severity level.

        Rules:
          - ImportError / RuntimeError from model loading → FATAL
          - API rate-limit / timeout errors               → RECOVERABLE
          - Everything else                               → AGENT_FAULT
        """
        msg = str(exc).lower()

        if isinstance(exc, (ImportError, SystemError)):
            return ErrorSeverity.FATAL

        if any(kw in msg for kw in ("rate limit", "429", "timeout", "connection")):
            return ErrorSeverity.RECOVERABLE

        if isinstance(exc, (ValueError, KeyError, AttributeError)):
            return ErrorSeverity.AGENT_FAULT

        return ErrorSeverity.AGENT_FAULT

    # ── Handle ────────────────────────────────────────────────────────────────

    def handle(
        self,
        exc:     Exception,
        agent:   str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Handle an exception from an agent.

        - FATAL:       re-raises as PipelineError
        - AGENT_FAULT: returns a safe fallback evidence/verdict dict
        - RECOVERABLE: returns an empty dict and logs a warning

        Args:
            exc:     the exception that was raised
            agent:   name of the failing agent (for logs)
            context: optional dict with criterion_id, bidder_id etc.

        Returns:
            safe fallback dict (only for non-fatal errors)

        Raises:
            PipelineError for FATAL severity
        """
        severity = self.classify(exc)
        tb       = traceback.format_exc()
        ctx      = context or {}

        record = {
            "agent":    agent,
            "severity": severity.value,
            "error":    str(exc),
            "context":  ctx,
        }
        self.errors.append(record)

        if severity == ErrorSeverity.FATAL:
            logger.critical(
                f"FATAL error in {agent}: {exc}\n{tb}"
            )
            raise PipelineError(str(exc), severity, agent, ctx) from exc

        elif severity == ErrorSeverity.AGENT_FAULT:
            logger.error(
                f"Agent fault in {agent}: {exc} | context={ctx}"
            )
            return self._evidence_fallback(str(exc), ctx)

        else:  # RECOVERABLE
            logger.warning(
                f"Recoverable error in {agent}: {exc} | context={ctx}"
            )
            return {}

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _evidence_fallback(reason: str, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns a minimal safe evidence dict that downstream agents
        (Agent 4 AuditorAgent) can consume without crashing.
        Verdict will be NEEDS_MANUAL_REVIEW.
        """
        return {
            "found":             False,
            "confidence":        0.0,
            "ambiguity_reason":  f"System Error: {reason}",
            "source_document":   ctx.get("bidder_id", ""),
            "criterion_id":      ctx.get("criterion_id", ""),
            "page_number":       None,
            "line_number":       None,
            "matching_line_text": None,
            "context_lines":     [],
            "extracted_value":   None,
            "extracted_numeric": None,
            "threshold_numeric": None,
            "meets_threshold":   None,
            "extraction_method": "error_fallback",
        }

    def get_error_log(self) -> list:
        """Return all errors recorded during this run."""
        return list(self.errors)

    def has_fatal(self) -> bool:
        """True if any fatal error was recorded."""
        return any(e["severity"] == ErrorSeverity.FATAL.value for e in self.errors)

    def clear(self) -> None:
        """Reset the error log between runs."""
        self.errors.clear()
