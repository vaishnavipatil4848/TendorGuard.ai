"""
message_bus.py
pipeline/ — TendorGuard.ai

Lightweight typed message bus for inter-agent communication.
Wraps payloads in AgentMessage envelopes so each pipeline step
has a clear, auditable record of what was passed and when.
"""

import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from config.settings import AgentMessage

logger = logging.getLogger(__name__)


class MessageBus:
    """
    In-process message bus used by the orchestrator to pass
    structured envelopes between agents.

    Supports:
      - publish / subscribe by message type
      - full message history for audit export
      - optional per-message callback (e.g. Streamlit progress update)
    """

    def __init__(self, on_message: Optional[Callable[[AgentMessage], None]] = None):
        """
        Args:
            on_message: optional hook called on every published message.
                        Useful for piping messages to a Streamlit status area.
        """
        self._subscribers: Dict[str, List[Callable[[AgentMessage], None]]] = {}
        self._history: List[AgentMessage] = []
        self._on_message = on_message

    # ── Pub / Sub ─────────────────────────────────────────────────────────────

    def subscribe(self, msg_type: str, handler: Callable[[AgentMessage], None]) -> None:
        """
        Register a handler for a specific message type.

        Args:
            msg_type: e.g. "TENDER_PARSED", "CRITERIA_EXTRACTED", "EVIDENCE_READY"
            handler:  callable that receives an AgentMessage
        """
        self._subscribers.setdefault(msg_type, []).append(handler)
        logger.debug(f"MessageBus: subscribed handler for '{msg_type}'")

    def publish(
        self,
        sender:   str,
        receiver: str,
        msg_type: str,
        payload:  Dict[str, Any]
    ) -> AgentMessage:
        """
        Publish a message and deliver it to all registered handlers.

        Args:
            sender:   name of the publishing agent (e.g. "Agent1_Architect")
            receiver: intended recipient (e.g. "Agent2_Vision", "Orchestrator")
            msg_type: message type string
            payload:  arbitrary data dict

        Returns:
            The created AgentMessage (also stored in history)
        """
        msg = AgentMessage(
            sender=sender,
            receiver=receiver,
            msg_type=msg_type,
            payload={**payload, "_timestamp": datetime.now().isoformat()}
        )
        self._history.append(msg)

        logger.debug(
            f"MessageBus: [{sender}] → [{receiver}] "
            f"type={msg_type} payload_keys={list(payload.keys())}"
        )

        # global hook
        if self._on_message:
            try:
                self._on_message(msg)
            except Exception as exc:
                logger.warning(f"MessageBus on_message hook failed: {exc}")

        # type-specific handlers
        for handler in self._subscribers.get(msg_type, []):
            try:
                handler(msg)
            except Exception as exc:
                logger.error(
                    f"MessageBus handler error for '{msg_type}': {exc}"
                )

        return msg

    # ── History ───────────────────────────────────────────────────────────────

    def get_history(self, msg_type: Optional[str] = None) -> List[AgentMessage]:
        """
        Return message history, optionally filtered by type.

        Args:
            msg_type: if given, only return messages of this type

        Returns:
            List of AgentMessage objects in chronological order
        """
        if msg_type:
            return [m for m in self._history if m.msg_type == msg_type]
        return list(self._history)

    def clear_history(self) -> None:
        """Clear the message history (e.g. between evaluation runs)."""
        self._history.clear()
        logger.debug("MessageBus: history cleared")

    def summary(self) -> Dict[str, int]:
        """Return count of messages per type."""
        counts: Dict[str, int] = {}
        for m in self._history:
            counts[m.msg_type] = counts.get(m.msg_type, 0) + 1
        return counts
