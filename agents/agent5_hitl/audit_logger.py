"""
audit_logger.py
Agent 5 — HITL Agent
Immutable audit log for every human review decision.

Design (per brainstorm doc):
  - PostgreSQL backend for production — immutable append-only records
  - SQLite fallback for local dev (no Docker dependency)
  - Every record is timestamped, signed with reviewer_id, and includes
    the original system verdict so drift can be tracked over time
  - Records are never updated or deleted — corrections are new records
  - Structured rejection taxonomy + free-text comment field

The audit log is the primary legal defensibility artefact —
every procurement decision that involved human review is traceable
to a specific reviewer action with full context.
"""

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Literal

logger = logging.getLogger(__name__)

DecisionType = Literal["PASS", "FAIL", "UNCERTAIN"]

# Structured rejection taxonomy (free-text comment is always also captured)
REJECTION_REASONS = [
    "document_not_submitted",
    "document_expired",
    "value_below_threshold",
    "document_not_authentic",
    "criterion_not_applicable",
    "ambiguous_requirement",
    "ocr_extraction_error",
    "other",
]

# SQLite schema — mirrors the PostgreSQL schema in migrations/001_initial.sql
_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS audit_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    log_id          TEXT NOT NULL UNIQUE,
    tender_id       TEXT NOT NULL,
    bidder_id       TEXT NOT NULL,
    criterion_id    TEXT NOT NULL,
    case_type       TEXT NOT NULL,
    system_verdict  TEXT NOT NULL,
    human_verdict   TEXT NOT NULL,
    reviewer_id     TEXT NOT NULL,
    rejection_reason TEXT,
    comment         TEXT NOT NULL,
    was_override    INTEGER NOT NULL,   -- 1 if human changed system verdict
    composite_confidence REAL,
    created_at      TEXT NOT NULL
);
"""


class AuditLogger:
    """
    Append-only audit log for human review decisions.
    Supports PostgreSQL (production) and SQLite (local dev fallback).
    """

    def __init__(
        self,
        postgres_url: Optional[str] = None,
        sqlite_path: str = "storage/audit_log.db"
    ):
        self.postgres_url = postgres_url or os.getenv("POSTGRES_URL")
        self.sqlite_path = Path(sqlite_path)
        self._use_postgres = bool(self.postgres_url)

        if self._use_postgres:
            logger.info("AuditLogger: using PostgreSQL backend")
        else:
            logger.info(
                f"AuditLogger: using SQLite fallback at {self.sqlite_path}"
            )
            self._init_sqlite()

    def log(
        self,
        tender_id: str,
        bidder_id: str,
        criterion_id: str,
        case_type: str,
        system_verdict: DecisionType,
        human_verdict: DecisionType,
        reviewer_id: str,
        comment: str,
        composite_confidence: float = 0.0,
        rejection_reason: Optional[str] = None
    ) -> str:
        """
        Append one immutable audit record.

        Args:
            tender_id:            tender being evaluated
            bidder_id:            bidder being reviewed
            criterion_id:         which criterion this decision covers
            case_type:            MISSING_EVIDENCE | MODEL_DISAGREEMENT | etc.
            system_verdict:       what the automated system recommended
            human_verdict:        what the reviewer decided
            reviewer_id:          reviewer's user ID
            comment:              mandatory free-text justification
            composite_confidence: automated confidence score at time of routing
            rejection_reason:     structured taxonomy tag (optional)

        Returns:
            log_id: unique identifier for this audit record
        """
        import uuid
        log_id = str(uuid.uuid4())
        was_override = int(system_verdict != human_verdict)
        created_at = datetime.now(timezone.utc).isoformat()

        record = {
            "log_id": log_id,
            "tender_id": tender_id,
            "bidder_id": bidder_id,
            "criterion_id": criterion_id,
            "case_type": case_type,
            "system_verdict": system_verdict,
            "human_verdict": human_verdict,
            "reviewer_id": reviewer_id,
            "rejection_reason": rejection_reason,
            "comment": comment,
            "was_override": was_override,
            "composite_confidence": composite_confidence,
            "created_at": created_at,
        }

        if self._use_postgres:
            self._insert_postgres(record)
        else:
            self._insert_sqlite(record)

        logger.info(
            f"Audit log: [{criterion_id}] reviewer={reviewer_id} "
            f"system={system_verdict} → human={human_verdict} "
            f"override={bool(was_override)}"
        )
        return log_id

    def get_log_for_bidder(
        self, tender_id: str, bidder_id: str
    ) -> List[Dict[str, Any]]:
        """Retrieve all audit records for a bidder in a tender."""
        if self._use_postgres:
            return self._query_postgres(
                "SELECT * FROM audit_log WHERE tender_id=%s AND bidder_id=%s "
                "ORDER BY created_at ASC",
                (tender_id, bidder_id)
            )
        return self._query_sqlite(
            "SELECT * FROM audit_log WHERE tender_id=? AND bidder_id=? "
            "ORDER BY created_at ASC",
            (tender_id, bidder_id)
        )

    def get_log_for_criterion(
        self, tender_id: str, criterion_id: str
    ) -> List[Dict[str, Any]]:
        """Retrieve audit records for a specific criterion across all bidders."""
        if self._use_postgres:
            return self._query_postgres(
                "SELECT * FROM audit_log WHERE tender_id=%s AND criterion_id=%s "
                "ORDER BY created_at ASC",
                (tender_id, criterion_id)
            )
        return self._query_sqlite(
            "SELECT * FROM audit_log WHERE tender_id=? AND criterion_id=? "
            "ORDER BY created_at ASC",
            (tender_id, criterion_id)
        )

    def get_override_stats(self, tender_id: str) -> Dict[str, Any]:
        """
        Return aggregate override statistics for a tender.
        Used by FeedbackLoop to track model accuracy over time.
        """
        if self._use_postgres:
            rows = self._query_postgres(
                "SELECT system_verdict, human_verdict, was_override "
                "FROM audit_log WHERE tender_id=%s",
                (tender_id,)
            )
        else:
            rows = self._query_sqlite(
                "SELECT system_verdict, human_verdict, was_override "
                "FROM audit_log WHERE tender_id=?",
                (tender_id,)
            )

        total = len(rows)
        overrides = sum(1 for r in rows if r["was_override"])
        return {
            "total_decisions": total,
            "overrides": overrides,
            "accuracy": round((total - overrides) / total, 3) if total else 0.0,
        }

    # ------------------------------------------------------------------ #
    # SQLite helpers
    # ------------------------------------------------------------------ #

    def _init_sqlite(self) -> None:
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.sqlite_path))
        conn.execute(_CREATE_TABLE)
        conn.commit()
        conn.close()

    def _insert_sqlite(self, record: Dict[str, Any]) -> None:
        conn = sqlite3.connect(str(self.sqlite_path))
        conn.execute(
            """INSERT INTO audit_log
               (log_id, tender_id, bidder_id, criterion_id, case_type,
                system_verdict, human_verdict, reviewer_id, rejection_reason,
                comment, was_override, composite_confidence, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                record["log_id"], record["tender_id"], record["bidder_id"],
                record["criterion_id"], record["case_type"],
                record["system_verdict"], record["human_verdict"],
                record["reviewer_id"], record["rejection_reason"],
                record["comment"], record["was_override"],
                record["composite_confidence"], record["created_at"],
            )
        )
        conn.commit()
        conn.close()

    def _query_sqlite(self, sql: str, params: tuple) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(str(self.sqlite_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params).fetchall()
        conn.close()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------ #
    # PostgreSQL helpers
    # ------------------------------------------------------------------ #

    def _insert_postgres(self, record: Dict[str, Any]) -> None:
        import psycopg2
        try:
            conn = psycopg2.connect(self.postgres_url)
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO audit_log
                   (log_id, tender_id, bidder_id, criterion_id, case_type,
                    system_verdict, human_verdict, reviewer_id, rejection_reason,
                    comment, was_override, composite_confidence, created_at)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (
                    record["log_id"], record["tender_id"], record["bidder_id"],
                    record["criterion_id"], record["case_type"],
                    record["system_verdict"], record["human_verdict"],
                    record["reviewer_id"], record["rejection_reason"],
                    record["comment"], record["was_override"],
                    record["composite_confidence"], record["created_at"],
                )
            )
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            logger.error(f"PostgreSQL insert failed: {e} — falling back to SQLite")
            self._use_postgres = False
            self._init_sqlite()
            self._insert_sqlite(record)

    def _query_postgres(self, sql: str, params: tuple) -> List[Dict[str, Any]]:
        import psycopg2
        import psycopg2.extras
        try:
            conn = psycopg2.connect(self.postgres_url)
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(sql, params)
            rows = cur.fetchall()
            cur.close()
            conn.close()
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"PostgreSQL query failed: {e}")
            return []