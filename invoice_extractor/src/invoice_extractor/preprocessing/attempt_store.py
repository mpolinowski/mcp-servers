"""Attempt history for the preprocessing retry loop — one SQLite table.

Deliberately NOT a column on the ``Invoice`` model: ``InvoiceStore`` generates
its schema from the model and hard-fails on missing columns, and attempts are
a per-cycle audit trail, not an attribute of the invoice. The table lives in
the same SQLite *file* as ``invoices`` (separate table, separate connection)
so there is still one ground-truth database per install.

A **cycle** is one extract→retries episode scoped to a resolved source-image
path: attempt 1 is the ``extract_invoice_from_image`` call, attempts 2..4 are
retries. Re-extracting the same image starts a fresh cycle (attempts restart
at 1) — the old cycle stays in the history as-is. Every cycle ends with a
finalize step that marks its best attempt (``best`` = 1); a cycle with no
finalized attempt is still open and may be retried.

Every attempt is self-contained in this table: the enhanced image (BLOB),
the JSON payload, and the HTML pass output are all stored, so the finalize
step can rebuild the winner from the database even when it runs in a later
call than the attempt itself. Only the winner's image is promoted to disk
(``<stem>_processed.<ext>``); the losers exist only here.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class AttemptStore:
    """Append-only-ish attempt log. Rows are never deleted or rewritten except
    by the one finalize update that marks the cycle's winner (``best``)."""

    def __init__(self, db_path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self.ensure_schema()

    # -- schema --

    def ensure_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS preprocessing_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_key TEXT NOT NULL,      -- resolved source-image path (what the agent passed)
                cycle INTEGER NOT NULL,       -- 1-based episode counter per image_key
                attempt INTEGER NOT NULL,     -- 1-based position within the cycle
                config TEXT NOT NULL,         -- JSON knob set the attempt ran with
                score REAL NOT NULL,          -- deterministic quality score [0..1]
                metrics TEXT NOT NULL,        -- JSON: present/missing fields, line count, sum-vs-total
                outcome TEXT NOT NULL,        -- SUCCESS | RETRYABLE | MANUAL_REVIEW | ERROR
                payload TEXT,                 -- JSON extraction payload (NULL on hard failure)
                html TEXT,                    -- HTML pass output (NULL when it failed / didn't run)
                enhanced_image BLOB,          -- enhanced bytes fed to the model (NULL if extraction failed)
                best INTEGER NOT NULL DEFAULT 0,
                best_processed_path TEXT,     -- set on the winner at finalize: <stem>_processed.<ext>
                created_at TEXT NOT NULL,
                UNIQUE (image_key, cycle, attempt)
            )
            """
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # -- cycle lifecycle --

    @staticmethod
    def _key(image_key) -> str:
        return str(Path(image_key).resolve())

    def latest_cycle(self, image_key) -> int | None:
        """Highest cycle for this image, or None when it has never started."""
        row = self._conn.execute(
            "SELECT MAX(cycle) AS c FROM preprocessing_attempts WHERE image_key = ?",
            (self._key(image_key),),
        ).fetchone()
        return int(row["c"]) if row["c"] is not None else None

    def attempts_in_cycle(self, image_key, cycle: int) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM preprocessing_attempts WHERE image_key = ? AND cycle = ? ORDER BY attempt",
            (self._key(image_key), cycle),
        ).fetchall()

    def cycle_is_closed(self, image_key) -> bool:
        """True once the cycle's finalize step has marked a winner."""
        cycle = self.latest_cycle(image_key)
        if cycle is None:
            return False
        row = self._conn.execute(
            "SELECT 1 FROM preprocessing_attempts WHERE image_key = ? AND cycle = ? AND best = 1 LIMIT 1",
            (self._key(image_key), cycle),
        ).fetchone()
        return row is not None

    def next_attempt(self, image_key) -> tuple[int, int]:
        """(cycle, next_attempt) for the *open* cycle of this image.

        Raises RuntimeError when the image has no cycle (call extract first)
        or when the cycle is already finalized (call extract to start a fresh
        one).
        """
        cycle = self.latest_cycle(image_key)
        if cycle is None:
            raise RuntimeError(
                f"no extraction cycle for {image_key} — call extract_invoice_from_image first"
            )
        if self.cycle_is_closed(image_key):
            raise RuntimeError(
                f"cycle {cycle} for {image_key} is already finalized — "
                "call extract_invoice_from_image to start a fresh cycle"
            )
        attempts = self.attempts_in_cycle(image_key, cycle)
        return cycle, len(attempts) + 1

    # -- records --

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def record_attempt(
        self,
        image_key,
        cycle: int,
        attempt: int,
        config: dict,
        score: float,
        metrics: dict,
        outcome: str,
        enhanced_image: bytes | None,
        payload: dict | None = None,
        html: str | None = None,
    ) -> int:
        key = self._key(image_key)
        existing = self._conn.execute(
            "SELECT id FROM preprocessing_attempts WHERE image_key = ? AND cycle = ? AND attempt = ?",
            (key, cycle, attempt),
        ).fetchone()
        payload_json = json.dumps(payload, ensure_ascii=False) if payload is not None else None
        if existing is not None:  # retrying a slot that's already taken — replace it in place
            cur = self._conn.execute(
                "UPDATE preprocessing_attempts SET config = ?, score = ?, metrics = ?, "
                "outcome = ?, payload = ?, html = ?, enhanced_image = ?, created_at = ? WHERE id = ?",
                (json.dumps(config), score, json.dumps(metrics), outcome, payload_json, html,
                 enhanced_image, self._now(), existing["id"]),
            )
            self._conn.commit()
            return int(existing["id"])
        cur = self._conn.execute(
            "INSERT INTO preprocessing_attempts "
            "(image_key, cycle, attempt, config, score, metrics, outcome, payload, html, enhanced_image, best, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)",
            (key, cycle, attempt, json.dumps(config), score, json.dumps(metrics), outcome,
             payload_json, html, enhanced_image, self._now()),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def best_score_before(self, image_key, cycle: int) -> float | None:
        """Best score kept by any *earlier* cycle, or None when nothing was kept yet.

        Powers the best-preservation guard: a later cycle must never clobber
        an extraction already kept in ``invoices`` with a worse one."""
        row = self._conn.execute(
            "SELECT MAX(score) AS s FROM preprocessing_attempts "
            "WHERE image_key = ? AND best = 1 AND cycle < ?",
            (self._key(image_key), cycle),
        ).fetchone()
        return float(row["s"]) if row["s"] is not None else None

    def enhanced_bytes(self, image_key, cycle: int, attempt: int) -> bytes | None:
        """The enhanced image the model saw for one attempt (BLOB copy)."""
        row = self._conn.execute(
            "SELECT enhanced_image FROM preprocessing_attempts "
            "WHERE image_key = ? AND cycle = ? AND attempt = ?",
            (self._key(image_key), cycle, attempt),
        ).fetchone()
        if row is None or row["enhanced_image"] is None:
            return None
        return bytes(row["enhanced_image"])

    def finalize_best(self, image_key, cycle: int, best_attempt_id: int, best_processed_path) -> None:
        """Mark the cycle's winner. Idempotent — re-finalizing just moves the flag."""
        key = self._key(image_key)
        self._conn.execute(
            "UPDATE preprocessing_attempts SET best = 0, best_processed_path = NULL "
            "WHERE image_key = ? AND cycle = ?",
            (key, cycle),
        )
        self._conn.execute(
            "UPDATE preprocessing_attempts SET best = 1, best_processed_path = ? "
            "WHERE id = ? AND image_key = ? AND cycle = ?",
            (str(best_processed_path), best_attempt_id, key, cycle),
        )
        self._conn.commit()

    def cycle_history(self, image_key) -> dict:
        """All attempts of the latest cycle + who won + the two canonical paths.

        Returns an empty dict when no cycle exists for the image yet.
        """
        cycle = self.latest_cycle(image_key)
        if cycle is None:
            return {}
        key = self._key(image_key)
        attempts = [
            self._row_to_dict(row) for row in self.attempts_in_cycle(image_key, cycle)
        ]
        best_row = next((a for a in attempts if a.get("best")), None)
        return {
            "cycle": cycle,
            "attempts": attempts,
            "best_attempt_id": (best_row or {}).get("id"),
            # The path the original extraction was run on (resolved), plus the
            # promoted winner image — together they say "what we started with,
            # what we ended up keeping".
            "original_image_path": key,
            "best_processed_path": (best_row or {}).get("best_processed_path"),
        }

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        return {
            "id": int(row["id"]),
            "attempt": int(row["attempt"]),
            "config": json.loads(row["config"]),
            "score": float(row["score"]),
            "metrics": json.loads(row["metrics"]),
            "outcome": row["outcome"],
            "payload": json.loads(row["payload"]) if row["payload"] is not None else None,
            "html": row["html"],
            "best": bool(row["best"]),
            "best_processed_path": row["best_processed_path"],
            "enhanced_image_bytes": len(row["enhanced_image"]) if row["enhanced_image"] else 0,
            "created_at": row["created_at"],
        }
