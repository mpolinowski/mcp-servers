"""SQLite ground truth: one row per ingested invoice.

The column schema is generated from the Pydantic ``Invoice`` model, so
changing the extraction schema is the single source of truth and the table
shape follows. Mapping rules (per field annotation):

- scalars keep their name (``str`` -> TEXT, ``float`` -> REAL, ``int`` -> INTEGER);
- ``Optional[Party]`` sub-models flatten to ``<parent>_<field>`` columns;
- list-shaped fields (``line_items``, ``payment_methods``) store JSON TEXT.

Re-ingestion is an upsert keyed by the *final* image path: the row id
survives updates, which is what keeps Chroma chunk metadata and BM25 hits
from dangling.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Union, get_args, get_origin
from types import UnionType

from pydantic import BaseModel

from ..tools.invoice_json import Invoice


# --- column generation -------------------------------------------------------


def _base(annotation: Any) -> Any:
    """Unwrap ``Optional[X]`` -> ``X``; anything else unchanged."""
    if get_origin(annotation) in (Union, UnionType):
        non_none = [a for a in get_args(annotation) if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    return annotation


def _sql_type(annotation: Any) -> str | None:
    """SQLite type for a scalar annotation, ``None`` for compound types."""
    base = _base(annotation)
    if base is str:
        return "TEXT"
    if base is float:
        return "REAL"
    if base in (int, bool):
        return "INTEGER"
    return None


def schema_columns(model: type[BaseModel] = Invoice) -> list[tuple[str, str]]:
    """Ordered ``(column, sql_type)`` for every field of ``model``."""
    cols: list[tuple[str, str]] = []
    for name, field in model.model_fields.items():
        base = _base(field.annotation)
        if get_origin(base) is list:
            cols.append((name, "TEXT"))
            continue
        if isinstance(base, type) and issubclass(base, BaseModel):
            for sub_name, sub_field in base.model_fields.items():
                cols.append((f"{name}_{sub_name}", _sql_type(sub_field.annotation) or "TEXT"))
            continue
        cols.append((name, _sql_type(field.annotation) or "TEXT"))
    return cols


# --- store --------------------------------------------------------------------


class InvoiceStore:
    """One SQLite file, one ``invoices`` table. ``id`` is the shared join key
    that Chroma chunks and BM25 hits carry back to the ground truth."""

    def __init__(self, db_path, model: type[BaseModel] = Invoice):
        self._model = model
        self._columns = schema_columns(model)
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self.ensure_schema()

    # -- schema --

    def ensure_schema(self) -> None:
        table = ", ".join(f"{name} {sql}" for name, sql in self._columns)
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS invoices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                clean_text TEXT,
                {table}
            )
            """
        )
        self._conn.commit()

    # -- writes --

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def _flatten(self, values: dict) -> dict[str, Any]:
        """Map a payload dict onto the generated columns.

        Works for both a validated ``model_dump()`` and a lenient payload
        (failing fields are ``"ERROR"`` strings or ``None`` — both store
        fine; the caller's report carries the failure paths). Every generated
        column is present in the result — absent sub-model fields land as
        ``None`` — so the upsert can build its statement blindly."""
        row: dict[str, Any] = {}
        for name, field in self._model.model_fields.items():
            raw = values.get(name)
            base = _base(field.annotation)
            if get_origin(base) is list:
                row[name] = None if raw is None else json.dumps(raw, ensure_ascii=False)
            elif isinstance(base, type) and issubclass(base, BaseModel):
                # The model routinely omits optional sub-model fields (a party
                # dict that only has a name is valid and common) — fill with
                # None instead of dropping the column. Unknown keys are junk
                # and deliberately ignored; the schema stays the source of truth.
                for k in base.model_fields:
                    row[f"{name}_{k}"] = raw.get(k) if isinstance(raw, dict) else None
            else:
                row[name] = raw
        return row

    def upsert_invoice(self, values: dict, image_path) -> int:
        """Insert or update (keyed by final image path) and return row id.

        The id is stable across re-ingestion of the same image — that is the
        contract Chroma and BM25 rely on."""
        row = self._flatten(values)
        col_names = [c for c, _ in self._columns]
        missing = [c for c in col_names if c not in row]
        if missing:  # invariant: flattening covers the schema
            raise ValueError(f"payload flattening missed columns: {missing}")
        img = str(Path(image_path).resolve())
        now = self._now()
        cols = ", ".join(col_names)
        set_clause = ", ".join(f"{c} = ?" for c in col_names)
        values_param = [row[c] for c in col_names]

        # SELECT-then-INSERT/UPDATE instead of ON CONFLICT DO UPDATE: SQLite's
        # upsert pre-evaluates the autoincrement default even when it takes the
        # UPDATE path, so a plain upsert burns one sequence value per re-ingest
        # (silent id gaps). Explicit branching spends the sequence only for
        # genuinely new rows.
        existing = self._conn.execute(
            "SELECT id FROM invoices WHERE image_path = ?", (img,)
        ).fetchone()
        if existing is not None:
            self._conn.execute(
                f"UPDATE invoices SET {set_clause}, updated_at = ? WHERE id = ?",
                [*values_param, now, existing["id"]],
            )
            self._conn.commit()
            return int(existing["id"])
        try:
            cur = self._conn.execute(
                f"INSERT INTO invoices "
                f"(image_path, created_at, updated_at, {cols}) "
                f"VALUES (?, ?, ?, {', '.join('?' * len(col_names))})",
                [img, now, now, *values_param],
            )
            row_id = int(cur.lastrowid)
        except sqlite3.IntegrityError:
            # Lost a race with another writer: the row exists by now — update it.
            row_id = self._conn.execute(
                "SELECT id FROM invoices WHERE image_path = ?", (img,)
            ).fetchone()["id"]
            self._conn.execute(
                f"UPDATE invoices SET {set_clause}, updated_at = ? WHERE id = ?",
                [*values_param, now, row_id],
            )
        self._conn.commit()
        return row_id

    def set_clean_text(self, row_id: int, text: str) -> None:
        self._conn.execute(
            "UPDATE invoices SET clean_text=?, updated_at=? WHERE id=?",
            (text, self._now(), row_id),
        )
        self._conn.commit()

    # -- reads --

    def get(self, row_id: int) -> dict | None:
        cur = self._conn.execute("SELECT * FROM invoices WHERE id=?", (row_id,))
        r = cur.fetchone()
        return dict(r) if r else None

    def get_by_image_path(self, image_path) -> dict | None:
        cur = self._conn.execute(
            "SELECT * FROM invoices WHERE image_path=?", (str(Path(image_path).resolve()),)
        )
        r = cur.fetchone()
        return dict(r) if r else None

    def all_rows(self):
        """``(id, clean_text)`` for the BM25 corpus — rows that have text."""
        cur = self._conn.execute(
            "SELECT id, clean_text FROM invoices "
            "WHERE clean_text IS NOT NULL AND clean_text <> '' ORDER BY id"
        )
        for r in cur:
            yield int(r["id"]), r["clean_text"]

    # -- lifecycle --

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "InvoiceStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
