"""Storage layer for extracted invoices.

Three stores, one ground truth:

- ``sqlite_store`` — one row per invoice (schema generated from the Pydantic
  ``Invoice`` model), keyed by the final image path; re-ingestion updates the
  row in place and keeps the same id.
- ``chroma_store``  — Markdown chunks from the HTML/Markdown pass, embedded
  and appended (never replaced) into a persistent ChromaDB collection. Every
  chunk carries the SQLite row id as metadata.
- ``bm25``          — lightweight in-memory BM25 ranking over the cleaned
  ``clean_text`` column of every ingested invoice.

Retrieval is always linkable back to its SQLite row. Retrieval *tools* on top
of these are a later phase; these are plain function/class APIs with offline
tests.
"""

from .bm25 import search as bm25_search
from .chroma_store import ChromaStore
from .sqlite_store import InvoiceStore, schema_columns
from .text_clean import clean_text, tokenize

__all__ = [
    "InvoiceStore",
    "schema_columns",
    "ChromaStore",
    "bm25_search",
    "clean_text",
    "tokenize",
]
