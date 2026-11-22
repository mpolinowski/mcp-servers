"""Lightweight BM25 ranking over the ingested invoices' clean text.

In-memory, no server: the corpus is the ``clean_text`` column of every
ingested row. Built fresh per query — the corpus is small (a handful of
invoices), so rebuild cost is trivial and this stays dependency-poor.

``bm25s`` 0.3.x API (this box): bare ``BM25()``, ``index(...)`` takes token
*lists* (raw strings are iterated per-character and score zero), and
``retrieve(..., corpus=<docs>, return_as="tuple")`` maps back to the
original corpus objects — here, the row ids.
"""

from typing import Any

from bm25s import BM25

from .text_clean import tokenize


def search(corpus: list[tuple[int, str]], query: str, k: int = 5) -> list[dict[str, Any]]:
    """Rank ``corpus`` rows by BM25 similarity to ``query``.

    ``corpus``: ``[(row_id, text), ...]``. Returns up to ``k`` dicts
    ``{"row_id": int, "score": float}`` best-first. Zero-score rows (no
    shared tokens) are dropped, so an empty result genuinely means "nothing
    matched" — and so do an empty corpus or a stopword-only query."""
    if not corpus or not (query or "").strip():
        return []
    q_tokens = tokenize(query)
    if not q_tokens:
        return []

    ids = [row_id for row_id, _ in corpus]
    tokenized = [text.split() for _, text in corpus]

    model = BM25()
    model.index(tokenized, show_progress=False)
    top_docs, top_scores = model.retrieve(
        [q_tokens],
        corpus=ids,
        k=min(k, len(ids)),
        sorted=True,
        return_as="tuple",
        show_progress=False,
    )
    results = [
        {"row_id": int(rid), "score": float(score)}
        for rid, score in zip(top_docs[0], top_scores[0])
        if float(score) > 0.0
    ][:k]
    return results
