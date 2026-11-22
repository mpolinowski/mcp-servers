"""ChromaDB chunk store for the HTML/Markdown pass.

Conventions follow the code-vector-ingestion reference (Markdown splitter,
cosine space, OpenAI-compatible embeddings pointed at Ollama), with one
deliberate difference: ingestion **appends**. Chunk ids are deterministic,
``inv<row_id>_chunk<i>``, so re-ingesting the same row upserts in place, and
orphan chunks left by a prior *longer* ingest of that row are pruned
individually. This store never deletes or recreates the collection.

Every chunk's metadata carries the SQLite row id (``invoice_id``), which is
the link back to the ground-truth row.
"""

from pathlib import Path

from chromadb import PersistentClient
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import MarkdownTextSplitter


def _chunk_id(row_id: int, index: int) -> str:
    return f"inv{row_id}_chunk{index}"


class ChromaStore:
    def __init__(
        self,
        db_dir,
        embeddings: Embeddings,
        collection_name: str = "invoices",
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
    ):
        self._db_dir = Path(db_dir)
        self._embeddings = embeddings
        self._collection_name = collection_name
        self._splitter = MarkdownTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self._db_dir.mkdir(parents=True, exist_ok=True)
        self._client = PersistentClient(path=str(self._db_dir))
        # ``hnsw:space`` takes effect at collection creation only; re-passing
        # it for an existing collection is harmless (verified).
        self._store = Chroma(
            client=self._client,
            collection_name=collection_name,
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"},
            create_collection_if_not_exists=True,
        )

    # -- writes ----------------------------------------------------------------

    def add_chunks(self, row_id: int, markdown: str) -> int:
        """Split, embed, and upsert the row's chunks. Returns the chunk count
        (0 when the markdown is empty).

        Deterministic ids mean a re-ingest replaces rather than duplicates,
        and any chunks of this row that no longer exist are deleted by id —
        targeted cleanup, never a collection reset."""
        if not markdown or not markdown.strip():
            return 0
        # langchain 1.x split_text() yields plain strings; older versions
        # yield Documents — accept both so this doesn't break on upgrade.
        parts = self._splitter.split_text(markdown)
        texts = [p if isinstance(p, str) else p.page_content for p in parts]
        if not texts:
            return 0
        new_ids = [_chunk_id(row_id, i) for i in range(len(texts))]
        metadatas = [{"invoice_id": row_id} for _ in texts]

        stale = self._existing_ids(row_id) - set(new_ids)

        # `add_texts` routes through `upsert`, so repeated ids replace in place.
        self._store.add_texts(texts=texts, ids=new_ids, metadatas=metadatas)
        if stale:
            self._store.delete(ids=sorted(stale))
        return len(texts)

    def _existing_ids(self, row_id: int) -> set[str]:
        try:
            result = self._store.get(where={"invoice_id": row_id}, include=[])
        except Exception:
            # Collection empty / query rejected — nothing can possibly be stale.
            return set()
        return set(result.get("ids") or [])

    # -- reads -----------------------------------------------------------------

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Best-first hits: ``[{"invoice_id", "text", "score"}, ...]``.

        ``score`` is the cosine *distance* Chroma reports — lower is closer.
        An empty (or missing) collection yields ``[]``, which means "not
        ingested yet", never a legitimate zero-hit answer."""
        try:
            if self._store._collection.count() == 0:  # noqa: SLF001 -- langchain 1.x doesn't expose count()
                return []
            hits = self._store.similarity_search_with_score(query, k=k)
        except Exception:
            return []
        return [
            {
                "invoice_id": doc.metadata.get("invoice_id"),
                "text": doc.page_content,
                "score": float(dist),
            }
            for doc, dist in hits
        ]

    def collection_count(self) -> int:
        return int(self._store._collection.count())  # noqa: SLF001 -- see above
