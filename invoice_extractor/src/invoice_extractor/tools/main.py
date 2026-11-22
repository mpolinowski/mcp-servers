"""Invoice extractor MCP server.

Two tools:
  * ``list_invoices``            — every image available in the assets dir (recursively).
  * ``extract_invoice_from_image`` — run BOTH extraction passes for one image,
    always, back-to-back, and persist the results:

      1. Structured JSON (strict Pydantic validation; fields that don't fit
         the schema get replaced by ``"ERROR"`` and the file is still written).
      2. HTML / Markdown (plain text + table recognition).
      3. Storage — if the JSON pass produced a payload, the image moves into
         today's date folder under the assets dir (the ``.json/.html/.md``
         outputs follow it), a row is upserted into the SQLite ground truth
         (schema generated from the ``Invoice`` model — the row id is the
         shared join key), the markdown's cleaned text goes into that same
         row, and the markdown is chunked into Chroma (append-only, chunk ids
         re-use the row id). Every retrieval mechanism — now and in the
         upcoming BM25 pass — therefore links back to the same ground-truth
         row.

    The return value is a per-pass success/failure report so an agent can
    tell, file by file, what worked and what didn't. Retrieval *tools* over
    the storage are a later phase; the store APIs and their offline tests
    exist now.

The module keeps the shared ``mcp`` symbol and a ``main()`` stdio entry,
so ``invoice-extractor`` (pyproject script) still starts the server.
"""

import json
import shutil
from datetime import date
from pathlib import Path

import requests
from fastmcp import FastMCP
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

# Relative imports first (package mode); this file is also run directly with
# ``python tools/main.py`` from the package, so fall back to flat imports.
try:
    from ..storage import ChromaStore, InvoiceStore, clean_text
    from .invoice_html import extract_invoice as extract_html_invoice
    from .invoice_html import render_markdown
    from .invoice_json import extract_invoice_lenient
except ImportError:
    from invoice_extractor.storage import ChromaStore, InvoiceStore, clean_text
    from invoice_html import extract_invoice as extract_html_invoice
    from invoice_html import render_markdown
    from invoice_json import extract_invoice_lenient

mcp = FastMCP("Invoice Extractor")

# Images live next to this package (src/invoice_extractor/assets/); stores in …/dbs/.
_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
_DB_DIR = Path(__file__).resolve().parent.parent / "dbs"
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".gif"}


def _image_files(directory: Path) -> list[Path]:
    """All supported images under ``directory``, recursively.

    Recursive on purpose: once ingested, invoices sit in per-day sub-folders
    under the assets dir, and ``list_invoices`` surfaces them for
    re-extraction (re-ingestion updates the row in place instead of
    duplicating it)."""
    if not directory.is_dir():
        raise FileNotFoundError(f"assets directory not found: {directory}")
    return sorted(
        p
        for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    )


def _report_failure(error: Exception) -> dict:
    if isinstance(error, requests.RequestException):
        status = "api_error"  # Ollama unreachable — both passes need it.
    else:
        status = "error"
    return {"success": False, "status": status, "error": f"{type(error).__name__}: {error}"}


def _embeddings() -> Embeddings:
    """Ollama-hosted embeddings via its OpenAI-compatible endpoint.

    Fully local (no external API), so the 'api key' is a placeholder required
    by the client. Tests monkeypatch :func:`_chroma_store` — or point this at
    a deterministic fake — instead of hitting the network."""
    return OpenAIEmbeddings(
        model="qwen3-embedding:0.6b",
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        check_embedding_ctx_length=False,  # Ollama's endpoint doesn't do OpenAI token-count
        chunk_size=32,
    )


def _sqlite_store() -> InvoiceStore:
    """Fresh store bound to the module-level DB dir — tests monkeypatch ``_DB_DIR``."""
    return InvoiceStore(_DB_DIR / "invoices.sqlite")


def _chroma_store() -> ChromaStore:
    """Fresh Chroma store (append-only) bound to ``_DB_DIR`` + the embeddings factory."""
    return ChromaStore(_DB_DIR / "chroma", _embeddings())


def _final_location_for(image: Path) -> tuple[Path, str]:
    """Where the image and its outputs should end up, decided before touching anything.

    The canonical location is today's date folder under the assets dir. If a
    *different* file already occupies that name, the image and its outputs
    stay at the source instead — never overwrite a foreign file, and never
    guess which invoice a name collision belongs to.

    Returns ``(final_path, disposition)`` where disposition is ``in-place``
    (already in today's folder), ``moved``, or ``kept-source`` (collision)."""
    target_dir = _ASSETS_DIR / date.today().isoformat()
    target = target_dir / image.name
    if image.resolve() == target.resolve():
        return image, "in-place"
    if target.exists():
        return image, "kept-source"
    target_dir.mkdir(parents=True, exist_ok=True)
    return target, "moved"


@mcp.tool()
def list_invoices() -> dict:
    """List every invoice image available for extraction in the assets directory (recursively).

    Includes invoices already ingested into their per-day folders — re-running
    extraction on one updates its row and chunks in place rather than creating
    a duplicate. Returns the assets path, the count, and a sorted list of
    image file paths. Call this first to know what to feed to
    ``extract_invoice_from_image``.
    """
    assets = _ASSETS_DIR
    files = _image_files(assets)
    return {
        "assets_dir": str(assets),
        "count": len(files),
        "images": [str(p) for p in files],
    }


@mcp.tool()
def extract_invoice_from_image(image_path: str) -> dict:
    """Extract data from one invoice image, run both passes, and persist the results.

    Runs both extraction passes unconditionally, in order:
      1. JSON — schema-validated; failing fields become ``"ERROR"`` (file still saved).
      2. HTML + Markdown — plain text plus any recognized tables.

    Then, when the JSON pass produced a payload, it stores: the image is moved
    into today's date folder under the assets dir (outputs follow it; a name
    collision with a different file keeps everything at the source instead),
    the JSON payload is upserted as one row in the SQLite ground truth (stable
    row id across re-ingestion), and the markdown becomes the row's cleaned
    text plus append-only Chroma chunks. Both stores carry that row id.

    If the JSON pass failed outright (e.g. Ollama down), nothing is stored —
    the markdown outputs are still written next to the source image — since
    there is no ground-truth row for a chunk to point at.

    Returns a per-pass report, e.g.::

        {
          "image": "/abs/src/invoice.jpg",
          "json":     {"success": true, "status": "ok", "failed_fields": [],
                       "output": "/abs/assets/2025-11-22/invoice.json"},
          "markdown": {"success": true, "status": "ok",
                       "output_md": ".../invoice.md", "output_html": ".../invoice.html"},
          "storage":  {"row_id": 3, "image": ".../invoice.jpg",
                       "disposition": "moved", "chunks": 2}
        }

    "success" is False for a pass if the model is unreachable or extraction
    errors out; "status" explains why.
    """
    image = Path(image_path)
    if not image.is_file():
        raise FileNotFoundError(f"image not found: {image}")

    report: dict = {"image": str(image), "json": {}, "markdown": {}, "storage": {}}

    # -- Pass 1: structured JSON (strict, but never refuses to emit a payload). --
    payload: dict | None = None
    try:
        payload, failed = extract_invoice_lenient(image)
        report["json"] = {
            "success": not failed,
            "status": "ok" if not failed else "errors_marked",
            "failed_fields": failed,
        }
    except FileNotFoundError:
        raise
    except Exception as exc:  # noqa: BLE001 — report, don't crash the agent.
        report["json"] = _report_failure(exc)
        report["json"]["output"] = None

    # -- Pass 2: HTML + Markdown, always runs even after an Ollama error. --
    html: str | None = None
    markdown = ""
    try:
        html = extract_html_invoice(image)
        markdown = render_markdown(html)
        non_empty = bool(markdown.strip())
        report["markdown"] = {"success": non_empty, "status": "ok" if non_empty else "empty"}
    except FileNotFoundError:
        raise
    except Exception as exc:  # noqa: BLE001
        report["markdown"] = _report_failure(exc)
        report["markdown"]["output_md"] = None
        report["markdown"]["output_html"] = None

    # -- Storage: ground-truth row first, then the derived text/chunks. --
    if payload is None:
        # No ground truth to point at: keep the legacy "outputs next to the
        # source image" behavior and store nothing.
        if html is not None:
            html_path = image.with_name(image.stem + ".html")
            md_path = image.with_name(image.stem + ".md")
            html_path.write_text(html, encoding="utf-8")
            md_path.write_text(markdown, encoding="utf-8")
            report["markdown"]["output_md"] = str(md_path)
            report["markdown"]["output_html"] = str(html_path)
        report["storage"] = {
            "row_id": None,
            "disposition": "none",
            "chunks": 0,
            "reason": "no ground-truth row (the JSON pass produced no payload)",
        }
        return report

    final, disposition = _final_location_for(image)
    out_dir = final.parent

    # Outputs are written at the final destination *before* the image moves,
    # so a failed move can't strand the image next to missing outputs.
    json_path = out_dir / (image.stem + ".json")
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if html is not None:
        html_path = out_dir / (image.stem + ".html")
        md_path = out_dir / (image.stem + ".md")
        html_path.write_text(html, encoding="utf-8")
        md_path.write_text(markdown, encoding="utf-8")
    if disposition == "moved":
        shutil.move(str(image), str(final))

    report["json"]["output"] = str(json_path)
    if html is not None:
        report["markdown"]["output_md"] = str(md_path)
        report["markdown"]["output_html"] = str(html_path)

    # Ground truth: one row per invoice, upserted by final image path.
    store = _sqlite_store()
    row_id = store.upsert_invoice(payload, str(final))

    storage_report: dict = {
        "row_id": row_id,
        "image": str(final),
        "disposition": disposition,
        "chunks": 0,
    }
    if html is not None and markdown.strip():
        store.set_clean_text(row_id, clean_text(markdown))
        storage_report["chunks"] = _chroma_store().add_chunks(row_id, markdown)
    else:
        storage_report["note"] = "clean_text/Chroma skipped (no non-empty markdown)"
    report["storage"] = storage_report

    return report


def main() -> None:
    """Start the MCP server over stdio."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
