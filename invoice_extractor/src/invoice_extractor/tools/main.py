"""Invoice extractor MCP server with a bounded preprocessing retry loop.

Tools (sync-callable, so tests drive them directly):

  * ``list_invoices`` — every image available in the assets dir (recursively).
  * ``extract_invoice_from_image(image_path, changes=None)`` — attempt #1 of a
    fresh cycle: preprocess with the given knobs (defaults when omitted), run
    both extraction passes on the enhanced image, score the result, and
    record the attempt.
  * ``retry_preprocessing(image_path, changes)`` — attempts #2..#4 on the
    still-open cycle; ``changes`` is a diff on the last attempt's knobs.
  * ``get_attempt_history(image_path)`` — the latest cycle's attempts, scores,
    metrics, outcomes, winner, and the promoted winner image.

Cycle policy (one source image = one cycle, max ``MAX_ATTEMPTS`` = 4):

  * attempt #1 is the extract call; attempts 2..4 are retries with different
    knob sets (cumulative — unlisted knobs keep the previous value);
  * every attempt is scored deterministically in [0..1] (no model
    confidence) and recorded in the same SQLite file as the ground truth;
  * a cycle closes on a successful attempt, or when the budget runs out —
    in both cases the cycle's *best* attempt is finalized and persisted:
    output files at the canonical (today's date folder) location, the payload
    upserted into the ground truth, the markdown chunked, and the winning
    enhanced image promoted to ``<stem>_processed.<ext>``;
  * best-preservation: the original image is untouched during the cycle, and
    a later cycle never clobbers a better extraction already kept.

Outcomes (decided by this module; ``assess`` only scores):

  * ``ERROR``         — the model pass produced no payload at all (score 0.0).
  * ``SUCCESS``       — score >= 0.5; the cycle finalizes immediately.
  * ``RETRYABLE``     — below threshold with attempts left; the report carries
                        a hint steering the agent to ``retry_preprocessing``.
  * ``MANUAL_REVIEW`` — budget exhausted below threshold; the best attempt is
                        kept and a human should look at the output.

Legacy behavior is preserved: the per-pass report blocks
(``image/json/markdown/storage``), the ``in-place|moved|kept-source`` final
location policy, and the storage wiring (SQLite row + clean text + Chroma
chunks keyed by one row id). The module keeps the shared ``mcp`` symbol and a
``main()`` stdio entry, so ``invoice-extractor`` still starts the server.
"""

import dataclasses
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
    from ..preprocessing import (
        AttemptStore,
        FIELD_RANGES,
        MAX_ATTEMPTS,
        OUTCOME_ERROR,
        OUTCOME_MANUAL_REVIEW,
        OUTCOME_RETRYABLE,
        OUTCOME_SUCCESS,
        SUCCESS_THRESHOLD,
        PreprocessConfig,
        assess,
        encode_image,
        load_image,
        process,
    )
    from ..storage import ChromaStore, InvoiceStore, clean_text
    from .invoice_html import extract_invoice as extract_html_invoice
    from .invoice_html import render_markdown
    from .invoice_json import extract_invoice_lenient
except ImportError:
    from invoice_extractor.preprocessing import (  # type: ignore[no-redef]
        AttemptStore,
        FIELD_RANGES,
        MAX_ATTEMPTS,
        OUTCOME_ERROR,
        OUTCOME_MANUAL_REVIEW,
        OUTCOME_RETRYABLE,
        OUTCOME_SUCCESS,
        SUCCESS_THRESHOLD,
        PreprocessConfig,
        assess,
        encode_image,
        load_image,
        process,
    )
    from invoice_extractor.storage import ChromaStore, InvoiceStore, clean_text  # type: ignore[no-redef]
    from invoice_html import extract_invoice as extract_html_invoice  # type: ignore[no-redef]
    from invoice_html import render_markdown  # type: ignore[no-redef]
    from invoice_json import extract_invoice_lenient  # type: ignore[no-redef]

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


def _attempt_store() -> AttemptStore:
    """Fresh attempt store on the *same* SQLite file as the ground truth — tests
    monkeypatch ``_DB_DIR``. Opened per call, closed per call."""
    return AttemptStore(_DB_DIR / "invoices.sqlite")


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


# --- attempt loop --------------------------------------------------------------


def _outcome_for(payload: dict | None, score: float, attempt: int) -> str:
    """Outcome policy. ``assess`` only scores — the status is this module's
    job, because only the tool layer knows the attempt budget."""
    if payload is None:
        # No model round-trip at all — Ollama down or an unparseable pass —
        # score 0.0 floor. A model that came back, however badly, is worth
        # reviewing at the end of the budget, so only a missing payload is
        # a hard ERROR.
        return OUTCOME_ERROR
    if score >= SUCCESS_THRESHOLD:
        return OUTCOME_SUCCESS
    if attempt < MAX_ATTEMPTS:
        return OUTCOME_RETRYABLE
    return OUTCOME_MANUAL_REVIEW  # budget exhausted; best attempt kept for review


def _should_finalize(outcome: str, attempt: int) -> bool:
    """A cycle closes as soon as it has a winner (SUCCESS) or has spent its
    whole budget — never in between."""
    return outcome == OUTCOME_SUCCESS or attempt >= MAX_ATTEMPTS


def _fresh_cycle_for_extract(store: AttemptStore, image: Path) -> int:
    """Cycle number for a new extract call.

    An *open* cycle must be finished with retries (or by spending its
    budget) — extract never silently restarts it under the agent's feet."""
    latest = store.latest_cycle(image)
    if latest is not None and not store.cycle_is_closed(image):
        raise RuntimeError(
            f"cycle {latest} for {image} is still open — call "
            f"retry_preprocessing for attempts 2-{MAX_ATTEMPTS}; only once the "
            "cycle is finalized can extract_invoice_from_image start a fresh one"
        )
    return (latest or 0) + 1


def _run_attempt(
    image: Path,
    store: AttemptStore,
    cycle: int,
    attempt: int,
    cfg: PreprocessConfig,
) -> dict:
    """One attempt, end to end.

    Preprocess → write the enhanced image to a scratch file (the source is
    never touched) → both extraction passes on the scratch → clean up the
    scratch → score → record. Returns that attempt's working report: outcome,
    score, config, metrics, and the legacy-shaped json/markdown pass blocks.
    Persistence happens only at finalize; this records the attempt row."""
    enhanced = process(load_image(image), cfg)
    enhanced_bytes = encode_image(enhanced, image)

    scratch = image.parent / f"{image.stem}__preproc_{cycle}_{attempt}{image.suffix}"
    payload: dict | None = None
    failed: list[str] = []
    html: str | None = None
    markdown = ""
    json_block: dict
    md_block: dict
    try:
        scratch.write_bytes(enhanced_bytes)

        # --- Pass 1: structured JSON (strict, but never refuses a payload). ---
        try:
            payload, failed = extract_invoice_lenient(scratch)
            json_block = {
                "success": not failed,
                "status": "ok" if not failed else "errors_marked",
                "failed_fields": failed,
            }
        except FileNotFoundError:
            raise
        except Exception as exc:  # noqa: BLE001 — record a failed attempt, don't crash the agent
            # Ollama unreachable / transport error: no model round-trip at all.
            json_block = _report_failure(exc)  # api_error — no payload
            json_block["output"] = None

        # --- Pass 2: HTML + Markdown, independent of the JSON pass outcome. ---
        try:
            html = extract_html_invoice(scratch)
            markdown = render_markdown(html)
            non_empty = bool(markdown.strip())
            md_block = {"success": non_empty, "status": "ok" if non_empty else "empty"}
        except FileNotFoundError:
            raise
        except Exception as exc:  # noqa: BLE001
            html = None
            md_block = _report_failure(exc)
            md_block["output_md"] = None
            md_block["output_html"] = None
    finally:
        scratch.unlink(missing_ok=True)

    quality = assess(payload, failed)
    outcome = _outcome_for(payload, quality.score, attempt)
    config_dict = dataclasses.asdict(cfg)
    store.record_attempt(
        image,
        cycle,
        attempt,
        config_dict,
        quality.score,
        quality.to_metrics_dict(),
        outcome,
        enhanced_bytes if payload is not None else None,  # BLOB contract: NULL when extraction failed
        payload,
        html,
    )
    return {
        "outcome": outcome,
        "score": quality.score,
        "config": config_dict,
        "metrics": quality.to_metrics_dict(),
        "json": json_block,
        "markdown": md_block,
    }


def _attempt_report(image: Path, cycle: int, attempt: int, result: dict) -> dict:
    """Mid-cycle answer: the attempt's numbers + a hint steering the agent."""
    report: dict = {"image": str(image), "cycle": cycle, "attempt": attempt, **result}
    if result["outcome"] == OUTCOME_RETRYABLE:
        report["hint"] = (
            f"score {result['score']:.4f} is below the {SUCCESS_THRESHOLD} pass "
            f"threshold; attempt {attempt} of {MAX_ATTEMPTS} is recorded. Inspect "
            "'json.failed_fields' and 'metrics.missing_fields', then call "
            "retry_preprocessing with a *different* knob set (changes are "
            "cumulative on this attempt's config) — remaining budget: "
            f"{MAX_ATTEMPTS - attempt} attempt(s)."
        )
    elif result["outcome"] == OUTCOME_ERROR:
        report["hint"] = (
            "the model pass failed entirely (see 'json.error') — score 0.0. "
            f"Attempt {attempt} of {MAX_ATTEMPTS} is recorded; check that the "
            "OCR service is reachable, then retry_preprocessing, or let the "
            "budget run out for manual review."
        )
    return report


def _render_markdown_guarded(html: str | None) -> str | None:
    """Re-derive markdown from a *stored* HTML copy at finalize time.

    ``render_markdown`` is deterministic on a string, but the input comes out
    of the database — a corrupt row shouldn't sink the whole finalize."""
    if html is None:
        return None
    try:
        return render_markdown(html)
    except Exception:  # noqa: BLE001
        return None


def _payload_failed_fields(payload: dict) -> list[str]:
    """Top-level fields marked ``"ERROR"`` in a lenient payload — the
    lenient pass's own failure markers, re-derived at finalize."""
    return [name for name, value in payload.items() if value == "ERROR"]


def _finalize(image: Path, store: AttemptStore, cycle: int) -> dict:
    """Close the cycle.

    Picks the best attempt of the cycle (max score, tie → earliest attempt),
    refuses to clobber a better extraction kept by an earlier cycle, and — on
    a win — writes the output files, upserts the ground-truth row, chunks the
    markdown, and promotes the winning enhanced image. Legacy report shape is
    preserved (``image/json/markdown/storage``) on top of the cycle keys."""
    attempts = store.cycle_history(image).get("attempts", [])
    if not attempts:
        raise RuntimeError(f"cycle {cycle} for {image} has no attempts to finalize")
    # attempts is attempt-ordered, so max(score, -attempt) = best score,
    # earliest attempt wins ties.
    best = max(attempts, key=lambda r: (r["score"], -r["attempt"]))
    best_score = best["score"]

    report: dict = {
        "image": str(image),
        "cycle": cycle,
        "best": {
            "attempt": best["attempt"],
            "score": best_score,
            "outcome": best["outcome"],
            "config": best["config"],
        },
    }

    # Best-preservation guard: a later cycle never overwrites an extraction
    # an earlier cycle already kept and scored at least this well.
    kept = store.best_score_before(image, cycle)
    if kept is not None and kept >= best_score:
        report["json"] = {"success": False, "status": "kept-previous", "output": None}
        report["markdown"] = {
            "success": False,
            "status": "kept-previous",
            "output_md": None,
            "output_html": None,
        }
        report["storage"] = {
            "row_id": None,
            "image": None,
            "disposition": "kept-previous",
            "chunks": 0,
            "reason": (
                f"an earlier cycle already kept an extraction scored {kept}; this "
                f"cycle's best was {best_score} — the existing row and outputs are "
                "untouched"
            ),
        }
        # Close the cycle so it can never be re-finalized on top of the kept one.
        store.finalize_best(image, cycle, best["id"], None)
        return report

    payload = best["payload"]
    html = best["html"]

    if payload is None:
        # No attempt in the cycle ever produced a payload (e.g. Ollama was
        # down the whole time): there is no ground-truth row, so nothing is
        # stored — legacy behavior, markdown/HTML outputs beside the source.
        markdown = _render_markdown_guarded(html)
        md_path = html_path = None
        if html is not None and markdown is not None:
            html_path = image.with_name(image.stem + ".html")
            md_path = image.with_name(image.stem + ".md")
            html_path.write_text(html, encoding="utf-8")
            md_path.write_text(markdown, encoding="utf-8")
        markdown_ok = md_path is not None and bool(markdown.strip())
        report["json"] = {
            "success": False,
            "status": "no_payload",
            "failed_fields": list(best["metrics"].get("missing_fields", [])),
            "output": None,
        }
        report["markdown"] = {
            "success": markdown_ok,
            "status": "ok" if markdown_ok else ("empty" if md_path is not None else "no_html"),
            "output_md": str(md_path) if md_path else None,
            "output_html": str(html_path) if html_path else None,
        }
        report["storage"] = {
            "row_id": None,
            "disposition": "none",
            "chunks": 0,
            "reason": "no ground-truth row (the JSON pass produced no payload)",
        }
        store.finalize_best(image, cycle, best["id"], None)
        return report

    # A winner: persist it, in the legacy shape.
    markdown = _render_markdown_guarded(html)
    final, disposition = _final_location_for(image)
    out_dir = final.parent

    # Outputs are written at the final destination *before* the image moves,
    # so a failed move can't strand the image next to missing outputs.
    json_path = out_dir / (image.stem + ".json")
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path = html_path = None
    if html is not None and markdown is not None:
        html_path = out_dir / (image.stem + ".html")
        md_path = out_dir / (image.stem + ".md")
        html_path.write_text(html, encoding="utf-8")
        md_path.write_text(markdown, encoding="utf-8")
    if disposition == "moved":
        shutil.move(str(image), str(final))

    failed_fields = _payload_failed_fields(payload)
    report["json"] = {
        "success": not failed_fields,
        "status": "ok" if not failed_fields else "errors_marked",
        "failed_fields": failed_fields,
        "output": str(json_path),
    }
    markdown_ok = md_path is not None and bool((markdown or "").strip())
    report["markdown"] = {
        "success": markdown_ok,
        "status": "ok" if markdown_ok else ("empty" if md_path is not None else "no_html"),
        "output_md": str(md_path) if md_path else None,
        "output_html": str(html_path) if html_path else None,
    }

    # Ground truth: one row per invoice, upserted by final image path.
    inv_store = _sqlite_store()
    row_id = inv_store.upsert_invoice(payload, str(final))
    storage_report: dict = {
        "row_id": row_id,
        "image": str(final),
        "disposition": disposition,
        "chunks": 0,
    }
    if md_path is not None and bool((markdown or "").strip()):
        inv_store.set_clean_text(row_id, clean_text(markdown))
        storage_report["chunks"] = _chroma_store().add_chunks(row_id, markdown)
    else:
        storage_report["note"] = "clean_text/Chroma skipped (no non-empty markdown)"
    report["storage"] = storage_report

    # Promote the winning enhanced image beside the final image; the losers
    # live only in the attempt table's BLOBs.
    processed_path = final.parent / f"{final.stem}_processed{final.suffix}"
    proc_bytes = store.enhanced_bytes(image, cycle, best["attempt"])
    if proc_bytes is not None:
        processed_path.write_bytes(proc_bytes)
        report["winner_image"] = str(processed_path)
        store.finalize_best(image, cycle, best["id"], str(processed_path))
    else:
        store.finalize_best(image, cycle, best["id"], None)
    return report


# --- tools ----------------------------------------------------------------------


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
def extract_invoice_from_image(image_path: str, changes: dict | None = None) -> dict:
    """Extract data from one invoice image and record attempt #1 of a fresh preprocessing cycle.

    What happens:
      1. The image is preprocessed (downscale to 2048 px, shadow removal,
         CLAHE on the lightness channel, optional unsharp-mask, small-angle
         deskew) with the knobs in ``changes`` (defaults when omitted) and the
         enhanced image goes to the OCR passes. The original is untouched.
      2. Both extraction passes run on the enhanced image: schema-validated
         JSON (failing fields become ``"ERROR"``, the payload still stands)
         and HTML + Markdown.
      3. The result is scored deterministically in [0..1] (fields present,
         line items, error markers — no model confidence) and the attempt is
         recorded in this image's cycle (max 4 attempts).

    Outcomes:
      * ``SUCCESS`` (score >= 0.5) — the cycle finalizes immediately and you
        get the full storage report (image moved into today's date folder,
        JSON upserted as a ground-truth row, Markdown chunked into Chroma,
        the winning enhanced image promoted to ``<stem>_processed.<ext>``).
      * ``RETRYABLE`` (below threshold, attempts remain) — inspect
        ``json.failed_fields`` and ``metrics``, then call
        ``retry_preprocessing`` with a different knob set.
      * A 4th attempt always finalizes the cycle with its best attempt; if
        that best fell short the outcome is ``MANUAL_REVIEW`` — review the
        output and fix the source quality.

    Knobs (all optional here; bounds are enforced at this boundary):
      * ``clahe_clip`` 1.0–4.0 (default 2.0) — CLAHE contrast strength;
        raise for low-contrast / faded prints.
      * ``clahe_tile`` 4–16, integer (default 8) — CLAHE tile grid; smaller
        tiles ≈ local contrast, larger ≈ global.
      * ``sharpen_strength`` 0.0–0.8 (default 0.0) — unsharp masking;
        amplifies sensor noise, so keep it low on grainy photos.

    Unknown knobs or out-of-range values raise ``ValueError`` with the
    allowed set.
    """
    image = Path(image_path)
    if not image.is_file():
        raise FileNotFoundError(f"image not found: {image}")
    # Knob validation at the boundary — the engine never sees garbage.
    cfg = PreprocessConfig.from_dict(changes)

    store = _attempt_store()
    try:
        cycle = _fresh_cycle_for_extract(store, image)
        result = _run_attempt(image, store, cycle, 1, cfg)
        if _should_finalize(result["outcome"], 1):
            return _finalize(image, store, cycle)
        return _attempt_report(image, cycle, 1, result)
    finally:
        store.close()


@mcp.tool()
def retry_preprocessing(image_path: str, changes: dict) -> dict:
    """Attempt #2–#4 on the image's still-open cycle, with different knobs.

    ``changes`` sets at least one of the three knobs and is applied on top of
    the previous attempt's settings (cumulative tuning — a knob you don't list
    keeps its last value). Outcomes are the same as the extract call: the
    report shows this attempt's score, metrics, and outcome; a ``SUCCESS``
    (or the 4th attempt, whichever comes first) finalizes the cycle with its
    best attempt and returns the full storage report.

    Refuses: an empty ``changes`` (a retry that changes nothing is a no-op),
    an out-of-range / unknown knob, an image with no open cycle (call
    ``extract_invoice_from_image`` first), or an exhausted budget (call
    ``get_attempt_history`` to see what happened, then extract a fresh cycle).
    """
    image = Path(image_path)
    if not changes:
        raise ValueError(f"changes must set at least one knob — allowed: {sorted(FIELD_RANGES)}")

    store = _attempt_store()
    try:
        # Cycle-state checks come first: on an open cycle the source still
        # sits in the work dir, but on a *finalized* one it was already
        # moved — the refusal must say the budget's gone, not "file missing".
        cycle, attempt = store.next_attempt(image)
        if attempt > MAX_ATTEMPTS:
            raise RuntimeError(
                f"cycle {cycle} for {image} already has {MAX_ATTEMPTS} attempts — "
                "the budget is exhausted; call get_attempt_history to inspect the "
                "cycle, then extract_invoice_from_image to start a fresh one"
            )
        if not image.is_file():
            raise FileNotFoundError(f"image not found: {image}")
        # Cumulative tuning: the last recorded attempt's knobs + this diff.
        attempts = store.cycle_history(image).get("attempts", [])
        last_config = attempts[-1]["config"] if attempts else None
        cfg = PreprocessConfig.from_dict(last_config).with_overrides(changes)
        result = _run_attempt(image, store, cycle, attempt, cfg)
        if _should_finalize(result["outcome"], attempt):
            return _finalize(image, store, cycle)
        return _attempt_report(image, cycle, attempt, result)
    finally:
        store.close()


@mcp.tool()
def get_attempt_history(image_path: str, max_attempts: int = 20) -> dict:
    """The latest preprocessing cycle for an image, for inspection and tuning.

    Returns every attempt's knob set, score, metrics, and outcome (in attempt
    order, up to ``max_attempts`` newest), plus who won (``best_attempt_id``)
    and the promoted winner image (``best_processed_path``). The image doesn't
    have to exist on disk anymore — a moved source still resolves to its cycle.

    When the image has never been through extraction the result is
    ``{"image": ..., "cycle": null, "attempts": []}``.
    """
    image = Path(image_path)
    if not isinstance(max_attempts, int) or isinstance(max_attempts, bool) or max_attempts < 1:
        raise ValueError(f"max_attempts must be a positive integer, got {max_attempts!r}")

    store = _attempt_store()
    try:
        hist = store.cycle_history(image)
        if not hist:
            return {"image": str(image), "cycle": None, "attempts": []}
        return {
            "image": str(image),
            "cycle": hist["cycle"],
            "attempts": hist["attempts"][-max_attempts:],
            "best_attempt_id": hist["best_attempt_id"],
            "original_image_path": hist["original_image_path"],
            "best_processed_path": hist["best_processed_path"],
        }
    finally:
        store.close()


def main() -> None:
    """Start the MCP server over stdio."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
