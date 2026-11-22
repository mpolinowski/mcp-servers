"""Offline tests for the invoice_extractor MCP server + agent adapter + storage.

No Ollama, no GPU: ``generate()`` is monkeypatched in both extraction
modules and the embeddings factory is swapped for a deterministic fake, so
every path (strict JSON, lenient ERROR, HTML/MD success, Ollama-down,
re-ingest, collision, storage wiring) can be exercised deterministically.

The extraction tool moves ingested photos into a per-day folder under the
assets dir, so the tool's directories are sandboxed before any test runs:
nothing inside the real ``assets/`` or ``dbs/`` is touched.

Run from the project root:

    uv run python tests/test_offline.py
"""

import asyncio
import json
import math
import shutil
import sys
import tempfile
import traceback
import zlib
from datetime import date
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from invoice_extractor import tools  # noqa: F401  (ensure package import works)
from invoice_extractor.tools import invoice_html, invoice_json
from invoice_extractor.tools import main as main_mod
from invoice_extractor.tools.main import (  # noqa: E402
    extract_invoice_from_image,
    list_invoices,
    mcp,
)
import invoice_extractor.agent as agent_mod  # noqa: E402
from invoice_extractor.storage import (  # noqa: E402
    ChromaStore,
    InvoiceStore,
    bm25_search,
    clean_text,
    schema_columns,
    tokenize,
)

from fastmcp.client import Client, FastMCPTransport  # noqa: E402
from fastmcp.exceptions import ToolError  # noqa: E402

_ASSETS_DIR = Path(__file__).resolve().parent.parent / "src" / "invoice_extractor" / "assets"
# Any fixture image works — content is irrelevant because generate() is
# monkeypatched.
ASSET = sorted(_ASSETS_DIR.rglob("*.jpg"))[0]

# --- sandbox -----------------------------------------------------------------
# main's real dirs live under the package; redirect them into a throwaway
# tree before the first tool call, and give it a deterministic, offline
# embeddings factory.
_SANDBOX = Path(tempfile.mkdtemp(prefix="invoice_extractor_test_"))
MAIN_ASSETS = _SANDBOX / "assets"
MAIN_ASSETS.mkdir()
main_mod._ASSETS_DIR = MAIN_ASSETS
main_mod._DB_DIR = _SANDBOX / "dbs"


class FakeEmbeddings:
    """Deterministic offline embeddings: normalized one-hot word-bag over
    16 dims, bucketed by crc32. No network, no model — just enough structure
    for ranking tests.

    langchain_core 1.x requires BOTH ``embed_documents`` and ``embed_query``
    (singular) on every Embeddings implementation."""

    _DIM = 16

    def _embed(self, words: list[str]) -> list[float]:
        v = [0.0] * self._DIM
        for w in words:
            v[zlib.crc32(w.encode()) % self._DIM] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts: list[str]):
        return [self._embed(t.lower().split()) for t in texts]

    def embed_query(self, text: str):
        return self._embed(text.lower().split())


main_mod._embeddings = lambda: FakeEmbeddings()

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [ok] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {detail}")


def date_dir() -> Path:
    """The per-day ingest folder the tool moves photos into."""
    return MAIN_ASSETS / date.today().isoformat()


def _cleanup(workdir: Path, stem: str) -> None:
    """Drop a test's artifacts wherever the tool may have landed them."""
    for suffix in (".jpg", ".json", ".md", ".html"):
        (workdir / (stem + suffix)).unlink(missing_ok=True)
        (date_dir() / (stem + suffix)).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Test fixtures: fake model outputs
# ---------------------------------------------------------------------------

VALID_JSON = """```json
{"invoice_number": "INV-2024-0847",
 "invoice_date": "2025-07-29", "due_date": "2025-08-28", "currency": "USD",
 "bill_from": {"name": "Northwind Supplies", "email": "sales@northwind.example"},
 "bill_to": {"name": "Contoso Ltd"},
 "line_items": [
    {"description": "Web Development", "quantity": "40", "unit": "hours", "unit_price": "$150.00", "amount": "$6,000.00"},
    {"description": "Design", "quantity": "20", "unit": "hours", "unit_price": "$125.00", "amount": "$2,500.00"},
    {"description": "Consulting", "quantity": "8", "unit": "hours", "unit_price": "$100.00", "amount": "$800.00"},
    {"description": "Support", "quantity": "1", "unit": "month", "unit_price": "$250.00", "amount": "$250.00"}
  ],
 "subtotal": "$9,550.00", "tax_rate": 8.5, "tax_amount": "$811.75", "total": "$10,361.75",
 "payment_terms": "Net 30",
 "payment_methods": ["wire", "card"],
 "notes": "Thank you"}
```"""

BAD_FIELD_JSON = """{
  "invoice_number": "INV-2024-0847",
  "line_items": [
    {"description": "Web Development", "quantity": "ABC", "unit_price": "150.00", "amount": "6000.00"}
  ],
  "subtotal": "not-a-number",
  "total": "10361.75"
}"""

GARBAGE_TEXT = "Sorry, I cannot read this document. It looks like a photo of a desk."

FALLBACK_TEXT = (
    "NORTHWIND SUPPLIES\nINVOICE INV-2024-0847\n"
    "Description Quantity Rate Amount\n"
    "Web Development 40 150.00 6000.00\n"
    "Subtotal 9550.00"
)
FALLBACK_TABLE = (
    "<table><thead><tr><th>Description</th><th>Quantity</th><th>Rate</th><th>Amount</th></tr></thead>"
    "<tbody><tr><td>Web Development</td><td>40</td><td>150.00</td><td>6000.00</td></tr></tbody>"
    "<tfoot><tr><td>Subtotal</td><td></td><td></td><td>9550.00</td></tr></tfoot></table>"
)


def _patch_generates(json_response: object, html_response: object) -> None:
    """Point both extraction modules at fixed responses (or an exception class)."""

    def make(response):
        def fake(prompt: str, image_base64: str):
            if isinstance(response, Exception):
                raise response
            return response
        return fake

    invoice_json.generate = make(json_response)
    invoice_html.generate = make(html_response)


def _down_response() -> Exception:
    return requests.ConnectionError("connection refused (simulated)")


def json_generate_only(response):
    """Patch only invoice_json.generate (to a fixed response); patch
    invoice_html.generate to raise ConnectionError (simulates Ollama down
    for the HTML pass)."""
    original_json = invoice_json.generate
    original_html = invoice_html.generate
    _patch_generates(response, _down_response())
    return lambda: (setattr(invoice_json, "generate", original_json),
                    setattr(invoice_html, "generate", original_html))


def html_generate_only(response):
    """JSON pass down, HTML pass returning a fixed response (dispatched by
    prompt so both Text and Table calls succeed)."""
    original_json = invoice_json.generate
    original_html = invoice_html.generate
    invoice_json.generate = lambda prompt, b64: (_ for _ in ()).throw(
        requests.ConnectionError("json pass down (simulated)"))
    invoice_html.generate = lambda prompt, b64: (
        FALLBACK_TABLE if prompt == invoice_html.TABLE_PROMPT else response)
    return lambda: (setattr(invoice_json, "generate", original_json),
                    setattr(invoice_html, "generate", original_html))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_list_invoices() -> None:
    print("\n== list_invoices (empty sandbox) ==")
    result = list_invoices()
    check("returns a dict with images", isinstance(result, dict) and "images" in result)
    check("empty sandbox: count 0", result["count"] == 0, str(result))
    check("empty sandbox: no images", result["images"] == [])
    check("reports sandbox assets dir", result["assets_dir"] == str(MAIN_ASSETS),
          str(result.get("assets_dir")))


def test_strict_json_success(workdir: Path) -> None:
    print("\n== extract: strict JSON success ==")
    img = workdir / "copy.jpg"
    shutil.copy(ASSET, img)
    restore = json_generate_only(VALID_JSON)
    try:
        report = extract_invoice_from_image(str(img))
    finally:
        restore()

    j = report["json"]
    check("json pass success", j["success"] is True, f"got: {j}")
    check("json status ok", j["status"] == "ok")
    check("json failed_fields empty", j["failed_fields"] == [])

    final_img = date_dir() / "copy.jpg"
    check("photo moved into per-day folder",
          final_img.is_file() and not img.exists(), f"final: {final_img} source: {img}")

    json_out = Path(j["output"])
    check("json output at final location", json_out == final_img.with_suffix(".json"),
          str(json_out))
    check("json file written", json_out.is_file())

    data = json.loads(json_out.read_text())
    check("total coerced to float", data["total"] == 10361.75, f"got: {data.get('total')!r}")
    check("subtotal coerced", data["subtotal"] == 9550.0, f"got: {data.get('subtotal')!r}")
    check("tax golden 811.75", data["tax_amount"] == 811.75, f"got: {data.get('tax_amount')!r}")
    check("4 line items", len(data["line_items"]) == 4, f"got: {len(data['line_items'])}")
    first = data["line_items"][0]
    check("first item quantity coerced", first["quantity"] == 40.0, f"got: {first}")
    check("first item amount 6000.0", first["amount"] == 6000.0, f"got: {first}")

    check("markdown pass failed (simulated down)", report["markdown"]["success"] is False)
    check("markdown status api_error", report["markdown"]["status"] == "api_error",
          f"got: {report['markdown']}")

    s = report["storage"]
    check("storage: disposition moved", s["disposition"] == "moved", str(s))
    check("storage: row id is int", isinstance(s["row_id"], int), str(s))
    check("storage: reports final image", Path(s["image"]) == final_img, s["image"])
    check("storage: chunks 0 (no markdown)", s["chunks"] == 0, str(s))
    check("storage: skip note present", "skipped" in s.get("note", ""), str(s))

    store = main_mod._sqlite_store()
    row = store.get(s["row_id"])
    check("row exists", row is not None, f"row_id={s['row_id']}")
    check("row: bill_from flattened", row["bill_from_name"] == "Northwind Supplies",
          f"got: {row.get('bill_from_name')!r}")
    check("row: bill_to flattened", row["bill_to_name"] == "Contoso Ltd",
          f"got: {row.get('bill_to_name')!r}")
    check("row: line_items JSON round-trip",
          json.loads(row["line_items"])[0]["amount"] == 6000.0, str(row.get("line_items")))
    check("row: total stored as float", row["total"] == 10361.75, f"got: {row.get('total')!r}")
    check("row: invoice_date", row["invoice_date"] == "2025-07-29", str(row.get("invoice_date")))
    check("row: due_date", row["due_date"] == "2025-08-28", str(row.get("due_date")))
    check("row: clean_text None (no markdown)", row["clean_text"] is None,
          f"got: {row.get('clean_text')!r}")
    check("row: image path stored resolved",
          Path(row["image_path"]).resolve() == final_img.resolve(), row["image_path"])
    hit = store.get_by_image_path(final_img)
    check("lookup by final image path", hit is not None and hit["id"] == s["row_id"],
          f"got: {hit}")
    _cleanup(workdir, "copy")


def test_lenient_error_markers(workdir: Path) -> None:
    print("\n== extract: validation failure -> ERROR markers, file still written ==")
    img = workdir / "bad.jpg"
    shutil.copy(ASSET, img)
    restore = json_generate_only(BAD_FIELD_JSON)
    try:
        report = extract_invoice_from_image(str(img))
    finally:
        restore()

    j = report["json"]
    check("json pass flagged failed", j["success"] is False, f"got: {j}")
    check("status errors_marked", j["status"] == "errors_marked", f"got: {j}")
    check("failed paths line up",
          "line_items.0.quantity" in j["failed_fields"] and "subtotal" in j["failed_fields"],
          f"got: {j['failed_fields']}")

    data = json.loads(Path(j["output"]).read_text())
    check("failing value replaced", data["line_items"][0]["quantity"] == "ERROR",
          f"got: {data['line_items'][0]}")
    check("second failing value replaced", data["subtotal"] == "ERROR",
          f"got: {data['subtotal']!r}")
    check("valid fields preserved", data["invoice_number"] == "INV-2024-0847")
    # On the failure path we return the raw model output (not a coerced dump),
    # so valid-but-uncoerced values stay as the model emitted them (strings).
    check("valid field value preserved raw", str(data["total"]) == "10361.75",
          f"got: {data.get('total')!r}")

    s = report["storage"]
    check("storage: row created despite failures", isinstance(s["row_id"], int), str(s))
    check("storage: disposition moved", s["disposition"] == "moved", str(s))
    check("photo moved despite failures",
          (date_dir() / "bad.jpg").is_file() and not img.exists())

    row = main_mod._sqlite_store().get(s["row_id"])
    check("row: failing field stored as ERROR string", row["subtotal"] == "ERROR",
          f"got: {row.get('subtotal')!r}")
    check("row: failing line-item stored as ERROR",
          json.loads(row["line_items"])[0]["quantity"] == "ERROR",
          str(row.get("line_items")))
    check("row: image path = final",
          Path(row["image_path"]).resolve() == (date_dir() / "bad.jpg").resolve(),
          row["image_path"])
    _cleanup(workdir, "bad")


def test_garbage_text_no_json(workdir: Path) -> None:
    print("\n== extract: model returns prose (no JSON object) ==")
    img = workdir / "garbage.jpg"
    shutil.copy(ASSET, img)
    restore = json_generate_only(GARBAGE_TEXT)
    try:
        report = extract_invoice_from_image(str(img))
    finally:
        restore()

    j = report["json"]
    check("json pass failed", j["success"] is False, f"got: {j}")
    check("status errors_marked", j["status"] == "errors_marked", f"got: {j}")
    data = json.loads(Path(j["output"]).read_text())
    check("all top-level fields flagged",
          all(v == "ERROR" for v in data.values()) and len(data) >= 3, f"got: {data}")
    check("required fields present in failed list",
          "invoice_number" in j["failed_fields"] and "line_items" in j["failed_fields"],
          f"got: {j['failed_fields']}")

    # A payload DID come back, so the pipeline still stores it and moves the
    # photo — the ERROR markers are the ground truth, not an extraction death.
    s = report["storage"]
    check("storage: row created for all-ERROR payload", isinstance(s["row_id"], int), str(s))
    check("storage: disposition moved", s["disposition"] == "moved", str(s))
    check("photo moved", (date_dir() / "garbage.jpg").is_file() and not img.exists())
    row = main_mod._sqlite_store().get(s["row_id"])
    check("row: total stored as ERROR", row["total"] == "ERROR", f"got: {row.get('total')!r}")
    _cleanup(workdir, "garbage")


def test_ollama_down_entirely(workdir: Path) -> None:
    print("\n== extract: Ollama unreachable on both passes ==")
    img = workdir / "down.jpg"
    shutil.copy(ASSET, img)

    original_json = invoice_json.generate
    original_html = invoice_html.generate
    _patch_generates(_down_response(), _down_response())
    try:
        report = extract_invoice_from_image(str(img))  # must not raise
    finally:
        setattr(invoice_json, "generate", original_json)
        setattr(invoice_html, "generate", original_html)

    check("json api_error", report["json"]["status"] == "api_error", f"got: {report['json']}")
    check("markdown api_error", report["markdown"]["status"] == "api_error",
          f"got: {report['markdown']}")
    check("no json file written", not (workdir / "down.json").exists(),
          str(list(workdir.glob('*'))))
    check("no html/md written",
          not (workdir / "down.html").exists() and not (workdir / "down.md").exists())
    check("has error message", "simulated" in report["json"].get("error", ""),
          str(report["json"]))

    # No payload -> nothing stored, and the photo is NOT moved.
    check("photo stays at source",
          img.exists() and not (date_dir() / "down.jpg").exists())
    s = report["storage"]
    check("storage: no row", s["row_id"] is None, str(s))
    check("storage: disposition none", s["disposition"] == "none", str(s))
    check("storage: reason given", "no ground-truth row" in s.get("reason", ""), str(s))
    check("storage: zero chunks", s["chunks"] == 0, str(s))
    _cleanup(workdir, "down")


def test_html_md_success(workdir: Path) -> None:
    print("\n== extract: JSON pass down, HTML/Markdown pass succeeds ==")
    img = workdir / "html_only.jpg"
    shutil.copy(ASSET, img)

    restore = html_generate_only(FALLBACK_TEXT)
    try:
        report = extract_invoice_from_image(str(img))
    finally:
        restore()

    check("json api_error", report["json"]["status"] == "api_error",
          f"got: {report['json']}")
    m = report["markdown"]
    check("markdown success", m["success"] is True, f"got: {m}")
    check("md written beside source", Path(m["output_md"]) == img.with_suffix(".md"),
          str(m["output_md"]))
    check("html written beside source", Path(m["output_html"]) == img.with_suffix(".html"),
          str(m["output_html"]))
    md_text = Path(m["output_md"]).read_text()
    check("md has pipe table", "| Web Development | 40 | 150.00 | 6000.00 |" in md_text,
          f"got: {md_text[:200]!r}")
    html_text = Path(m["output_html"]).read_text()
    check("html has <table>", "<table" in html_text)

    # No JSON payload -> markdown alone doesn't make a ground-truth row.
    s = report["storage"]
    check("storage: no row without payload", s["row_id"] is None, str(s))
    check("storage: disposition none", s["disposition"] == "none", str(s))
    check("photo NOT moved (no ground truth)",
          img.exists() and not (date_dir() / "html_only.jpg").exists())
    _cleanup(workdir, "html_only")


def test_reingest_updates_in_place(workdir: Path) -> None:
    print("\n== extract: re-ingestion updates the same row + chunks in place ==")
    img = workdir / "re.jpg"
    shutil.copy(ASSET, img)

    # Phase 1: JSON pass works, HTML down. Row gets created, no chunks.
    restore = json_generate_only(VALID_JSON)
    try:
        r1 = extract_invoice_from_image(str(img))
    finally:
        restore()
    s1 = r1["storage"]
    check("phase 1: moved", s1["disposition"] == "moved", str(s1))
    row_id1 = s1["row_id"]
    check("phase 1: row created", isinstance(row_id1, int), str(s1))
    check("phase 1: chunks 0 (html down)", s1["chunks"] == 0, str(s1))
    moved = date_dir() / "re.jpg"
    check("phase 1: photo in per-day folder", moved.is_file() and not img.exists())

    # Phase 2: both passes work, on the already-ingested photo.
    def good_html(prompt: str, image_base64: str) -> str:
        return FALLBACK_TABLE if prompt == invoice_html.TABLE_PROMPT else FALLBACK_TEXT

    original_json = invoice_json.generate
    original_html = invoice_html.generate
    invoice_json.generate = lambda prompt, b64: VALID_JSON
    invoice_html.generate = good_html
    try:
        r2 = extract_invoice_from_image(str(moved))
    finally:
        setattr(invoice_json, "generate", original_json)
        setattr(invoice_html, "generate", original_html)

    check("phase 2: json ok", r2["json"]["success"] is True, str(r2["json"]))
    check("phase 2: markdown ok", r2["markdown"]["success"] is True, str(r2["markdown"]))
    s2 = r2["storage"]
    check("phase 2: disposition in-place", s2["disposition"] == "in-place", str(s2))
    check("phase 2: SAME row id (stable identity)", s2["row_id"] == row_id1,
          f"{s2['row_id']} != {row_id1}")
    check("phase 2: chunks written", s2["chunks"] >= 1, str(s2))
    check("outputs at final location",
          Path(r2["markdown"]["output_md"]).parent == date_dir(),
          str(r2["markdown"]["output_md"]))

    row = main_mod._sqlite_store().get(row_id1)
    check("row: clean_text set on re-ingest",
          bool((row.get("clean_text") or "").strip()), f"got: {row.get('clean_text')!r}")

    hits = main_mod._chroma_store().search("Web Development invoice", k=3)
    check("chroma: has hits", len(hits) >= 1, f"got: {hits}")
    if hits:
        check("chroma: top hit is the row", hits[0]["invoice_id"] == row_id1, str(hits[:2]))
        check("chroma: hit shape", set(hits[0]) == {"invoice_id", "text", "score"},
              str(sorted(hits[0])))
        check("chroma: hit text non-empty", bool(hits[0]["text"].strip()))
    _cleanup(workdir, "re")


def test_collision_keeps_source(workdir: Path) -> None:
    print("\n== extract: name collision keeps photo + outputs at source ==")
    stem = "clash"
    dd = date_dir()
    dd.mkdir(parents=True, exist_ok=True)
    foreign = dd / f"{stem}.jpg"
    foreign.write_bytes(b"foreign-bytes")

    img = workdir / f"{stem}.jpg"
    shutil.copy(ASSET, img)
    restore = json_generate_only(VALID_JSON)
    try:
        report = extract_invoice_from_image(str(img))
    finally:
        restore()

    s = report["storage"]
    check("disposition kept-source", s["disposition"] == "kept-source", str(s))
    check("foreign file intact", foreign.read_bytes() == b"foreign-bytes")
    check("source photo untouched", img.is_file())
    check("json output beside source",
          Path(report["json"]["output"]) == workdir / "clash.json",
          str(report["json"]["output"]))
    check("row still created", isinstance(s["row_id"], int), str(s))
    row = main_mod._sqlite_store().get(s["row_id"])
    check("row path points at the SOURCE photo",
          Path(row["image_path"]).resolve() == img.resolve(), row["image_path"])
    _cleanup(workdir, stem)


def test_agent_adapter_bridge(workdir: Path) -> None:
    print("\n== agent adapter: in-process MCP bridge (list_tools -> _build_tool -> call) ==")

    async def _discover():
        # The exact transport the agent uses: FastMCP Client over the in-process server.
        async with Client(FastMCPTransport(agent_mod.mcp)) as client:
            return await client.list_tools()

    defs = asyncio.run(_discover())
    names = [d.name for d in defs]
    check("bridge: both tools exposed",
          {"list_invoices", "extract_invoice_from_image"} <= set(names), str(names))
    extract_def = next(d for d in defs if d.name == "extract_invoice_from_image")
    tool = agent_mod._build_tool(extract_def)
    check("tool: bound as StructuredTool", tool.name == "extract_invoice_from_image")
    check("tool: sync + async entrypoints",
          tool.func is not None and tool.coroutine is not None)
    check("tool: input schema knows image_path",
          "image_path" in tool.args_schema.model_fields,
          str(list(tool.args_schema.model_fields)))

    img = workdir / "adapter.jpg"
    shutil.copy(ASSET, img)

    # Phase 1: one extraction through the bridge (sync entrypoint), JSON pass only.
    restore = json_generate_only(VALID_JSON)
    try:
        r1 = tool.func(image_path=str(img))
    finally:
        restore()
    check("bridge: report comes back as a dict", isinstance(r1, dict), f"got: {r1!r}")
    if not isinstance(r1, dict):
        _cleanup(workdir, "adapter")
        return

    check("bridge: json pass ok", r1["json"]["success"] is True, f"got: {r1.get('json')}")
    check("bridge: photo moved into per-day folder",
          (date_dir() / "adapter.jpg").is_file() and not img.exists())
    check("bridge: storage reports moved", r1["storage"]["disposition"] == "moved",
          str(r1.get("storage")))
    row_id1 = r1["storage"]["row_id"]

    # Phase 2: re-extraction at the MOVED location -> same row, in place.
    # (The original path is gone — the tool moves the photo — so the second
    # call must target the date-folder copy, exactly like the re-ingest test.)
    restore = json_generate_only(VALID_JSON)
    try:
        r2 = tool.func(image_path=str(date_dir() / "adapter.jpg"))
    finally:
        restore()
    check("bridge: re-extract via bridge ok",
          isinstance(r2, dict) and r2["json"]["success"] is True, f"got: {r2!r}")
    if isinstance(r2, dict) and "storage" in r2:
        check("bridge: in-place disposition", r2["storage"]["disposition"] == "in-place",
              str(r2["storage"]))
        check("bridge: same row id across bridge calls", r2["storage"]["row_id"] == row_id1,
              f"{r2['storage']['row_id']} != {row_id1}")

    # A source path that no longer holds the photo must come back as an error
    # (FileNotFoundError "image not found: ..."), never as a silent report.
    try:
        r3 = tool.func(image_path=str(img))
        missing_ok = (isinstance(r3, dict) and r3.get("is_error") is True) or (
            isinstance(r3, str) and ("not found" in r3.lower() or "error" in r3.lower())
        )
        check("bridge: moved-away source returns an error", missing_ok, f"got: {r3!r}")
    except (FileNotFoundError, ToolError) as e:
        # The bridge surfaces the tool's FileNotFoundError as a ToolError.
        check("bridge: moved-away source errors out", "not found" in str(e).lower(), str(e))
    _cleanup(workdir, "adapter")


# ---------------------------------------------------------------------------
# Storage layer (unit-level: no tool wiring, no tool directories)
# ---------------------------------------------------------------------------


def test_storage_layer() -> None:
    print("\n== storage layer: extract_lenient / InvoiceStore / BM25 / ChromaStore ==")
    tmp = Path(tempfile.mkdtemp(prefix="invoice_storage_test_"))

    # --- extract_lenient: pure function, no model round-trip ------------------
    payload, failed = invoice_json.extract_lenient(VALID_JSON)
    check("lenient: valid -> no failed fields", failed == [], str(failed))
    check("lenient: valid -> total coerced to float", payload["total"] == 10361.75,
          f"got: {payload.get('total')!r}")
    check("lenient: valid -> 4 line items", len(payload["line_items"]) == 4, str(payload))

    payload, failed = invoice_json.extract_lenient(BAD_FIELD_JSON)
    check("lenient: bad -> subtotal marked", payload["subtotal"] == "ERROR", f"got: {payload}")
    check("lenient: bad -> nested quantity marked",
          payload["line_items"][0]["quantity"] == "ERROR", f"got: {payload['line_items']}")
    check("lenient: bad -> valid values kept raw (no partial coercion)",
          payload["total"] == "10361.75", f"got: {payload.get('total')!r}")
    check("lenient: bad -> failed paths sorted+unique",
          failed == ["line_items.0.quantity", "subtotal"], str(failed))

    payload, failed = invoice_json.extract_lenient(GARBAGE_TEXT)
    check("lenient: no-JSON -> every field ERROR",
          all(v == "ERROR" for v in payload.values()), f"got: {payload}")
    check("lenient: no-JSON -> failed == all model fields",
          sorted(failed) == sorted(invoice_json.Invoice.model_fields), str(failed))

    # --- InvoiceStore: identity, flattening, clean text -----------------------
    store_path = tmp / "nested" / "invoices.sqlite"
    store = InvoiceStore(store_path)
    check("store: db file created (parents included)", store_path.is_file(), str(store_path))

    cols = [c for c, _ in schema_columns()]
    check("schema: core columns present",
          {"invoice_number", "line_items", "total"} <= set(cols), str(cols))
    check("schema: parties flattened to parent_field",
          {"bill_from_name", "bill_to_name"} <= set(cols), str(cols))
    types = dict(schema_columns())
    check("schema: total REAL, line_items TEXT",
          types.get("total") == "REAL" and types.get("line_items") == "TEXT", str(types))

    valid_payload = invoice_json.extract_lenient(VALID_JSON)[0]
    img_a = tmp / "a.jpg"; img_a.touch()
    img_b = tmp / "b.jpg"; img_b.touch()
    id_a = store.upsert_invoice(valid_payload, img_a)
    id_b = store.upsert_invoice(valid_payload, img_b)
    check("store: fresh rows get fresh sequential ids", (id_a, id_b) == (1, 2),
          f"got: {(id_a, id_b)}")

    again_a = store.upsert_invoice(valid_payload, img_a)
    check("store: re-upsert same path -> same id (stable identity)", again_a == id_a,
          f"{again_a} != {id_a}")
    check("store: re-upsert did not duplicate rows",
          store.get_by_image_path(img_a)["id"] == id_a
          and store.get_by_image_path(img_b)["id"] == id_b)
    check("store: path that was never ingested -> None",
          store.get_by_image_path(tmp / "nope.jpg") is None)
    check("store: unknown id -> None", store.get(9999) is None)

    text_a = clean_text("Northwind invoice 40 hours wire payment net 30")
    text_b = clean_text("Contoso design retainer, card accepted, monthly billing")
    check("clean_text: drops nothing it shouldn't",
          "wire" in text_a.split() and "net" in text_a.split(), str(text_a))
    store.set_clean_text(id_a, text_a)
    store.set_clean_text(id_b, text_b)
    rows = list(store.all_rows())
    check("store: all_rows -> ordered (id, clean_text) pairs",
          rows == [(id_a, text_a), (id_b, text_b)], str(rows))

    # --- BM25 over the cleaned-text corpus -------------------------------------
    corpus = list(store.all_rows())
    hits = bm25_search(corpus, "wire payment net 30")
    check("bm25: doc A ranks first for its terms", hits and hits[0]["row_id"] == id_a,
          str(hits))
    check("bm25: hit shape {row_id, score}",
          bool(hits) and set(hits[0]) == {"row_id", "score"}, str(hits[:1]))
    hits_b = bm25_search(corpus, "design retainer card accepted")
    check("bm25: distinct doc ranks first for its terms",
          hits_b and hits_b[0]["row_id"] == id_b, str(hits_b))
    check("bm25: stopword-only query -> no hits",
          bm25_search(corpus, "the of and but") == [])
    check("bm25: empty corpus -> no hits", bm25_search([], "wire payment") == [])

    # --- ChromaStore: append-only chunks keyed by row id -----------------------
    chroma = ChromaStore(tmp / "chroma", FakeEmbeddings(), chunk_size=120, chunk_overlap=0)
    check("chroma: empty store -> zero hits", chroma.search("zebra northwind", k=3) == [])
    check("chroma: empty store -> count 0", chroma.collection_count() == 0)

    long_md = "\n\n".join(
        f"## Section {i}\n\nnorthwind invoice section {i} zebra token" for i in range(8)
    )
    n_a = chroma.add_chunks(id_a, long_md)
    check("chroma: long markdown splits into several chunks", n_a >= 2, f"got {n_a}")
    check("chroma: collection holds exactly those chunks",
          chroma.collection_count() == n_a, f"count {chroma.collection_count()} != {n_a}")

    hits = chroma.search("zebra northwind invoice", k=3)
    check("chroma: search returns hits", len(hits) >= 1, str(hits))
    if hits:
        check("chroma: hit keys {invoice_id, text, score}",
              set(hits[0]) == {"invoice_id", "text", "score"}, str(hits[0]))
        check("chroma: top hit is row A", hits[0]["invoice_id"] == id_a, str(hits[:2]))
        check("chroma: text non-empty, score is a float",
              bool(hits[0]["text"].strip()) and isinstance(hits[0]["score"], float),
              str(hits[0]))

    chroma.add_chunks(id_b, "## beta\n\ncontoso beta row content")
    check("chroma: other rows accumulate without touching row A",
          chroma.collection_count() == n_a + 1, f"got {chroma.collection_count()}")

    # Re-add row A with a SHORTER doc: its old chunks must be pruned, not kept.
    short_a = "## revised\n\nnorthwind revised single chunk"
    n_a2 = chroma.add_chunks(id_a, short_a)
    check("chroma: re-add replaces own chunks (stale pruned)",
          chroma.collection_count() == n_a2 + 1,
          f"got {chroma.collection_count()}, expected {n_a2 + 1}")
    check("chroma: idempotent re-add of same content",
          chroma.add_chunks(id_a, short_a) == n_a2
          and chroma.collection_count() == n_a2 + 1)
    check("chroma: whitespace markdown -> 0 chunks", chroma.add_chunks(id_b, "   \n\t  ") == 0)
    check("chroma: row B left intact", chroma.collection_count() == n_a2 + 1)

    print(f"\n    (storage sandbox: {tmp})")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    """Run every test in order, report totals, exit non-zero on any failure."""
    workdir = Path(tempfile.mkdtemp(prefix="invoice_extractor_run_"))
    print(f"sandbox:  {_SANDBOX}")
    print(f"workdir:  {workdir}")
    # test_list_invoices asserts an empty sandbox, so it runs first.
    plan = [
        test_list_invoices,
        lambda: test_strict_json_success(workdir),
        lambda: test_lenient_error_markers(workdir),
        lambda: test_garbage_text_no_json(workdir),
        lambda: test_ollama_down_entirely(workdir),
        lambda: test_html_md_success(workdir),
        lambda: test_reingest_updates_in_place(workdir),
        lambda: test_collision_keeps_source(workdir),
        lambda: test_agent_adapter_bridge(workdir),
        test_storage_layer,
    ]
    for step in plan:
        try:
            step()
        except Exception:  # noqa: BLE001 -- one blown test must not hide the rest.
            global FAIL
            FAIL += 1
            print(f"  [FAIL] {step.__name__} raised:")
            traceback.print_exc()

    print(f"\n{'=' * 50}")
    print(f"RESULT: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
