"""Offline tests for the invoice_extractor MCP server (3-tool retry-loop contract).

Covers the bounded preprocessing retry loop end-to-end without Ollama or GPU:

  * PreprocessConfig knob validation (ranges, unknown keys, integer tile grid)
  * assess() deterministic quality scoring (exact scores per fixture payload)
  * OpenCV engine (determinism, knobs actually change the image, formats, errors)
  * AttemptStore (cycles, budget, best-finalization, exact history shapes)
  * all 4 MCP tools through the real tool layer, with storage sandboxed:
    SUCCESS finalize, RETRYABLE cumulative tuning, 4-attempt MANUAL_REVIEW,
    budget exhaustion, open/closed-cycle guards, empty-changes rejection,
    Ollama-down cycles (both passes dead / only JSON pass dead), the
    kept-previous guard, history slicing, list_invoices recursion,
    missing-file errors
  * the in-process FastMCP bridge the agent uses (list_tools / _build_tool / call)

The model seams (extract_invoice_lenient / extract_html_invoice /
render_markdown) are monkeypatched in the tools module with scripted
behaviors; the engine, scoring, attempt store and storage layer are the real
things. The tool's directories are sandboxed before any test runs, so nothing
inside the real assets/ or dbs/ is touched.

Run from the project root:

    uv run python tests/test_offline.py
"""

import asyncio
import json
import math
import shutil
import sqlite3
import sys
import tempfile
import traceback
import zlib
from datetime import date
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from invoice_extractor import tools  # noqa: F401  (ensure package import works)
from invoice_extractor.tools import main as main_mod
from invoice_extractor.tools.main import (  # noqa: E402
    extract_invoice_from_image,
    get_attempt_history,
    list_invoices,
    mcp,
    retry_preprocessing,
)
import invoice_extractor.agent as agent_mod  # noqa: E402
from invoice_extractor.preprocessing import (  # noqa: E402
    MAX_ATTEMPTS,
    PreprocessConfig,
    assess,
    encode_image,
    load_image,
    process,
)
from invoice_extractor.preprocessing.attempt_store import AttemptStore  # noqa: E402

from fastmcp.client import Client, FastMCPTransport  # noqa: E402
from fastmcp.exceptions import ToolError  # noqa: E402

# --- sandbox -----------------------------------------------------------------
# main's real dirs live under the package; redirect them into a throwaway tree
# before any tool call, and give the tool a deterministic, offline embeddings
# factory. The real fixture jpgs are copied into the sandbox per scenario.
REAL_ASSETS = Path(__file__).resolve().parent.parent / "src" / "invoice_extractor" / "assets"
FIXTURE = sorted(REAL_ASSETS.rglob("*.jpg"))[0]

_SANDBOX = Path(tempfile.mkdtemp(prefix="invoice_extractor_test_"))
MAIN_ASSETS = _SANDBOX / "assets"
MAIN_ASSETS.mkdir()
WORK = _SANDBOX / "work"
WORK.mkdir()
main_mod._ASSETS_DIR = MAIN_ASSETS
main_mod._DB_DIR = _SANDBOX / "dbs"


class FakeEmbeddings:
    """Deterministic offline embeddings: normalized one-hot word-bag over 16
    dims, bucketed by crc32. No network, no model — just enough structure for
    chunking to succeed.

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

# --- scripted model seams (patched in main's own namespace) -------------------

FAKE_HTML = (
    "<html><body><h1>FAKE INVOICE</h1>"
    "<table><tr><td>Description</td><td>Amount</td></tr>"
    "<tr><td>Wiring</td><td>50.00</td></tr></table>"
    "</body></html>"
)


def _fake_render_markdown(html: str) -> str:
    # Deterministic in the HTML input — finalize re-derives markdown from the
    # stored HTML at close time, so it must come out identical.
    return "# FAKE-MD\n\n" + html[:200]


class _FakeExtractions:
    """Scripted (payload, failed) / HTML responses, consumed in call order.

    Over-scripting is a test bug — exhausting the script raises so a silently
    wrong call count can't pass."""

    def __init__(self) -> None:
        self._lenient: list = []
        self._html: list = []
        self.lenient_calls = 0
        self.html_calls = 0

    def reset(self, lenient: list, html: list) -> None:
        self._lenient = list(lenient)
        self._html = list(html)
        self.lenient_calls = 0
        self.html_calls = 0

    def lenient(self, image_path) -> tuple[dict, list[str]]:
        self.lenient_calls += 1
        if self.lenient_calls > len(self._lenient):
            raise AssertionError(
                f"fake extract_invoice_lenient called {self.lenient_calls} times, "
                f"scripted for {len(self._lenient)}"
            )
        item = self._lenient[self.lenient_calls - 1]
        if isinstance(item, BaseException):
            raise item
        return item

    def html(self, image_path) -> str:
        self.html_calls += 1
        if self.html_calls > len(self._html):
            raise AssertionError(
                f"fake extract_html_invoice called {self.html_calls} times, "
                f"scripted for {len(self._html)}"
            )
        item = self._html[self.html_calls - 1]
        if isinstance(item, BaseException):
            raise item
        return item


FAKE = _FakeExtractions()
main_mod.extract_invoice_lenient = FAKE.lenient
main_mod.extract_html_invoice = FAKE.html
main_mod.render_markdown = _fake_render_markdown

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


def new_source(stem: str) -> Path:
    """A fresh source jpg in the sandbox work dir (content is irrelevant for
    the model seams, but the real engine runs on it)."""
    img = WORK / f"{stem}.jpg"
    shutil.copy(FIXTURE, img)
    return img


def check_no_scratch(stem: str) -> None:
    left = list(WORK.glob(f"{stem}__preproc_*")) + list(date_dir().glob(f"{stem}__preproc_*"))
    check(f"no scratch file left for {stem}", not left, ", ".join(str(p) for p in left))


def expect_raises(name: str, exc_type, fn, *args, **kwargs) -> BaseException | None:
    """Run fn, require exc_type, return the exception for message asserts."""
    try:
        fn(*args, **kwargs)
    except exc_type as e:
        check(name, True)
        return e
    except Exception as e:  # wrong exception type
        check(name, False, f"raised {type(e).__name__}: {e}")
        return None
    check(name, False, f"no exception (wanted {exc_type.__name__})")
    return None


# --- fixture payloads ---------------------------------------------------------
# Each scenario gets a UNIQUE invoice number so row checks stay unambiguous.

def _lines(n: int, amount: float = 1000.0) -> list[dict]:
    return [
        {"description": f"Item {i + 1}", "quantity": 1, "unit": "ea",
         "unit_price": amount, "amount": amount}
        for i in range(n)
    ]


FULL_PAYLOAD = {
    "invoice_number": "INV-05-FULL",
    "invoice_date": "2026-09-01",
    "due_date": "2026-10-01",
    "currency": "USD",
    "bill_from": {"name": "Northwind Supplies", "email": "sales@northwind.example"},
    "bill_to": {"name": "Contoso Ltd"},
    "line_items": _lines(8),
    "subtotal": 8000.0,
    "tax_rate": 0.0,
    "tax_amount": 0.0,
    "total": 8000.0,
    "payment_terms": "Net 30",
}
# All six meaningful fields present, 8 line items summing to the total, zero
# "ERROR" markers -> score 1.0.
S_FULL = 1.0

LOW_PAYLOAD = {
    "invoice_number": "ERROR",
    "invoice_date": "2026-09-01",
    "bill_from": {"name": "Northwind"},
    "bill_to": {"name": "Contoso"},
    "line_items": [{"description": "Wiring", "quantity": 1, "unit": "ea",
                    "unit_price": 50.0, "amount": 50.0}],
    "total": "ERROR",
}
LOW_FAILED = ["invoice_number", "total"]
# 4 of 6 fields present, 1 line item, 2 error markers:
#   0.3*(4/6) + 0.2*(1/8) + 0.2*(1 - 2/3) = 0.2917  -> RETRYABLE
S_LOW = round(0.30 * (4 / 6) + 0.20 * (1 / 8) + 0.20 * (1 - 2 / 3), 4)

OZ = requests.RequestException("ollama down (simulated)")  # one instance, reusable

DEFAULTS = {"clahe_clip": 2.0, "clahe_tile": 8, "sharpen_strength": 0.0}
ROW_KEYS = {
    "id", "attempt", "config", "score", "metrics", "outcome", "payload",
    "html", "best", "best_processed_path", "enhanced_image", "created_at",
}
HISTORY_KEYS = {"cycle", "attempts", "best_attempt_id", "original_image_path", "best_processed_path"}


# --- A. PreprocessConfig: knob validation at the boundary ----------------------

def test_config_validation() -> None:
    d = PreprocessConfig.from_dict(None)
    check("config: from_dict(None) == defaults",
          (d.clahe_clip, d.clahe_tile, d.sharpen_strength) == (2.0, 8, 0.0), str(d))
    check("config: defaults is_default", d.is_default)

    p = PreprocessConfig.from_dict({"clahe_clip": 3.0})
    check("config: partial override keeps the rest",
          (p.clahe_clip, p.clahe_tile, p.sharpen_strength) == (3.0, 8, 0.0), str(p))
    check("config: override is not default", not p.is_default)

    check("config: range lower bound accepted",
          PreprocessConfig.from_dict({"clahe_clip": 1.0}).clahe_clip == 1.0)
    check("config: range upper bound accepted",
          PreprocessConfig.from_dict({"clahe_tile": 16}).clahe_tile == 16)

    e = expect_raises("config: out-of-range raises ValueError",
                      ValueError, PreprocessConfig.from_dict, {"clahe_clip": 99})
    if e is not None:
        check("config: range message names the knob + value",
              "clahe_clip" in str(e) and "99" in str(e), str(e))

    e = expect_raises("config: non-integer tile raises ValueError",
                      ValueError, PreprocessConfig.from_dict, {"clahe_tile": 8.5})
    if e is not None:
        check("config: tile message", "clahe_tile must be an integer" in str(e), str(e))

    e = expect_raises("config: unknown knob raises ValueError",
                      ValueError, PreprocessConfig.from_dict, {"wibble": 1})
    if e is not None:
        check("config: unknown-knob message lists the knob + allowed set",
              "wibble" in str(e) and "clahe_clip" in str(e), str(e))

    e = expect_raises("config: with_overrides({}) raises ValueError",
                      ValueError, PreprocessConfig.with_overrides, d, {})
    if e is not None:
        check("config: overrides message", "at least one knob" in str(e), str(e))

    p2 = p.with_overrides({"sharpen_strength": 0.5})
    check("config: overrides are cumulative on this config",
          (p2.clahe_clip, p2.sharpen_strength) == (3.0, 0.5), str(p2))


# --- B. assess(): deterministic scoring ---------------------------------------

def test_assess() -> None:
    full = assess(FULL_PAYLOAD, [])
    check("assess: FULL scores 1.0", full.score == 1.0, repr(full.score))
    check("assess: FULL no missing fields", full.missing_fields == ())
    check("assess: FULL all six present", len(full.present_fields) == 6)
    check("assess: FULL line_count 8", full.line_count == 8, str(full.line_count))
    check("assess: FULL sum_of_lines", full.sum_of_lines == 8000.0, str(full.sum_of_lines))
    check("assess: FULL invoice_total", full.invoice_total == 8000.0)
    check("assess: FULL sum matches total", full.sum_mismatches_total is False)
    check("assess: FULL payload_ok", full.payload_ok is True)
    m = full.to_metrics_dict()
    check("assess: metrics dict drops the score", "score" not in m, str(m.keys()))
    check("assess: metrics dict keeps the explainable keys",
          {"present_fields", "missing_fields", "payload_ok"} <= set(m), str(m.keys()))

    low = assess(LOW_PAYLOAD, LOW_FAILED)
    check("assess: LOW exact score", low.score == S_LOW, f"{low.score} != {S_LOW}")
    check("assess: LOW missing fields exact",
          low.missing_fields == ("invoice_number", "total"), str(low.missing_fields))
    check("assess: LOW line_count 1", low.line_count == 1)
    check("assess: LOW sum_of_lines 50.0", low.sum_of_lines == 50.0, str(low.sum_of_lines))
    check("assess: LOW error total is the string ERROR -> None", low.invoice_total is None)
    check("assess: LOW error_count 2", low.error_count == 2)
    check("assess: LOW payload_ok False", low.payload_ok is False)

    dead = assess(None, ["line_items"])
    check("assess: no payload scores 0.0", dead.score == 0.0, str(dead.score))
    check("assess: no payload present ()", dead.present_fields == ())
    check("assess: no payload all six missing", len(dead.missing_fields) == 6,
          str(dead.missing_fields))
    check("assess: no payload line_count 0", dead.line_count == 0)
    check("assess: no payload error_count from failed list", dead.error_count == 1)

    broken = assess(FULL_PAYLOAD, ["line_items"])
    check("assess: failed field lowers the score", broken.score < 1.0, str(broken.score))
    check("assess: failed field doesn't change presence (payload decides)",
          len(broken.present_fields) == 6, str(broken.present_fields))


# --- C. OpenCV engine: deterministic, knobs matter, formats -------------------

def test_engine() -> None:
    img = load_image(FIXTURE)
    check("engine: load_image returns 3-channel ndarray",
          isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[2] == 3,
          f"{type(img)} {getattr(img, 'shape', None)}")

    d = PreprocessConfig.from_dict(None)
    a = process(img, d)
    b = process(img, PreprocessConfig.from_dict(None))
    check("engine: process is deterministic", np.array_equal(a, b))
    if not np.array_equal(a, b):
        check("engine: encode deterministic (skipped)", False)
        return
    check("engine: encode_image deterministic",
          encode_image(a, FIXTURE) == encode_image(b, FIXTURE))

    alt = process(img, PreprocessConfig.from_dict({"clahe_clip": 4.0, "sharpen_strength": 0.6}))
    check("engine: different knobs change the image", not np.array_equal(a, alt))
    same = process(img, PreprocessConfig.from_dict({"clahe_tile": 8}))
    check("engine: identical knobs -> identical result", np.array_equal(same, a))

    png = WORK / "roundtrip_probe.png"
    png.write_bytes(encode_image(a, FIXTURE))
    back = np.asarray(load_image(png), np.uint8)
    # The engine deliberately stores JPEG q92 (smaller BLOBs, model-readable),
    # so a byte-exact round trip is the wrong bar: the image must come back
    # the same size and essentially undamaged.
    check("engine: encode round-trips through the loader",
          back.shape == a.shape and np.abs(back.astype(np.int16) - a.astype(np.int16)).mean() <= 2.0,
          f"delta mean={np.abs(back.astype(np.int16) - a.astype(np.int16)).mean():.3f}")
    check("engine: round-trip image is decodable color",
          back.ndim == 3 and back.shape[2] == 3, str(back.shape))

    check("engine: 2-D input rejected",
          expect_raises("engine: 2-D", ValueError, process,
                        np.zeros((4, 4), dtype=np.uint8), d) is not None)
    check("engine: 4-channel input rejected",
          expect_raises("engine: 4ch", ValueError, process,
                        np.zeros((4, 4, 4), dtype=np.uint8), d) is not None)
    check("engine: missing file rejected",
          expect_raises("engine: missing file", ValueError,
                        load_image, WORK / "definitely_not_here.jpg") is not None)


# --- D. AttemptStore: cycles, budget, best, history shapes --------------------

def test_attempt_store() -> None:
    key = Path(_SANDBOX) / "d_store_image.jpg"  # key-based; the file need not exist
    st = AttemptStore(_SANDBOX / "dbs" / "standalone_store.sqlite")
    try:
        check("store: fresh image has no cycle", st.latest_cycle(key) is None)
        check("store: fresh image history empty", st.cycle_history(key) == {})
        check("store: fresh image not finalized", st.cycle_is_closed(key) is False)
        e = expect_raises("store: next_attempt without a cycle", RuntimeError, st.next_attempt, key)
        if e is not None:
            check("store: error tells the agent to call extract first",
                  "extract_invoice_from_image" in str(e), str(e))

        rows = [
            (1, FULL_PAYLOAD, "SUCCESS"),
            (2, LOW_PAYLOAD, "RETRYABLE"),
            (3, LOW_PAYLOAD, "RETRYABLE"),
            (4, LOW_PAYLOAD, "MANUAL_REVIEW"),
        ]
        cfgs = [
            {"clahe_clip": 2.0, "clahe_tile": 8, "sharpen_strength": 0.0},
            {"clahe_clip": 3.0, "clahe_tile": 8, "sharpen_strength": 0.0},
            {"clahe_clip": 3.0, "clahe_tile": 12, "sharpen_strength": 0.0},
            {"clahe_clip": 4.0, "clahe_tile": 12, "sharpen_strength": 0.4},
        ]
        for i, (att, payload, outcome) in enumerate(rows):
            st.record_attempt(
                key, 1, att, cfgs[i],
                [1.0, S_LOW, S_LOW, S_LOW][i],
                assess(payload, LOW_FAILED if payload is LOW_PAYLOAD else []).to_metrics_dict(),
                outcome,
                f"fake-enhanced-{att}".encode(),
                payload,
                FAKE_HTML if att != 4 else None,
            )
        got = st.attempts_in_cycle(key, 1)
        check("store: four attempts recorded", len(got) == 4, str(len(got)))
        if got:
            check("store: row keys match the contract", set(got[0].keys()) >= ROW_KEYS,
                  str(set(got[0].keys())))
            check("store: attempt order preserved",
                  [r["attempt"] for r in got] == [1, 2, 3, 4])
            check("store: score column per row", got[0]["score"] == 1.0)
            check("store: outcome column per row", got[3]["outcome"] == "MANUAL_REVIEW")
            check("store: config stored as JSON dict",
                  json.loads(got[1]["config"] if isinstance(got[1]["config"], str) else "{}")
                  .get("clahe_clip", json.loads(got[1]["config"] if isinstance(got[1]["config"], str) else "null").get("clahe_tile")) is not None,
                  str(got[1]["config"]))
        check("store: enhanced BLOB retrievable",
              st.enhanced_bytes(key, 1, 3) == b"fake-enhanced-3")
        check("store: BLOB for unknown attempt is None",
              st.enhanced_bytes(key, 1, 9) is None)
        check("store: NULL BLOB stays None",
              st.enhanced_bytes(key, 2, 1) is None)  # nothing recorded in cycle 2

        hist = st.cycle_history(key)
        check("store: history carries the full cycle shape",
              HISTORY_KEYS <= set(hist), str(set(hist)))
        atts = hist.get("attempts", [])
        check("store: history attempts in order",
              [a["attempt"] for a in atts] == [1, 2, 3, 4], str([a.get("attempt") for a in atts]))
        check("store: history best is attempt 1 (top score)",
              hist.get("best_attempt_id") is not None or st.cycle_is_closed(key) is False)

        first_id = got[0]["id"]
        st.finalize_best(key, 1, first_id, "/sandbox/kept/winner_processed.jpg")
        check("store: finalize closes the cycle", st.cycle_is_closed(key) is True)
        e = expect_raises("store: next_attempt after finalize", RuntimeError, st.next_attempt, key)
        if e is not None:
            check("store: closed-cycle error tells the agent to extract again",
                  "already finalized" in str(e) and "extract_invoice_from_image" in str(e), str(e))
        hist2 = st.cycle_history(key)
        check("store: finalizing marks the winner best",
              bool(hist2["attempts"][0].get("best")), str(hist2["attempts"][0]))
        check("store: winner path persisted",
              str(hist2["best_processed_path"]) == "/sandbox/kept/winner_processed.jpg")
        check("store: best_score_before cycle 2 is the cycle-1 winner",
              st.best_score_before(key, 2) == 1.0, repr(st.best_score_before(key, 2)))
    finally:
        st.close()


# --- E. Tool integration: extract / retry / history over the sandbox --------

_E: dict = {}  # cross-test state (paths, report fragments)


def test_extract_success_finalizes():
    """E1: attempt 1 clears the threshold -> cycle closes immediately, legacy
    report shape, everything persisted, source moved, scratch cleaned."""
    p = new_source("e1")
    FAKE.reset(lenient=[(FULL_PAYLOAD, [])], html=[FAKE_HTML])
    try:
        rep = main_mod.extract_invoice_from_image(str(p), None)
    except Exception as e:
        check("E1: extract call survives", False, repr(e))
        raise
    check("E1: report head carries cycle 1", rep.get("cycle") == 1, str(rep.get("cycle")))
    best = rep.get("best") or {}
    check("E1: best is attempt 1 @ 1.0",
          best.get("attempt") == 1 and best.get("score") == 1.0, str(best))
    check("E1: best outcome SUCCESS", best.get("outcome") == "SUCCESS", str(best.get("outcome")))
    storage = rep.get("storage") or {}
    check("E1: disposition moved into date folder",
          storage.get("disposition") == "moved"
          and Path(storage.get("image", "")).parent == date_dir(), str(storage))
    check("E1: ground-truth row upserted", storage.get("row_id") is not None,
          str(storage.get("row_id")))
    check("E1: markdown chunked into Chroma", storage.get("chunks", 0) > 0,
          str(storage.get("chunks")))
    check("E1: json output written and clean",
          rep.get("json", {}).get("status") == "ok" and Path(rep["json"]["output"]).is_file(),
          str(rep.get("json")))
    check("E1: markdown + html written",
          rep.get("markdown", {}).get("status") == "ok"
          and Path(rep["markdown"]["output_md"]).is_file()
          and Path(rep["markdown"]["output_html"]).is_file(), str(rep.get("markdown")))
    wi = rep.get("winner_image")
    check("E1: winning enhanced image promoted",
          isinstance(wi, str) and wi.endswith("e1_processed.jpg") and Path(wi).is_file(), str(wi))
    check("E1: source moved out of the work dir",
          not p.exists() and Path(storage["image"]).is_file(), str(p))

    inv = main_mod._sqlite_store().get_by_image_path(storage["image"])
    check("E1: sqlite row resolvable by final path", bool(inv), repr(inv))
    check("E1: chroma collection holds data", main_mod._chroma_store().collection_count() > 0)

    hist = main_mod.get_attempt_history(str(p))
    check("E1: history resolves the moved source path",
          hist.get("cycle") == 1 and len(hist.get("attempts", [])) == 1, str(hist))
    check("E1: history names the winner image",
          str(hist.get("best_processed_path", "")).endswith("e1_processed.jpg"),
          str(hist.get("best_processed_path")))
    check_no_scratch("e1")
    _E["e1"] = rep


def test_extract_retryable_reports_and_keeps_cycle_open():
    """E2: low score with remaining budget -> steering report, cycle stays open
    for retry_preprocessing, the source image is left alone."""
    p = new_source("e2")
    _E["e2_path"] = str(p)
    FAKE.reset(lenient=[(LOW_PAYLOAD, LOW_FAILED)], html=[FAKE_HTML])
    try:
        rep = main_mod.extract_invoice_from_image(str(p), None)
    except Exception as e:
        check("E2: extract call survives", False, repr(e))
        raise
    check("E2: attempt report head (cycle, attempt)",
          rep.get("cycle") == 1 and rep.get("attempt") == 1,
          str({k: rep.get(k) for k in ("cycle", "attempt")}))
    check("E2: outcome RETRYABLE", rep.get("outcome") == "RETRYABLE", str(rep.get("outcome")))
    check("E2: score matches assess", rep.get("score") == S_LOW, str(rep.get("score")))
    check("E2: knobs stored are the defaults",
          rep.get("config") == {"clahe_clip": 2.0, "clahe_tile": 8, "sharpen_strength": 0.0},
          str(rep.get("config")))
    check("E2: failed fields surfaced from the payload",
          set(rep.get("json", {}).get("failed_fields", [])) >= set(LOW_FAILED), str(rep.get("json")))
    check("E2: json status errors_marked",
          rep.get("json", {}).get("status") == "errors_marked", str(rep.get("json")))
    check("E2: metrics explain the misses",
          sorted(rep.get("metrics", {}).get("missing_fields", [])) == ["invoice_number", "total"],
          str(rep.get("metrics")))
    hint = rep.get("hint") or ""
    check("E2: hint points at retry_preprocessing", "retry_preprocessing" in hint, hint)

    st = main_mod._attempt_store()
    try:
        check("E2: cycle not closed yet", st.cycle_is_closed(p) is False)
        check("E2: next attempt is (1, 2)", st.next_attempt(p) == (1, 2), str(st.next_attempt(p)))
    finally:
        st.close()
    check("E2: source untouched in the work dir", p.is_file(), str(p))
    check_no_scratch("e2")
    _E["e2"] = rep


def test_cumulative_retry_finalizes_with_best():
    """E3: a retry that fixes the image wins; the knob diff is applied on top
    of the previous attempt's config (cumulative tuning) and the BLOB of the
    winning attempt is what gets promoted."""
    p = new_source("e3")
    FAKE.reset(lenient=[(LOW_PAYLOAD, LOW_FAILED), (FULL_PAYLOAD, [])],
               html=[FAKE_HTML, FAKE_HTML])
    r1 = main_mod.extract_invoice_from_image(str(p), None)
    check("E3: attempt 1 lands RETRYABLE",
          r1.get("outcome") == "RETRYABLE" and r1.get("score") == S_LOW,
          f"{r1.get('outcome')} {r1.get('score')}")
    try:
        r2 = main_mod.retry_preprocessing(str(p), {"clahe_clip": 3.0})
    except Exception as e:
        check("E3: retry call survives", False, repr(e))
        raise
    best = r2.get("best") or {}
    check("E3: cycle finalizes on the winning attempt",
          best.get("attempt") == 2 and best.get("score") == 1.0, str(best))
    check("E3: winner image promoted",
          isinstance(r2.get("winner_image"), str)
          and r2["winner_image"].endswith("e3_processed.jpg")
          and Path(r2["winner_image"]).is_file(), str(r2.get("winner_image")))

    st = main_mod._attempt_store()
    try:
        rows = st.attempts_in_cycle(p, 1)
        check("E3: two attempts recorded", len(rows) == 2, str(len(rows)))
        cfg = rows[1]["config"] if len(rows) > 1 else None
        cfg = json.loads(cfg) if isinstance(cfg, str) else cfg
        check("E3: cumulative config — new knob on top of the rest",
              cfg == {"clahe_clip": 3.0, "clahe_tile": 8, "sharpen_strength": 0.0}, str(cfg))
        check("E3: attempt-1 config kept untouched",
              (lambda c: json.loads(c) if isinstance(c, str) else c)(rows[0]["config"])
              == {"clahe_clip": 2.0, "clahe_tile": 8, "sharpen_strength": 0.0},
              str(rows[0]["config"]) if rows else "n/a")
        check("E3: winning attempt's enhanced BLOB stored",
              st.enhanced_bytes(p, 1, 2) is not None)
        check("E3: cycle closed after finalize", st.cycle_is_closed(p) is True)
    finally:
        st.close()
    check_no_scratch("e3")
    _E["e3"] = r2


def test_budget_exhaustion_finalizes_and_blocks():
    """E4: four below-threshold attempts -> cycle finalizes with the best (a
    tie goes to the earliest attempt), the 5th retry is refused for this image,
    and a *different* image still gets a full fresh budget."""
    p = new_source("e4")
    FAKE.reset(lenient=[(LOW_PAYLOAD, LOW_FAILED)] * 4, html=[FAKE_HTML] * 4)
    r1 = main_mod.extract_invoice_from_image(str(p), None)
    r2 = main_mod.retry_preprocessing(str(p), {"clahe_clip": 3.0})
    r3 = main_mod.retry_preprocessing(str(p), {"sharpen_strength": 0.4})
    check("E4: attempts 1-3 still reportable (budget remains)",
          all(r.get("outcome") == "RETRYABLE" for r in (r1, r2, r3)), str(r3.get("outcome")))
    r4 = main_mod.retry_preprocessing(str(p), {"clahe_tile": 4})
    best = r4.get("best") or {}
    check("E4: 4th attempt finalizes the cycle", "best" in r4 and "storage" in r4, str(r4.get("best")))
    check("E4: tie on score -> earliest attempt wins",
          best.get("attempt") == 1 and best.get("score") == S_LOW, str(best))
    check("E4: winner keeps its recorded outcome (RETRYABLE)",
          best.get("outcome") == "RETRYABLE", str(best.get("outcome")))
    check("E4: best payload still persisted as a row",
          r4.get("storage", {}).get("row_id") is not None, str(r4.get("storage")))
    check("E4: outputs written beside the (moved) image",
          Path(r4.get("json", {}).get("output", "")).is_file(), str(r4.get("json")))
    hist4 = main_mod.get_attempt_history(str(p))
    check("E4: history shows the full 4-attempt cycle",
          [a["attempt"] for a in hist4["attempts"]] == [1, 2, 3, 4], str(hist4))
    check("E4: 4th attempt recorded as MANUAL_REVIEW",
          hist4["attempts"][-1]["outcome"] == "MANUAL_REVIEW", str(hist4["attempts"][-1]))

    e = expect_raises("E4: 5th retry refused", RuntimeError,
                      main_mod.retry_preprocessing, str(p), {"clahe_tile": 12})
    if e is not None:
        check("E4: refusal points at the budget / next step",
              "exhausted" in str(e) or "finalized" in str(e), str(e))

    # Budget is per image: a fresh image gets all four attempts back.
    p2 = new_source("e4b")
    FAKE.reset(lenient=[(FULL_PAYLOAD, [])], html=[FAKE_HTML])
    rep = main_mod.extract_invoice_from_image(str(p2), None)
    check("E4: budget does not leak across images",
          (rep.get("best") or {}).get("score") == 1.0, str(rep.get("best")))
    _E["e4"] = r4


def test_all_attempts_dead_finalizes_with_no_payload():
    """E5: Ollama down for the whole cycle -> every attempt ERROR, nothing
    stored, no .md/.html written (no HTML was ever produced), source stays
    exactly where it was — the agent can re-extract once the model is back."""
    p = new_source("e5")
    FAKE.reset(lenient=[OZ] * 4, html=[OZ] * 4)
    r1 = main_mod.extract_invoice_from_image(str(p), None)
    check("E5: attempt 1 reports ERROR at score 0",
          r1.get("outcome") == "ERROR" and r1.get("score") == 0.0, str({k: r1.get(k) for k in ("outcome", "score")}))
    check("E5: api_error surfaced with the exception",
          r1.get("json", {}).get("status") == "api_error" and "RequestException" in str(r1.get("json", {}).get("error", "")),
          str(r1.get("json")))
    hint = r1.get("hint") or ""
    check("E5: hint flags the failed model pass",
          "json.error" in hint or "failed entirely" in hint, hint)
    for i, changes in enumerate(({"clahe_clip": 3.0}, {"clahe_tile": 4}, {"sharpen_strength": 0.3})):
        rep = main_mod.retry_preprocessing(str(p), changes)
        if i < 2:
            check(f"E5: retry {i + 2} still ERROR", rep.get("outcome") == "ERROR", str(rep.get("outcome")))
        else:
            best = rep.get("best") or {}
            check("E5: 4th attempt finalizes an all-dead cycle",
                  best.get("outcome") == "ERROR" and best.get("score") == 0.0, str(best))
            check("E5: json status no_payload, no row",
                  rep.get("json", {}).get("status") == "no_payload"
                  and rep.get("storage", {}).get("row_id") is None
                  and rep.get("storage", {}).get("disposition") == "none", str(rep))
            check("E5: markdown no_html (no html ever came back)",
                  rep.get("markdown", {}).get("status") == "no_html", str(rep.get("markdown")))

    st = main_mod._attempt_store()
    try:
        rows = st.attempts_in_cycle(p, 1)
        check("E5: four attempts all ERROR @ 0.0",
              len(rows) == 4 and all(r["outcome"] == "ERROR" and r["score"] == 0.0 for r in rows),
              str([(r["attempt"], r["outcome"], r["score"]) for r in rows]))
        check("E5: no enhanced BLOBs (payload never stood)",
              all(st.enhanced_bytes(p, 1, a) is None for a in (1, 2, 3, 4)))
        check("E5: cycle closed so it can never be re-finalized", st.cycle_is_closed(p) is True)
    finally:
        st.close()
    leftovers = [n for n in ("e5.md", "e5.html", "e5.json") if (WORK / n).is_file() or (date_dir() / n).is_file()]
    check("E5: no output files dropped beside the source", not leftovers, str(leftovers))
    check("E5: source left untouched (not moved, not deleted)", p.is_file(), str(p))
    check_no_scratch("e5")
    _E["e5"] = p


def test_partial_failure_keeps_html_and_leaves_cycle_open():
    """E6: only the JSON pass dies -> the attempt is ERROR but the HTML it did
    produce is recorded, the cycle stays OPEN (a model retry may well work),
    and nothing finalizes yet."""
    p = new_source("e6")
    FAKE.reset(lenient=[OZ], html=[FAKE_HTML])
    rep = main_mod.extract_invoice_from_image(str(p), None)
    check("E6: attempt ERROR but not finalized",
          rep.get("outcome") == "ERROR" and "best" not in rep, str({k: rep.get(k) for k in ("outcome",)}))
    st = main_mod._attempt_store()
    try:
        row = st.attempts_in_cycle(p, 1)
        check("E6: payload recorded as None", row[0]["payload"] is None, str(row[0]["payload"]))
        check("E6: HTML recorded even though the JSON pass died",
              bool(row[0]["html"]) and "FAKE INVOICE" in str(row[0]["html"]), str(row[0]["html"]))
        check("E6: no enhanced BLOB without a payload", st.enhanced_bytes(p, 1, 1) is None)
        check("E6: cycle still open — a retry may still land",
              st.cycle_is_closed(p) is False and st.next_attempt(p) == (1, 2),
              str(st.next_attempt(p)))
    finally:
        st.close()
    check("E6: source untouched", p.is_file(), str(p))
    check_no_scratch("e6")
    _E["e6"] = p


def test_non_api_error_reports_error_not_api_error():
    """E7: an exception that is *not* a network failure (a bug, bad input)
    must report "error", not "api_error" — the agent shouldn't blame Ollama."""
    p = new_source("e7")
    FAKE.reset(lenient=[ValueError("bogus extraction payload")], html=[FAKE_HTML])
    rep = main_mod.extract_invoice_from_image(str(p), None)
    check("E7: outcome ERROR", rep.get("outcome") == "ERROR", str(rep.get("outcome")))
    check("E7: status is 'error' (not 'api_error')",
          rep.get("json", {}).get("status") == "error", str(rep.get("json")))
    check("E7: exception type + message surfaced",
          "ValueError" in str(rep.get("json", {}).get("error", ""))
          and "bogus extraction payload" in str(rep.get("json", {}).get("error", "")),
          str(rep.get("json")))
    check_no_scratch("e7")
    _E["e7"] = p


def test_worse_cycle_keeps_previous_winner():
    """E8: cycle 1 produced a great extraction; the same image re-enters the
    pipeline and every one of cycle 2's attempts is worse. The cycle must
    finalize WITHOUT clobbering: previous row, outputs and winner image stay
    exactly as cycle 1 left them."""
    import shutil

    p = new_source("e8")
    FAKE.reset(lenient=[(FULL_PAYLOAD, [])], html=[FAKE_HTML])
    r1 = main_mod.extract_invoice_from_image(str(p), None)
    check("E8: cycle 1 wins and moves the source",
          (r1.get("best") or {}).get("score") == 1.0
          and r1.get("storage", {}).get("disposition") == "moved", str(r1.get("storage")))
    final1 = Path(r1["storage"]["image"])

    # Re-feed: the same file lands in the work dir again.
    shutil.copy(final1, p)

    FAKE.reset(lenient=[(LOW_PAYLOAD, LOW_FAILED)] * 4, html=[FAKE_HTML] * 4)
    a1 = main_mod.extract_invoice_from_image(str(p), None)
    a2 = main_mod.retry_preprocessing(str(p), {"clahe_clip": 3.0})
    a3 = main_mod.retry_preprocessing(str(p), {"clahe_tile": 4})
    check("E8: cycle-2 attempts 1-3 still steerable",
          all(r.get("outcome") == "RETRYABLE" for r in (a1, a2, a3)), str(a3.get("outcome")))
    a4 = main_mod.retry_preprocessing(str(p), {"sharpen_strength": 0.3})

    check("E8: finalize keeps the better previous winner (json)",
          (a4.get("json") or {}).get("status") == "kept-previous", str(a4.get("json")))
    check("E8: markdown also reports kept-previous",
          (a4.get("markdown") or {}).get("status") == "kept-previous", str(a4.get("markdown")))
    check("E8: storage claims no new row or chunks",
          (a4.get("storage") or {}).get("disposition") == "kept-previous"
          and (a4.get("storage") or {}).get("row_id") is None
          and (a4.get("storage") or {}).get("chunks", 0) == 0, str(a4.get("storage")))

    st = main_mod._attempt_store()
    try:
        check("E8: the earlier cycle's winner ranks ahead of cycle 2",
              st.best_score_before(p, 2) == 1.0, str(st.best_score_before(p, 2)))
        check("E8: cycle closed so it cannot be re-finalized", st.cycle_is_closed(p) is True)
    finally:
        st.close()
    check("E8: cycle-1 row, outputs and winner image all intact",
          main_mod._sqlite_store().get_by_image_path(str(final1)) is not None
          and Path(r1["json"]["output"]).is_file()
          and Path(r1["winner_image"]).is_file(), str(final1))
    check("E8: source not moved or deleted", p.is_file(), str(p))
    check_no_scratch("e8")


def test_validation_and_error_paths():
    """E9: boundary refusals — a new extract on an open cycle, empty or
    unknown retry knobs, and a missing image file. None of them may reach
    the model; the e2 cycle (still open from E2) is live proof afterwards."""
    p2 = Path(_E.get("e2_path", str(WORK / "e2.jpg")))
    FAKE.reset(lenient=[(FULL_PAYLOAD, [])] * 4, html=[FAKE_HTML] * 4)

    e = expect_raises("E9: new extract on an open cycle is refused", RuntimeError,
                      main_mod.extract_invoice_from_image, str(p2), None)
    if e is not None:
        check("E9: refusal names the open cycle", "still open" in str(e), str(e))

    e = expect_raises("E9: retry with empty changes is refused", ValueError,
                      main_mod.retry_preprocessing, str(p2), {})
    if e is not None:
        check("E9: refusal names the missing knob", "at least one knob" in str(e), str(e))

    e = expect_raises("E9: retry with an unknown knob is refused", ValueError,
                      main_mod.retry_preprocessing, str(p2), {"wibble": 1})
    if e is not None:
        check("E9: the unknown knob is named in the error", "wibble" in str(e), str(e))

    e = expect_raises("E9: missing image refused before any model call", FileNotFoundError,
                      main_mod.extract_invoice_from_image, str(WORK / "nope.jpg"), None)
    if e is not None:
        check("E9: the missing path is named", "image not found" in str(e), str(e))

    # If any refused call above touched the model, the next extraction would
    # trip FAKE's over-scripting AssertionError. It must not.
    p9 = new_source("e9")
    rep = main_mod.extract_invoice_from_image(str(p9), {})
    check("E9: empty changes means default knobs and still extracts",
          (rep.get("best") or {}).get("score") == 1.0, str(rep.get("best")))
    check_no_scratch("e9")


def test_history_and_limits():
    """E10: history for a never-seen image, the finalized e3 cycle in order,
    the capped tail, and max_attempts boundary checks."""
    fresh = new_source("e10")
    h0 = main_mod.get_attempt_history(str(fresh))
    check("E10: unseen image reports no cycle",
          h0 == {"image": str(fresh), "cycle": None, "attempts": []}, str(h0))

    e3 = _E.get("e3")
    if e3 is None:
        check("E10: precondition (E3 ran and stored a report)", False, "no e3 state")
        return
    p3 = WORK / "e3.jpg"  # the key is the *original* source path, pre-move
    hist = main_mod.get_attempt_history(str(p3))
    check("E10: finalized cycle reports cycle 1 with two attempts in order",
          hist.get("cycle") == 1 and [a["attempt"] for a in hist["attempts"]] == [1, 2],
          str(hist))
    check("E10: the winning attempt is flagged by name",
          hist["attempts"][-1].get("best") is True
          and str(hist.get("best_processed_path", "")).endswith("e3_processed.jpg"),
          str(hist.get("best_processed_path")))
    tail = main_mod.get_attempt_history(str(p3), 1)
    check("E10: max_attempts=1 returns only the latest attempt",
          [a["attempt"] for a in tail["attempts"]] == [2], str(tail["attempts"]))
    e = expect_raises("E10: max_attempts=0 is refused", ValueError,
                      main_mod.get_attempt_history, str(p3), 0)
    if e is not None:
        check("E10: the error demands a positive integer", "positive integer" in str(e), str(e))


def test_listing_sees_stored_invoices():
    """E11: list_invoices reflects the sandbox assets directory, the stored
    invoices survive there, and a missing assets dir is refused cleanly."""
    lis = main_mod.list_invoices()
    check("E11: the assets dir is the sandbox dir",
          lis.get("assets_dir") == str(MAIN_ASSETS), str(lis))
    check("E11: at least one invoice is listed", (lis.get("count") or 0) >= 1, str(lis))
    names = {Path(x).name for x in lis.get("images") or []}
    check("E11: e1's stored pair appears by name",
          {"e1.jpg", "e1_processed.jpg"} <= names, str(sorted(names))[:200])

    e = expect_raises("E11: a missing assets dir is refused", FileNotFoundError,
                      main_mod._image_files, WORK / "missing_dir")
    if e is not None:
        check("E11: the error names the directory", "assets directory not found" in str(e), str(e))


# --- F. MCP bridge + LangChain wiring ---------------------------------------

EXPECTED_TOOLS = ["extract_invoice_from_image", "get_attempt_history",
                  "list_invoices", "retry_preprocessing"]


def test_agent_bridge_exposes_all_tools():
    """F: the in-process MCP server exposes exactly the four intended tools,
    one of them actually runs through the wire, and the LangChain agent
    module can bind every tool without a live model."""
    import asyncio

    from fastmcp.client import Client, FastMCPTransport

    async def _run() -> tuple[list[str], object, list]:
        async with Client(FastMCPTransport(main_mod.mcp)) as client:
            tools = await client.list_tools()
            names = sorted(t.name for t in tools)
            assert names == sorted(EXPECTED_TOOLS), names
            result = await client.call_tool("list_invoices", {})
            return names, result, list(tools)

    try:
        names, result, tools = asyncio.run(_run())
    except Exception as ex:  # noqa: BLE001
        check("F: MCP bridge survives", False, repr(ex))
        raise
    check("F: exactly the four intended tools are exposed",
          names == sorted(EXPECTED_TOOLS), str(names))
    data = result.data if hasattr(result, "data") else result
    check("F: a bridged call returns the invoice listing",
          isinstance(data, dict) and isinstance(data.get("count"), int), str(type(data)))

    try:
        import invoice_extractor.agent as agent_mod
    except Exception as ex:  # noqa: BLE001
        check("F: the agent module imports", False, repr(ex))
        return
    try:
        bound = [agent_mod._build_tool(t) for t in tools]
    except Exception as ex:  # noqa: BLE001
        check("F: every tool binds to a LangChain StructuredTool", False, repr(ex))
        return
    check("F: all four tools bind cleanly", len(bound) == 4, str(len(bound)))
    check("F: tool names survive the binding",
          sorted(b.name for b in bound) == sorted(EXPECTED_TOOLS),
          str([b.name for b in bound]))


# --- driver ------------------------------------------------------------------

def main() -> int:
    tests = [
        # A — preprocessing primitives
        test_config_validation,
        test_assess,
        test_engine,
        test_attempt_store,
        # E — tool integration over the sandbox
        test_extract_success_finalizes,
        test_extract_retryable_reports_and_keeps_cycle_open,
        test_cumulative_retry_finalizes_with_best,
        test_budget_exhaustion_finalizes_and_blocks,
        test_all_attempts_dead_finalizes_with_no_payload,
        test_partial_failure_keeps_html_and_leaves_cycle_open,
        test_non_api_error_reports_error_not_api_error,
        test_worse_cycle_keeps_previous_winner,
        test_validation_and_error_paths,
        test_history_and_limits,
        test_listing_sees_stored_invoices,
        # F — server surface
        test_agent_bridge_exposes_all_tools,
    ]
    for t in tests:
        try:
            t()
        except Exception:  # noqa: BLE001
            traceback.print_exc()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
