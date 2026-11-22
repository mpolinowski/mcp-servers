# invoice-extractor

MCP server + LangChain ReactAgent that extracts invoice data from images using
the GLM-OCR vision model served locally by Ollama (`http://localhost:11434`).
No cloud calls.

Per source image, **both** passes run unconditionally, in order:

1. **Strict JSON** — fixed Chinese extraction prompt, schema-validated
   (Pydantic `Invoice` model in `tools/invoice_json.py`). On validation
   failure, failing values are replaced with `"ERROR"` and the JSON file is
   still written; the report lists every failed field path.
2. **HTML / Markdown** — loose `Table Recognition` / `Text Recognition`
   prompts rendered to `<stem>.html` and `<stem>.md`.

Outputs are written beside the source image: `<stem>.json`, `<stem>.html`,
`<stem>.md`.

## Setup

```bash
uv sync
```

Requires Ollama running with the `glm-ocr:latest` model pulled.

## MCP server

`invoice-extractor` starts a FastMCP server over stdio
(`src/invoice_extractor/main.py` → `tools/main.py`):

| Tool | Args | Returns |
|------|------|---------|
| `list_invoices` | — | `assets_dir`, `count`, `images` (every image in `src/invoice_extractor/assets/`) |
| `extract_invoice_from_image` | `image_path` | per-pass report: `json` / `markdown`, each with `success`, `status` (`ok` \| `errors_marked` \| `api_error` \| `error` \| `empty`), failed field paths, output file paths |

## Agent

```bash
uv run python -m invoice_extractor.agent "Extract every invoice in the assets directory and report per-file success/failure."
```

Drives a `create_agent` ReactAgent (`qwen3.8:latest` via Ollama's
OpenAI-compatible endpoint) that lists the images, extracts each one through
the in-process MCP bridge, and compiles a per-file Markdown report.

The bridge (`src/invoice_extractor/agent.py`) does **not** use
`langchain-mcp-adapters` (incompatible with the installed `mcp` SDK — it
imports `mcp.shared.context.RequestContext`, which was removed). It wraps
each FastMCP tool in a `StructuredTool` with a pydantic args model generated
from the tool's JSON schema, and calls the tools through
`fastmcp.client.Client(FastMCPTransport(mcp))`. The client is created per
tool call so sync and async invocation paths never share a session across
event loops.

## Tests

No test framework is a dependency; the suite is a plain script:

```bash
uv run python tests/test_offline.py
```

It monkeypatches `invoice_json.generate` / `invoice_html.generate` (no model
round-trips, no GPU) and covers: tool inventory, JSON success, validation
failure → `"ERROR"` markers (valid fields still preserved, file still
written), prose output (no JSON object), Ollama down on both passes,
HTML/Markdown success, and both the async and sync LangChain bridge paths.

The fixture is `src/invoice_extractor/assets/*.jpg` (test data is copied to
the temp dir before running; outputs are cleaned up).

## Known model quirks

- `currency` comes back as `"$"` rather than `USD`.
- The model occasionally drops array elements (seen with `payment_methods`).
- Dates are returned ISO-normalized even when the invoice prints prose dates.

These are model behavior, not schema problems — downstream consumers should
treat them as such.
