# deepdive-mcp

The protocol playground — the same minimal MCP server explored three ways, to
see the **SDK → FastMCP → client** layers in one place.

> Part of the [MCP training ladder](../README.md). Read top-to-bottom.

| # | Folder | What it shows |
| :-- | :-- | :-- |
| ① | `01_stdio_mcp_server/` | A FastMCP **server** (`mcp_server_stdio.py`: `fetch`, `process` over stdio) + two **clients** that consume it — a raw-SDK-style client and a **LangChain MCP-adapter** client (`mcp_langchain_client.py`). |
| ② | `02_http_mcp_server/` | The same tools served over **streamable-HTTP** (`mcp_server_http.py`: `fetch_http`, `process_http`), plus a LangChain client. |
| ③ | `03_community_mcp_server/` | A **LangChain client** hitting a *community* server (`duckduckgo-mcp-server` via `uvx`) — no local server of our own. |

All five local tools are tiny mocks with no side effects. Each declares the
four MCP **tool hints** via `@mcp.tool(annotations=…)` — here
`read-only / idempotent / non-destructive / closed-world` — so a host knows up
front that calling them is safe.

```bash
cd deepdive-mcp
uv sync

python 01_stdio_mcp_server/mcp_server_stdio.py    # start the stdio server
python 02_http_mcp_server/mcp_server_http.py      # …or serve over streamable-HTTP
```
