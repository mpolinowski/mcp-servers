# mcp-server-gateway

The **aggregate / proxy pattern**: one FastMCP server that `mount(create_proxy(…))`s
several *upstream* servers and re-exposes them all over a **single**
streamable-HTTP endpoint. A client only ever connects to *this* server.

> Part of the [MCP training ladder](../README.md).

## What it exposes

| Source | Tools | How it's attached |
| :-- | :-- | :-- |
| **local (this package)** | `fetch`, `process` | `@mcp.tool(annotations=…)` — pure mocks, so declared *read-only / idempotent / non-destructive / closed-world*. |
| upstream `duckduckgo-mcp-server` | web search, etc. | `create_proxy` → `uvx duckduckgo-mcp-server` (stdio). Hints come from that server. |
| `agent_terminal_tools` | 15 shell & file tools | `create_proxy` → the installed `agent_terminal_tools` entry point. Hints come from that package (see [`mcp-server-deployment`](../mcp-server-deployment/README.md)). |

So the gateway adds **one** upstream's worth of tools per `mount`; the hints on
those proxied tools are whatever the upstream server declares — the gateway
doesn't override them.

## Run it

```bash
cd mcp-server-gateway
uv sync
python src/mcp_server_gateway/mcp_gateway.py   # serves streamable-HTTP on 127.0.0.1:8888
```

The DuckDuckGo proxy needs `uvx` on PATH (the Docker build pre-installs the
server); the `agent_terminal_tools` proxy resolves the sibling package's venv
entry point at import time.
