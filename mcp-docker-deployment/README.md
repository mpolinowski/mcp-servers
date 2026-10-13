# mcp-docker-deployment

The gateway pattern from [`mcp-server-gateway`](../mcp-server-gateway/README.md),
**containerised**: ship an MCP server as a Docker image.

> Part of the [MCP training ladder](../README.md).

## Image recipe

[`Dockerfile`](Dockerfile) on `python:3.14-slim`:

1. Copies `uv`/`uvx` in from the official `astral-sh/uv` image.
2. **Pre-installs the community server** the gateway will proxy
   (`uv tool install duckduckgo-mcp-server`) so the client doesn't wait for a
   first-run download.
3. `pip install -r requirements.txt` (just `fastmcp`).
4. Copies the app and runs `python app/mcp_gateway.py`.

[`app/mcp_gateway.py`](app/mcp_gateway.py) is the same proxy gateway: local
`fetch`/`process` mocks (declared *read-only / idempotent / non-destructive /
closed-world* via tool annotations) + `create_proxy` mounts for DuckDuckGo and
`agent_terminal_tools`, served over **streamable-HTTP**.

## Build & run

```bash
cd mcp-docker-deployment
docker build -t mcp-gateway .
docker run --rm -p 8888:8888 mcp-gateway
# → streamable-HTTP MCP endpoint at http://localhost:8888/mcp
```

> For the `agent_terminal_tools` proxy to resolve inside the container, that
> package must be installed on the image; the DuckDuckGo proxy is pre-installed
> by the Dockerfile.
