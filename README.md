# Model Context Protocol (FastMCP & LangChain)

> 🎓 **Personal training repo** — hands-on notes on building & consuming MCP
> servers. Pre-work for the [instar-mcp-server](https://github.com/mpolinowski/instar-mqtt-mcp).

* [MCP Server Foundation](https://mpolinowski.github.io/docs/IoT-and-Machine-Learning/ML/2026-09-08--mcp-server-foundation/2026-09-08)
* [MCP Protocol Deepdive](https://mpolinowski.github.io/docs/IoT-and-Machine-Learning/ML/2026-09-09--mcp-protocol-deepdive/2026-09-09)
* [MCP Server Deployment](https://mpolinowski.github.io/docs/IoT-and-Machine-Learning/ML/2026-10-08--mcp-server-deployment/2026-10-08)
* [MCP Gateway](https://mpolinowski.github.io/docs/IoT-and-Machine-Learning/ML/2026-10-09--mcp-gateway/2026-10-09)
* [INSTAR MQTT Camera AI Agent Interface](https://mpolinowski.github.io/docs/IoT-and-Machine-Learning/ML/2026-09-10--instar-mqtt-mcp-server/2026-09-10)

---

## 🛠️ What is MCP used for?

**MCP (Model Context Protocol)** is an open, JSON-RPC-based standard for how
LLM-powered apps talk to the outside world. Instead of wiring a bespoke
integration per tool or data source, an **MCP server** exposes a set of
*tools* (callable actions), *resources* (readable data), and *prompts* — and
any **MCP client** (Claude, Cursor, a LangChain agent, …) can use them.
One standard plug: **model in, capability out.**

| Layer | What it is |
| :-- | :-- |
| **MCP server** 🖥️ | Exposes tools / resources / prompts over *stdio* or *streamable-HTTP*. |
| **MCP client** 🖱️ | Connects to a server, lists its tools, calls them. |
| **FastMCP** ⚡ | Python framework atop the official `mcp` SDK — build servers in a few lines. |
| **LangChain ↔ MCP** 🤖 | The adapter that turns MCP tools into LangChain tools (client side). |

This repo is that stack in miniature: **build servers (FastMCP)** + **use them
(LangChain / raw SDK)**, then **package & deploy** them.

---

## 📚 Research — a tour of the folders

**The learning ladder (suggested reading order):**

```
🔬 deepdive-mcp          learn the protocol + both kinds of client
🌦️ weather-node-mcp      the same thing in TypeScript (not just Python)
📁 file-access-python-mcp build a real, clean FastMCP server
📦 mcp-server-deployment → package it as an installable library
🚪 mcp-server-gateway    → aggregate many servers behind one endpoint
🐳 mcp-docker-deployment → ship it in a container
```

| 📁 Repo | What's inside | 🎯 Takeaway |
| :-- | :-- | :-- |
| **deepdive-mcp** 🔬 | Three parallel scenarios — ① raw `mcp` SDK over `stdio`, ② FastMCP over stdio **+** `streamable-HTTP`, ③ a LangChain MCP-adapter client hitting a community `duckduckgo` server. | The 3 layers of MCP in one place: **SDK server → FastMCP → LangChain client.** The core “what / where / how” playground. |
| **weather-node-mcp** 🌦️ | Minimal MCP server in **TypeScript** using the official `@modelcontextprotocol` v2 client **+** server and `zod`. One tool (`get-alerts`, US NWS) + one resource, and a demo client. | MCP in the **other language** — and that server *and* client can live in the same folder. |
| **file-access-python-mcp** 📁 | Production-flavoured FastMCP server: exposes a local folder of text docs as **9 tools** (search / list / glob / get / edit / create / delete) over stdio **and** streamable-HTTP+CORS, with `--allow-*` permission flags. | The **cleanest example here** — a solid template (domain / tools / transport split + e2e tests). Copy this one. |
| **mcp-server-deployment** 📦 | `agent_terminal_tools`: a FastMCP server of shell & file tools (run command, list, glob, grep, read/write/append/delete/copy/move) packaged as an **installable Python package** (`uv` build + entry point) with tests. | How to turn a server into a **package** others can `pip install` / `uvx`. |
| **mcp-server-gateway** 🚪 | A FastMCP **gateway**: one server that `mount(create_proxy(...))`s several upstreams (DuckDuckGo **+** `agent_terminal_tools`) and re-exposes them over streamable-HTTP. | The **aggregate / proxy pattern** — many servers behind a single endpoint. |
| **mcp-docker-deployment** 🐳 | The gateway above, **containerised**: `python:3.14-slim` + `uv`, pre-installs a `uvx` community server, runs it over streamable-HTTP. | How to **ship** an MCP server as a Docker image. |

> 🚧 `weaviate-mcp/` is a vendored **Go** MCP server for the Weaviate vector DB —
> **WIP, not yet published**, so it's left out of the tour.
