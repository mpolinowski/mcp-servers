# weather-node-mcp

The same MCP idea **in TypeScript** — proof that the protocol isn't Python-only.
One small server + one client that consumes it.

> Part of the [MCP training ladder](../README.md).

## What's in it

| File | Purpose |
| :-- | :-- |
| [`src/index.ts`](src/index.ts) | The server: one resource (`weather://about`) and one tool, [`get-alerts`](#tools), that queries the **US National Weather Service** (`api.weather.gov`) for a state's active alerts. Serves over **stdio**. |
| [`src/client.ts`](src/client.ts) | A demo client: connects over stdio, lists resources, reads the resource, lists tools, and calls `get-alerts` for a state. |
| `package.json` | Deps: `@modelcontextprotocol/server` + `/client` (v2), `zod` (schema), `tsx`. |

### Tools

`get-alerts({ state: "CA" })` → the active NWS alert headlines for that US state.

It declares the four MCP **tool hints** via the `annotations` config:
`readOnlyHint: true` (read-only), `destructiveHint: false`, `idempotentHint: true`,
and `openWorldHint: true` (it reaches an external service, NWS). Per the MCP spec
these are *hints, not guarantees*.

## Run it

```bash
cd weather-node-mcp
npm install

npm tsx src/index.ts          # start the server over stdio
tsx src/client.ts             # run the demo client (lists tools, then calls get-alerts)
```

Requires network access to `api.weather.gov` for the tool to return real alerts.
