# Security audit — findings & verdict

> **Context.** An automated security audit rated this repository **C (71/100)**
> and raised several findings: "No license file", "a tool describes read intent
> but its handler mutates (grep)", "no tools have read-only/destructive/
> idempotent/openWorld annotations", and "only 0/33 tools referenced in tests".
>
> This file records **what I found by reading the actual code**, and what I
> changed in response. Where the audit was wrong or overstated, it says so and
> shows why — the goal was to fix the *real* issues, not to appease a
> misread. Where a suggestion was legitimate but optional, it is labelled as
> such.

**Scope.** `weaviate-mcp/` is **out of scope** — it is git-ignored, not
committed, and was not read or modified. Every statement below concerns the
other sub-projects only.

---

## Verdict at a glance

| # | Audit finding | Verdict | Response |
| :- | :-- | :- | :-- |
| 1 | "No license file" | ✅ **Real** | Added a repo-root [`LICENSE`](LICENSE) (MIT). |
| 2 | "`grep` describes read intent but its handler mutates (the `subprocess.run` line)" | ❌ **False positive** | Left the tool **as-is** — it is genuinely read-only. Declared `read_only_hint: true` so that fact is explicit, instead of renaming/splitting a correct tool. |
| 3 | "No tools have read-only/destructive/idempotent/openWorld annotations" | ⚠️ **Real gap, but optional** | Declared all four hints explicitly on **all 33** in-scope local tools, matching each handler's real behaviour. |
| 4 | "Only 0/33 tools referenced in tests" | ❌ **Mostly false** | The `file-access` server's **9/9** tools are thoroughly e2e-tested. Recorded the *true* gap: `agent_terminal_tools` (15 tools) and the small mocks have no real tests. |
| 5 | *(lists 29 tools, yet says "33/33" in the counts)* | ℹ️ **Caveat** | Noted the audit was not fully rigorous; verified counts directly against the source. |

---

## Per-finding evidence

### 1. "No license file" — ✅ real, fixed

There was no `LICENSE` anywhere in the repository (only vendored dependency
licenses under `.venv/`, which are irrelevant). Added a single MIT `LICENSE` at
the repo root, which covers all sub-projects. This was the one finding that
was unambiguously correct.

### 2. "`grep` … handler mutates (line 128: `subprocess.run`) " — ❌ false positive

This is the only *specific* code claim in the audit, so it deserves scrutiny.
The tool actually looks like this
(`mcp-server-deployment/src/agent_terminal_tools/tools.py`):

```python
@mcp.tool(annotations=_READ_ONLY_LOCAL)
def grep(pattern: str, path: str = ".", include: str = ""):
    '''Search file contents for a regex pattern (like: grep -rn) ...'''
    command = ["grep", "-rn"]
    if include:
        command.append(f"--include={include}")
    command += [pattern, path]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, check=False)
```

Reading it carefully:

- It runs the **`grep` binary** with `-rn --include=X PATTERN PATH`. `grep` is a
  read-only utility — it *reads* files and prints matching lines. It does not
  create, modify, or delete anything.
- `subprocess.run(command, …)` here is **not** "executes arbitrary commands"; it
  is the plain mechanism for invoking the `grep` *program*. The audit appears to
  have pattern-matched on `subprocess.run` and assumed a mutating shell call —
  the same call is used by a dozen read-only tools in this file (`ls`, `stat`,
  the `glob`/`read` helpers, etc.).
- Crucially, `shell=False` (the default) and `pattern` / `path` / `include` are
  passed as **separate argv tokens**, so there is **no shell-metacharacter
  interpretation** and no way for an argument to turn into an extra command, a
  write, or a delete. `include` is folded into one argv element
  (`--include=…`) and passed literally to `grep`.

So the description ("Search file contents … like `grep -rn`") **exactly matches**
the behaviour, and the handler **mutates nothing**. The right response was not to
rename or split a correct tool, but to make its nature explicit by declaring
`read_only_hint: true` (done — see the `_READ_ONLY_LOCAL` constant).

### 3. "No tools have … annotations" — ⚠️ real gap, but optional by spec

**Verified real:** no in-scope server declared MCP tool annotations —
`grep -r annotations= …` found nothing in the tool layer before this change.

**But the framing is overstated.** The MCP spec (and the docstring on
`ToolAnnotations` shipped in both the Python `mcp` SDK and the TypeScript
`@modelcontextprotocol/server` package) states, verbatim:

> *NOTE: all properties in ToolAnnotations are **hints**. … Clients should
> never make tool use decisions based on ToolAnnotations received from
> untrusted servers.*

All four hints (`readOnlyHint`, `destructiveHint`, `idempotentHint`,
`openWorldHint`) are **optional** — a tool *may* carry a subset or none. The
audit's assertion that "OpenAI's directory **rejects** tools where any of the
four hints are missing or non-boolean" could not be verified against a primary
source and is **not** supported by the spec text above, so it is treated as an
overstatement, not a hard requirement.

Adding the hints was still a **legitimate, low-risk hygiene improvement** (hosts
can then warn a user before a destructive call), so it was done for every tool,
each value chosen to match the handler:

- **Python (FastMCP):** `@mcp.tool(annotations={ "read_only_hint": …,
  "destructive_hint": …, "idempotent_hint": …, "open_world_hint": … })`
- **TypeScript (MCP server v2):** `annotations: { readOnlyHint: …,
  destructiveHint: …, idempotentHint: …, openWorldHint: … }` in the
  `registerTool` config.

All four are set explicitly on every tool, so a client always gets a complete,
unambiguous picture.

### 4. "Only 0/33 tools referenced in tests" — ❌ mostly false

The flat "0/33" is contradicted by the code:

- **`file-access-python-mcp`** ships a real e2e suite,
  [`testing.py`](file-access-python-mcp/testing.py), that spawns the actual
  server over **stdio and streamable-HTTP** and drives it with a real MCP
  client. It references and exercises **all nine** tools by name across four
  scenarios (read-only, per-flag permissions, `--allow-all`, HTTP+CORS), plus
  path-escape and symlink-containment checks. It must print
  `ALL TESTS PASSED` — and it did after the annotation change.
- **What is true:** [`mcp-server-deployment`](mcp-server-deployment/README.md)
  (`agent_terminal_tools`, 15 tools) has **no real test** — its
  `test_client.py` / `test_package.py` are *client launchers/demos*, not
  assertions. The small training mocks (`deepdive` `fetch`/`process`, the
  gateway/docker local tools, `weather` `get-alerts`) also have no tests.
- Per the agreed scope, **no new test suite was added**; the untested-server
  gap is recorded here as a known open item rather than silently "fixed".

In short: the audit's *direction* (some servers are untested) is fair, but the
"0/33" number is flatly wrong — 9 tools on the most security-sensitive server
are well covered.

---

## What changed (commit by commit)

Each step was committed separately on `master` (branch was `up to date with
origin/master`; **nothing has been pushed**):

| Commit | Change |
| :-- | :-- |
| `Add MIT LICENSE` | New repo-root `LICENSE` — resolves finding #1. |
| `mcp-server-deployment: add tool annotations + fill README` | Annotations on all 15 `agent_terminal_tools` tools; filled the previously empty `README.md`. |
| `file-access-python-mcp: add tool annotations; keep docs in sync` | Annotations on all 9 tools; `AGENTS.md` + `README.md` updated; **e2e suite re-run: ALL TESTS PASSED**. |
| `deepdive-mcp: add tool annotations + fill README` | Annotations on `fetch`/`process` (01) and `fetch_http`/`process_http` (02); filled the previously empty `README.md`. |
| `mcp-server-gateway: annotate local tools + fill README` | Annotations on the 2 local mock tools (proxied tools keep their upstream hints); filled the previously empty `README.md`. |
| `mcp-docker-deployment: annotate local tools + add README` | Annotations on the 2 local mock tools; new `README.md` for the image. |
| `weather-node-mcp: annotate get-alerts + add README` | Annotations on `get-alerts` (verified by booting the server and reading them back through a client); new `README.md`. |

**Verification done along the way:** every Python server was imported and its
tools enumerated to confirm all four hints are present and correct; the
TypeScript server was booted over stdio and `get-alerts`' annotations were read
back through a real MCP client; the `file-access` e2e suite passed in full.

## Known open item (not acted on, by agreement)

- **Real tests for the untested servers.** `agent_terminal_tools`
  (15 shell/file tools) and the small mocks have no assertions. Out of scope
  for this pass, but this is a genuine gap worth closing before any of those
  servers are consumed by a public client — especially `bash`,
  `execute_file`, `run_python`, `delete_file`, `delete_folder`.

## What was deliberately NOT changed

- **The `grep` tool** (or any tool's behaviour, name, or signature). The audit
  misread a read-only `grep` invocation as "mutating"; splitting or renaming it
  would have degraded a correct tool to satisfy a misreading. The one change to
  it is the now-explicit `read_only_hint: true`.
- **The mounted/proxied upstream tools** in the gateway & docker servers — their
  hints come from the servers they proxy (`agent_terminal_tools` now sets them;
  community servers set their own).
- **`weaviate-mcp/`** — out of scope and git-ignored; untouched.

## A note on the audit itself

Two internal inconsistencies stood out — it lists **29** tools in the body but
references **"33/33"** in the hint/test counts, and its single code-level claim
(the `grep` line) does not describe what that line does. The *direction* of its
suggestions (add a license; add annotations; write more tests) is sound, and the
actionable parts were taken. The specific claims were checked against the source
before being trusted.
