# mcp-server-deployment — `agent_terminal_tools`

A **FastMCP** server that exposes common shell & file operations as tools,
packaged as an **installable Python package** so another server (or an MCP
client) can launch it with a single command.

This is the "how do I turn a server into a package others can `pip install` /
`uvx`" rung of the [learning ladder](../README.md).

> ⚠️ **Trust boundary:** every one of these tools can run real commands or
> modify real files on the machine that starts the server. Treat it as
> powerful as a shell. The MCP [tool annotations](#tool-annotations) below tell
> a host which tools are read-only vs. destructive so it can confirm with you
> first — but a host is not required to honour them.

## Tools

| Tool | What it does | Annotations (readOnly / destructive / idempotent / openWorld) |
| :-- | :-- | :-- |
| `bash` | Run a shell command | `false / true / false / true` |
| `execute_file` | Run a file/script via bash | `false / true / false / true` |
| `run_python` | Run a script with this server's interpreter | `false / true / false / true` |
| `list_directory` | List a directory (`ls -la`) | `true / false / true / false` |
| `glob_files` | Find files by glob pattern | `true / false / true / false` |
| `grep` | Search file contents (`grep -rn`, read-only) | `true / false / true / false` |
| `read_file` | Read a text file | `true / false / true / false` |
| `write_file` | Write/overwrite a file | `false / true / true / false` |
| `append_file` | Append to a file | `false / false / false / false` |
| `delete_file` | Delete a file (`rm -v`) | `false / true / false / false` |
| `create_folder` | Create directories (`mkdir -p`) | `false / false / true / false` |
| `delete_folder` | Delete a directory tree (`rm -r`) | `false / true / false / false` |
| `copy_file` | Copy a file/dir (`cp -r`) | `false / true / true / false` |
| `move_file` | Move/rename a file/dir (`mv`) | `false / true / false / false` |
| `file_info` | Show file metadata (`stat`) | `true / false / true / false` |

### Tool annotations

Each tool declares the four MCP hints via `@mcp.tool(annotations=…)`. Per the
MCP spec these are **hints, not guarantees** — a client *may* use them to warn
a user before invoking (e.g. "this is destructive"), and some host/directory
policies make them easier to reason about. All four are set explicitly on every
tool, matching the handler's actual behaviour.

## Run it

```bash
cd mcp-server-deployment
uv sync                       # installs deps + the console entry point
uv run agent_terminal_tools   # start the server over stdio (FastMCP)
# or, to see what a client would see:
uv run python test_client.py  # connect with a LangChain MCP-adapter client
```

> `test_client.py` and `test_package.py` are **client launchers/demos**, not an
> automated test suite — see [../SECURITY_AUDIT.md](../SECURITY_AUDIT.md) for
> the test-coverage note.

## Package

- `src/agent_terminal_tools/tools.py` — the FastMCP server + all tools
- `src/agent_terminal_tools/main.py` — `main()` entry point (stdio)
- `pyproject.toml` — build metadata + the `agent_terminal_tools` console script
