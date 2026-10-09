"""Entry point: run the terminal-tools MCP server over stdio.

Runs the FastMCP server (and all of its tools) defined in ``tools.py``.
``main()`` is what the ``mcp-server-deployment`` console script calls; the
``if __name__ == "__main__"`` block also lets you start it directly with
``python main.py`` from within the package directory.
"""

try:  # Package mode: imported as ``agent_terminal_tools.main``.
    from agent_terminal_tools.tools import mcp
except ImportError:  # Script mode: launched as ``python main.py`` from the package dir.
    from tools import mcp


def main() -> None:
    """Start the MCP server over stdio."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
