"""agent_terminal_tools — an MCP server exposing common terminal / shell tools.

Structure:

- ``agent_terminal_tools.tools``  defines the FastMCP server (``mcp``) and the
  actual ``@mcp.tool()`` tools.
- ``agent_terminal_tools.main``   holds the ``main()`` entry point that runs the
  server over stdio.

``main`` and ``mcp`` are re-exported here so both
``from agent_terminal_tools import main`` and
``from agent_terminal_tools import mcp`` work (and so the
``mcp-server-deployment`` console script keeps resolving ``main``).
"""

from .main import main
from .tools import mcp

__all__ = ["main", "mcp"]
