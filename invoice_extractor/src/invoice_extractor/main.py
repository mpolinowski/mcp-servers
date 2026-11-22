try:  # Package mode: imported as ``agent_terminal_tools.main``.
    from invoice_extractor.tools.main import mcp
except ImportError:  # Script mode: launched as ``python main.py`` from the package dir.
    from tools.main import mcp


def main() -> None:
    """Start the MCP server over stdio."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
