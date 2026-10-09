import os
import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import Connection
from langchain_mcp_adapters.tools import load_mcp_tools

venv_path = os.path.join(os.path.dirname(os.path.abspath(__name__)), ".venv")

# MCP Server config
connections: dict[str, Connection] = {
    "agent_terminal_tools": {
        "transport": "stdio",
        "command": os.path.join(venv_path, "bin", "python"),
        "args": [os.path.join(venv_path, "bin", "agent_terminal_tools")],
    }
}

async def main():
    client = MultiServerMCPClient(connections)

    async with client.session("agent_terminal_tools") as session:
        # list all the tools
        tools = await load_mcp_tools(session)
        for tool in tools:
            print(f"Tool: {tool.name}")

if __name__ == "__main__":
    asyncio.run(main())
