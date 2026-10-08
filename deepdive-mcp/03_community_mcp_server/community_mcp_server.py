import asyncio
import json

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import Connection
from langchain_mcp_adapters.tools import load_mcp_tools

# MCP Server config
connections: dict[str, Connection] = {
    "data_fetch_mcp_stdio": {
        "transport": "stdio",
        "command": "uvx",
        "args": ["duckduckgo-mcp-server"],
        "env": {
            "DDG_SAFE_SEARCH": "OFF",
            "DDG_REGION": "cn-zh"
        }
    }
}

async def main():
    client = MultiServerMCPClient(connections)

    # one persistent connection to the server for the whole block —
    # both tools below execute on the same server process, like the raw SDK client
    async with client.session("data_fetch_mcp_stdio") as session:
        # list all the tools
        tools = await load_mcp_tools(session)
        tools_by_name = {t.name: t for t in tools}
        # for tool in tools:
        #     print(f"Tool: {tool.name}\n")
        #     print(f"Tool description: {tool.description}\n")

        def payload_of(result):
            """helper function: ainvoke returns a text block -> str."""
            if isinstance(result, list):
                result = result[0]["text"]
            try:
                return json.loads(result)
            except (TypeError, ValueError):
                return result

        # search
        search_results = payload_of(await tools_by_name["search"].ainvoke({"query": "What is the Model Context Protocol for?"}))
        print("Search results:    ", search_results)

if __name__ == "__main__":
    asyncio.run(main())
