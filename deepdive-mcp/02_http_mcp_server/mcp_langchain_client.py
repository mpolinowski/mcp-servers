import os
import asyncio
import json

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import Connection
from langchain_mcp_adapters.tools import load_mcp_tools

mcp_server_script = os.path.join((os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "01_stdio_mcp_server")
venv_path = os.path.join((os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".venv")

# MCP Server config
connections: dict[str, Connection] = {
    "data_fetch_mcp_stdio": {
        "transport": "stdio",
        "command": os.path.join(venv_path, "bin", "python"),
        "args": [os.path.join(mcp_server_script, "mcp_server_stdio.py")],
    },
    "data_fetch_mcp_http": {
        "transport": "streamable_http",
        "url": "http://127.0.0.1:8888/mcp",
    }
}

def payload_of(result):
    """helper function: ainvoke returns a text block -> str."""
    if isinstance(result, list):
        result = result[0]["text"]
    try:
        return json.loads(result)
    except (TypeError, ValueError):
        return result


async def main():
    client = MultiServerMCPClient(connections)

    # Get all tools from the STDIO server.
    # NOTE: tools are bound to the session they were loaded from, so the stdio
    # session MUST stay open (nested below) while any stdio tool is invoked.
    async with client.session("data_fetch_mcp_stdio") as session:
        # list all the tools
        tools = await load_mcp_tools(session)
        tools_by_name = {t.name: t for t in tools}
        print("Available tools: ", [tools_by_name])

        # Get all tools from the HTTP server
        async with client.session("data_fetch_mcp_http") as session_http:
            # list all the tools
            tools_http = await load_mcp_tools(session_http)
            tools_by_name_http = {t.name: t for t in tools_http}
            print("Available tools: ", [tools_by_name_http])

            # fetch
            for tool in [tools_by_name["fetch"], tools_by_name_http["fetch_http"]]:
                customer_data = payload_of(await tool.ainvoke({"customerID": 6870}))
                print(f"Fetched by {tool.name}:    ", customer_data)

            # pass the fetched payload into process
            for tool in [tools_by_name["process"], tools_by_name_http["process_http"]]:
                process_result = payload_of(await tool.ainvoke({"data": customer_data}))
                print(f"Processed by {tool.name}:  ", process_result)

if __name__ == "__main__":
    asyncio.run(main())
