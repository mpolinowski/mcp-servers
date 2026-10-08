import os
import asyncio
import json

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import Connection
from langchain_mcp_adapters.tools import load_mcp_tools

mcp_server_script = os.path.join((os.path.dirname(os.path.abspath(__file__))), "mcp_server_stdio.py")
venv_path = os.path.join((os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".venv")

# MCP Server config
connections: dict[str, Connection] = {
    "data_fetch_mcp_stdio": {
        "transport": "stdio",
        "command": os.path.join(venv_path, "bin", "python"),
        "args": [str(mcp_server_script)],
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
        print("Available tools: ", [tools_by_name])

        def payload_of(result):
            """helper function: ainvoke returns a text block -> str."""
            if isinstance(result, list):
                result = result[0]["text"]
            try:
                return json.loads(result)
            except (TypeError, ValueError):
                return result

        # fetch
        customer_data = payload_of(await tools_by_name["fetch"].ainvoke({"customerID": 6870}))
        print("Fetched:    ", customer_data)

        # pass the fetched payload into process
        process_result = payload_of(await tools_by_name["process"].ainvoke({"data": customer_data}))
        print("Processed:  ", process_result)

if __name__ == "__main__":
    asyncio.run(main())
