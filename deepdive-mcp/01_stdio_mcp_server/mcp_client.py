import os

import asyncio
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters, client

# MCP server path
mcp_server_script = os.path.join((os.path.dirname(os.path.abspath(__file__))), "mcp_server_stdio.py")

server_params = StdioServerParameters(
    command="python",
    args=[str(mcp_server_script)],
    env={}
)

# Create client session
async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # list available tools
            tools = await session.list_tools()
            print("Available tools: ", tools)

            # call the fetch and process tool
            data = await session.call_tool("fetch", arguments={"customerID": 6870})
            if data.is_error:
                raise RuntimeError(f"fetch failed: {data.content}")
            # extract the actual CustomerData payload
            customer_data = data.structured_content
            process_result = await session.call_tool("process", arguments={"data": customer_data})
            print("Processed:  ", process_result.structured_content)


if __name__ == "__main__":
    asyncio.run(main())
