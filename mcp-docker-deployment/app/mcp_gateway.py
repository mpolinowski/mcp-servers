from typing import Dict
import os
from fastmcp import FastMCP
from fastmcp.server import create_proxy
from pydantic import BaseModel

mcp = FastMCP("My Server")

venv_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".venv"
)

class CustomerData(BaseModel):
    data: str
    id: int

# Tool hints — the two local tools are pure mocks (no side effects). The
# proxy-mounted upstream tools carry their own hints from their servers.
_READ_MOCK = {
    "read_only_hint": True,
    "destructive_hint": False,
    "idempotent_hint": True,
    "open_world_hint": False,
}

@mcp.tool(annotations=_READ_MOCK)
def fetch(customerID: int) -> CustomerData:
    '''Use this tool to fetch customer data'''
    # mock-api fetch request
    return CustomerData(data=f"Data retrieved for: ", id=customerID)

@mcp.tool(annotations=_READ_MOCK)
def process(data: CustomerData) -> Dict:
    '''Use this tool to process customer data'''
    # mock-data processing
    return {"data": f"Data for {data.id} has been processed"}


mcp.mount(
    create_proxy({
        "mcpServers": {
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
    })
)


mcp.mount(
    create_proxy({
        "mcpServers": {
          "agent_terminal_tools": {
                "transport": "stdio",
                "command": os.path.join(venv_path, "bin", "python"),
                "args": [os.path.join(venv_path, "bin", "agent_terminal_tools")],
            }
        }
    })
)



if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8888)
