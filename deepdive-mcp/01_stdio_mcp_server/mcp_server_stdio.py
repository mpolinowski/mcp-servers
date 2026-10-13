from typing import Dict

from fastmcp import FastMCP
from pydantic import BaseModel

mcp = FastMCP()

class CustomerData(BaseModel):
    data: str
    id: int

# Tool hints — both tools are pure mocks (no side effects), so read-only,
# safe to repeat, and closed-world. Per the MCP spec these are *hints*, not
# guarantees; we set all four explicitly.
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

if __name__ == "__main__":
    mcp.run(transport="stdio")
