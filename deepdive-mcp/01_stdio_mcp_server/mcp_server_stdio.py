from typing import Dict

from fastmcp import FastMCP
from pydantic import BaseModel

mcp = FastMCP()

class CustomerData(BaseModel):
    data: str
    id: int

@mcp.tool()
def fetch(customerID: int) -> CustomerData:
    '''Use this tool to fetch customer data'''
    # mock-api fetch request
    return CustomerData(data=f"Data retrieved for: ", id=customerID)

@mcp.tool()
def process(data: CustomerData) -> Dict:
    '''Use this tool to process customer data'''
    # mock-data processing
    return {"data": f"Data for {data.id} has been processed"}

if __name__ == "__main__":
    mcp.run(transport="stdio")
