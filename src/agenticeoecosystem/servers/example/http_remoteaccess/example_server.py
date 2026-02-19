import requests

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Example MCP Server")

@mcp.tool()
def greet(name: str) -> str:
    return f"Hello, {name}!"

if __name__ == "__main__":
    mcp.run(transport="http", host="127.0.0.1", port=8000)