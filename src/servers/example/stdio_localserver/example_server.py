import requests

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Example MCP Server")

@mcp.tool()
def greet(name: str) -> str:
    """Greet a person by name."""
    return f"Hello, {name}!"

if __name__ == "__main__":
    mcp.run()