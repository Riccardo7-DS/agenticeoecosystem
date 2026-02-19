# ESA documents MCP
An MCP server implementation for a RAG containing ESA documentation scraped from online sources.

Server info:

```{
  "mcpServers": {
    "esa-rag": {
      "command": "uv",
      "args": [
        "--directory",
        "/mcp_server/esa_rag", # PATH to project directory
        "run",
        "main.py" 
      ]
    }
  }
}
```