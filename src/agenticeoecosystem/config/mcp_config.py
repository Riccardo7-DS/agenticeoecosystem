"""MCP (Model Context Protocol) servers configuration."""

from agenticeoecosystem.definitions import ROOT_DIR


MCP_SERVERS = {
    # "example": {
    #     "transport": "stdio",
    #     "command": "python",
    #     "args": [
    #         f"{ROOT_DIR}/src/agenticeoecosystem/servers/example/stdio_localserver/example_server.py"
    #     ],
    # },
    "eodashboard": {
        "transport": "stdio",
        "command": "python",
        "args": [
            f"{ROOT_DIR}/src/agenticeoecosystem/servers/EOdashboard/eodashboard_server.py"
        ],
    },
    "gee-mcp": {
      "transport": "stdio",
      "command": "python",
      "args": [f"{ROOT_DIR}/src/agenticeoecosystem/servers/gee/server.py"],
      "env": {}
    },
}
