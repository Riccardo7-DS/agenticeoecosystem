import asyncio
import os

from dotenv import load_dotenv

load_dotenv()

import requests
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport, SSETransport, NodeStdioTransport, PythonStdioTransport

EMAIL = os.environ.get("EMAIL")
PASSWORD = os.environ.get("PASSWORD")

def login(base_url: str, email: str, password: str) -> str:
    print(f"Logging in with email: {email} and password: ***")
    resp = requests.post(
        f"{base_url}/login",
        json={"email": email, "password": password},
        timeout=30,
    )
    if resp.status_code != 200:
        print(f"Login failed ({resp.status_code}): {resp.text}")
    print(resp.json())
    token = resp.json()["access_token"]
    return token


async def main() -> None:
    email = EMAIL
    password = PASSWORD
    if not email or not password:
        raise ValueError(
            "Set EMAIL and PASSWORD environment variables before running this script."
        )

    access_token = login(
        "https://agentic-api.eve-chat.chat", email, password
    )
    client = Client[
        PythonStdioTransport
        | NodeStdioTransport
        | SSETransport
        | StreamableHttpTransport
    ](
        "https://agentic-api.eve-chat.chat/mcp/eodashboard",
        auth=access_token,
    )

    async with client:
        tools = await client.list_tools()
        print(tools)


if __name__ == "__main__":
    asyncio.run(main())
