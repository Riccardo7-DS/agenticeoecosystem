import asyncio
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv()


async def main():
    # Configure MCP servers
    mcp_servers = {
        "example": {
            "transport": "stdio",
            "command": "python",
            "args": [
                "/shared/home/egm/Projects/AgenticEOEcosystem/agenticeoecosystem/src/servers/example/stdio_localserver/example_server.py"
            ],
        },
        "eodashboard": {
            "transport": "stdio",
            "command": "python",
            "args": [
                "/shared/home/egm/Projects/AgenticEOEcosystem/agenticeoecosystem/src/servers/EOdashboard/eodashboard_server.py"
            ],
        },
    }

    # Initialize MCP client service
    mcp_client = MultiServerMCPClient(mcp_servers)

    # Get tools from all MCP servers
    tools = await mcp_client.get_tools()
    print(f"Loaded {len(tools)} tools from MCP servers")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description}")

    llm = ChatOllama(model="mistral")

    system_prompt = """You are a helpful assistant with access to tools.
        Complete the ENTIRE user request step by step.
        After using tools, always provide a clear summary of what you found.
        If the user asks for multiple things, make sure to do ALL of them before responding."""

    agent_executor = create_agent(llm, tools, system_prompt=system_prompt)

    result = await agent_executor.ainvoke(
        {
            "messages": [
                (
                    "user",
                    """Find a narrative about agriculture or crops. 
                    Then get the collection codes it references. 
                    Then search for one of those collections to see its details.
                    Finally, try to get 3 data items from that collection.""",
                )
            ]
        }
    )

    print(result)

    # Clean output (showing just the final response)
    print("\n" + "=" * 50)
    print("AGENT RESPONSE:")
    print("=" * 50)
    final_message = result["messages"][-1]
    print(f"\n{final_message.content}\n")

    # # Uncomment to see full conversation with tool calls:
    # print("\nFull conversation:")
    # for msg in result["messages"]:
    #     print(f"\n{msg.type}: {msg.content}")


if __name__ == "__main__":
    asyncio.run(main())
