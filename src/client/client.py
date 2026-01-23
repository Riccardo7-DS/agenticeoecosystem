import asyncio
import os
from langchain_mcp_adapters.client import MultiServerMCPClient  
from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate

# # Load environment variables from .env file (if it exists)
# try:
#     from dotenv import load_dotenv
#     load_dotenv()
# except ImportError:
#     pass  # dotenv not installed, use terminal exports instead

# # Optional: Enable LangSmith tracing (web UI for debugging)
# if os.getenv("LANGCHAIN_TRACING_V2") == "true":
#     print("✓ LangSmith tracing enabled - view at https://smith.langchain.com/")
# else:
#     print("X LangSmith tracing disabled. To enable, set LANGCHAIN_TRACING_V2=true")

async def main():
    # Configure MCP servers
    mcp_servers = {
        "example": {
            "transport": "stdio",
            "command": "python",
            "args": ["/shared/home/egm/Projects/AgenticEOEcosystem/agenticeoecosystem/src/servers/example/stdio_localserver/example_server.py"]
        },
        "eodashboard": {
            "transport": "stdio",
            "command": "python",
            "args": ["/shared/home/egm/Projects/AgenticEOEcosystem/agenticeoecosystem/src/servers/EOdashboard/eodashboard_server.py"]
        },
        # Add more servers as you build them:
        # "rag": {
        #     "transport": "stdio",
        #     "command": "python",
        #     "args": ["path/to/rag_server.py"]
        # }
    }
    
    # Initialize MCP client service
    mcp_client = MultiServerMCPClient(mcp_servers)
    
    # Get tools from all MCP servers
    tools = await mcp_client.get_tools()
    print(f"Loaded {len(tools)} tools from MCP servers")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description}")
    
    # Option 1: OpenAI (requires API key: export OPENAI_API_KEY=...)
    # from langchain_openai import ChatOpenAI
    # llm = ChatOpenAI(model="gpt-4", temperature=0.7)
    
    # # Option 2: Anthropic Claude (requires API key: export ANTHROPIC_API_KEY=...)
    # from langchain_anthropic import ChatAnthropic
    # llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.7)

    # Option 3: Ollama (local, no API key, but needs ollama installed and running)
    from langchain_ollama import ChatOllama
    
    # Recommended models for complex tool calling (install with: ollama pull <model>)
    llm = ChatOllama(model="mistral")  # 7B, excellent for tool calling
    # llm = ChatOllama(model="qwen2.5:7b")  # Bigger version, better reasoning
    # llm = ChatOllama(model="llama3.1:8b")  # Meta's model, very capable
    # llm = ChatOllama(model="qwen2.5:3b")  # Smallest, faster but limited reasoning
    
    # Option 4: Your custom VLLM model (when ready)
    # from langchain_community.llms import VLLM
    # llm = VLLM(
    #     model="path/to/your/model",  # Update with your model path
    #     trust_remote_code=True,
    #     temperature=0.7,
    #     max_tokens=2048,
    # )
    
    # Create agent using LangGraph's React agent (simpler, more modern approach)
    # Add custom system prompt to control agent behavior
    system_prompt = """You are a helpful assistant with access to tools.
Complete the ENTIRE user request step by step.
After using tools, always provide a clear summary of what you found.
If the user asks for multiple things, make sure to do ALL of them before responding."""
    
    agent_executor = create_agent(
        llm, 
        tools,
        system_prompt=system_prompt
    )
    
    # Example usage - full pipeline test (narrative → data)
    result = await agent_executor.ainvoke({
        "messages": [("user", """Find a narrative about agriculture or crops. 
Then get the collection codes it references. 
Then search for one of those collections to see its details.
Finally, try to get 3 data items from that collection.""")]
    })

    print(result)
    
    # Clean output - show just the final response
    print("\n" + "="*50)
    print("AGENT RESPONSE:")
    print("="*50)
    final_message = result["messages"][-1]
    print(f"\n{final_message.content}\n")
    
    # # Uncomment to see full conversation with tool calls:
    # print("\nFull conversation:")
    # for msg in result["messages"]:
    #     print(f"\n{msg.type}: {msg.content}")

if __name__ == "__main__":
    asyncio.run(main())

