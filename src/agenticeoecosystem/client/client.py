import asyncio
from typing import Union, List, Literal
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from agenticeoecosystem.config.mcp_config import MCP_SERVERS
import os 
from dotenv import load_dotenv
import logging 

logger = logging.getLogger(__name__)
load_dotenv()

async def run_eodashboard_agent(query: str,
               model:Literal["mistral","eve"] = "mistral", 
               temperature=0.3):
    
    # Initialize MCP client service
    mcp_client = MultiServerMCPClient(MCP_SERVERS)

    # Get tools from all MCP servers
    tools = await mcp_client.get_tools()
    print(f"Loaded {len(tools)} tools from MCP servers")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description}")

    if model not in ["mistral","eve"]:
        raise ValueError(f"Unsupported model: {model}")

    if model == "eve":
        logger.info("Using EVE model via OpenAI-compatible API")
        llm = ChatOpenAI(
            base_url=os.getenv("MAIN_MODEL_URL"),
            api_key=os.getenv("MAIN_MODEL_API_KEY"),
            model=os.getenv("MAIN_MODEL_NAME"),
            temperature=temperature,       
    )
    else:
        logger.info(f"Using {model} model via Ollama")
        llm = ChatOllama(model=model, temperature=temperature)


    system_prompt = """You are a helpful assistant with access to tools.
                        Complete the ENTIRE user request step by step.
                        After using tools, always provide a clear summary of what you found.
                        If no tool results are returned:
                        - Do not answer
                        - State explicitly that no data is available.
                        If the user asks for multiple things, make sure to do ALL of them before responding."""

    agent_executor = create_agent(llm, tools, system_prompt=system_prompt)

    result = await agent_executor.ainvoke(
        {
            "messages": [
                (
                    "user",
                    query,
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

    return final_message.content


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MCP Client Agent")
    parser.add_argument("--model", type=str, default=os.environ.get("model", "mistral"),
                        help="Model to use: 'mistral' (Ollama) or 'eve' (OpenAI-compatible)")
    parser.add_argument("--temperature", type=float, default=os.environ.get("temperature", 0.3),
                        help="Temperature for the LLM")
    args = parser.parse_args()

    narrative_query = (
        """Find a narrative about agriculture or crops. 
            Then get the collection codes it references. 
            Then search for one of those collections to see its details.
            Finally, try to get 3 data items from that collection."""
    )
    asyncio.run(run_eodashboard_agent(narrative_query, model=args.model, temperature=args.temperature))