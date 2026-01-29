import asyncio
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from agenticeoecosystem.config.mcp_config import MCP_SERVERS
import logging 
import os 
from langchain_core.messages import SystemMessage, HumanMessage
import json 

logger = logging.getLogger(__name__)

load_dotenv()

MODEL_NAME = os.getenv("MAIN_MODEL_NAME")

class EVEAPIClient:
    """Client for chatting with a model through an OpenAI-compatible API."""
    
    def __init__(self, 
                 api_url: str, 
                 api_key: str, 
                 model: str = MODEL_NAME, 
                 temperature: float = 0.7):
        """
        Initialize the ModelAPIClient.
        
        Args:
            api_url: Base URL of the API (e.g., http://localhost:8000/v1)
            api_key: API key for authentication
            model: Model name to use
            temperature: Temperature for model responses (0.0 to 1.0)
        """
        self.llm = ChatOpenAI(
            base_url=api_url,
            api_key=api_key,
            model=model,
            temperature=temperature,
        )
    
    def invoke(self, message: str) -> str:
        """
        Send a synchronous message to the model.
        
        Args:
            message: The message to send
            
        Returns:
            The model's response
        """
        response = self.llm.invoke(message)
        return response.content
    
    async def ainvoke(self, message: str) -> str:
        """
        Send an asynchronous message to the model.
        
        Args:
            message: The message to send
            
        Returns:
            The model's response
        """
        response = await self.llm.ainvoke(message)
        return response.content

async def test_query_mcp_with_llm(llm=None, 
                                  message:str="""Find a narrative about agriculture or crops. 
                        Then get the collection codes it references. 
                        Then search for one of those collections to see its details.
                        Finally, try to get 3 data items from that collection."""):
    
    # Initialize MCP client service with configured servers
    mcp_client = MultiServerMCPClient(MCP_SERVERS)

    # Get tools from all MCP servers
    tools = await mcp_client.get_tools()
    logger.info(f"Loaded {len(tools)} tools from MCP servers")
    for tool in tools:
        logger.info(f"  - {tool.name}: {tool.description}")

    if llm is None:
        logger.info("No LLM provided, using default Ollama Mistral model")
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
                    message,
                )
            ]
        }
    )

    logger.info(result)

    # Clean output (showing just the final response)
    logger.info("\n" + "=" * 50)
    logger.info("AGENT RESPONSE:")
    logger.info("=" * 50)
    final_message = result["messages"][-1]
    logger.info(f"\n{final_message.content}\n")

    # # Uncomment to see full conversation with tool calls:
    # logger.info("\nFull conversation:")
    # for msg in result["messages"]:
    #     logger.info(f"\n{msg.type}: {msg.content}")

async def load_mcp_tools():
    mcp_client = MultiServerMCPClient(MCP_SERVERS)
    tools = await mcp_client.get_tools()

    logger.info("Loaded MCP tools:")
    for t in tools:
        logger.info(f"  - {t.name}")

    return {t.name: t for t in tools}  # return dict for easier access

async def test_mcp_tools(tools):
    print("\n" + "="*60)
    print("TESTING MCP TOOLS INDIVIDUALLY")
    print("="*60)

    default_inputs = {
        "greet": {"name": "Riccardo"},
        "list_narratives": {"query": "agriculture"},
        "search_narratives": {"query": "crops"},
        "get_narrative_content": {"narrative_id": "1"},
        "list_themes": {},
        "get_narrative_indicators": {"narrative_id": "1"},
        "search_collections": {"query": "crops"},
        "get_collection_items": {"collection_code": "1"},
        "find_narratives_for_indicator": {"indicator_code": "1"},
    }

    for tool_name, tool in tools.items():
        try:
            input_data = default_inputs.get(tool_name, {"query": "test"})
            output = await tool.ainvoke(input_data)
            print(f"\nTool: {tool_name}")
            print(f"Input: {input_data}")
            print(f"Output: {output}")
        except Exception as e:
            print(f"\nTool: {tool_name} raised an error: {e}")

# ---------------------------------------------------------------------
# MANUAL PLAN -> EXECUTE -> SUMMARIZE
# ---------------------------------------------------------------------
async def run_mcp_query_with_execution_loop(llm, tools, user_query: str):
    """
    1. Ask the model to produce a JSON plan of which tools to call
    2. Execute the tools in Python
    3. Feed outputs back to the model for final summary
    """
    # 1️⃣ Ask for a plan
    system_prompt = SystemMessage(
        content=(
            "You are an intelligent assistant with access to the following tools:\n"
            + "\n".join([f"- {name}" for name in tools.keys()])
            + "\n\nGiven a user request, create a PLAN in JSON format describing which tools to call, "
              "with required arguments for each tool. Do NOT provide the final answer yet."
        )
    )
    human_prompt = HumanMessage(content=user_query)

    plan_response = await llm.ainvoke([system_prompt, human_prompt])

    try:
        plan_text = plan_response.content
    except AttributeError:
        plan_text = plan_response["choices"][0]["message"]["content"]

    # 2️⃣ Parse JSON plan
    try:
        plan = json.loads(plan_text)
    except json.JSONDecodeError:
        print("Failed to parse plan as JSON. Raw plan:")
        print(plan_text)
        return plan_text

    # Ensure plan is a list
    if not isinstance(plan, list):
        print("Plan is not a list. Wrapping in list.")
        plan = [plan]

    # 3️⃣ Execute each step
    execution_results = []
    for step in plan:
        tool_name = step.get("tool")
        input_args = step.get("input", {})

        if tool_name not in tools:
            execution_results.append({tool_name: f"Tool not found"})
            continue

        try:
            output = await tools[tool_name].ainvoke(input_args)
            execution_results.append({tool_name: output})
            print(f"Executed tool '{tool_name}' with input {input_args} -> {output}")
        except Exception as e:
            execution_results.append({tool_name: f"Error: {e}"})

    # 4️⃣ Ask LLM to summarize
    summary_prompt = SystemMessage(
        content="You now have the results of each tool execution. Provide a clear final answer to the user request."
    )
    summary_human = HumanMessage(
        content=f"User request: {user_query}\n\nTool results:\n{json.dumps(execution_results, indent=2)}"
    )

    summary_response = await llm.ainvoke([summary_prompt, summary_human])

    try:
        return summary_response.content
    except AttributeError:
        return summary_response["choices"][0]["message"]["content"]

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
if __name__ == "__main__":
    llm = ChatOpenAI(
        base_url=os.getenv("MAIN_MODEL_URL"),
        api_key=os.getenv("MAIN_MODEL_API_KEY"),
        model=os.getenv("MAIN_MODEL_NAME"),
        temperature=0.3,
    )

    user_query = (
        "Find a narrative about agriculture or crops. "
        "Then get the collection codes it references. "
        "Then search for one of those collections to see its details. "
        "Finally, try to get 3 data items from that collection."
    )

    async def main():
        # Load MCP tools
        tools = await load_mcp_tools()

        # Test MCP tools individually
        await test_mcp_tools(tools)

        # Run the LLM query with execution loop
        result = await run_mcp_query_with_execution_loop(llm, tools, user_query)

        print("\n" + "="*60)
        print("FINAL AGENT RESPONSE")
        print("="*60)
        print(result)

    asyncio.run(main())