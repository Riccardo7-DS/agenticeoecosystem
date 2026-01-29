import asyncio
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from definitions import ROOT_DIR
from agenticeoecosystem.config.mcp_config import MCP_SERVERS
import logging 
import os 

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


if __name__ == "__main__":
    from langchain_openai import ChatOpenAI
    import os 
    import asyncio

    api_url = os.getenv("MAIN_MODEL_URL")  # or FALLBACK_MODEL_URL
    api_key = os.getenv("MAIN_MODEL_API_KEY")  # or FALLBACK_MODEL_API_KEY
    model_name = os.getenv("MAIN_MODEL_NAME")


    llm = ChatOpenAI(
                base_url=api_url,
                api_key=api_key,
                model=model_name,
                temperature=0.3,
            )

    asyncio.run(test_query_mcp_with_llm(llm=llm))

