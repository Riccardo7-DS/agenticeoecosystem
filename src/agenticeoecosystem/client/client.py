import asyncio
from typing import Union, List, Literal
from langchain.agents import create_agent
from agenticeoecosystem import print_and_execute
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama, OllamaEmbeddings

from agenticeoecosystem.config.mcp_config import MCP_SERVERS
import os 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda
import os
from typing import List, Dict, TypedDict, Annotated

from pydantic import BaseModel, Field

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from langchain_mcp_adapters.client import MultiServerMCPClient

import logging 

logger = logging.getLogger(__name__)
load_dotenv()


SYSTEM_AGENTIC_PROMPT = """You are a helpful assistant with access to tools.
                    Complete the ENTIRE user request step by step.
                    After using tools, always provide a clear summary of what you found.
                    If no tool results are returned:
                    - Do not answer
                    - State explicitly that no data is available.
                    If the user asks for multiple things, make sure to do ALL of them before responding."""

SYSTEM_CODING_PROMPT = """
                    You are an expert python programmer.
                    Return ONLY Python Code.
                    Do NOT say you cannot execute code.
                    """


async def run_code_pipeline(model_name, user_prompt, temperature=0):
    """
    Code to run python code agent
    """
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system", SYSTEM_CODING_PROMPT

            ),
            ("human", "{input}"),
        ]
    )

    llm = ChatOllama(model=model_name, temperature=temperature)

    chain = (
        prompt_template
        | llm
        | StrOutputParser()
        | RunnableLambda(print_and_execute)
    )

    response = await chain.ainvoke({"input": user_prompt})
    return response

async def run_eve_agent(user_prompt: str,
               model_name:Literal["mistral","eve","llama3.1:8b",""] = "mistral", 
               temperature=0.3):
    
    # Initialize MCP client service
    mcp_client = MultiServerMCPClient(MCP_SERVERS)

    # Get tools from all MCP servers
    tools = await mcp_client.get_tools()
    print(f"Loaded {len(tools)} tools from MCP servers")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description}")

    if model_name not in ["mistral","eve", "llama3.1:8b", "glm-4.7-flash:q8_0"]:
        raise ValueError(f"Unsupported model: {model_name}")

    if model_name == "eve":
        logger.info("Using EVE model via OpenAI-compatible API")
        llm = ChatOpenAI(
            base_url=os.getenv("MAIN_MODEL_URL"),
            api_key=os.getenv("MAIN_MODEL_API_KEY"),
            model=os.getenv("MAIN_MODEL_NAME"),
            temperature=temperature,       
    )
    else:
        logger.info(f"Using {model_name} model via Ollama")
        llm = ChatOllama(model=model_name, temperature=temperature)


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
                    user_prompt,
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




class MultiToolPipeline:

    class State(TypedDict):
        messages: Annotated[list, add_messages]
        selected_tools: list[str]
        retry_count: int

    class QueryForTools(BaseModel):
        query: str = Field(...)

    def __init__(self, mcp_servers: List[str], user_query: str, model_name:str, temperature:float,
                 embedder_model:str="nomic-embed-text"):
        self.mcp_servers = mcp_servers
        self.user_query = user_query
        self.tool_registry = {}
        self.vector_store = None
        self.llm = self._initialize_model(model_name, temperature)
        self.graph = None
        self.embedder_model = embedder_model


    def _initialize_model(self, model_name, temperature):
        return ChatOllama(model=model_name, temperature=temperature)

    # ----------------------------
    # Fetch MCP tools
    # ----------------------------
    async def fetch_tools(self):
        client = MultiServerMCPClient(self.mcp_servers)
        tools = await client.get_tools()

        print(f"Loaded {len(tools)} tools")
        for i, t in enumerate(tools):
            print(f"  - {t.name}")

        self.tool_registry = {str(i): tool for i, tool in enumerate(tools)}

    # ----------------------------
    # Vector store
    # ----------------------------
    def setup_vector_store(self):
        docs = [
            Document(
                page_content=tool.description,
                id=id,
                metadata={"tool_name": tool.name},
            )
            for id, tool in self.tool_registry.items()
        ]

        embeddings = OllamaEmbeddings(model=self.embedder_model)
        self.vector_store = InMemoryVectorStore(embedding=embeddings)
        self.vector_store.add_documents(docs)

    # ----------------------------
    # Nodes
    # ----------------------------
    def _select_tools_node(self, state):
        last_message = state["messages"][-1]

        if isinstance(last_message, HumanMessage):
            query_text = last_message.content
        else:
            system_prompt = SystemMessage(
                "Generate a short query describing what tools are needed next."
            )
            llm_query = self.llm.bind_tools([self.QueryForTools], tool_choice=True)
            response = llm_query.invoke([system_prompt] + state["messages"])

            if response.tool_calls:
                query_text = response.tool_calls[0]["args"]["query"]
            else:
                # fallback: use last LLM message text
                query_text = response.content if hasattr(response, "content") else ""
                if not query_text:
                    # last resort: repeat user message
                    query_text = state["messages"][-1].content

        docs = self.vector_store.similarity_search(query_text)
        selected_ids = [doc.id for doc in docs]
        return {"selected_tools": selected_ids}


    def agent_node(self, state: State):
        selected_tools = [self.tool_registry[id] for id in state["selected_tools"]]
        llm_with_tools = self.llm.bind_tools(selected_tools)

        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def should_continue(self, state: State):
        """
        Custom retry logic:
        - If tool call happened → go to tools
        - If no tool call → finish
        """
        last_message = state["messages"][-1]

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"

        return END

    # ----------------------------
    # Build graph
    # ----------------------------
    def build_graph(self):
        builder = StateGraph(self.State)

        builder.add_node("select_tools", self._select_tools_node)
        builder.add_node("agent", self.agent_node)

        tool_node = ToolNode(tools=list(self.tool_registry.values()))
        builder.add_node("tools", tool_node)

        builder.add_edge(START, "select_tools")
        builder.add_edge("select_tools", "agent")

        builder.add_conditional_edges(
            "agent",
            self.should_continue
        )

        builder.add_edge("tools", "select_tools")

        self.graph = builder.compile()

    # ----------------------------
    # Run
    # ----------------------------
    async def run(self):
        # Use async graph invocation
        result = await self.graph.ainvoke({
            "messages": [("user", self.user_query)],
            "retry_count": 0
        })
        return result["messages"]



# ----------------------------
# Usage
# ----------------------------
import asyncio

async def main(args):
    from agenticeoecosystem import MCP_SERVERS
    filepath = "/home/egm/Desktop/agenticecosystems/data/Final_assignment.xlsx"

    user_query = f"""
    For the file in the filepath {filepath} in the "Final_assignment" sheet. The headers are in the second row. 
    Do the following: get the dataframe info, get the list of available columns.
    Clean the expert column. Summarize the description column. Categorize the commercial status and the cofunding type and also the cofunding binary.
    Subsequently, categorize the ideas with the weighted and the imposed categories.
    If tools are available, use them before answering.
    Return the processed dataframe.
    """

    pipeline = MultiToolPipeline(MCP_SERVERS, 
                                 user_query, 
                                 model_name=args.model, 
                                 temperature=args.temperature)

    await pipeline.fetch_tools()
    pipeline.setup_vector_store()
    pipeline.build_graph()

    messages = await pipeline.run()

    for msg in messages:
        msg.pretty_print()
    




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
    # asyncio.run(run_eve_agent(narrative_query, model=args.model, temperature=args.temperature))

    asyncio.run(main(args))