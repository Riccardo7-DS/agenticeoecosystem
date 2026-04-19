from langchain_openai import ChatOpenAI
import base64
from langchain_mcp_adapters.client import MultiServerMCPClient
from agenticeoecosystem.config.mcp_config import MCP_SERVERS
import asyncio
import logging
from dotenv import load_dotenv
from typing import List, Optional, Literal
from langchain_core.language_models.chat_models import BaseChatModel
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
from langchain_core.messages import HumanMessage, trim_messages

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

import os
username = os.getenv("EVE_USERNAME")
password = os.getenv("EVE_PASSWORD")
credentials = base64.b64encode(f"{username}:{password}".encode()).decode() 




logger = logging.getLogger(__name__)
load_dotenv()
async def load_tools_from_mcp():

    mcp_client = MultiServerMCPClient(MCP_SERVERS)
    tools = await mcp_client.get_tools()
    logger.info(f"Loaded {len(tools)} tools from MCP servers")
    for tool in tools:
        logger.info(f"  - {tool.name}: {tool.description}")
    return tools

mcp_tools = asyncio.run(load_tools_from_mcp())

def load_llm(eve=False):
    # Declare the LLM with the EVE model and credentials
    if eve:
        return ChatOpenAI(
            base_url="https://phi-gateway.tce.ples.dedyn.io/neohpc-eve-internal/18000/v1",
            api_key="EMPTY",
            model="eve_v0.4",
                    temperature=0,
                    default_headers={
            "Authorization": f"Basic {credentials}"
        })
    else:
        llm = init_chat_model('rnj-1:latest', model_provider="ollama", temperature=0)
        return llm

eve_llm = load_llm()

def make_agents(llm, tools, system_prompt):
    pass



class State(MessagesState):
    next: str


def make_supervisor_node(llm: BaseChatModel, members: list[str]) -> str:
    # The routing options for the supervisor are the workers or FINISH if no more workers are needed
    options = ["FINISH"] + members 

    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        f" following workers: {members}. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When finished,"
        " respond with FINISH."
    )

    class Router(TypedDict):
        """Worker to route to next. If no workers needed, route to FINISH."""

        next: Literal[*options]

    def supervisor_node(state: State) -> Command[Literal[*members, "__end__"]]:
        """An LLM-based router."""
        messages = [
            {"role": "system", "content": system_prompt},
        ] + state["messages"]
        # response = llm.with_structured_output(Router).invoke(messages)
        response = llm.invoke(messages)
        logger.info(f"Supervisor response: {response}")
        # logger.info(f"Supervisor routing decision: {response['next']}")
        goto = response["next"]
        if goto == "FINISH":
            goto = END

        return Command(goto=goto, update={"next": goto})

    return supervisor_node



eo_agent = create_agent(eve_llm, tools=mcp_tools[:4])
stac_agent = create_agent(eve_llm, tools=mcp_tools[4:8])
gee_agent = create_agent(eve_llm, tools=mcp_tools[8:12])


def stac_node(state: State) -> Command[Literal["supervisor"]]:
    result = stac_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="stac")
            ]
        },
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="supervisor",
    )



def eo_node(state: State) -> Command[Literal["supervisor"]]:
    result = eo_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="eo")
            ]
        },
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="supervisor",
    )

def gee_node(state: State) -> Command[Literal["supervisor"]]:
    result = gee_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="gee")
            ]
        },
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="supervisor",
    )

supervisor_node = make_supervisor_node(eve_llm, ["eo", "gee", "stac"])

eve_ma_system = StateGraph(State)
eve_ma_system.add_node("supervisor", supervisor_node)
eve_ma_system.add_node("eo", eo_node)
eve_ma_system.add_node("gee", gee_node)
eve_ma_system.add_node("stac", stac_node)

eve_ma_system.add_edge(START, "supervisor")
eve_graph = eve_ma_system.compile()

from IPython.display import Image, display

# display(Image(eve_graph.get_graph().draw_mermaid_png()))
with open("output.png", "wb") as f:
    image_bytes = eve_graph.get_graph().draw_mermaid_png()
    f.write(image_bytes)

for s in eve_graph.stream(
    {"messages": [("user", "Find the narratives related to Covid-19")]},
    {"recursion_limit": 100},
):
    print(s)
    print("---")
# # Example usage
# response = eve_llm.invoke("What is the capital of France?")
# print(response)