# from __future__ import annotations
import asyncio
from typing import Union, List, Literal, Dict, TypedDict, Annotated
from langchain.agents import create_agent
from agenticeoecosystem import print_and_execute
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.chat_models import init_chat_model

from agenticeoecosystem.config.mcp_config import MCP_SERVERS
import os 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda

from pydantic import BaseModel, Field

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from collections import defaultdict

import logging 

logger = logging.getLogger(__name__)
load_dotenv()


# this is the output from the orchestrator agent, which decides which worker.
# Make sure to include the sanitized name of the worker, meaning replace ":" with "_" and so on, to match the safe_worker_agent_names list. 
class RoutingOutput(BaseModel):
    next_agent: Literal["phi4_mini_latest", "qwen2_5_coder_7b", "eve", "END"]

class State(TypedDict):
        """
        Should be improved later to have structure, prep for production use. For now, it's just a dict with messages and next_agent.
        """

        messages: Annotated[list, add_messages]
        next_agent: str

class multi_agent_pipeline:
    

    def __init__(self, orchestrator_agent_name : str, worker_agent_names: [str], mcp_servers:List[str]):
        self.worker_agent_names = worker_agent_names
        self.safe_worker_agent_names = [self._sanitize_name(name) for name in worker_agent_names] # for lang graph node naming
        self.mcp_tools, self.tool_registry = self.__init_mcp_tools__(mcp_servers)
        self.orchestrator_model = self.__init_models__(orchestrator_agent_name)
        self.worker_models = [self.__init_models__(worker_agent_name) for worker_agent_name in self.worker_agent_names]
        self.__init_agents__()
        self.build_graph()

        # self.orchestrator_agent = self.__init_agent__(orchestrator_agent_name)
        # self.worker_agent = self.__init_agent__(worker_agent_name)
    
    def _sanitize_name(self, name: str) -> str:
        return name.replace(":", "_",).replace("-", "_").replace(".", "_")

    def __init_models__(self, agent_name: str) -> BaseModel:
        """
        Initialize a language model based on the agent name.
        For "eve", we use ChatOpenAI. For others, we use Ollama.
        """
        if agent_name == 'eve':
            logger.info(f"Initializing {agent_name} with ChatOpenAI")
            llm = ChatOpenAI(
            base_url=os.getenv("MAIN_MODEL_URL"),
            api_key=os.getenv("MAIN_MODEL_API_KEY"),
            model=os.getenv("MAIN_MODEL_NAME"),
            temperature=0,
        )
        else:  
            logger.info(f"Initializing {agent_name} with Ollama")
            llm = init_chat_model(
            agent_name,
            model_provider="ollama",
            temperature=0
        )
        return llm

    def __init_mcp_tools__(self, mcp_servers:List[str]):
        """
        Initialize MCP tools by connecting to the specified MCP servers and retrieving their tools.
        Currently tools are all assigned to the worker agent, indexed at 0, but in the future we could assign to other agents as required.
        """
        tool_registry = defaultdict(list)
        mcp_client = MultiServerMCPClient(mcp_servers)
        tools =  asyncio.run(mcp_client.get_tools())
        logger.info(f"Loaded {len(tools)} tools from MCP servers")
        for tool in tools:
            logger.info(f"  - {tool.name}: {tool.description}")
            tool_registry[self.safe_worker_agent_names[0]].append(tool.name)
        return tools, tool_registry

    def __init_agents__(self):
        """
        Initialize orchestrator and worker agents.
        Workers get tools bound.
        Orchestrator decides which worker to call.
        """
        orchestrator_prompt = ChatPromptTemplate.from_messages([
                                                    ("system",
                                                    "You are a routing controller.\n"
                                                    f"Available workers: {self.safe_worker_agent_names}.\n"
                                                    "Return ONLY one of these worker names or END."),
                                                    MessagesPlaceholder(variable_name="messages")
                                                ])
        # For now Orchestrator does NOT get tools directly, but the worker agents do.
        self.orchestrator_agent = orchestrator_prompt | self.orchestrator_model.with_structured_output(RoutingOutput)
        # Workers get tools
        self.worker_agents = {}

        for i, worker_name in enumerate(self.safe_worker_agent_names):
            worker_model = self.worker_models[i]

            # Assign tools specific to that worker
            assigned_tools = [
                tool for tool in self.mcp_tools
                if tool.name in self.tool_registry[self.safe_worker_agent_names[i]]
            ]
            tool_bound_model = worker_model.bind_tools(assigned_tools)
            tools_with_descriptions = "\n".join([f"- {tool.name}: {tool.description}" for tool in assigned_tools])
            worker_prompt = ChatPromptTemplate.from_messages([
                                        ("system",
                                        f"You are worker: {worker_name}.\n"
                                        f"You've the following MCP Tools available: {tools_with_descriptions}\n"
                                        "You can use tools when required.\n"
                                        "If a tool is relevant, call it properly."),
                                        MessagesPlaceholder(variable_name="messages")
                                    ])
            agent = worker_prompt | tool_bound_model

            self.worker_agents[worker_name] = agent

    

    def orchestrator_node(self, state: State):

        # structured_model = self.orchestrator_agent.with_structured_output(RoutingOutput)
        structured_model = self.orchestrator_agent

        try:
            response = structured_model.invoke(state["messages"])
            selected_worker = response.next_agent
        except Exception as e:
            logger.error(f"Routing failed: {e}")
            selected_worker = "END"

        return {
            "messages": state["messages"],  # do NOT append routing response
            "next_agent": selected_worker
        }

    # def worker_node(self, worker_name: str):
    #     def _node(state: State):
    #         agent = self.worker_agents[worker_name]
    #         response = agent.invoke(state["messages"])
    #         return {"messages": [response]}
    #     return _node

    def worker_node(self, worker_name: str):
        def _node(state: State):
            agent = self.worker_agents[worker_name]
            response = agent.invoke(state["messages"])

            return {
                "messages": [response],
                "next_agent": "tools" if response.tool_calls else "orchestrator"
            }

        return _node

    
    
    # def build_graph(self):

    #     builder = StateGraph(State)

    #     # Add orchestrator
    #     builder.add_node("orchestrator", self.orchestrator_node)

    #     # Add worker nodes dynamically
    #     for worker_name in self.worker_agent_names:
    #         builder.add_node(worker_name, self.worker_node(worker_name))

    #     # Tool node
    #     from langgraph.prebuilt import ToolNode
    #     tool_node = ToolNode(self.mcp_tools)
    #     builder.add_node("tools", tool_node)

    #     # Flow
    #     builder.add_edge(START, "orchestrator")

    #     # Routing logic
    #     builder.add_conditional_edges(
    #         "orchestrator",
    #         lambda state: state["next_agent"]
    #     )

    #     # After worker → tools if needed
    #     for worker_name in self.worker_agent_names:
    #         builder.add_edge(worker_name, "tools")

    #     builder.add_edge("tools", "orchestrator")

    #     self.graph = builder.compile()
    def build_graph(self):

        builder = StateGraph(State)

        builder.add_node("orchestrator", self.orchestrator_node)

        for worker_name in self.safe_worker_agent_names:
            builder.add_node(worker_name, self.worker_node(worker_name))

        tool_node = ToolNode(self.mcp_tools)
        builder.add_node("tools", tool_node)

        builder.add_edge(START, "orchestrator")

        # Safe routing
        routing_map = {name: name for name in self.safe_worker_agent_names}
        routing_map["END"] = END

        builder.add_conditional_edges(
            "orchestrator",
            lambda state: state["next_agent"],
            routing_map
        )

        # Worker conditional routing
        for worker_name in self.safe_worker_agent_names:
            builder.add_conditional_edges(
                worker_name,
                lambda state: state["next_agent"],
                {
                    "tools": "tools",
                    "orchestrator": "orchestrator"
                }
            )

        builder.add_edge("tools", "orchestrator")

        self.graph = builder.compile()

    def print_graph_mermaid(self):
        if not self.graph:
            raise ValueError("Graph not built yet.")

        print(self.graph.get_graph().draw_mermaid())


    def run(self, query: str, max_steps: int = 15):

        initial_state = {
            "messages": [HumanMessage(content=query)],
            "next_agent": ""
        }

        steps = 0
        state = initial_state

        for step in self.graph.stream(state):
            logger.info("================================================")
            logger.info(f"Step {steps}: Current state: {state}")
            steps += 1
            state = step

            if steps > max_steps:
                logger.warning("Max steps exceeded. Forcing termination.")
                break
        logger.info("================================================")
        logger.info("Pipeline execution completed. with final state:")
        logger.info(state)
        return state['orchestrator']['messages'][-1].content

pipeline = multi_agent_pipeline(
    orchestrator_agent_name="rnj-1:latest",
    worker_agent_names=["phi4-mini:latest"],
    mcp_servers=MCP_SERVERS
)


print(type(pipeline.orchestrator_model))
print(type(pipeline.worker_models[0]))
# print(pipeline.mcp_tools)
print(pipeline.tool_registry)
print(pipeline.orchestrator_agent)
print(pipeline.worker_agents)
pipeline.print_graph_mermaid()
response = pipeline.run("What are the available MCP tools?")
print(response)