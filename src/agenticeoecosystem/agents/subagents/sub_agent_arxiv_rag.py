from agenticeoecosystem.agents.subagents.llm_init import llm
from google.adk.agents.llm_agent import Agent
from agenticeoecosystem.definitions import ROOT_DIR
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters
from typing import Optional

import logging
logger = logging.getLogger(__name__)



_ARXIV_SYSTEM_PROMPT = (
    "You are an assistant for querying and retrieving scientific papers from an ArXiv MCP server. "
    "Use the available tools to search for papers, retrieve abstracts, download full texts, and perform semantic searches. "
    "When a user asks a question, determine which tool(s) to use and extract the necessary parameters from the query. "
    "For example, if a user asks 'Find recent papers on climate change and get their abstracts', you might use "
    "'search_papers' with the keyword 'climate change' and a recent date filter, and proper category, "
    "then use 'get_abstract' for each relevant paper found.\n"
    "IMPORTANT CONSTRAINTS:\n"
    "- Prefer 'get_abstract' over 'download_paper' unless the user explicitly requests the full text. "
    "  Abstracts contain enough information for most questions and avoid context overflows.\n"
    "- NEVER call 'download_paper' for more than one paper in a single task — full PDFs are very large "
    "  and will exceed the model's context window.\n"
    "- NEVER call 'get_abstract' for more than 5 papers in a single task; abstracts can still overflow context when batched too aggressively.\n"
    "- If the user only asks to find/fetch relevant papers, use discovery tools (e.g., search/list) and do not call 'get_abstract' or 'download_paper' unless explicitly requested.\n"
    "- If asked to summarise or analyse multiple papers, use 'get_abstract' for all of them."
)

toolset  = [McpToolset(
                     connection_params=StdioConnectionParams(
                         server_params = StdioServerParameters(
                             command='uv',
                             args=[
                                    "tool",
                                    "run",
                                    "--python", "3.12",
                                    "arxiv-mcp-server",
                                    "--storage-path", f"{ROOT_DIR}/output/arxiv"
                                ]),
                             timeout=300))]
def _fetch_agent(toolset) -> Optional[Agent]:
        if toolset:
            return Agent(
                model=llm,
                name='arxiv_rag_agent',
                description='A helpful assistant for answering user questions related to ArXiv paper retrieval.',
                instruction=_ARXIV_SYSTEM_PROMPT,
                tools=toolset,
                mode='single_turn'
            )
        else:
            logger.warning("ArXiv RAG agent not initialized due to missing toolset.")
            return None


arxiv_rag_agent = _fetch_agent(toolset)