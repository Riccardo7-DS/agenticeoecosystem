from agenticeoecosystem.agents.subagents.llm_init import llm
from google.adk.agents.llm_agent import Agent
import logging
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters
from typing import Optional
from agenticeoecosystem.definitions import ROOT_DIR

logger = logging.getLogger(__name__)

_MOOC_SYSTEM_PROMPT = (
    "You are a retrieval assistant for Earth Observation MOOC (Massive Open Online Course) transcripts. "
    "Use the search_moocs tool to find relevant content from EO educational courses and return "
    "the retrieved chunks to the caller. Do not summarise or interpret — return the raw context."
)

toolset  = [McpToolset(
                     connection_params=StdioConnectionParams(
                         server_params = StdioServerParameters(
                             command='python',
                             args=[f"{ROOT_DIR}/src/agenticeoecosystem/servers/esa_moocs/server.py"]),
                             timeout=300))]
def _fetch_agent(toolset) -> Optional[Agent]:
        if toolset:
            return Agent(
                model=llm,
                name='mooc_rag_agent',
                description='A helpful assistant for answering user questions related to MOOC transcripts.',
                instruction=_MOOC_SYSTEM_PROMPT,
                tools=toolset,
                mode='single_turn'
            )
        else:
            logger.warning("MOOC RAG agent not initialized due to missing toolset.")
            return None


mooc_rag_agent = _fetch_agent(toolset)