# agenticeoecosystem/agents/subagents/vlm_agent.py

import logging
from typing import Optional

from google.adk.agents.llm_agent import Agent
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

from agenticeoecosystem.agents.subagents.vlm_init import vlm
from agenticeoecosystem.definitions import ROOT_DIR

logger = logging.getLogger(__name__)

_VLM_SYSTEM_PROMPT = """You are a VLM reasoning agent.

You receive a file path (a single image or a directory of images) and a
natural-language request from an orchestrator agent. Use the `resolve_images`
tool first to find out what images are available at the given path, then use
`load_images` to actually load them into context before answering.

Rules:
- If the path is a directory, only the first 10 images (sorted by name) will
  ever be loaded — mention it if the directory contained more.
- If a path doesn't resolve to any valid image, say so clearly instead of
  guessing.
- Always ground your answer in the images you were actually given.
"""

toolset = [
    McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command="python",
                args=[f"{ROOT_DIR}/src/agenticeoecosystem/servers/vlm_tools/server.py"],
            ),
            timeout=300,
        )
    )
]


def _fetch_agent(toolset: list) -> Optional[Agent]:
    if not toolset:
        logger.warning("VLM agent not initialized due to missing toolset.")
        return None
    return Agent(
        model=vlm,
        name="vlm_agent",
        description="A helpful assistant for answering user questions related to reasoning over image data.",
        instruction=_VLM_SYSTEM_PROMPT,
        tools=toolset,
    )


vlm_agent = _fetch_agent(toolset)