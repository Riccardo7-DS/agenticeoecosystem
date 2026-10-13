import importlib
import logging
import pkgutil
import os
from google.adk.agents.llm_agent import Agent
from agenticeoecosystem.agents.subagents.llm_init import llm


logger = logging.getLogger(__name__)

ALL_SUB_AGENTS: list[Agent] = []

_PACKAGE = __name__ if __name__ != "__main__" else "agenticeoecosystem.agents.subagents"
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))

# Comma-separated list of suffixes (the part after "sub_agent_") to load, e.g.
# "vlm,gee". Unset/empty means "load every sub_agent_*.py module" (default).
_ENABLED = {
    name.strip()
    for name in os.environ.get("AGENTICEO_SUBAGENTS", "").split(",")
    if name.strip()
}

for _, module_name, _ in pkgutil.iter_modules([_PKG_DIR]):
    if not module_name.startswith("sub_agent_"):
        continue
    if _ENABLED and module_name[len("sub_agent_"):] not in _ENABLED:
        continue
    module = importlib.import_module(f"{_PACKAGE}.{module_name}")
    for value in vars(module).values():
        if isinstance(value, Agent) and value not in ALL_SUB_AGENTS:
            ALL_SUB_AGENTS.append(value)
            logger.info("Loaded sub-agent '%s' from %s", value.name, module_name)

if not ALL_SUB_AGENTS:
    logger.warning("No sub-agents were initialized — check MCP toolset/env configs.")
print(f"Loaded {len(ALL_SUB_AGENTS)} sub-agents: {[agent.name for agent in ALL_SUB_AGENTS]}")