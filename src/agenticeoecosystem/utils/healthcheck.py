"""
healthcheck.py — Pre-flight reachability checks for the pipeline="multi"
backing services (VLM/orchestrator LLM endpoints, MCP tool servers), plus a
shared bordered-block log formatter used for both the pre-flight report and
the end-of-run summary so both stand out from the per-task log noise around
them.
"""

import asyncio
import logging
import urllib.request

logger = logging.getLogger(__name__)

_BORDER = "=" * 70


def log_block(title: str, lines: list[str]) -> None:
    """Log a clearly bordered, multi-line block."""
    body = "\n".join(lines) if lines else "(nothing to report)"
    logger.info("\n%s\n%s\n%s\n%s", _BORDER, title, body, _BORDER)


def _describe(e: Exception) -> str:
    """Format an exception for a log line. Some exceptions (e.g.
    asyncio.TimeoutError) stringify to '', which would otherwise produce an
    uninformative 'UNREACHABLE ()' — always include the exception type."""
    msg = str(e)
    return f"{type(e).__name__}: {msg}" if msg else type(e).__name__


def check_llm_endpoint(
    name: str, base_url: str | None, headers: dict | None = None, timeout: float = 3.0
) -> bool:
    """Ping an OpenAI-compatible endpoint's /models route, with the same auth
    headers the real client would send. Non-fatal — never raises."""
    if not base_url:
        logger.warning("%s: no base_url configured.", name)
        return False
    try:
        req = urllib.request.Request(f"{base_url.rstrip('/')}/models", headers=headers or {})
        urllib.request.urlopen(req, timeout=timeout)
        return True
    except Exception as e:
        logger.warning("%s UNREACHABLE at %s (%s)", name, base_url, _describe(e))
        return False


async def check_mcp_toolset(name: str, toolset, timeout: float = 180.0) -> bool:
    """Confirm an MCP stdio server responds, via ADK's own McpToolset.get_tools()
    rather than a hand-rolled session. Non-fatal — never raises.

    All MCP subprocesses start concurrently (see asyncio.gather below), so
    slow-starting stdio servers need more than a few seconds of headroom here
    to avoid a false-negative UNREACHABLE result.
    """
    try:
        await asyncio.wait_for(toolset.get_tools(), timeout=timeout)
        return True
    except Exception as e:
        logger.warning("MCP server '%s' UNREACHABLE (%s)", name, _describe(e))
        return False


async def run_preflight_healthcheck() -> dict[str, bool]:
    """
    Check reachability of every backing service the "multi" pipeline depends
    on: the VLM endpoint, the orchestrator LLM endpoint, and every MCP tool
    server behind a loaded sub-agent. Logs a bordered summary. Non-fatal —
    failures are logged, the caller decides whether/how to proceed.

    Must be called after `agenticeoecosystem.agents.multi_agent` has been
    imported (so ALL_SUB_AGENTS and the LLM clients are already built).
    """
    from google.adk.tools.mcp_tool import McpToolset
    from agenticeoecosystem.agents.subagents import ALL_SUB_AGENTS
    from agenticeoecosystem.agents.subagents.vlm_init import VLM_BASE_URL
    from agenticeoecosystem.agents.subagents.llm_init import LLM_BASE_URL, LLM_AUTH_HEADERS

    results: dict[str, bool] = {
        "VLM endpoint": check_llm_endpoint("VLM endpoint", VLM_BASE_URL),
        "Orchestrator LLM endpoint": check_llm_endpoint(
            "Orchestrator LLM endpoint", LLM_BASE_URL, headers=LLM_AUTH_HEADERS
        ),
    }

    mcp_names: list[str] = []
    mcp_checks = []
    for agent in ALL_SUB_AGENTS:
        for i, tool in enumerate(getattr(agent, "tools", [])):
            if isinstance(tool, McpToolset):
                name = agent.name if i == 0 else f"{agent.name}[{i}]"
                mcp_names.append(name)
                mcp_checks.append(check_mcp_toolset(name, tool))

    if mcp_checks:
        outcomes = await asyncio.gather(*mcp_checks, return_exceptions=True)
        for name, outcome in zip(mcp_names, outcomes):
            results[f"MCP: {name}"] = outcome is True

    n_ok = sum(results.values())
    lines = [f"{'OK  ' if ok else 'FAIL'}  {name}" for name, ok in results.items()]
    log_block(f"HEALTHCHECK: {n_ok}/{len(results)} reachable", lines)

    return results
