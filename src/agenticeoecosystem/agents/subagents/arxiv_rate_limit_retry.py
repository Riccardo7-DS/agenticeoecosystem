"""
Retries arXiv MCP tool calls that fail due to upstream rate limiting,
instead of surfacing the error straight to the LLM.

ROOT CAUSE:
arxiv-mcp-server (the third-party stdio MCP server used by
sub_agent_arxiv_rag.py) raises a plain RuntimeError when arXiv starts
rate-limiting the host IP:

    raise RuntimeError(
        f"arXiv is rate limiting this IP (HTTP {response.status_code}). "
        "Please wait 60 seconds before retrying."
    )

Per the MCP protocol, an exception raised inside a tool handler is not a
transport failure — it comes back as an ordinary CallToolResult with
isError=True and the exception text as content. So
McpTool._run_async_impl (google/adk/tools/mcp_tool/mcp_tool.py) returns
normally with that text as the tool's output, and it's up to the LLM to
decide what to do next. There is no code path that actually waits;
observed in practice the agent either gives up immediately or re-calls
right away and gets rate-limited again.

FIX:
Monkeypatch McpTool._run_async_impl: after each call, if the result is
an arXiv rate-limit error, actually `await asyncio.sleep(...)` and retry
the same tool call (bounded), before ever handing control back to the
LLM. Matches narrowly on the arXiv server's exact error text, so it's a
no-op for every other MCP tool/server in the system.

USAGE:
    from agenticeoecosystem.agents.subagents import arxiv_rate_limit_retry
    arxiv_rate_limit_retry.apply()

Call once, anywhere before the first tool call — alongside the other
patches in multi_agent.py.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from google.adk.tools.mcp_tool.mcp_tool import McpTool

logger = logging.getLogger(__name__)

_RATE_LIMIT_MARKER = "rate limiting this IP"
_MAX_RETRIES = 3
_WAIT_SECONDS = 60

_original_run_async_impl = McpTool._run_async_impl
_applied = False


def _is_rate_limit_error(result: Any) -> bool:
    if not isinstance(result, dict) or not result.get("isError"):
        return False
    content = result.get("content") or []
    text = " ".join(
        block.get("text", "") for block in content if isinstance(block, dict)
    )
    return _RATE_LIMIT_MARKER in text


async def _patched_run_async_impl(self, *, args, tool_context, credential):
    result = await _original_run_async_impl(
        self, args=args, tool_context=tool_context, credential=credential
    )
    attempt = 0
    while _is_rate_limit_error(result) and attempt < _MAX_RETRIES:
        attempt += 1
        logger.info(
            "arxiv_rate_limit_retry: %s hit arXiv's rate limit (retry %d/%d), "
            "waiting %ds before retrying",
            self.name, attempt, _MAX_RETRIES, _WAIT_SECONDS,
        )
        await asyncio.sleep(_WAIT_SECONDS)
        result = await _original_run_async_impl(
            self, args=args, tool_context=tool_context, credential=credential
        )
    if _is_rate_limit_error(result):
        logger.warning(
            "arxiv_rate_limit_retry: %s still rate-limited after %d retries, "
            "giving up and surfacing the error",
            self.name, _MAX_RETRIES,
        )
    return result


def apply() -> None:
    """Idempotently monkeypatch McpTool._run_async_impl to retry-with-backoff
    on arXiv rate-limit errors. Safe to call multiple times."""
    global _applied
    if _applied:
        logger.debug("arxiv_rate_limit_retry: already applied, skipping")
        return
    McpTool._run_async_impl = _patched_run_async_impl
    _applied = True
    logger.info(
        "arxiv_rate_limit_retry: patched McpTool._run_async_impl to retry on "
        "arXiv rate-limit errors (up to %d retries, %ds wait)",
        _MAX_RETRIES, _WAIT_SECONDS,
    )


def remove() -> None:
    """Restore the original McpTool._run_async_impl."""
    global _applied
    McpTool._run_async_impl = _original_run_async_impl
    _applied = False
    logger.info("arxiv_rate_limit_retry: removed patch, original restored")
