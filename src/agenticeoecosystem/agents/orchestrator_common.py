"""Code shared by multi_agent.py and multi_agent_research.py.

Moved here, unchanged, so the research orchestrator can reuse it without importing
multi_agent.py, which builds its own root agent over the same subagent objects
(an ADK agent can have only one parent).
"""
import logging
import re

from agenticeoecosystem.agents.subagents.eve_patch import _PLACEHOLDER_ASSISTANT_CONTENT

logger = logging.getLogger(__name__)

# ============================ Supervisor prompt ============================
# Cut _SUPERVISOR_SYSTEM_PROMPT out of multi_agent.py and paste it here UNCHANGED.
# (`git diff --color-moved` will show the block as moved, not edited.)
SUPERVISOR_SYSTEM_PROMPT = (
    "You are a routing agent for an Earth Observation / GIS assistant. "
    # ... rest of the current prompt, unchanged ...
)

_head, _marker, _tail = SUPERVISOR_SYSTEM_PROMPT.partition("OUTPUT RULES:")
if not _marker:
    raise RuntimeError("'OUTPUT RULES:' not found in SUPERVISOR_SYSTEM_PROMPT; update the split.")
SUPERVISOR_ROUTING = _head                  # which sub-agent to use, and when
SUPERVISOR_OUTPUT_RULES = _marker + _tail   # how to answer, conversation, scope


# ============================ Tool-error recovery ============================

def on_tool_error(tool, args, tool_context, error):
    # Paste the original _on_tool_error docstring here (it explains the storm-guard exception).
    from agenticeoecosystem.agents.subagents.orphaned_tool_call_guard import (
        OrphanedToolCallStormError,
    )
    if isinstance(error, OrphanedToolCallStormError):
        logger.error("Tool '%s' hit storm guard, aborting task: %s", getattr(tool, "name", "?"), error)
        return None

    logger.warning("Tool '%s' failed: %s", getattr(tool, "name", "?"), error)
    return {"error": f"Tool '{getattr(tool, 'name', '?')}' failed: {error}"}


# ============================ Response text cleaning ============================
# Move the explanatory comments above these from multi_agent.py too.

_LEAKED_CHANNEL_PREFIX_RE = re.compile(
    r"^\s*<\|?channel\|?>\s*[\w-]*\s*\n?\s*<\s*channel\|?>\s*",
    re.IGNORECASE,
)


def strip_leaked_channel_marker(text: str) -> str:
    """Remove a leaked reasoning-channel marker prefix, if present."""
    if not text:
        return text
    return _LEAKED_CHANNEL_PREFIX_RE.sub("", text, count=1)


_THINK_BLOCK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.IGNORECASE | re.DOTALL)
_UNCLOSED_THINK_RE = re.compile(r"^\s*<think>", re.IGNORECASE)


def strip_leaked_think_block(text: str) -> str | None:
    """Remove a leaked <think>...</think> block. None if the block never closes."""
    if not text:
        return text
    stripped = _THINK_BLOCK_RE.sub("", text, count=1)
    if stripped != text:
        return stripped
    if _UNCLOSED_THINK_RE.match(text):
        return None
    return text


def clean_text(content) -> str | None:
    """First real text part of a Content. Skips thought parts, eve_patch's
    role-alternation placeholder, and leaked reasoning markers."""
    if not (content and getattr(content, "parts", None)):
        return None
    for part in content.parts:
        if getattr(part, "thought", False):
            continue
        text = getattr(part, "text", None)
        if text and text.strip() == _PLACEHOLDER_ASSISTANT_CONTENT:
            continue
        if text:
            cleaned = strip_leaked_think_block(strip_leaked_channel_marker(text))
            if cleaned is not None:
                return cleaned
    return None


def extract_final_text(event) -> str | None:
    """Plain text of a final-response event (see clean_text)."""
    return clean_text(event.content)