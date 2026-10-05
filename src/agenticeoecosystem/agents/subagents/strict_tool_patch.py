"""
Patches litellm's acompletion/completion to mark every function-tool
declaration as OpenAI "strict" mode, which forces constrained decoding of
tool-call arguments into schema-valid JSON.

ROOT CAUSE:
google-adk's AgentTool wraps each sub-agent as a callable tool with the
trivial schema {"request": {"type": "string"}}. When converting this to the
OpenAI-compatible tool-call format (_function_declaration_to_tool_param in
google/adk/models/lite_llm.py), ADK never sets "strict": true on the
function declaration — that flag is only ever applied to top-level
*response* schemas (structured final output), never to *tool-call*
argument schemas. Some models/providers (observed: meta-llama/llama-4-maverick
via OpenRouter/DigitalOcean) then intermittently fail to properly escape
embedded quotes inside the free-text `request` string, producing malformed
JSON such as:

    json.decoder.JSONDecodeError: Unterminated string starting at: ...

which crashes that turn's tool call. The model then sometimes gives up and
emits the literal tool-call syntax as its plain-text "final answer" instead
of a real result.

FIX:
Monkeypatch litellm.acompletion / litellm.completion so that, right before
the request is sent, every entry in kwargs["tools"] of type "function" gets
"strict": true set on its function dict, plus "additionalProperties": false
on its parameters schema (both required by OpenAI's strict-mode spec).
Verified empirically (5/5 trials) to eliminate the malformed-JSON failure
for llama-4-maverick on a request string containing embedded quotes that
reliably triggers it without this patch.

Gated behind the LITELLM_STRICT_TOOL_SCHEMA=1 env var — off by default so it
never changes behavior for already-validated candidates (EVE-Instruct,
Gemma4-31B, Qwen3.6-27B, Qwen3-VL-8B). Enable it only for models known to
need it.

USAGE:
    from agenticeoecosystem.agents.subagents.strict_tool_patch import apply as apply_strict_tool_patch
    apply_strict_tool_patch()
"""

from __future__ import annotations

import logging
import os
from typing import Any

import litellm

logger = logging.getLogger(__name__)

_original_acompletion = litellm.acompletion
_original_completion = litellm.completion

_applied = False


def _enable() -> bool:
    return os.getenv("LITELLM_STRICT_TOOL_SCHEMA", "0") == "1"


def _add_strict_to_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Reuse ADK's own schema normalizer (used correctly for response_format,
    # never for tool-call schemas) so nested/optional-property schemas — not
    # just the trivial {"request": str} AgentTool wrapper — get a fully
    # strict-mode-compliant transform (every property in `required`,
    # additionalProperties: false, $ref siblings stripped, recursing into
    # $defs/anyOf/items). A naive partial transform (e.g. only setting
    # additionalProperties without fixing `required`) produces a schema that
    # violates the strict-mode spec and can make providers behave worse, not
    # better — observed empirically before this fix.
    from google.adk.models.lite_llm import _enforce_strict_openai_schema

    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        fn = tool.get("function")
        if not isinstance(fn, dict):
            continue
        fn["strict"] = True
        params = fn.get("parameters")
        if isinstance(params, dict):
            _enforce_strict_openai_schema(params)
    return tools


async def _patched_acompletion(*args: Any, **kwargs: Any):
    if _enable() and isinstance(kwargs.get("tools"), list):
        kwargs["tools"] = _add_strict_to_tools(kwargs["tools"])
    return await _original_acompletion(*args, **kwargs)


def _patched_completion(*args: Any, **kwargs: Any):
    if _enable() and isinstance(kwargs.get("tools"), list):
        kwargs["tools"] = _add_strict_to_tools(kwargs["tools"])
    return _original_completion(*args, **kwargs)


def apply() -> None:
    """Idempotently monkeypatch litellm.acompletion/completion to add strict
    tool schemas when LITELLM_STRICT_TOOL_SCHEMA=1. Safe to call multiple
    times, and safe to call even when the env var is unset (becomes a no-op
    passthrough)."""
    global _applied
    if _applied:
        logger.debug("strict_tool_patch: already applied, skipping")
        return

    litellm.acompletion = _patched_acompletion
    litellm.completion = _patched_completion
    _applied = True
    logger.info(
        "strict_tool_patch: patched litellm.acompletion/completion "
        "(active=%s via LITELLM_STRICT_TOOL_SCHEMA)", _enable(),
    )


def remove() -> None:
    """Restore the original litellm.acompletion/completion functions."""
    global _applied
    litellm.acompletion = _original_acompletion
    litellm.completion = _original_completion
    _applied = False
    logger.info("strict_tool_patch: removed litellm patch, originals restored")
