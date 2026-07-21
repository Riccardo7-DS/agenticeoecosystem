"""
Patches litellm's acompletion/completion to fix a token-budget explosion
caused by a gap in google-adk's MCP tool-result handling for images.

ROOT CAUSE:
`McpTool._run_async_impl` (google/adk/tools/mcp_tool/mcp_tool.py) does

    result = response.model_dump(exclude_none=True, mode="json")

on the raw MCP `CallToolResult`, with no special-casing for `ImageContent`
blocks — google-adk has no reference to `ImageContent` anywhere in its
codebase. The base64 image payload gets flattened into the same plain JSON
dict as everything else, which becomes the string `content` of a
`tool`-role chat message. Sent to an OpenAI-compatible endpoint, that
string is tokenized as ordinary text instead of routed through the vision
encoder: measured directly against a vLLM endpoint, a 512x512 JPEG costs
~260 tokens as a proper `image_url` part, but ~40,000+ tokens once
flattened into base64-in-JSON text — a ~150x blowup that can single-
handedly exhaust a 131k-token context window with just 2-3 MCP-tool-
delivered images (e.g. the vlm_agent's `load_images` tool).

FIX:
Monkeypatch litellm.acompletion/completion to scan outgoing `tool`-role
messages for MCP ImageContent blocks embedded in their JSON-stringified
content, strip the base64 `data` out of that string (replacing it with a
short placeholder so the tool message stays informative), and re-insert
each extracted image as a proper `image_url` content part in a new
synthetic `user`-role message placed immediately after. OpenAI's schema
requires `tool` message content to be a plain string (images can't go
there directly) and requires an `assistant` message between any `tool`
and `user` message — this patch is designed to compose with eve_patch
(apply that one first), whose role-sanitizer inserts the placeholder
`assistant` message this creates the need for.

USAGE:
    from agenticeoecosystem.agents.subagents.eve_patch import apply as apply_eve_patch
    from agenticeoecosystem.agents.subagents.mcp_image_patch import apply as apply_mcp_image_patch
    apply_eve_patch()
    apply_mcp_image_patch()

Call both once, in that order, before constructing any Runner/Agent,
ideally at the top of your entrypoint module (multi_agent.py).
"""

from __future__ import annotations

import json
import logging
from typing import Any

import litellm

logger = logging.getLogger(__name__)

_IMAGE_PLACEHOLDER = "<image data extracted into a following user message>"

_original_acompletion = litellm.acompletion
_original_completion = litellm.completion

_applied = False


def _extract_images(value: Any, found: list[dict[str, str]]) -> Any:
    """Recursively walk a parsed JSON structure, replacing any MCP
    ImageContent-shaped dict's base64 `data` with a placeholder and
    appending the extracted (data, mimeType) pair to `found`.

    Returns a new structure; does not mutate the input.
    """
    if isinstance(value, dict):
        if value.get("type") == "image" and isinstance(value.get("data"), str):
            found.append(
                {
                    "data": value["data"],
                    "mimeType": value.get("mimeType", "image/jpeg"),
                }
            )
            stripped = dict(value)
            stripped["data"] = _IMAGE_PLACEHOLDER
            return stripped
        return {k: _extract_images(v, found) for k, v in value.items()}
    if isinstance(value, list):
        return [_extract_images(v, found) for v in value]
    return value


def _split_images_out_of_tool_message(
    msg: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """If a `tool`-role message's content is a JSON string embedding MCP
    ImageContent blocks, strip the base64 data out and return the cleaned
    message plus the list of extracted images.

    Returns (msg, []) unchanged if content isn't JSON or has no images.
    """
    content = msg.get("content")
    if not isinstance(content, str):
        return msg, []
    try:
        parsed = json.loads(content)
    except (ValueError, TypeError):
        return msg, []

    found: list[dict[str, str]] = []
    cleaned = _extract_images(parsed, found)
    if not found:
        return msg, []

    new_msg = dict(msg)
    new_msg["content"] = json.dumps(cleaned)
    return new_msg, found


def _sanitize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rewrite MCP-tool-delivered images out of `tool` messages and into
    proper `image_url` content parts on synthetic `user` messages.
    Returns a new list; does not mutate the input.
    """
    if not messages:
        return messages

    fixed: list[dict[str, Any]] = []
    total_images = 0
    for msg in messages:
        if msg.get("role") == "tool":
            new_msg, images = _split_images_out_of_tool_message(msg)
            fixed.append(new_msg)
            if images:
                total_images += len(images)
                fixed.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{img['mimeType']};base64,{img['data']}"
                                },
                            }
                            for img in images
                        ],
                    }
                )
        else:
            fixed.append(msg)

    if total_images:
        logger.debug(
            "mcp_image_patch: extracted %d image(s) from tool message(s) into "
            "proper image_url content parts",
            total_images,
        )
    return fixed


async def _patched_acompletion(*args: Any, **kwargs: Any):
    if "messages" in kwargs:
        kwargs["messages"] = _sanitize_messages(kwargs["messages"])
    elif args:
        args = list(args)
        for i, a in enumerate(args):
            if isinstance(a, list) and a and isinstance(a[0], dict) and "role" in a[0]:
                args[i] = _sanitize_messages(a)
                break
        args = tuple(args)
    return await _original_acompletion(*args, **kwargs)


def _patched_completion(*args: Any, **kwargs: Any):
    if "messages" in kwargs:
        kwargs["messages"] = _sanitize_messages(kwargs["messages"])
    elif args:
        args = list(args)
        for i, a in enumerate(args):
            if isinstance(a, list) and a and isinstance(a[0], dict) and "role" in a[0]:
                args[i] = _sanitize_messages(a)
                break
        args = tuple(args)
    return _original_completion(*args, **kwargs)


def apply() -> None:
    """Idempotently monkeypatch litellm.acompletion/completion with the
    MCP-image-extracting wrappers. Safe to call multiple times."""
    global _applied
    if _applied:
        logger.debug("mcp_image_patch: already applied, skipping")
        return

    litellm.acompletion = _patched_acompletion
    litellm.completion = _patched_completion
    _applied = True
    logger.info(
        "mcp_image_patch: patched litellm.acompletion/completion for MCP image handling"
    )


def remove() -> None:
    """Restore the original litellm.acompletion/completion functions."""
    global _applied
    litellm.acompletion = _original_acompletion
    litellm.completion = _original_completion
    _applied = False
    logger.info("mcp_image_patch: removed litellm patch, originals restored")
