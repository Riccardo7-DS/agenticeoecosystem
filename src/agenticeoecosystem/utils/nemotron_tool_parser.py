"""
vLLM tool parser plugin for NVIDIA Nemotron models.

Nemotron emits tool calls in this format:
    <TOOLCALL>[{"name": "fn", "arguments": {...}}]</TOOLCALL>

Usage — pass to vLLM via:
    --tool-call-parser nemotron
    --tool-parser-plugin <path-to-this-file>
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.logger import init_logger
from vllm.tool_parsers.abstract_tool_parser import ToolParser, ToolParserManager
from vllm.tokenizers import TokenizerLike

logger = init_logger(__name__)

_TAG_RE = re.compile(r"<TOOLCALL>(.*?)</TOOLCALL>", re.DOTALL)


@ToolParserManager.register_module("nemotron")
class NemotronToolParser(ToolParser):
    """
    Parser for NVIDIA Nemotron models that output:
        <TOOLCALL>[{"name": "...", "arguments": {...}}]</TOOLCALL>
    """

    def __init__(self, tokenizer: TokenizerLike):
        super().__init__(tokenizer)

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        match = _TAG_RE.search(model_output)
        if not match:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        try:
            raw_calls = json.loads(match.group(1).strip())
            if isinstance(raw_calls, dict):
                raw_calls = [raw_calls]

            tool_calls = [
                ToolCall(
                    type="function",
                    function=FunctionCall(
                        name=c["name"],
                        arguments=json.dumps(c.get("arguments", {}), ensure_ascii=False),
                    ),
                )
                for c in raw_calls
            ]

            content = model_output[: match.start()].strip() or None
            return ExtractedToolCallInformation(
                tools_called=True, tool_calls=tool_calls, content=content
            )

        except Exception:
            logger.exception("NemotronToolParser: failed to parse tool call.")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        # Only emit tool calls once the closing tag is seen
        if "</TOOLCALL>" not in current_text:
            return DeltaMessage(content=delta_text)

        # Closing tag just arrived — parse the full call
        if "</TOOLCALL>" in current_text and "</TOOLCALL>" not in previous_text:
            match = _TAG_RE.search(current_text)
            if not match:
                return None
            try:
                raw_calls = json.loads(match.group(1).strip())
                if isinstance(raw_calls, dict):
                    raw_calls = [raw_calls]

                delta_tool_calls = [
                    DeltaToolCall(
                        index=i,
                        type="function",
                        function=DeltaFunctionCall(
                            name=c["name"],
                            arguments=json.dumps(c.get("arguments", {}), ensure_ascii=False),
                        ),
                    )
                    for i, c in enumerate(raw_calls)
                ]
                return DeltaMessage(tool_calls=delta_tool_calls)
            except Exception:
                logger.exception("NemotronToolParser streaming: parse error.")
                return None

        return None
