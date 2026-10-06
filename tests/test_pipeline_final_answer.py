"""Verify that single-agent candidate traces retain their final answer."""

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import patch

import agenticeoecosystem.client
from agenticeoecosystem.evaluation.pipeline_runner import run_pipeline


class FakeSingleAgent:
    """Emit a tool result followed by a final assistant response."""

    def __init__(self, content: str | list) -> None:
        self.content = content

    async def astream(self, *args: object, **kwargs: object):
        yield {"tools": {"messages": [SimpleNamespace(type="tool", content="raw tool result")]}}
        yield {"agent": {"messages": [SimpleNamespace(type="ai", tool_calls=[], content=self.content)]}}


def test_single_agent_preserves_final_answer_and_tool_result() -> None:
    """A candidate must contain both the final response and recorded tool evidence."""
    module = SimpleNamespace(
        agent=FakeSingleAgent("January mean: 280 K."),
        reset_step_log=lambda: None,
        _extract_steps_from_messages=lambda messages: None,
        _token_counts={"input": 123, "output": 12},
        _step_log=[{"function_name": "inspect_netcdf", "arguments": {}, "tool_result": {"mean": 280}}],
    )
    asyncio.set_event_loop(asyncio.new_event_loop())
    with patch.dict(sys.modules, {"agenticeoecosystem.client.client_single_agent": module}):
        solution, input_tokens, output_tokens = run_pipeline("Inspect the file.", pipeline="single")
    assert solution.final_answer == "January mean: 280 K."
    assert solution.steps[0].tool_result == {"mean": 280}
    assert (input_tokens, output_tokens) == (123, 12)


def test_single_agent_text_blocks_exclude_reasoning() -> None:
    """Structured assistant content retains answer text without reasoning blocks."""
    module = SimpleNamespace(
        agent=FakeSingleAgent([{"type": "reasoning", "text": "private reasoning"},
                               {"type": "text", "text": "Answer"}]),
        reset_step_log=lambda: None,
        _extract_steps_from_messages=lambda messages: None,
        _token_counts={"input": 1, "output": 1}, _step_log=[],
    )
    asyncio.set_event_loop(asyncio.new_event_loop())
    with patch.dict(sys.modules, {"agenticeoecosystem.client.client_single_agent": module}):
        solution, _, _ = run_pipeline("Answer.", pipeline="single")
    assert solution.final_answer == "Answer"
