"""
Caps how many times the same orphaned tool_call_id can be silently re-patched
by google-adk's LiteLlm._ensure_tool_results() healer, to stop the
"retry-storm" failure mode where one dropped/malformed tool call keeps
getting patched over on every subsequent LLM turn for the rest of a task —
sometimes hundreds of turns and thousands of seconds — without ever producing
a real answer.

ROOT CAUSE:
When a tool call's arguments fail to parse (or a turn is otherwise cut off
before the matching tool response is recorded), the assistant's tool_call
message is left permanently in the conversation history with no response.
google-adk's `_ensure_tool_results()` (google/adk/models/lite_llm.py) runs on
every subsequent turn to keep the OpenAI-compatible request schema valid: it
finds any such orphaned tool_call_id, logs a
"Missing tool results for tool_call_id(s): [...]" warning, and inserts a
placeholder tool message just for that one request. Nothing ever removes the
original broken message from history, so this repeats — and gets logged
again — on every single turn for the remainder of the task. Analysis of a
267-task benchmark run showed this is not a real "retry": tasks where the
same id recurred beyond ~9 times never converged to an answer resembling the
reference solution (typically stuck calling one tool identically dozens of
times), while the small number of genuine recoveries all resolved within 5
occurrences. There is no benefit to letting it continue past a single-digit
count — only wasted wall-clock/token cost.

FIX:
Attach a logging.Filter to the "google_adk.google.adk.models.lite_llm"
logger that watches for that warning and counts, per tool_call_id, how many
times it has recurred within the current task. Once any id exceeds
_MAX_RETRIES, raise OrphanedToolCallStormError. Because logging.Logger.filter()
has no exception-suppression around it (unlike per-handler dispatch), this
propagates out of _ensure_tool_results() -> LiteLlm.generate_content_async()
exactly like a genuine model/API error, and is caught by
pipeline_runner.py's per-task try/except the same way the existing
"Max number of llm calls exceeded" hard failure already is — the task is
recorded as unsolved and the batch moves on, instead of burning through the
rest of its retry budget.

SECOND MECHANISM — IDENTICAL REPEATED TOOL CALLS:
The orphaned-tool_call_id storm above only catches loops caused by a
malformed/dropped tool call whose id never resolves. A distinct loop shape
was found in TASK_260713_000071_500050, which hit google-adk's built-in
500-LLM-call ceiling: the agent kept calling the *same* tool with the
*same* arguments over and over, each call getting a fresh, well-formed
tool_call_id (so the id-based filter above never triggers) but the model
never changing its approach in response to the (successful) result. To
catch this, McpTool.run_async (google/adk/tools/mcp_tool/mcp_tool.py) is
also patched: every call is counted by (tool name, arguments) within the
current task, and once the same pair exceeds _REPEAT_CALL_MAX, the same
OrphanedToolCallStormError is raised. Patching the outer run_async (rather
than _run_async_impl, which arxiv_rate_limit_retry.py separately patches)
keeps the two patches decoupled regardless of apply() order: a tool's
internal retry-with-backoff loop happens entirely inside one run_async
call, so it is counted once, not once per internal retry.

THIRD MECHANISM — REPEATED ERRORS WITH VARYING ARGUMENTS:
The 2026-08-07 09:22 gemma4-31b run (generate_Gemma4-31B_[multi-no-scope]
..._092230.log) crashed with litellm.ContextWindowExceededError after the
cdse_agent sub-agent made 7 consecutive failed fetch_sentinel1_grd calls
(bbox tweaked, then height/width added, then polarization swapped, then
custom_bands swapped) chasing a single Sentinel Hub 400 error, plus a
similar shorter run against fetch_sentinel2. Each attempt used different
arguments, so the identical-args counter above never triggered, but every
attempt's error text stayed in history, and the accumulated conversation
eventually exceeded the model's 32768-token context window and aborted
the whole task. To catch this, the same McpTool.run_async patch also
counts *errors* (isError result) per tool name alone, ignoring arguments;
once a single tool's error count exceeds _ERROR_COUNT_MAX within a task,
OrphanedToolCallStormError is raised instead of letting the model keep
trying new argument combinations.

USAGE:
    from agenticeoecosystem.agents.subagents.orphaned_tool_call_guard import apply as apply_orphan_guard
    apply_orphan_guard()

Call once, before constructing any Runner/Agent (multi_agent.py). Call
reset() at the start of each task (get_ans_tool_path) so counts don't
persist across unrelated tasks in the same batch run.
"""

from __future__ import annotations

import logging

from google.adk.tools.mcp_tool.mcp_tool import McpTool

logger = logging.getLogger(__name__)

_TARGET_LOGGER_NAME = "google_adk.google.adk.models.lite_llm"
_WARNING_PREFIX = "Missing tool results for tool_call_id(s):"
_MAX_RETRIES = 10
_REPEAT_CALL_MAX = 5
_ERROR_COUNT_MAX = 4

_counts: dict[str, int] = {}
_call_counts: dict[tuple, int] = {}
_error_counts: dict[str, int] = {}
_filter_instance: "_OrphanedToolCallFilter | None" = None
_original_mcp_run_async = McpTool.run_async
_mcp_patch_applied = False


class OrphanedToolCallStormError(RuntimeError):
    """Raised once the same orphaned tool_call_id has recurred more than
    _MAX_RETRIES times, or the same tool has been called with identical
    arguments more than _REPEAT_CALL_MAX times, within a single task —
    either way, indicating the conversation is stuck in a loop rather than
    making progress."""


def _make_call_key(tool_name: str, args: dict) -> tuple:
    try:
        normalized = tuple(sorted((k, repr(v)) for k, v in args.items()))
    except Exception:
        normalized = (repr(args),)
    return (tool_name, normalized)


def _is_error_result(result) -> bool:
    return isinstance(result, dict) and bool(result.get("isError"))


async def _counting_run_async(self, *, args, tool_context):
    key = _make_call_key(self.name, args)
    count = _call_counts[key] = _call_counts.get(key, 0) + 1
    if count > _REPEAT_CALL_MAX:
        raise OrphanedToolCallStormError(
            f"tool {self.name!r} has been called with identical arguments "
            f"{args!r} {count} times in this task (limit {_REPEAT_CALL_MAX}) "
            "without the agent changing its approach — aborting this task "
            "instead of continuing the loop."
        )

    result = await _original_mcp_run_async(self, args=args, tool_context=tool_context)

    if _is_error_result(result):
        error_count = _error_counts[self.name] = _error_counts.get(self.name, 0) + 1
        if error_count > _ERROR_COUNT_MAX:
            raise OrphanedToolCallStormError(
                f"tool {self.name!r} has failed {error_count} times in this "
                f"task (limit {_ERROR_COUNT_MAX}), most recently with "
                f"arguments {args!r} — aborting this task instead of letting "
                "the agent keep trying new argument combinations and "
                "growing the context unboundedly."
            )

    return result


class _OrphanedToolCallFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not record.msg or _WARNING_PREFIX not in str(record.msg):
            return True
        ids = record.args[0] if record.args else []
        if isinstance(ids, str):
            ids = [ids]
        for tool_call_id in ids:
            count = _counts[tool_call_id] = _counts.get(tool_call_id, 0) + 1
            if count > _MAX_RETRIES:
                raise OrphanedToolCallStormError(
                    f"tool_call_id {tool_call_id!r} has been orphaned/re-warned "
                    f"about {count} times (limit {_MAX_RETRIES}) without ever "
                    "getting a real tool result — aborting this task instead "
                    "of continuing the retry-storm."
                )
        return True


def reset() -> None:
    """Clear per-id, per-(tool, args), and per-tool-error counts. Call at the
    start of each new task."""
    _counts.clear()
    _call_counts.clear()
    _error_counts.clear()


def apply() -> None:
    """Idempotently attach the retry-counting filter and the identical-repeated-
    call guard. Safe to call multiple times."""
    global _filter_instance, _mcp_patch_applied
    if _filter_instance is None:
        _filter_instance = _OrphanedToolCallFilter()
        logging.getLogger(_TARGET_LOGGER_NAME).addFilter(_filter_instance)
        logger.info(
            "orphaned_tool_call_guard: capping orphaned tool_call_id retries at %d",
            _MAX_RETRIES,
        )
    else:
        logger.debug("orphaned_tool_call_guard: logging filter already applied, skipping")

    if not _mcp_patch_applied:
        McpTool.run_async = _counting_run_async
        _mcp_patch_applied = True
        logger.info(
            "orphaned_tool_call_guard: capping identical repeated tool calls at %d",
            _REPEAT_CALL_MAX,
        )
    else:
        logger.debug("orphaned_tool_call_guard: repeat-call guard already applied, skipping")


def remove() -> None:
    """Detach the filter and restore McpTool.run_async."""
    global _filter_instance, _mcp_patch_applied
    if _filter_instance is not None:
        logging.getLogger(_TARGET_LOGGER_NAME).removeFilter(_filter_instance)
        _filter_instance = None
        logger.info("orphaned_tool_call_guard: removed logging filter")
    if _mcp_patch_applied:
        McpTool.run_async = _original_mcp_run_async
        _mcp_patch_applied = False
        logger.info("orphaned_tool_call_guard: removed repeat-call guard")
