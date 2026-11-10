"""Interaction reviewer and hypothesis ledger.

Orchestrators call, in this order:
    configure(mode)              once, before building any agent
    attach_subagents(agents)     before the root agent is built
    root_callbacks(on_tool_error) as keyword arguments of the root Agent(...)
The task runner calls finish_task(loop) after each task, outside its timeout.
"""
import asyncio
import logging

from agenticeoecosystem.agents.helper import config

logger = logging.getLogger(__name__)


def configure(mode: str) -> None:
    if mode not in config.MODES:
        raise ValueError(f"helper mode must be one of {config.MODES}, got {mode!r}")
    config.MODE = mode
    if mode != "off":
        logger.info("Helper mode %s; reports under %s", mode, config.output_root())


def _chain(existing, ours):
    """Ours runs first and always returns None, so the existing callback keeps full control."""
    if existing is None:
        return ours
    if isinstance(existing, list):
        return [ours, *existing]

    def chained(*args, **kwargs):
        ours(*args, **kwargs)
        return existing(*args, **kwargs)   # may be a coroutine; ADK awaits it
    return chained


def attach_subagents(agents) -> None:
    """Adds recording callbacks to the evaluated sub-agents only."""
    if config.MODE == "off":
        return
    from agenticeoecosystem.agents.helper import hooks
    for a in agents:
        if a.name not in config.EVALUATED_AGENTS:
            continue
        a.before_agent_callback = _chain(a.before_agent_callback, hooks.sub_before_agent)
        a.after_tool_callback = _chain(a.after_tool_callback, hooks.sub_after_tool)
        a.after_model_callback = _chain(a.after_model_callback, hooks.sub_after_model)
        logger.info("Helper recording %s", a.name)


def root_callbacks(on_tool_error=None) -> dict:
    """Keyword arguments for the root Agent(...). Just on_tool_error when the helper is off."""
    if config.MODE == "off":
        return {"on_tool_error_callback": on_tool_error} if on_tool_error else {}
    from agenticeoecosystem.agents.helper import hooks
    return {
        "before_agent_callback": hooks.root_before_agent,
        "before_tool_callback": hooks.root_before_tool,
        "after_tool_callback": hooks.root_after_tool,
        "on_tool_error_callback": _chain(on_tool_error, hooks.root_on_tool_error),
    }


def finish_task(loop=None) -> None:
    """Let pending reviews finish. Call after a task, OUTSIDE its timeout."""
    if config.MODE == "off":
        return
    from agenticeoecosystem.agents.helper.reviewer import drain, pending_count
    if pending_count():
        (loop or asyncio.get_event_loop()).run_until_complete(drain(timeout=config.DRAIN_TIMEOUT_S))