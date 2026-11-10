"""
ledger.py — Provenance log for a Virtual Lab run, plus the ADK plugin that feeds it.

The plugin is registered on the lab's Runner. ADK's AgentTool propagates
runner plugins into the sub-runs it spawns, so the same plugin sees both the
lab participants' calls to existing sub-agents *and* the leaf MCP tool calls
those sub-agents make. That makes it the single choke point for:
  - attributing every tool call to the lab step that caused it (via a
    contextvar the coordinator sets around each participant call),
  - enforcing plan_only permissions (leaf tools outside the read-only list),
  - enforcing the tool-call and participant-LLM-call budgets, per-step limits on what one
    participant may call (proposal, critique, revision) and on what one sub-agent run may call,
  - answering repeated read-only calls from the cache (free: not counted),
  - giving sub-agents today's date and marking replies where a sub-agent could not reach
    any data source ("no_answer"), so they are never used as evidence,
  - counting tokens.
"""

from __future__ import annotations

import json
import logging
import re
import time
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.agent_tool import AgentTool
from google.genai import types

from agenticeoecosystem.servers.cdse_geocode.catalog import PRODUCT_ALIASES, normalise_collection

from .config import VirtualLabConfig

logger = logging.getLogger(__name__)

# Label of the lab step currently running, e.g. "r0.propose.optical_specialist".
current_step: ContextVar[str | None] = ContextVar("virtual_lab_step", default=None)

_RESULT_PREVIEW_CHARS = 2000

# Sub-agent replies that say it could not consult any data source. They are not evidence:
# the ledger marks them "no_answer", so they are never linked to claims.
_NO_ANSWER = re.compile(
    r"\b(?:(?:do not|don't|doesn't|does not) (?:currently )?have (?:the )?(?:access|tools?|capabilit"
    r"(?:y|ies)|data access)|(?:cannot|can't|can not|unable to|am not able to) "
    r"(?:access|retrieve|query|fetch|list|provide|directly)|(?:not available|no data) for future"
    r" dates|future dates?)", re.IGNORECASE)
# A claim that the requested dates lie in the future. The lab only asks about past events
# (sub-agents get today's date), so such a reply rests on the model's training cutoff, not on
# data, even when the sub-agent also called a tool.
_FUTURE_CLAIM = re.compile(
    r"\b(?:dates?|period|range|window|time ?frame)\b[^.\n]{0,40}\b(?:is|are|lies|lie) in the future"
    r"|\bfuture (?:dates?|period|time)", re.IGNORECASE)
NO_ANSWER_NOTE = ("Lab note: this sub-agent did not get any data from a tool (none called, or "
                  "all blocked or failed), so its reply is not evidence. Do not cite it; treat the "
                  "point as unchecked.")

CATALOG_TOOL = "search_cdse_catalog"
# Product/granule searches returning {total, listed, days: {day: [HH:MM]}}, keyed by this argument.
DAYS_TOOLS = {"count_eumetsat_products": "collection_id", "search_earthdata_granules": "short_name"}
_TODAY_PREFIX = re.compile(r"^\[Today is [^\]]*\]\s*")


def _normalised(value: Any) -> Any:
    if isinstance(value, str):
        return " ".join(value.lower().split())
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, (list, tuple)):
        return [_normalised(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalised(v) for k, v in sorted(value.items()) if v is not None}
    return value


def cache_key(tool: str, args: dict) -> str:
    """Identity of a call for the cache: tool plus arguments, ignoring case, whitespace,
    unset arguments and the date prefix the guard adds to sub-agent requests."""
    args = dict(args or {})
    if isinstance(args.get("request"), str):
        args["request"] = _TODAY_PREFIX.sub("", args["request"])
    return json.dumps([tool, _normalised(args)], sort_keys=True, default=str)


def _days_scope(tool: str, args: dict) -> tuple | None:
    try:
        bbox = tuple(round(float(x), 3) for x in args.get("bbox") or ())
    except (TypeError, ValueError):
        return None
    key = str(args.get(DAYS_TOOLS[tool]) or "").strip().upper()
    return (tool, key, bbox) if key and len(bbox) == 4 else None


def catalog_scope(args: dict) -> tuple | None:
    """What a catalogue search asks for apart from its dates: (collection, product filter,
    bbox, cloud limit), with names normalised as the catalogue tool does. None if invalid."""
    collection, implied = normalise_collection(str(args.get("collection") or ""))
    try:
        bbox = tuple(round(float(x), 3) for x in args.get("bbox") or ())
    except (TypeError, ValueError):
        return None
    if collection is None or len(bbox) != 4:
        return None
    product = args.get("product_type")
    product = PRODUCT_ALIASES.get(str(product or "").upper(), product) or implied
    return collection, str(product or "").upper(), bbox, args.get("max_cloud_cover")


def is_no_answer(text: str) -> bool:
    """True when a sub-agent reply only says it could not access data or tools."""
    return bool(_NO_ANSWER.search(text or ""))


def claims_future_dates(text: str) -> bool:
    """True when a reply says the requested dates are in the future."""
    return bool(_FUTURE_CLAIM.search(text or ""))


def _result_json(result: Any) -> dict | None:
    """The JSON object an MCP tool returned as text, or None."""
    if not isinstance(result, dict):
        return None
    for block in result.get("content") or []:
        text = block.get("text", "") if isinstance(block, dict) else ""
        try:
            body = json.loads(text)
        except (TypeError, ValueError):
            continue
        if isinstance(body, dict):
            return body
    return None


def catalog_summary(result: Any) -> dict[str, Any] | None:
    """Compact record of a search_cdse_catalog result: which collection and date range was
    searched, under which product filter, the acquisition dates found, and whether the list is
    complete (every match listed). The full product list would not fit in result_preview."""
    body = result if isinstance(result, dict) and "products" in result else _result_json(result)
    if not body or body.get("error") or "products" not in body:
        return None
    return {"collection": body.get("collection"), "product_type": body.get("product_type"),
            "start": body.get("start_date"), "end": body.get("end_date"),
            "complete": body.get("listed") == body.get("total"),
            "dates": sorted({(p.get("start") or "")[:10] for p in body["products"] if p.get("start")})}


def reports_error(result: Any) -> bool:
    """True for a failed tool result, including MCP servers that return their error as JSON
    text inside a normal reply (isError is then false)."""
    if not isinstance(result, dict):
        return False
    if result.get("error") or result.get("isError"):
        return True
    for block in result.get("content") or []:
        text = block.get("text", "") if isinstance(block, dict) else ""
        if text.lstrip().startswith("{"):
            try:
                body = json.loads(text)
            except ValueError:
                continue
            if isinstance(body, dict) and body.get("error"):
                return True
    return False


def _request_with_date(tool_args: dict) -> None:
    """Prefix a sub-agent request with today's date, so its model does not treat recent
    dates as the future."""
    request = tool_args.get("request")
    if isinstance(request, str) and not request.startswith("[Today is "):
        tool_args["request"] = (f"[Today is {datetime.now().date().isoformat()}; earlier dates are "
                                f"in the past.] {request}")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class LabLedger:
    """Append-only record of everything that happened in one lab run."""

    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []      # proposals, critiques, decisions, ...
        self.tool_calls: list[dict[str, Any]] = []
        self.llm_calls: list[dict[str, Any]] = []
        self.budget_exhausted: str | None = None
        self._cache: dict[str, tuple[str, Any]] = {}            # cache_key -> (call id, result)
        self._catalog_cache: list[dict[str, Any]] = []          # complete catalogue searches
        self._days_cache: list[dict[str, Any]] = []             # complete product/granule searches

    def remember(self, entry: dict[str, Any], result: Any) -> None:
        """Keep a successful read-only result for later identical (or, for a complete catalogue
        search, narrower) calls."""
        self._cache[cache_key(entry["tool"], entry.get("args") or {})] = (entry["id"], result)
        body = result if isinstance(result, dict) and "products" in result else _result_json(result)
        args = entry.get("args") or {}
        if (entry["tool"] in DAYS_TOOLS and isinstance(result, dict) and "days" in result
                and result.get("listed") == result.get("total")
                and (scope := _days_scope(entry["tool"], args))):
            self._days_cache.append({"scope": scope, "start": result.get("start_date"),
                                     "end": result.get("end_date"), "id": entry["id"], "body": result})
        if (entry["tool"] == CATALOG_TOOL and body and not body.get("error")
                and body.get("listed") == body.get("total")
                and (scope := catalog_scope(entry.get("args") or {}))):
            self._catalog_cache.append({"scope": scope, "start": body.get("start_date"),
                                        "end": body.get("end_date"), "id": entry["id"], "body": body})

    def recall(self, tool: str, args: dict) -> tuple[str, Any] | None:
        """(id of the call that answered it, result) for a call the cache can answer."""
        hit = self._cache.get(cache_key(tool, args))
        if hit:
            return hit
        if tool in DAYS_TOOLS:
            return self._recall_days(tool, args)
        if tool != CATALOG_TOOL:
            return None
        scope = catalog_scope(args)
        start, end = str(args.get("start_date") or "")[:10], str(args.get("end_date") or "")[:10]
        if scope is None or not start or not end:
            return None
        collection, product, bbox, cloud = scope
        for c in self._catalog_cache:
            c_collection, c_product, c_bbox, c_cloud = c["scope"]
            if not (c_collection == collection and c_bbox == bbox
                    and c_product in ("", product) and c_cloud in (None, cloud)
                    and (c["start"] or "") <= start and end <= (c["end"] or "")):
                continue
            # A complete broader search answers a narrower one: filter what it listed.
            products = [p for p in c["body"].get("products", [])
                        if start <= (p.get("start") or "")[:10] <= end
                        and (not product or c_product or product in str(p.get("name", "")).upper())
                        and (cloud is None or c_cloud is not None
                             or (isinstance(p.get("cloudCover"), (int, float)) and p["cloudCover"] <= cloud))]
            limit = max(1, min(int(args.get("limit") or 20), 50))
            return c["id"], {**c["body"], "product_type": product or None, "start_date": start,
                             "end_date": end, "total": len(products),
                             "listed": min(len(products), limit), "products": products[:limit]}
        return None

    def _recall_days(self, tool: str, args: dict) -> tuple[str, Any] | None:
        """A complete product/granule search answers one over a narrower window."""
        scope = _days_scope(tool, args)
        start, end = str(args.get("start_date") or "")[:10], str(args.get("end_date") or "")[:10]
        if scope is None or not start or not end:
            return None
        for c in self._days_cache:
            cached_days = c["body"].get("days", {})
            # A granule that began before the cached window (e.g. a monthly product) can't be
            # placed in a narrower one by its start day: leave such searches to the provider.
            if any(d < (c["start"] or "") for d in cached_days):
                continue
            if c["scope"] == scope and (c["start"] or "") <= start and end <= (c["end"] or ""):
                days = {d: t for d, t in cached_days.items() if start <= d <= end}
                n = sum(len(t) for t in days.values())
                return c["id"], {**c["body"], "start_date": start, "end_date": end,
                                 "total": n, "listed": n, "days": days}
        return None

    def log(self, kind: str, step: str | None = None, **payload: Any) -> None:
        self.entries.append({"time": _now(), "kind": kind, "step": step, **payload})

    def add_tool_call(self, **fields: Any) -> dict[str, Any]:
        entry = {"id": f"tc-{len(self.tool_calls) + 1:04d}", "time": _now(), **fields}
        self.tool_calls.append(entry)
        return entry

    def calls_for(self, step: str) -> list[dict[str, Any]]:
        return [c for c in self.tool_calls if c["step"] == step and c["status"] == "ok"]

    def used_tool_calls(self) -> int:
        """Tool calls that count against the agents' budget: everything the guard did not block
        or answer from the cache, except the lab's own data inventory and literature checks."""
        return sum(1 for c in self.tool_calls if c["status"] != "blocked" and not c.get("cached_from")
                   and not c.get("inventory") and not c.get("literature"))

    def literature_tool_calls(self) -> int:
        """Tool calls the literature checks made (their own budget)."""
        return sum(1 for c in self.tool_calls if c.get("literature") and c["status"] != "blocked")

    def catalog_results(self) -> list[dict[str, Any]]:
        """Summaries of successful CDSE catalogue searches (see catalog_summary)."""
        return [c["catalog"] for c in self.tool_calls if c["status"] == "ok" and c.get("catalog")]

    def lab_llm_calls(self) -> int:
        return sum(1 for c in self.llm_calls if c["lab_agent"])

    def usage(self) -> dict[str, int]:
        return {
            "input_tokens": sum(c["input_tokens"] for c in self.llm_calls),
            "output_tokens": sum(c["output_tokens"] for c in self.llm_calls),
            "llm_calls": len(self.llm_calls),
            "lab_llm_calls": self.lab_llm_calls(),
            "tool_calls": len(self.tool_calls),
            "blocked_tool_calls": sum(1 for c in self.tool_calls if c["status"] == "blocked"),
            "cached_tool_calls": sum(1 for c in self.tool_calls if c.get("cached_from")),
            "literature_tool_calls": self.literature_tool_calls(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {"entries": self.entries, "tool_calls": self.tool_calls,
                "llm_calls": self.llm_calls, "usage": self.usage(),
                "budget_exhausted": self.budget_exhausted}

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        return path


class LabGuardPlugin(BasePlugin):
    """Records, permits and budgets every tool and model call in a lab run."""

    def __init__(self, config: VirtualLabConfig, ledger: LabLedger, lab_agents: set[str]) -> None:
        super().__init__(name="virtual_lab_guard")
        self.config = config
        self.ledger = ledger
        self.lab_agents = lab_agents
        self._pending: dict[str, dict[str, Any]] = {}
        # (invocation_id, agent) -> (start ISO time, monotonic start) of the model call in flight
        self._model_start: dict[tuple[str, str], tuple[str, float]] = {}

    def _via(self, tool_context) -> str | None:
        """None for a lab participant's own call, else the sub-agent making the leaf call."""
        agent = tool_context.agent_name
        return None if agent in self.lab_agents else agent

    def _block(self, tool, tool_args, tool_context, reason: str) -> dict:
        self.ledger.add_tool_call(step=current_step.get(), agent=tool_context.agent_name,
                                  tool=tool.name, via=self._via(tool_context), args=tool_args,
                                  run=tool_context.invocation_id, status="blocked", reason=reason)
        logger.info("Blocked tool '%s': %s", tool.name, reason)
        return {"error": reason}

    def _from_cache(self, tool, tool_args, tool_context) -> dict | None:
        if not self.config.cache_tool_calls:
            return None
        hit = self.ledger.recall(tool.name, tool_args)
        if hit is None:
            return None
        source_id, result = hit
        entry = self.ledger.add_tool_call(
            step=current_step.get(), agent=tool_context.agent_name, tool=tool.name,
            via=self._via(tool_context), args=tool_args, run=tool_context.invocation_id,
            status="ok", cached_from=source_id, result_preview=str(result)[:_RESULT_PREVIEW_CHARS])
        if tool.name == CATALOG_TOOL and (summary := catalog_summary(result)):
            entry["catalog"] = summary
        logger.info("Tool '%s' answered from cache (%s)", tool.name, source_id)
        return result if isinstance(result, dict) else {"result": result}

    async def before_tool_callback(self, *, tool, tool_args, tool_context) -> dict | None:
        is_sub_agent = isinstance(tool, AgentTool)
        if (cached := self._from_cache(tool, tool_args, tool_context)) is not None:
            return cached
        if self.ledger.used_tool_calls() >= self.config.max_tool_calls:
            self.ledger.budget_exhausted = self.ledger.budget_exhausted or "tool_calls"
            return self._block(tool, tool_args, tool_context,
                               f"Tool-call budget ({self.config.max_tool_calls}) exhausted. Do not call any more "
                               "tools: return the requested JSON object now, using what you "
                               "already have and marking unchecked points as assumptions.")
        step = current_step.get() or ""
        limit = self._step_limit(step)
        if (limit is not None and tool_context.agent_name in self.lab_agents
                and self._lab_calls_in(step) >= limit[1]):
            return self._block(tool, tool_args, tool_context,
                               f"{limit[0]} tool limit ({limit[1]} calls) reached. Do not call more "
                               "tools: return the requested JSON object now, marking unchecked "
                               "points as assumptions.")
        if (tool_context.agent_name not in self.lab_agents
                and self._run_calls(tool_context.invocation_id) >= self.config.max_sub_agent_tool_calls):
            return self._block(tool, tool_args, tool_context,
                               f"Sub-agent tool limit ({self.config.max_sub_agent_tool_calls} calls "
                               "per request) reached. Answer now with the data you retrieved; if you "
                               "retrieved none, say so instead of guessing.")
        if is_sub_agent:
            _request_with_date(tool_args)
        if not is_sub_agent and tool.name not in self.config.allowed_tools:
            # Read-only in execute mode too: the lab's own fixed analyses run the experiment.
            return self._block(tool, tool_args, tool_context,
                               f"'{tool.name}' is not permitted in {self.config.mode} mode (it may "
                               "download or modify data). Use catalog/search/metadata tools only.")
        entry = self.ledger.add_tool_call(
            step=current_step.get(), agent=tool_context.agent_name, tool=tool.name,
            via=self._via(tool_context), args=tool_args, run=tool_context.invocation_id,
            status="pending",
        )
        if tool_context.function_call_id:
            self._pending[tool_context.function_call_id] = entry
        return None

    def _step_limit(self, step: str) -> tuple[str, int] | None:
        """(label, max tool calls per participant) for steps that have a limit."""
        if step.endswith(".formulate"):
            return "Formulation", self.config.max_formulate_tool_calls
        if step.endswith((".decide", ".redecide")):
            return "Decision", self.config.max_decide_tool_calls
        if step.endswith(".conclude"):
            return "Conclusion", 0      # the conclusion rests on the executed analyses only
        if ".propose." in step:
            return "Proposal", self.config.max_proposal_tool_calls
        if ".critique." in step:
            return "Critique", self.config.max_critique_tool_calls
        if ".revise." in step:
            return "Revision", self.config.max_revise_tool_calls
        return None

    def _lab_calls_in(self, step: str) -> int:
        """Tool calls lab participants made in one step (blocked and cached ones are free)."""
        return sum(1 for c in self.ledger.tool_calls
                   if c["step"] == step and c["via"] is None and c["status"] != "blocked"
                   and not c.get("cached_from"))

    def _run_calls(self, run: str | None) -> int:
        """Leaf tool calls one sub-agent run made (blocked and cached ones are free)."""
        return sum(1 for c in self.ledger.tool_calls
                   if run and c.get("run") == run and c["via"] is not None
                   and c["status"] != "blocked" and not c.get("cached_from"))

    def _leaf_calls_ok(self, entry: dict, sub_agent: str) -> bool:
        """Did the sub-agent run behind `entry` get any successful leaf tool result? If not
        (no call, or all blocked or failed) its reply rests on no data. Parallel runs of the
        same sub-agent in one step are not told apart."""
        later = self.ledger.tool_calls[self.ledger.tool_calls.index(entry) + 1:]
        return any(c["via"] == sub_agent and c["step"] == entry["step"] and c["status"] == "ok"
                   for c in later)

    async def after_tool_callback(self, *, tool, tool_args, tool_context, result) -> dict | None:
        entry = self._pending.pop(tool_context.function_call_id or "", None)
        if entry is None:
            return None
        is_error = reports_error(result)
        entry["status"] = "error" if is_error else "ok"
        entry["result_preview"] = str(result)[:_RESULT_PREVIEW_CHARS]
        if not is_error and tool.name == "search_cdse_catalog" and (summary := catalog_summary(result)):
            entry["catalog"] = summary
        text = str(result)
        if (not is_error and isinstance(tool, AgentTool)
                and (not self._leaf_calls_ok(entry, tool.name) or claims_future_dates(text))):
            entry["status"] = "no_answer"
            return {"result": result, "lab_note": NO_ANSWER_NOTE}
        if entry["status"] == "ok" and self.config.cache_tool_calls:
            self.ledger.remember(entry, result)
        return None

    async def on_tool_error_callback(self, *, tool, tool_args, tool_context, error) -> dict | None:
        from agenticeoecosystem.agents.subagents.orphaned_tool_call_guard import (
            OrphanedToolCallStormError,
        )
        entry = self._pending.pop(tool_context.function_call_id or "", None)
        if entry is not None:
            entry["status"] = "error"
            entry["result_preview"] = str(error)[:_RESULT_PREVIEW_CHARS]
        if isinstance(error, OrphanedToolCallStormError):
            return None  # let the storm guard abort the run, as in multi_agent._on_tool_error
        logger.warning("Tool '%s' failed: %s", tool.name, error)
        return {"error": f"Tool '{tool.name}' failed: {error}"}

    async def before_model_callback(self, *, callback_context, llm_request) -> LlmResponse | None:
        if (callback_context.agent_name in self.lab_agents
                and self.ledger.lab_llm_calls() >= self.config.max_llm_calls):
            self.ledger.budget_exhausted = self.ledger.budget_exhausted or "llm_calls"
            return LlmResponse(content=types.Content(role="model", parts=[
                types.Part(text="BUDGET_EXHAUSTED")]))
        self._model_start[(callback_context.invocation_id, callback_context.agent_name)] = (
            _now(), time.monotonic())
        return None

    async def after_model_callback(self, *, callback_context, llm_response) -> LlmResponse | None:
        usage = llm_response.usage_metadata
        start, t0 = self._model_start.pop((callback_context.invocation_id, callback_context.agent_name),
                                          (None, None))
        self.ledger.llm_calls.append({
            "step": current_step.get(),
            "agent": callback_context.agent_name,
            "lab_agent": callback_context.agent_name in self.lab_agents,
            "input_tokens": (usage.prompt_token_count or 0) if usage else 0,
            "output_tokens": (usage.candidates_token_count or 0) if usage else 0,
            "start": start,
            "duration_s": round(time.monotonic() - t0, 2) if t0 is not None else None,
        })
        return None
