"""
app/virtual_lab_app.py — Live Gradio view of a Virtual Lab meeting.

Top  : phase tracker (formulate → inventory → proposals → rounds → decision → plan) and the
       lab members, highlighting who is speaking now.
Left : the meeting as cards, one per contribution (problem, data inventory, proposals,
       critiques, validator findings, literature checks, revisions, decision, plan),
       colour-coded by member. Tabs hold the full data inventory and the final answer.
Right: elapsed time and budget use, and a live feed of tool calls.

Everything is rendered from the lab's ledger (LabLedger.to_dict()), the same record the result
JSON stores, so a finished run can be replayed from logs/ with identical rendering.

Run (from the repo root, or the /scratch copy for faster start-up):
    python app/virtual_lab_app.py [--port 7861] [--host 127.0.0.1]
All participants and sub-agents run on EVE (EVE_ENDPOINT), as in
scripts/run_virtual_lab_example.py.
"""

from __future__ import annotations

import asyncio
import html
import importlib.util
import json
import logging
import re
import time
from datetime import datetime
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from agenticeoecosystem.definitions import ROOT_DIR  # noqa: E402


def _load_example():
    """The EMSR859 query and force_eve() from the example script (no package import path)."""
    path = ROOT_DIR / "scripts" / "run_virtual_lab_example.py"
    spec = importlib.util.spec_from_file_location("run_virtual_lab_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_EXAMPLE = _load_example()
_EXAMPLE.force_eve()   # before any agent module is imported: llm_init reads the env at import

import gradio as gr  # noqa: E402
from markdown_it import MarkdownIt  # noqa: E402  (a gradio dependency)

logger = logging.getLogger(__name__)
LOGS_DIR = ROOT_DIR / "logs"
REFRESH_S = 1.5
PANEL_HEIGHT_PX = 760   # transcript and side panel scroll inside this height
# Model text is untrusted: raw HTML in it is escaped, never rendered.
_MD = MarkdownIt("commonmark", {"html": False, "linkify": False}).enable("table")

# ---------------------------------------------------------------------------
# Lab members
# ---------------------------------------------------------------------------

# agent -> (icon, display name, accent colour)
_ROLES: dict[str, tuple[str, str, str]] = {
    "lab_pi": ("🧑‍🔬", "Principal Investigator", "#4f46e5"),
    "optical_specialist": ("🛰️", "Optical", "#0891b2"),
    "thermal_specialist": ("🔥", "Thermal", "#ea580c"),
    "sar_specialist": ("📡", "SAR", "#7c3aed"),
    "atmosphere_specialist": ("🌫️", "Atmosphere", "#0284c7"),
    "foundation_model_specialist": ("🧠", "Foundation models", "#db2777"),
    "scientific_critic": ("🧐", "Scientific critic", "#b91c1c"),
    "validator": ("⚖️", "Validator", "#64748b"),
    "literature_reviewer": ("📚", "Literature reviewer", "#15803d"),
    "experiment_designer": ("🧪", "Experiment designer", "#a16207"),
    "lab": ("🗂️", "Lab", "#475569"),
    "lab_analyses": ("📊", "Lab analyses", "#0f766e"),
}
_FIXED_MEMBERS = ("lab_pi", "scientific_critic", "literature_reviewer", "experiment_designer")
_SEVERITY = {"blocking": ("blocking", "sev-blocking"), "major": ("major", "sev-major"),
             "minor": ("minor", "sev-minor")}
_VERDICT = {"supports": "✅ supports", "contradicts": "❌ contradicts", "mixed": "⚖️ mixed",
            "inconclusive": "❔ inconclusive", "not_searched": "· not searched"}


def _role(agent: str | None) -> tuple[str, str, str]:
    return _ROLES.get(agent or "", ("🤖", agent or "?", "#64748b"))


def _clip(text: Any, n: int | None = None) -> str:
    """Whitespace collapsed; cut to n characters only when n is given."""
    text = " ".join(str(text or "").split())
    return text if n is None or len(text) <= n else text[: n - 1] + "…"


def _esc(text: Any) -> str:
    return html.escape(str(text or ""))


# ---------------------------------------------------------------------------
# Ledger entry -> card
# ---------------------------------------------------------------------------

def _card(agent: str, title: str, body_md: str, time_: str = "", badge: str = "",
          badge_class: str = "", extra_class: str = "", extra_html: str = "") -> str:
    icon, name, colour = _role(agent)
    badge_html = f'<span class="vl-badge {badge_class}">{_esc(badge)}</span>' if badge else ""
    return (f'<article class="vl-card {extra_class}" style="--accent:{colour}">'
            f'<header><span class="vl-avatar">{icon}</span>'
            f'<span class="vl-who">{_esc(name)}</span><span class="vl-title">{_esc(title)}</span>'
            f'{badge_html}<time>{_esc(time_)}</time></header>'
            f'<div class="vl-body">{_MD.render(body_md)}{extra_html}</div></article>')


def _issues_md(issues: list[dict]) -> str:
    order = {"blocking": 0, "major": 1, "minor": 2}
    rows = []
    for i in sorted(issues, key=lambda i: order.get(i.get("severity"), 3)):
        row = (f"- **{i.get('severity', '').upper()}** · *{i.get('category')}* — "
               f"**{_clip(i.get('claim'))}**: {_clip(i.get('rationale'))}")
        if i.get("suggested_fix"):
            row += f" *Fix:* {_clip(i['suggested_fix'])}"
        rows.append(row)
    return "\n".join(rows) or "No issues."


def _worst(issues: list[dict]) -> tuple[str, str]:
    for sev in ("blocking", "major", "minor"):
        n = sum(i.get("severity") == sev for i in issues)
        if n:
            label, cls = _SEVERITY[sev]
            return f"{n} {label}", cls
    return "no issues", "sev-ok"


def _proposal_card(e: dict) -> str:
    p = e["proposal"]
    revised = e["kind"] == "revision"
    body = [_clip(p.get("summary"))]
    if revised and p.get("changes_from_previous"):
        body.append(f"**What changed:** {_clip(p['changes_from_previous'])}")
    if p.get("hypotheses"):
        body.append("**Hypotheses**\n\n" + "\n".join(
            f"- `{h.get('id')}` {_clip(h.get('statement'))}"
            + (f"\n  - *Predicts:* {_clip(h['prediction'])}" if h.get("prediction") else "")
            + (f"\n  - *Refuted if:* {_clip(h['falsification'])}" if h.get("falsification") else "")
            for h in p["hypotheses"]))
    if p.get("datasets"):
        body.append("**Data**\n\n" + "\n".join(
            f"- {d.get('name')} *({d.get('role')})*"
            + (f" — {_clip(d.get('observable'))}" if d.get("observable") else "")
            + (f". {_clip(d.get('justification'))}" if d.get("justification") else "")
            for d in p["datasets"]))
    for key, label in (("methods", "Methods"), ("assumptions", "Assumptions"), ("limitations", "Limitations")):
        if p.get(key):
            body.append(f"**{label}**\n\n" + "\n".join(f"- {_clip(x)}" for x in p[key]))
    kinds: dict[str, int] = {}
    for ev in p.get("evidence", []):
        kinds[ev.get("kind", "?")] = kinds.get(ev.get("kind", "?"), 0) + 1
    if kinds:
        body.append("**Evidence** — " + ", ".join(f"{n} {k}" for k, n in kinds.items()))
    title = f"revision {p.get('revision')}" if revised else "proposal"
    return _card(e.get("agent"), title, "\n\n".join(body), e.get("_t", ""),
                 f"confidence {p.get('confidence')}", "conf")


def _critique_card(e: dict) -> str:
    c = e["critique"]
    target = c.get("target") or "?"
    issues = c.get("issues", [])
    badge, cls = _worst(issues)
    if c.get("author") == "validator":
        title = "checks the plan" if target == "experiment_plan" else f"checks {_role(target)[1]}"
        return _card("validator", title, _issues_md(issues), e.get("_t", ""), badge, cls, "vl-quiet")
    title = "reviews the plan" if target == "experiment_plan" else f"→ critiques {_role(target)[1]}"
    return _card(c.get("author"), title, _issues_md(issues), e.get("_t", ""), badge, cls)


def _literature_card(e: dict) -> str:
    c = e["check"]
    body = [f"> {_clip(c.get('claim'))}",
            f"Raised by **{_role(c.get('raised_by'))[1]}**"
            + (f" against **{_role(c.get('target'))[1]}**" if c.get("target") else ""),
            _clip(c.get("summary"))]
    papers = [f"- arXiv:{p.get('arxiv_id')} — {_clip(p.get('title'))} ({p.get('year') or '?'}"
              + (f", {p['citation_count']} citations" if p.get("citation_count") is not None else "") + ")"
              for p in c.get("papers", [])]
    if papers:
        body.append("\n".join(papers))
    verdict = c.get("verdict", "not_searched")
    return _card("literature_reviewer", "literature check", "\n\n".join(b for b in body if b),
                 e.get("_t", ""), _VERDICT.get(verdict, verdict), f"verdict-{verdict}")


def _problem_card(e: dict) -> str:
    p = e["problem"]
    area, period = p.get("study_area") or {}, p.get("time_period") or {}
    body = [f"**{_clip(p.get('question'))}**",
            "\n".join(f"- {_clip(o)}" for o in p.get("objectives", [])),
            f"**Area** {area.get('name')} `{area.get('bbox')}` · **Period** {period.get('start')} → {period.get('end')}"]
    if p.get("participants"):
        body.append("**Specialists** — " + ", ".join(_role(n)[1] for n in p["participants"]))
    if p.get("sub_agents"):
        body.append("**Sub-agents** — " + ", ".join(f"`{n}`" for n in p["sub_agents"]))
    return _card("lab_pi", "frames the problem", "\n\n".join(b for b in body if b), e.get("_t", ""))


def _decision_card(e: dict) -> str:
    d = e["decision"]
    body = [_clip(d.get("summary"))]
    if d.get("status_note"):
        body.append(f"⚠️ *{d['status_note']}*")
    if d.get("recommended_workflow"):
        body.append("**Recommended workflow**\n\n" + "\n".join(f"- {_clip(s)}" for s in d["recommended_workflow"]))
    for dis in d.get("disagreements", []):
        body.append(f"**Open disagreement — {dis.get('topic')}**\n\n" + "\n".join(
            f"- *{_role(a)[1]}*: {_clip(pos)}" for a, pos in (dis.get("positions") or {}).items()))
    for j in d.get("justification", []):
        if j.get("kind") == "verified":
            body.append(f"✔ *verified by {j.get('result_id')}*: {_clip(j.get('claim'))}")
    status = d.get("status", "?")
    title = "concludes from the results" if e.get("step") == "final.conclude" else "decides"
    return _card("lab_pi", title, "\n\n".join(body), e.get("_t", ""), status.replace("_", " "),
                 f"status-{status}", "vl-highlight")


def _image_html(path: str) -> str:
    """Quicklooks are PNGs the lab wrote itself; embed them inline."""
    import base64
    from pathlib import Path
    p = Path(path)
    if p.suffix.lower() != ".png" or not p.is_file():
        return ""
    b64 = base64.b64encode(p.read_bytes()).decode()
    return f'<img class="vl-figure" src="data:image/png;base64,{b64}" alt="{_esc(p.name)}">'


def _result_card(e: dict) -> str:
    r = e["result"]
    metrics = r.get("metrics") or {}
    body = [_clip(r.get("summary"))]
    if metrics:
        body.append("| metric | value |\n|---|---|\n" + "\n".join(
            f"| {k.replace('_', ' ')} | {v:g} |" for k, v in metrics.items()))
    days = (r.get("details") or {}).get("days")
    if days:
        body.append("| date | phase | cloud | aerosol |\n|---|---|---|---|\n" + "\n".join(
            f"| {d['date']} | {d['phase']} | {100 * d['cloud']:.0f}% | {d['aot']:.2f} |" for d in days))
    images = "".join(_image_html(a) for a in r.get("artifacts", []))
    ok = r.get("status") == "ok"
    return _card("lab_analyses", f"{r.get('id')} · {r.get('analysis')}", "\n\n".join(body), e.get("_t", ""),
                 "ok" if ok else "failed", "sev-ok" if ok else "sev-blocking", "vl-highlight" if ok else "",
                 images)


def _plan_card(e: dict) -> str:
    p = e["plan"]
    body = [f"**{_clip(p.get('research_question'))}**",
            "\n".join(f"{s.get('order')}. `{s.get('agent')}` — {_clip(s.get('description'))}"
                      for s in p.get("workflow", []))]
    if p.get("metrics"):
        body.append("**Metrics** — " + "; ".join(_clip(m) for m in p["metrics"]))
    issues = (e.get("validation") or {}).get("issues", [])
    if issues:
        body.append("**Validator on the plan**\n\n" + _issues_md(issues))
    badge, cls = _worst(issues)
    return _card("experiment_designer", "experiment plan", "\n\n".join(body), e.get("_t", ""),
                 badge, cls, "vl-highlight")


def _inventory_card(e: dict) -> str:
    lines = (e.get("inventory") or "").splitlines()
    sections = [ln for ln in lines if ln and not ln.startswith(("-", " "))]
    body = ("Searched every provider's public catalogue for the study area and period. "
            "Full text in the **Data inventory** tab.\n\n" + "\n".join(f"- {_clip(s, 160)}" for s in sections))
    return _card("lab", "data inventory", body, e.get("_t", ""), f"{len(lines)} lines", "conf")


_CARDS = {"problem": _problem_card, "inventory": _inventory_card, "proposal": _proposal_card,
          "revision": _proposal_card, "critique": _critique_card, "validation": _critique_card,
          "literature": _literature_card, "decision": _decision_card, "plan": _plan_card,
          "result": _result_card}


def transcript_html(ledger: dict, query: str = "") -> str:
    cards = []
    if query:
        cards.append(f'<article class="vl-card vl-query"><header><span class="vl-avatar">❓</span>'
                     f'<span class="vl-who">Question</span></header>'
                     f'<div class="vl-body">{_MD.render(query)}</div></article>')
    for entry in ledger.get("entries", []):
        render = _CARDS.get(entry.get("kind"))
        if render is None:
            continue
        if entry["kind"] == "validation" and not (entry.get("critique") or {}).get("issues"):
            continue   # a clean validation adds nothing to read
        try:
            cards.append(render({**entry, "_t": (entry.get("time") or "")[11:16] + " UTC"}))
        except Exception as exc:  # a malformed entry must not break the view
            cards.append(_card("lab", f"could not render {entry.get('kind')}", _esc(exc)))
    if len(cards) <= (1 if query else 0):
        cards.append('<div class="vl-empty">The meeting appears here as members speak.</div>')
    # Newest first, so a live meeting shows the latest contribution without scrolling;
    # the question stays at the bottom, where the meeting starts.
    return f'<div class="vl-feed">{"".join(reversed(cards))}</div>'


# ---------------------------------------------------------------------------
# Phase tracker, roster, stats, tool feed
# ---------------------------------------------------------------------------

def _phases(rounds: int, execute: bool = False) -> list[tuple[str, str]]:
    out = [("formulate", "Formulate"), ("inventory", "Data inventory"), ("propose", "Proposals")]
    out += [(f"r{r}", f"Round {r}") for r in range(1, rounds + 1)]
    out += [("decide", "Decision"), ("design", "Plan")]
    return out + ([("execute", "Experiment"), ("conclude", "Conclusion")] if execute else [])


def _current_phase(step: str | None, ledger: dict) -> str | None:
    if step:
        if step.startswith("r0.formulate"):
            return "formulate"
        if step.startswith("r0.inventory"):
            return "inventory"
        if step.startswith("r0."):
            return "propose"
        if m := re.match(r"r(\d+)\.", step):
            return f"r{m.group(1)}"
        if step.startswith("final.execute"):
            return "execute"
        if step.startswith("final.conclude"):
            return "conclude"
        if step.startswith("final.") and any(k in step for k in ("design", "plan_review")):
            return "design"
        if step.startswith("final."):
            return "decide"
    kinds = {e.get("kind") for e in ledger.get("entries", [])}
    return ("execute" if "result" in kinds else "design" if "plan" in kinds
            else "decide" if "decision" in kinds
            else "inventory" if "inventory" in kinds else None)


def tracker_html(ledger: dict, live: dict | None, rounds: int) -> str:
    execute = bool((live or {}).get("execute")) or any(e.get("kind") == "result" for e in ledger.get("entries", []))
    phases = _phases(rounds, execute)
    keys = [k for k, _ in phases]
    current = _current_phase((live or {}).get("step"), ledger)
    finished = bool(live and live.get("finished"))
    idx = keys.index(current) if current in keys else -1
    steps = []
    for i, (_key, label) in enumerate(phases):
        cls = "done" if finished or i < idx else "active" if i == idx else ""
        steps.append(f'<li class="{cls}"><span class="dot"></span>{_esc(label)}</li>')

    problem = next((e["problem"] for e in ledger.get("entries", []) if e.get("kind") == "problem"), {})
    members = list(_FIXED_MEMBERS[:1]) + list(problem.get("participants") or (live or {}).get("participants") or []) \
        + list(_FIXED_MEMBERS[1:])
    spoke = {e.get("agent") for e in ledger.get("entries", [])}
    spoke |= {(e.get("critique") or {}).get("author") for e in ledger.get("entries", [])}
    if any(e.get("kind") == "literature" for e in ledger.get("entries", [])):
        spoke.add("literature_reviewer")
    if any(e.get("kind") == "plan" for e in ledger.get("entries", [])):
        spoke.add("experiment_designer")
    if execute:
        members.append("lab_analyses")
        if any(e.get("kind") == "result" for e in ledger.get("entries", [])):
            spoke.add("lab_analyses")
    if problem:
        spoke.add("lab_pi")
    now = None if finished else (live or {}).get("agent")
    if not finished and ((live or {}).get("step") or "").startswith("final.execute"):
        now = "lab_analyses"
    chips = []
    for m in dict.fromkeys(members):
        icon, name, colour = _role(m)
        state = "speaking" if m == now else "spoke" if m in spoke else ""
        chips.append(f'<span class="vl-chip {state}" style="--accent:{colour}">{icon} {_esc(name)}</span>')
    return (f'<div class="vl-top"><ol class="vl-phases">{"".join(steps)}</ol>'
            f'<div class="vl-roster">{"".join(chips)}</div></div>')


def _tool_kind(c: dict) -> tuple[str, str]:
    if c.get("analysis"):
        return ("analysis", "k-ana") if c.get("status") == "ok" else ("analysis ✗", "k-err")
    if c.get("inventory"):
        return "inventory", "k-inv"
    if c.get("cached_from"):
        return "cache", "k-cache"
    if c.get("literature"):
        return "literature", "k-lit"
    return {"pending": ("running", "k-run"), "ok": ("ok", "k-ok"), "blocked": ("blocked", "k-block"),
            "error": ("error", "k-err"), "no_answer": ("no data", "k-err")}.get(c.get("status"), ("?", ""))


def side_html(ledger: dict, live: dict | None, max_calls: int = 30) -> str:
    usage = ledger.get("usage") or {}
    state = (live or {}).get("state", "Idle")
    tiles = [("Elapsed", (live or {}).get("elapsed") or "–"),
             ("LLM calls", usage.get("lab_llm_calls", 0)),
             ("Tool calls", usage.get("tool_calls", 0)),
             ("From cache", usage.get("cached_tool_calls", 0)),
             ("Literature", usage.get("literature_tool_calls", 0)),
             ("Blocked", usage.get("blocked_tool_calls", 0))]
    tiles_html = "".join(f'<div class="vl-tile"><b>{_esc(v)}</b><span>{_esc(k)}</span></div>' for k, v in tiles)
    now = ""
    if live and live.get("step") and not live.get("finished"):
        now = f'<div class="vl-now">Step <code>{_esc(live["step"])}</code></div>'
    warn = (f'<div class="vl-warn">Budget exhausted: {_esc(ledger["budget_exhausted"])}</div>'
            if ledger.get("budget_exhausted") else "")
    rows = []
    for c in reversed(ledger.get("tool_calls", [])[-max_calls:]):
        label, cls = _tool_kind(c)
        who = _role(c.get("agent"))[1] if c.get("agent") in _ROLES else c.get("agent")
        via = f" via {c['via']}" if c.get("via") else ""
        detail = _clip(c.get("reason"), 120) if c.get("status") == "blocked" else \
            _clip(json.dumps(c.get("args", {}), default=str), 110)
        rows.append(f'<li><span class="vl-kind {cls}">{label}</span><div><b>{_esc(c.get("tool"))}</b>'
                    f'<small> · {_esc(who)}{_esc(via)} · {_esc(c.get("id"))}</small>'
                    f'<p>{_esc(detail)}</p></div></li>')
    feed = "".join(rows) or '<li class="vl-empty">No tool calls yet.</li>'
    return (f'<div class="vl-state">{_esc(state)}</div>{now}<div class="vl-tiles">{tiles_html}</div>{warn}'
            f'<h4>Tool calls</h4><ul class="vl-tools">{feed}</ul>')


def inventory_md(ledger: dict) -> str:
    entry = next((e for e in ledger.get("entries", []) if e.get("kind") == "inventory"), None)
    return (f"```text\n{entry['inventory']}\n```" if entry
            else "_The data inventory is built right after the PI formulates the problem._")


def final_md(ledger: dict, answer: str | None = None) -> str:
    if answer:
        return answer
    return "_The final answer appears when the meeting ends._"


# ---------------------------------------------------------------------------
# Live run and replay
# ---------------------------------------------------------------------------

_BRANCH_STEP = re.compile(r"^(?P<step>.+?)\.a\d+(?:\.|$)")


def _step_of(branch: str | None, coordinator_name: str) -> str | None:
    """'virtual_lab.r1.critique.thermal_specialist.a0...' -> 'r1.critique.thermal_specialist'."""
    if not branch or not branch.startswith(coordinator_name + "."):
        return None
    m = _BRANCH_STEP.match(branch[len(coordinator_name) + 1:])
    return m.group("step") if m else None


def _elapsed(t0: float) -> str:
    s = int(time.monotonic() - t0)
    return f"{s // 60}:{s % 60:02d}"


def _empty_ledger() -> dict:
    return {"entries": [], "tool_calls": [], "llm_calls": [], "usage": {}, "budget_exhausted": None}


def _view(ledger: dict, live: dict, query: str, rounds: int, answer: str | None) -> tuple:
    return (tracker_html(ledger, live, rounds), transcript_html(ledger, query), side_html(ledger, live),
            inventory_md(ledger), final_md(ledger, answer))


async def run_lab(query: str, rounds: int, execute: bool = False):
    """Gradio streaming handler. Yields the five views every REFRESH_S, skipping unchanged ones
    so the transcript only re-renders (and scrolls) when a member has spoken."""
    from agenticeoecosystem.agents.virtual_lab.config import VirtualLabConfig
    from agenticeoecosystem.agents.virtual_lab.runner import stream_virtual_lab

    query, rounds = (query or "").strip(), int(rounds)
    if not query:
        gr.Warning("Enter a scientific question first.")
        return
    config = VirtualLabConfig.from_env(max_rounds=rounds, mode="execute" if execute else "plan_only")
    live: dict[str, Any] = {"state": "Starting — loading agents…", "step": None, "agent": None,
                            "participants": list(config.participants), "execute": bool(execute)}
    holder: dict[str, Any] = {"coordinator": None, "ledger": None, "answer": None}
    t0 = time.monotonic()

    async def consume() -> None:
        async for event, coordinator, ledger in stream_virtual_lab(query, config):
            holder["coordinator"], holder["ledger"] = coordinator, ledger
            live["state"] = "Meeting in progress"
            if step := _step_of(event.branch, coordinator.name):
                live["step"] = step
            if event.author and event.author not in ("user", coordinator.name):
                live["agent"] = event.author
            if (event.author == coordinator.name and event.content and event.content.parts
                    and (text := "".join(p.text or "" for p in event.content.parts))):
                holder["answer"] = text

    last: list = [None] * 5

    def changed(views: tuple) -> list:
        out = [gr.skip() if v == last[i] else v for i, v in enumerate(views)]
        last[:] = views
        return out

    def snapshot(final: bool = False) -> tuple:
        ledger = holder["ledger"].to_dict() if holder["ledger"] else _empty_ledger()
        live["elapsed"] = _elapsed(t0)
        return _view(ledger, live, query, rounds, holder["answer"] if final else None)

    task = asyncio.create_task(consume())
    try:
        while not task.done():
            yield changed(snapshot())
            await asyncio.wait({task}, timeout=REFRESH_S)
        if task.exception() is not None:
            logger.error("Virtual Lab run failed", exc_info=task.exception())
            live["state"] = f"Failed: {task.exception()!r}"[:300]
        else:
            live.update(state="Finished", agent=None, finished=True)
        yield changed(snapshot(final=True))
        _save(query, config, holder)
    finally:
        if not task.done():   # the user pressed Stop or closed the page
            task.cancel()


def _save(query: str, config, holder: dict) -> None:
    """Write the same result JSON as the example script, so app runs can be replayed."""
    if holder["ledger"] is None:
        return
    record = holder["coordinator"].record if holder["coordinator"] else None
    out = LOGS_DIR / f"virtual_lab_app_result_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps({"query": query, "config": config.model_dump(mode="json"),
                               "final_answer": holder["answer"],
                               "lab_record": record.model_dump(mode="json") if record else None,
                               "ledger": holder["ledger"].to_dict()}, indent=2, default=str))
    logger.info("Result written to %s", out)


def result_files() -> list[str]:
    files = sorted(LOGS_DIR.glob("virtual_lab_*result_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.name for p in files]


def replay(name: str):
    if not name:
        gr.Warning("Pick a result file first.")
        return [gr.skip()] * 5
    data = json.loads((LOGS_DIR / name).read_text())
    ledger = data.get("ledger") or _empty_ledger()
    rounds = int((data.get("config") or {}).get("max_rounds", 2))
    live = {"state": f"Replay · {name}", "finished": True,
            "participants": (data.get("config") or {}).get("participants", [])}
    return _view(ledger, live, data.get("query", ""), rounds, data.get("final_answer"))


# ---------------------------------------------------------------------------
# Look and feel
# ---------------------------------------------------------------------------

_THEME = gr.themes.Soft(
    primary_hue="indigo", secondary_hue="sky", neutral_hue="slate", radius_size="lg",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
).set(body_text_size="15px", block_title_text_weight="600")

_CSS = """
.gradio-container { max-width: 1600px !important; margin: 0 auto !important; }
body, .gradio-container { -webkit-font-smoothing: antialiased; }
#vl-header { align-items: center; gap: 16px; padding: 4px 4px 12px;
             border-bottom: 1px solid var(--border-color-primary); margin-bottom: 4px; }
#vl-header h1 { font-size: 1.45rem; font-weight: 700; letter-spacing: -0.01em; margin: 0; }
#vl-header p  { margin: 2px 0 0; color: var(--body-text-color-subdued); font-size: 0.92rem; }
#vl-question textarea { font-size: 0.92rem; line-height: 1.5; }

/* phase tracker + roster */
.vl-top { display: flex; flex-direction: column; gap: 10px; padding: 10px 2px; }
.vl-phases { display: flex; flex-wrap: wrap; gap: 6px; list-style: none; margin: 0; padding: 0; }
.vl-phases li { display: flex; align-items: center; gap: 6px; padding: 5px 12px; border-radius: 999px;
                font-size: 0.82rem; font-weight: 500; color: var(--body-text-color-subdued);
                background: var(--background-fill-secondary); border: 1px solid var(--border-color-primary); }
.vl-phases .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--border-color-primary); }
.vl-phases li.done { color: var(--body-text-color); }
.vl-phases li.done .dot { background: #16a34a; }
.vl-phases li.active { color: #fff; background: var(--color-accent); border-color: var(--color-accent); }
.vl-phases li.active .dot { background: #fff; animation: vl-pulse 1.2s ease-in-out infinite; }
.vl-roster { display: flex; flex-wrap: wrap; gap: 6px; }
.vl-chip { font-size: 0.8rem; padding: 4px 10px; border-radius: 999px; opacity: 0.55;
           border: 1px solid var(--border-color-primary); background: var(--background-fill-primary); }
.vl-chip.spoke { opacity: 1; border-color: var(--accent); }
.vl-chip.speaking { opacity: 1; color: #fff; background: var(--accent); border-color: var(--accent);
                    box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 25%, transparent); }
@keyframes vl-pulse { 50% { opacity: 0.35; } }

/* transcript */
#vl-transcript { overflow-y: auto !important; padding-right: 6px; }
.vl-feed { display: flex; flex-direction: column; gap: 12px; padding: 4px 2px; }
.vl-card { border: 1px solid var(--border-color-primary); border-left: 4px solid var(--accent, #94a3b8);
           border-radius: 10px; background: var(--background-fill-primary); padding: 12px 16px; }
.vl-card.vl-quiet { opacity: 0.85; background: var(--background-fill-secondary); }
.vl-card.vl-highlight { box-shadow: 0 1px 8px color-mix(in srgb, var(--accent) 22%, transparent); }
.vl-card.vl-query { --accent: #94a3b8; background: var(--background-fill-secondary); }
.vl-card header { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-bottom: 6px; }
.vl-avatar { width: 30px; height: 30px; display: grid; place-items: center; border-radius: 50%; font-size: 16px;
             background: color-mix(in srgb, var(--accent, #94a3b8) 14%, transparent); }
.vl-who { font-weight: 650; color: var(--accent, var(--body-text-color)); }
.vl-title { color: var(--body-text-color-subdued); font-size: 0.9rem; }
.vl-card time { margin-left: auto; font-size: 0.75rem; color: var(--body-text-color-subdued);
                font-family: var(--font-mono); }
.vl-body { font-size: 0.92rem; line-height: 1.55; }
.vl-body p { margin: 0.35em 0; }
.vl-body ul, .vl-body ol { margin: 0.3em 0 0.3em 1.2em; padding: 0; }
.vl-body li { margin: 0.15em 0; }
.vl-body blockquote { margin: 0.4em 0; padding: 2px 10px; border-left: 3px solid var(--border-color-primary);
                      color: var(--body-text-color-subdued); }
.vl-body code { font-size: 0.82em; padding: 1px 5px; border-radius: 4px; background: var(--background-fill-secondary); }
.vl-badge { font-size: 0.72rem; font-weight: 600; padding: 2px 8px; border-radius: 999px; text-transform: uppercase;
            letter-spacing: 0.03em; background: var(--background-fill-secondary); color: var(--body-text-color-subdued); }
.sev-blocking, .verdict-contradicts, .status-insufficient_evidence { background: #fee2e2; color: #991b1b; }
.sev-major, .verdict-mixed, .status-needs_experiment { background: #ffedd5; color: #9a3412; }
.sev-minor, .verdict-inconclusive { background: #fef9c3; color: #854d0e; }
.sev-ok, .verdict-supports, .status-decided { background: #dcfce7; color: #166534; }
.vl-empty { color: var(--body-text-color-subdued); text-align: center; padding: 40px 0; font-size: 0.92rem; }

/* side panel */
#vl-side { overflow-y: auto !important; padding-right: 4px; }
.vl-state { font-weight: 650; font-size: 1rem; margin-bottom: 4px; }
.vl-now { font-size: 0.82rem; color: var(--body-text-color-subdued); margin-bottom: 8px; }
.vl-tiles { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin: 8px 0 12px; }
.vl-tile { border: 1px solid var(--border-color-primary); border-radius: 10px; padding: 8px 10px;
           background: var(--background-fill-secondary); display: flex; flex-direction: column; }
.vl-tile b { font-size: 1.15rem; font-variant-numeric: tabular-nums; }
.vl-tile span { font-size: 0.72rem; color: var(--body-text-color-subdued); text-transform: uppercase;
                letter-spacing: 0.04em; }
.vl-warn { background: #fee2e2; color: #991b1b; border-radius: 8px; padding: 6px 10px; font-size: 0.85rem; }
#vl-side h4 { margin: 8px 0 6px; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em;
              color: var(--body-text-color-subdued); }
.vl-tools { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 6px; }
.vl-tools li { display: flex; gap: 8px; align-items: flex-start; font-size: 0.82rem;
               border-bottom: 1px dashed var(--border-color-primary); padding-bottom: 6px; }
.vl-tools p { margin: 2px 0 0; color: var(--body-text-color-subdued); font-family: var(--font-mono);
              font-size: 0.72rem; word-break: break-all; }
.vl-tools small { color: var(--body-text-color-subdued); }
.vl-kind { flex: none; min-width: 64px; text-align: center; font-size: 0.68rem; font-weight: 600;
           text-transform: uppercase; padding: 2px 6px; border-radius: 6px;
           background: var(--background-fill-secondary); color: var(--body-text-color-subdued); }
.k-ok { background: #dcfce7; color: #166534; } .k-cache { background: #e0f2fe; color: #075985; }
.k-inv { background: #ede9fe; color: #5b21b6; } .k-lit { background: #dcfce7; color: #14532d; }
.k-block { background: #fee2e2; color: #991b1b; } .k-err { background: #ffedd5; color: #9a3412; }
.k-run { background: #fef9c3; color: #854d0e; }
.k-ana { background: #ccfbf1; color: #115e59; }
.vl-figure { width: 100%; height: auto; margin-top: 10px; border-radius: 8px;
             border: 1px solid var(--border-color-primary); background: #fff; }
.vl-body table { border-collapse: collapse; font-size: 0.82rem; margin: 6px 0; }
.vl-body th, .vl-body td { border: 1px solid var(--border-color-primary); padding: 3px 8px; text-align: left; }
#vl-inventory pre, #vl-inventory code { font-size: 0.78rem; line-height: 1.45; }
"""


def _logo_html() -> str:
    import base64
    path = ROOT_DIR / "app" / "ESA_philab.png"
    if not path.exists():
        return ""
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{b64}" alt="ESA Φ-lab" style="height:64px;width:auto;">'


def build_app() -> gr.Blocks:
    with gr.Blocks(title="EO Virtual Lab") as demo:   # normal page scroll; panels scroll inside
        with gr.Row(elem_id="vl-header"):
            gr.HTML(_logo_html(), min_width=80, scale=0)
            gr.HTML("<h1>EO Virtual Scientific Lab</h1><p>The PI frames the problem, specialists "
                    "propose, critique and revise, the PI decides and the designer writes the "
                    "experiment plan. With <b>Execute analyses</b> the lab then tests the plan on "
                    "Sentinel-1/2 data and the PI concludes from the results.</p>", scale=1)
        with gr.Row(equal_height=False):
            with gr.Column(scale=8):
                query = gr.Textbox(value=_EXAMPLE.QUERY, lines=3, max_lines=12, show_label=False,
                                   elem_id="vl-question", placeholder="Scientific question…")
            with gr.Column(scale=2, min_width=220):
                rounds = gr.Slider(0, 3, value=2, step=1, label="Deliberation rounds (max)")
                execute = gr.Checkbox(value=False, label="Execute analyses",
                                      info="Test the claims on Sentinel-1/2 data (Sentinel Hub, CDSE quota) "
                                           "and conclude from the results.")
                with gr.Row():
                    run_btn = gr.Button("Run meeting", variant="primary", icon=None)
                    stop_btn = gr.Button("Stop", variant="stop")
        tracker = gr.HTML(tracker_html(_empty_ledger(), None, 2))
        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                with gr.Tabs():
                    with gr.Tab("Meeting"):
                        transcript = gr.HTML(transcript_html(_empty_ledger()), elem_id="vl-transcript",
                                             min_height=420, max_height=PANEL_HEIGHT_PX)
                    with gr.Tab("Data inventory"):
                        inventory = gr.Markdown(inventory_md(_empty_ledger()), elem_id="vl-inventory")
                    with gr.Tab("Final answer"):
                        final = gr.Markdown(final_md(_empty_ledger()))
            with gr.Column(scale=1, min_width=330):
                side = gr.HTML(side_html(_empty_ledger(), None), elem_id="vl-side",
                               min_height=420, max_height=PANEL_HEIGHT_PX)
                with gr.Accordion("Replay a finished run", open=False):
                    files = gr.Dropdown(choices=result_files(), show_label=False, interactive=True)
                    with gr.Row():
                        load_btn = gr.Button("Load", size="sm", variant="secondary")
                        refresh_btn = gr.Button("Refresh list", size="sm")

        outputs = [tracker, transcript, side, inventory, final]
        run_event = run_btn.click(run_lab, inputs=[query, rounds, execute], outputs=outputs)
        stop_btn.click(None, cancels=[run_event])
        load_btn.click(replay, inputs=[files], outputs=outputs)
        refresh_btn.click(lambda: gr.update(choices=result_files()), outputs=[files])
    return demo


if __name__ == "__main__":
    import argparse

    from agenticeoecosystem.utils.logging import setup_logging

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--host", default="127.0.0.1",
                        help="Interface to bind (0.0.0.0 exposes the app to the network).")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--share", action="store_true", help="Create a public gradio.live link.")
    args = parser.parse_args()
    setup_logging()
    # Our own file handler on the loggers that matter: Gradio/uvicorn reconfigure the root
    # logger when the server starts, which silently dropped the lab's messages from the log.
    log_path = LOGS_DIR / f"virtual_lab_app_{datetime.now():%Y%m%d_%H%M%S}.log"
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s — %(message)s"))
    for name in ("agenticeoecosystem", "google_adk", "LiteLLM", "__main__"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.INFO)
        lg.addHandler(file_handler)
    logger.info("Logging to %s", log_path)
    build_app().queue().launch(server_name=args.host, server_port=args.port, share=args.share,
                               theme=_THEME, css=_CSS)
