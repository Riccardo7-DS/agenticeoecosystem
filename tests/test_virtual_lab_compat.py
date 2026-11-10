"""Backward compatibility: the Virtual Lab is optional and leaves `multi` untouched.

Run:
    uv run --with pytest pytest tests/test_virtual_lab_compat.py
"""

import asyncio
import json
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

from _virtual_lab_fakes import ScriptedLlm, fake_existing_agents
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.agent_tool import AgentTool

from agenticeoecosystem.agents.virtual_lab.config import VirtualLabConfig
from agenticeoecosystem.agents.virtual_lab.participants import build_participants
from agenticeoecosystem.evaluation.runners.pipeline_runner import run_pipeline


def test_importing_the_lab_has_no_side_effects():
    """No sub-agent loading, no orchestrator, no Runner at import time."""
    code = ("import sys, agenticeoecosystem.agents.virtual_lab as v; "
            "import agenticeoecosystem.agents.virtual_lab.runner; "
            "print(any(m.startswith('agenticeoecosystem.agents.subagents') or "
            "m == 'agenticeoecosystem.agents.multi_agent' for m in sys.modules))")
    out = subprocess.run([sys.executable, "-W", "ignore", "-c", code], capture_output=True, text=True,
                         env={**os.environ, "AGENTICEO_SUBAGENTS": "none"}, check=True)
    assert out.stdout.strip().splitlines()[-1] == "False"


def _fake_pipeline_module(answer: str) -> SimpleNamespace:
    async def get_ans_tool_path(query: str) -> str:
        return json.dumps({"final_answer": answer, "tools_called": [
            {"function_name": "search_eumetsat_products", "arguments": {}, "tool_result": {"count": 1}}],
            "agents_called": ["x"], "input_tokens": 3, "output_tokens": 4})
    return SimpleNamespace(get_ans_tool_path=get_ans_tool_path)


def test_pipeline_runner_routes_multi_and_virtual_lab_separately():
    import agenticeoecosystem.agents.virtual_lab as lab_pkg

    fake_multi, fake_lab = _fake_pipeline_module("from multi"), _fake_pipeline_module("from lab")
    modules = {"agenticeoecosystem.agents.multi_agent": fake_multi,
               "agenticeoecosystem.agents.virtual_lab.runner": fake_lab}
    asyncio.set_event_loop(asyncio.new_event_loop())
    # `import a.b.c as x` prefers the attribute on an already-imported parent package.
    with patch.dict(sys.modules, modules), patch.object(lab_pkg, "runner", fake_lab, create=True):
        multi, *_ = run_pipeline("q", pipeline="multi")
        lab, in_tok, out_tok = run_pipeline("q", pipeline="virtual_lab")
    assert multi.final_answer == "from multi"
    assert lab.final_answer == "from lab"
    assert lab.steps[0].function_name == "search_eumetsat_products" and (in_tok, out_tok) == (3, 4)


def test_existing_agents_are_wrapped_not_reparented():
    """multi_agent's orchestrator owns the sub-agents; the lab must not steal them."""
    llm = ScriptedLlm(seen=[])
    existing = fake_existing_agents(llm)
    orchestrator = LlmAgent(name="orchestrator_agent", model=llm, sub_agents=list(existing.values()))
    parts = build_participants(VirtualLabConfig(literature_enabled=False), llm, existing)
    assert all(a.parent_agent is orchestrator for a in existing.values())
    assert all(a.mode == "single_turn" for a in existing.values())  # originals unchanged for multi
    [toolset] = parts.specialists["thermal_specialist"].tools   # filtered by the PI's choice
    tools = toolset.all_tools
    assert all(isinstance(t, AgentTool) for t in tools)
    assert all(t.agent.mode == "chat" and t.agent is not existing[t.name] for t in tools)
    assert {t.name for t in tools} == {"eumetsat_agent", "gee_agent"}  # tle_agent absent here


def test_unknown_participants_are_ignored_and_config_reads_env(monkeypatch):
    monkeypatch.setenv("VIRTUAL_LAB_PARTICIPANTS", "sar_specialist,not_a_scientist")
    monkeypatch.setenv("VIRTUAL_LAB_MAX_ROUNDS", "4")
    monkeypatch.setenv("VIRTUAL_LAB_CRITIC", "false")
    cfg = VirtualLabConfig.from_env()
    assert (cfg.max_rounds, cfg.critic_enabled) == (4, False)
    llm = ScriptedLlm(seen=[])
    parts = build_participants(cfg, llm, fake_existing_agents(llm))
    assert list(parts.specialists) == ["sar_specialist"] and parts.critic is None
