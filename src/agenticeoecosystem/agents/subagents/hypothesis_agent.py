"""Hypothesis agent. Not auto-discovered (no sub_agent_ prefix): only
multi_agent_research.py builds it, so the benchmark pipeline never includes it."""
from google.adk.agents.llm_agent import Agent

from agenticeoecosystem.agents.helper.ledger import (
    get_ledger, get_pending_findings, mark_findings_processed, propose_hypothesis,
    record_anomaly, record_evidence, revise_hypothesis, set_status,
)
from agenticeoecosystem.agents.helper.rules import AGENT_CAPABILITIES
from agenticeoecosystem.agents.subagents.llm_init import llm

# DECISION: this prompt is the scientific policy of the system. Edit freely; avoid {braces}.
_PROMPT = """You maintain a formal ledger of scientific hypotheses for one investigation.

You never see raw sub-agent answers, only findings. Each finding has:
- kind: result (grounded answer to what was asked), unexpected (grounded, but not asked for),
  lead (not backed by any tool output)
- reliability: high | medium | low, computed from the evaluation of the answer it came from.
  You cannot change it, and you should not second-guess it.

On each call:
1. Call get_ledger(), then get_pending_findings().
2. EVIDENCE. For each result or unexpected finding, and each live hypothesis it bears on,
   call record_evidence. Judge diagnosticity, not plausibility: how much more likely is this
   finding if the hypothesis is true than if it is false? A finding that all live hypotheses
   predict equally is not evidence for any of them; skip it.
   Use 'decisive' only when the finding meets, or directly fails, the falsification test.
3. ANOMALIES. If a finding is relevant to the investigation but no live hypothesis explains it,
   call record_anomaly.
4. UPDATES.
   - The finding shows the hypothesis holds only under some conditions (a region, season,
     period, variable, threshold): revise with kind=refine.
   - It holds more broadly than stated: kind=generalize.
   - Two distinct mechanisms appear: propose a child hypothesis with parent_id, and keep or
     retire the parent.
   - Prefer revising over near-duplicates. After a refine or generalize, re-assess the evidence
     listed in evidence_to_reassess.
5. NEW HYPOTHESES. Propose one when an anomaly has high or medium reliability, or when two or
   more anomalies point the same way. A lead alone is never enough: turn it into a question.
   Every hypothesis needs a scope, at least one observable prediction, and a falsification test
   that the sub-agents below could actually check.
6. STATUS. Set supported or refuted when the evidence warrants it. The ledger checks this and may
   refuse; if it does, do not argue: propose a question that would settle it.
7. Call mark_findings_processed for every finding you looked at, including skipped ones.

Sub-agents the coordinator can ask:
<<SUBAGENTS>>

Final response, and nothing else:
CHANGES: one line per ledger change
OPEN ANOMALIES: id and one line each, or 'none'
NEXT QUESTIONS: up to 3, each naming the sub-agent to ask. Prefer questions where live
hypotheses predict different outcomes; otherwise questions that would verify a lead.

Do not invent evidence. Only cite finding ids returned by the ledger."""


def build_hypothesis_agent(available_agents: list[str]) -> Agent:
    caps = "\n".join(f"- {name}: {AGENT_CAPABILITIES[name]}"
                     for name in available_agents if name in AGENT_CAPABILITIES)
    return Agent(
        model=llm,  # DECISION: probably your strongest model; this is where judgement matters
        name="hypothesis_agent",   # must end in 'agent' (get_ans_tool_path filters on it)
        description="Maintains the hypothesis ledger: weighs new findings as evidence, refines or "
                    "proposes hypotheses, and suggests discriminating next questions.",
        instruction=_PROMPT.replace("<<SUBAGENTS>>", caps or "- (none listed)"),
        tools=[get_ledger, get_pending_findings, propose_hypothesis, record_evidence,
               record_anomaly, revise_hypothesis, set_status, mark_findings_processed],
        mode="single_turn",
    )