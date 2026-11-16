"""
coordinator.py — The Scientific Coordinator: a bounded, phase-structured lab meeting.

  A  formulate   PI -> ScientificProblem, selects specialists
  B  propose     specialists, each blind to the others -> ScientificProposal
  C  deliberate  up to max_rounds of: peer review + Scientific Critic + validator
                 -> ScientificCritique; contested claims -> literature check (arXiv search,
                 ranked by citations, judged by the literature reviewer); then specialists
                 revise; stops early on convergence
  D  decide      PI -> ScientificDecision (status capped by validation.finalize_status);
                 open disagreements -> literature check -> PI decides again;
                 Experiment Designer -> ExperimentPlan, reviewed by the validator and the
                 critic, one redesign if either finds a blocking or major issue

Implemented as a custom BaseAgent rather than Sequential/Parallel/LoopAgent
(deprecated in ADK 2.x in favour of Workflow) so that phase order, per-call
context isolation, retries and budgets are explicit code. Each participant
call runs on its own event branch with include_contents='none'; what it may
see is exactly what its InstructionProvider reads from session state.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any
from collections.abc import AsyncGenerator, Callable

import json5
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.genai import types
from pydantic import BaseModel, PrivateAttr

from . import prompts
from .analyses import ANALYSES, RUNNERS, AnalysisContext, with_dependencies
from .config import VirtualLabConfig
from . import inventory as inv
from .ledger import LabLedger, catalog_summary, current_step, data_records
from .literature import (broaden_query, is_duplicate_claim, is_relevant, limit_and,
                         normalise_arxiv_id, query_terms)
from .participants import CRITIC, LabParticipants
from .schemas import (
    CritiqueBatch,
    DecisionStatus,
    EvidenceKind,
    ExperimentPlan,
    ExperimentResult,
    LabRecord,
    LiteratureAssessmentBatch,
    LiteratureCheck,
    LiteratureQueryBatch,
    PaperRef,
    ProvenanceLink,
    ScientificCritique,
    ScientificDecision,
    ScientificProblem,
    ScientificProposal,
)
from .validation import (
    TerminationReason,
    bboxes_overlap,
    off_area_calls,
    finalize_conclusion,
    finalize_status,
    link_results,
    normalise_conclusion,
    link_evidence,
    grounding_metrics,
    link_items,
    literature_uptake,
    open_issues,
    tool_call_evidence,
    validate_plan,
    validate_proposal,
)

logger = logging.getLogger(__name__)

RECORD_ARTIFACT = "virtual_lab_record.json"
_PREVIEW_CHARS = 1500
_ABSTRACT_CHARS = 1200
# Critique categories that dispute a claim (rather than, say, a missing control), and are
# therefore worth checking against the literature.
_CONTESTABLE = ("unsupported_claim", "physical_inconsistency", "dataset_suitability")
PLAN_TARGET = "experiment_plan"


def parse_json_object(text: str | None) -> dict:
    """Extract the first top-level JSON object from model text (tolerates fences/prose)."""
    if not text or not text.strip():
        raise ValueError("empty output")
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in output")
    blob = text[start:end + 1]
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        data = json5.loads(blob)
    if not isinstance(data, dict):
        raise ValueError("output is not a JSON object")
    return data


async def _merge(gens: list[AsyncGenerator[Event, None]]) -> AsyncGenerator[Event, None]:
    """Interleave participant runs; each waits until the runner has appended its last event
    (same contract as ADK's ParallelAgent merge)."""
    sentinel = object()
    queue: asyncio.Queue = asyncio.Queue()

    async def drain(gen: AsyncGenerator[Event, None]) -> None:
        try:
            async for event in gen:
                resume = asyncio.Event()
                await queue.put((event, resume))
                await resume.wait()
        finally:
            await queue.put((sentinel, None))

    async with asyncio.TaskGroup() as tg:
        for gen in gens:
            tg.create_task(drain(gen))
        finished = 0
        while finished < len(gens):
            event, resume = await queue.get()
            if event is sentinel:
                finished += 1
                continue
            yield event
            resume.set()


def _drop_checklist_echoes(issues: list) -> list:
    """Drop plan-review issues whose 'claim' restates a checklist item instead of quoting the
    plan: they point at nothing the designer can fix."""
    checklist = [" ".join(line.lower().split())[:40] for line in prompts.plan_review_checklist_lines()]
    kept = [i for i in issues if not any(" ".join(i.claim.lower().split()).startswith(c) for c in checklist)]
    if len(kept) < len(issues):
        logger.info("Dropped %d plan-review issues that echoed the checklist", len(issues) - len(kept))
    return kept


def _year(published: str | None) -> int | None:
    try:
        return int((published or "")[:4])
    except ValueError:
        return None


def _results_dir() -> Path:
    """Quicklooks of one execute-mode run: output/virtual_lab/run_<timestamp>/."""
    from agenticeoecosystem.definitions import ROOT_DIR
    return ROOT_DIR / "output" / "virtual_lab" / f"run_{datetime.now():%Y%m%d_%H%M%S}"


def _dump(models: dict[str, BaseModel] | list[BaseModel]) -> Any:
    if isinstance(models, dict):
        return {k: m.model_dump(mode="json") for k, m in models.items()}
    return [m.model_dump(mode="json") for m in models]


class VirtualLabCoordinator(BaseAgent):
    lab_config: VirtualLabConfig

    _participants: LabParticipants = PrivateAttr()
    _ledger: LabLedger = PrivateAttr()
    _record: LabRecord | None = PrivateAttr(default=None)
    _search: Callable | None = PrivateAttr(default=None)   # async (ctx, query, n) -> [paper dict]
    _rank: Callable | None = PrivateAttr(default=None)     # async (arxiv_ids) -> {"ranked": [...]}
    _catalogs: Any = PrivateAttr(default=None)            # inventory.Catalogs (metadata clients)
    _checked_claims: list[str] = PrivateAttr(default_factory=list)
    _query_cache: dict[str, tuple[str, list[dict]]] = PrivateAttr(default_factory=dict)
    _lit_check_start: int = PrivateAttr(default=0)
    _arxiv_unresponsive: bool = PrivateAttr(default=False)    # a search timed out in this check
    _analysis_client: Any = PrivateAttr(default=None)         # execute mode: Sentinel Hub client
    _embedder: Any = PrivateAttr(default=None)                # data discovery ranking (discovery.py)

    def __init__(self, *, lab_config: VirtualLabConfig, participants: LabParticipants,
                 ledger: LabLedger, name: str = "virtual_lab",
                 literature_search: Callable | None = None,
                 citation_ranker: Callable | None = None, catalogs: Any = None,
                 analysis_client: Any = None, embedder: Any = None) -> None:
        super().__init__(
            name=name,
            description="Scientific Coordinator running a structured EO virtual-lab meeting.",
            sub_agents=participants.all_agents,
            lab_config=lab_config,
        )
        self._participants = participants
        self._ledger = ledger
        self._search = literature_search
        self._rank = citation_ranker
        self._catalogs = catalogs
        self._analysis_client = analysis_client
        self._embedder = embedder

    @property
    def record(self) -> LabRecord | None:
        return self._record

    # ------------------------------------------------------------------ plumbing

    def _state_event(self, ctx: InvocationContext, delta: dict[str, Any]) -> Event:
        return Event(invocation_id=ctx.invocation_id, author=self.name, branch=ctx.branch,
                     actions=EventActions(state_delta=delta))

    async def _call(
        self,
        ctx: InvocationContext,
        agent: LlmAgent,
        step: str,
        model_cls: type[BaseModel],
        out: dict[str, BaseModel],
        state: dict[str, Any] | None = None,
        prepare: Callable[[dict], dict] | None = None,
        initial_repair: str | None = None,
    ) -> AsyncGenerator[Event, None]:
        """Run one participant, parse + validate its JSON, retry with the error on failure.
        The validated artifact lands in out[step]; failures are recorded, never raised."""
        current_step.set(step)
        repair = initial_repair
        for attempt in range(self.lab_config.max_repair_attempts + 1):
            registry = self._registry()
            delta = {**(state or {}), prompts.raw_key(agent.name): None,
                     prompts.repair_key(agent.name): repair,
                     prompts.STATE_REGISTRY: inv.format_registry(registry) if registry else ""}
            yield self._state_event(ctx, delta)
            # Own branch per call: the participant can't see any other participant's events.
            sub_ctx = ctx.model_copy(update={"branch": f"{self.name}.{step}.a{attempt}"})
            try:
                async for event in agent.run_async(sub_ctx):
                    yield event
            except Exception as exc:  # a failed participant must not sink the meeting
                logger.warning("Lab step %s (%s) raised: %s", step, agent.name, exc, exc_info=True)
                self._record.invalid_outputs.append(
                    {"step": step, "agent": agent.name, "attempt": attempt, "error": repr(exc)[:_PREVIEW_CHARS]})
                break
            raw = ctx.session.state.get(prompts.raw_key(agent.name))
            try:
                data = parse_json_object(raw)
                out[step] = model_cls.model_validate(prepare(data) if prepare else data)
                break
            except ValueError as exc:  # includes pydantic.ValidationError and json5 errors
                self._record.invalid_outputs.append(
                    {"step": step, "agent": agent.name, "attempt": attempt,
                     "error": str(exc)[:_PREVIEW_CHARS], "raw": str(raw)[:_PREVIEW_CHARS]})
                repair = f"{str(exc)[:_PREVIEW_CHARS]}\nPrevious output (truncated): {str(raw)[:_PREVIEW_CHARS]}"
        yield self._state_event(ctx, {prompts.repair_key(agent.name): None})

    async def _fan_out(self, ctx, calls: list[dict[str, Any]]) -> AsyncGenerator[Event, None]:
        gens = [self._call(ctx, **c) for c in calls]
        if self.lab_config.parallel and len(gens) > 1:
            async for event in _merge(gens):
                yield event
        else:
            for gen in gens:
                async for event in gen:
                    yield event

    def _budget_allows(self, upcoming_llm_calls: int, reserve: int = 2) -> bool:
        """reserve: calls kept back for what must still follow (by default the decision and plan).
        Both counts are worst cases: every call may need max_repair_attempts retries, so a phase
        full of invalid outputs can never eat the calls the decision needs.
        A spent tool budget does not stop LLM-only steps: the guard already blocks further tool
        calls, and revisions, literature assessment and plan review still need to run."""
        if self._ledger.budget_exhausted == "llm_calls":
            return False
        per_call = 1 + self.lab_config.max_repair_attempts
        needed = (upcoming_llm_calls + reserve) * per_call
        return self._ledger.lab_llm_calls() + needed <= self.lab_config.max_llm_calls

    # ------------------------------------------------------------------ artifacts

    def _ingest_proposal(self, name: str, proposal: ScientificProposal, step: str,
                         round_: int) -> tuple[ScientificProposal, ScientificCritique]:
        """Link evidence to tool calls, namespace hypothesis ids, validate, log."""
        prefix = name.split("_")[0]
        hyps = [h if h.id.startswith(f"{prefix}-") else h.model_copy(update={"id": f"{prefix}-{h.id}"})
                for h in proposal.hypotheses]
        proposal = proposal.model_copy(update={"hypotheses": hyps})
        # A proposal may cite its calls in this step, any paper a literature check retrieved, and
        # (revisions restate earlier findings) its calls in earlier steps to tools it did not
        # call again in this one.
        current = self._ledger.calls_for(step)
        called_now = {c["tool"] for c in current} | {c["via"] for c in current if c.get("via")}
        earlier = [c for c in self._ledger.tool_calls
                   if c["status"] == "ok" and c["step"] != step and (c["step"] or "").endswith(f".{name}")
                   and c["tool"] not in called_now and c.get("via") not in called_now]
        shared = [c for c in self._ledger.tool_calls
                  if c["status"] == "ok" and (c.get("paper_ids") or c.get("inventory"))]
        calls = list({c["id"]: c for c in current + earlier + shared}.values())
        proposal, link_issues = link_evidence(proposal, calls)
        proposal = proposal.model_copy(update={"evidence": proposal.evidence + tool_call_evidence(
            name, self._ledger.calls_for(step), proposal.evidence)})
        proposal = self._carry_linked_evidence(name, proposal)
        grounded = bool(self._participants.grounding.get(name))
        link_issues += literature_uptake(proposal, self._record.literature)
        link_issues += off_area_calls(self._ledger.calls_for(step), self._record.problem)
        check = validate_proposal(proposal, self._record.problem, grounded, round_, link_issues,
                                  registry=self._registry())
        self._record.proposal_history.append(proposal)
        self._ledger.log("proposal" if round_ == 0 else "revision", step, agent=name,
                         proposal=proposal.model_dump(mode="json"))
        self._ledger.log("validation", step, critique=check.model_dump(mode="json"))
        return proposal, check

    def _carry_linked_evidence(self, name: str, proposal: ScientificProposal) -> ScientificProposal:
        """Keep the previous version's tool-linked evidence that a revision left out: models
        rewrite proposals from scratch and drop retrieved claims they did not mean to retract.
        A claim the revision restates (similar wording) is not duplicated."""
        previous = next((p for p in reversed(self._record.proposal_history) if p.agent == name), None)
        if previous is None:
            return proposal
        kept = [e.claim for e in proposal.evidence]
        carried = []
        for e in previous.evidence:
            if e.kind == EvidenceKind.RETRIEVED and e.tool_call_ids and not is_duplicate_claim(e.claim, kept):
                carried.append(e)
                kept.append(e.claim)
        return proposal.model_copy(update={"evidence": proposal.evidence + carried}) if carried else proposal

    @staticmethod
    def _prepare_proposal(name: str, revision: int) -> Callable[[dict], dict]:
        return lambda d: {**d, "agent": name, "revision": revision}

    @staticmethod
    def _prepare_critiques(author: str, round_: int) -> Callable[[dict], dict]:
        def prepare(d: dict) -> dict:
            items = d.get("critiques", [])
            return {"critiques": [{**c, "author": author, "round": round_} for c in items
                                  if isinstance(c, dict)]}
        return prepare

    # ------------------------------------------------------------------ meeting

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        cfg, parts = self.lab_config, self._participants
        question = "".join(p.text or "" for p in (ctx.user_content.parts if ctx.user_content else []))
        self._record = LabRecord(question=question, config=cfg.model_dump(mode="json"))
        self._checked_claims, self._query_cache = [], {}
        record, out = self._record, {}

        # Phase A — problem formulation
        async for event in self._call(ctx, parts.pi, "r0.formulate", ScientificProblem, out,
                                      state={prompts.STATE_PHASE: "formulate"}):
            yield event
        problem = out.get("r0.formulate")
        if problem is None:
            async for event in self._finish(ctx, TerminationReason.FORMULATION_FAILED, {}, [], []):
                yield event
            return
        selected = [p for p in problem.participants if p in parts.specialists] or list(parts.specialists)
        problem = problem.model_copy(update={"participants": selected})
        offered = list(parts.sub_agents)
        chosen = [n for n in problem.sub_agents if n in offered] or offered
        problem = problem.model_copy(update={"sub_agents": chosen})
        problem = await self._locate(problem)
        record.problem = problem
        self._ledger.log("problem", "r0.formulate", problem=problem.model_dump(mode="json"))
        # Specialists and the critic only see the sub-agents the PI chose (participants.py).
        yield self._state_event(ctx, {prompts.STATE_SUB_AGENTS: chosen})

        # Inventory — what data exist for this area and period, retrieved once for everyone
        if cfg.inventory_enabled and self._catalogs:
            text = await self._build_inventory(problem)
            self._ledger.log("inventory", "r0.inventory", inventory=text)
            yield self._state_event(ctx, {prompts.STATE_INVENTORY: text})

        # Phase B — independent proposals (each sees only the problem)
        calls = [dict(agent=parts.specialists[n], step=f"r0.propose.{n}", model_cls=ScientificProposal,
                      out=out, prepare=self._prepare_proposal(n, 0),
                      state={prompts.STATE_PHASE: "propose",
                             prompts.STATE_PROBLEM: problem.model_dump(mode="json")})
                 for n in selected]
        async for event in self._fan_out(ctx, calls):
            yield event
        proposals: dict[str, ScientificProposal] = {}
        checks: dict[str, ScientificCritique] = {}
        for n in selected:
            if (p := out.get(f"r0.propose.{n}")) is not None:
                proposals[n], checks[n] = self._ingest_proposal(n, p, f"r0.propose.{n}", 0)
        if not proposals:
            async for event in self._finish(ctx, TerminationReason.NO_PROPOSALS, {}, [], []):
                yield event
            return

        # Phase C — deliberation
        termination = TerminationReason.MAX_ROUNDS
        last_critiques: list[ScientificCritique] = []
        for r in range(1, cfg.max_rounds + 1):
            reviewers = [n for n in proposals if cfg.peer_review and len(proposals) > 1]
            round_cost = len(reviewers) + (1 if parts.critic else 0) + len(proposals)
            if not self._budget_allows(round_cost):
                termination = TerminationReason.BUDGET
                break
            record.rounds_run = r
            review_state = {prompts.STATE_PHASE: "critique", prompts.STATE_ROUND: r,
                            prompts.STATE_PROPOSALS: _dump(proposals),
                            prompts.STATE_VALIDATION: _dump(list(checks.values()))}
            calls = [dict(agent=parts.specialists[n], step=f"r{r}.critique.{n}", model_cls=CritiqueBatch,
                          out=out, state=review_state, prepare=self._prepare_critiques(n, r))
                     for n in reviewers]
            if parts.critic:
                calls.append(dict(agent=parts.critic, step=f"r{r}.critique.{CRITIC}", model_cls=CritiqueBatch,
                                  out=out, state=review_state, prepare=self._prepare_critiques(CRITIC, r)))
            async for event in self._fan_out(ctx, calls):
                yield event

            critiques = [c for call in calls if (batch := out.get(call["step"])) is not None
                         for c in batch.critiques if c.target in proposals and c.target != c.author]
            critiques += list(checks.values())
            for c in critiques:
                self._ledger.log("critique", f"r{r}.critique", critique=c.model_dump(mode="json"))
            record.critiques.extend(critiques)
            last_critiques = critiques
            if not open_issues(critiques):
                termination = TerminationReason.CONVERGED
                break

            async for event in self._literature_check(
                    ctx, f"r{r}.literature", self._contested_claims(critiques, r), r,
                    reserve=2 + len(proposals)):
                yield event

            calls = [dict(agent=parts.specialists[n], step=f"r{r}.revise.{n}", model_cls=ScientificProposal,
                          out=out, prepare=self._prepare_proposal(n, r),
                          state={prompts.STATE_PHASE: "revise",
                                 prompts.STATE_PROPOSALS: _dump(proposals),
                                 prompts.STATE_CRITIQUES: _dump(critiques)})
                     for n in proposals]
            async for event in self._fan_out(ctx, calls):
                yield event
            for n in list(proposals):
                if (p := out.get(f"r{r}.revise.{n}")) is not None:
                    proposals[n], checks[n] = self._ingest_proposal(n, p, f"r{r}.revise.{n}", r)
            if r < cfg.max_rounds and cfg.next_round_only_if_open and not self._needs_another_round(
                    critiques, list(checks.values())):
                termination = TerminationReason.SETTLED
                break

        async for event in self._finish(ctx, termination, proposals, list(checks.values()), last_critiques):
            yield event

    @staticmethod
    def _needs_another_round(critiques: list[ScientificCritique],
                             revised_checks: list[ScientificCritique]) -> bool:
        """Another round is worth its cost only if this round raised a blocking issue (the
        revision must be re-reviewed) or the validator still finds blocking/major issues in
        the revised proposals."""
        blocking = any(i.severity == "blocking" for c in critiques for i in c.issues)
        return blocking or bool(open_issues(revised_checks))

    async def _finish(self, ctx, termination: TerminationReason,
                      proposals: dict[str, ScientificProposal],
                      checks: list[ScientificCritique],
                      last_critiques: list[ScientificCritique]) -> AsyncGenerator[Event, None]:
        """Phase D — decision and experiment design, then status capping and persistence."""
        parts, record, out = self._participants, self._record, {}
        record.termination = termination.value
        record.final_proposals = proposals
        decision: ScientificDecision | None = None
        plan: ExperimentPlan | None = None

        if proposals and self._ledger.budget_exhausted != "llm_calls":
            async for event in self._call(
                    ctx, parts.pi, "final.decide", ScientificDecision, out,
                    state={prompts.STATE_PHASE: "decide", prompts.STATE_PROPOSALS: _dump(proposals),
                           prompts.STATE_CRITIQUES: _dump(last_critiques),
                           prompts.STATE_VALIDATION: _dump(checks)}):
                yield event
            decision = out.get("final.decide")

        if decision is not None and decision.disagreements:
            n_checks = len(record.literature)
            async for event in self._literature_check(
                    ctx, "final.literature", self._disagreement_claims(decision, record.rounds_run),
                    record.rounds_run, reserve=3):
                yield event
            if len(record.literature) > n_checks and self._budget_allows(1):
                # The PI reads the new checks from STATE_LITERATURE.
                async for event in self._call(
                        ctx, parts.pi, "final.redecide", ScientificDecision, out,
                        state={prompts.STATE_PHASE: "decide", prompts.STATE_PROPOSALS: _dump(proposals),
                               prompts.STATE_CRITIQUES: _dump(last_critiques),
                               prompts.STATE_VALIDATION: _dump(checks)}):
                    yield event
                decision = out.get("final.redecide") or decision

        if decision is not None:
            all_calls = [c for c in self._ledger.tool_calls if c["status"] == "ok"]
            justification, _ = link_items(decision.justification, all_calls)
            decision = decision.model_copy(update={"justification": justification})
            state = {prompts.STATE_PHASE: "design", prompts.STATE_DECISION: decision.model_dump(mode="json")}
            async for event in self._call(ctx, parts.designer, "final.design", ExperimentPlan, out,
                                          state=state):
                yield event
            plan = out.get("final.design")
            available = set(parts.executors)
            if plan is not None:
                catalog = self._ledger.catalog_results()
                plan_check = validate_plan(plan, record.problem, available, catalog,
                                                   registry=self._registry())
                review = None
                async for event in self._review_plan(ctx, plan, plan_check, out):
                    yield event
                if (batch := out.get("final.plan_review")) is not None:
                    review = ScientificCritique(author=CRITIC, target=PLAN_TARGET, round=record.rounds_run,
                                                issues=_drop_checklist_echoes(
                                                    [i for c in batch.critiques for i in c.issues]))
                    record.critiques.append(review)
                    self._ledger.log("critique", "final.plan_review", critique=review.model_dump(mode="json"))
                findings = open_issues([plan_check] + ([review] if review else []))
                if findings and self._budget_allows(1, reserve=0):
                    listed = json.dumps([i.model_dump(mode="json") for i in findings], indent=1)
                    async for event in self._call(
                            ctx, parts.designer, "final.redesign", ExperimentPlan, out, state=state,
                            initial_repair="This is not a format error: the validator and the Scientific "
                                           "Critic found these problems in your plan. Fix each one, or "
                                           f"state in 'assumptions' why it cannot be fixed:\n{listed}\n"
                                           "A step's 'agent' must be one of: "
                                           + ", ".join(sorted(available))):
                        yield event
                    if (redesigned := out.get("final.redesign")) is not None:
                        plan = redesigned
                        plan_check = validate_plan(plan, record.problem, available, catalog,
                                                   registry=self._registry())
                record.plan_validation = plan_check
                self._ledger.log("plan", "final.design", plan=plan.model_dump(mode="json"),
                                 validation=plan_check.model_dump(mode="json"))

        if decision is None:
            if self._ledger.budget_exhausted or termination == TerminationReason.BUDGET:
                status, summary = DecisionStatus.BUDGET_EXHAUSTED, "Stopped by a resource budget before a decision."
            else:
                status, summary = DecisionStatus.INSUFFICIENT_EVIDENCE, (
                    f"No valid decision could be produced (termination: {termination.value}).")
            decision = ScientificDecision(status=status, summary=summary)
        else:
            decision = self._cap_and_link(decision, proposals, checks, plan, record)

        decision_step = "final.decide"
        if self.lab_config.mode == "execute" and record.problem is not None:
            self._ledger.log("decision", "final.decide", decision=decision.model_dump(mode="json"),
                             termination=termination.value)
            async for event in self._execute(ctx, plan):
                yield event
            if record.results and self._ledger.budget_exhausted != "llm_calls":
                async for event in self._call(
                        ctx, parts.pi, "final.conclude", ScientificDecision, out,
                        state={prompts.STATE_PHASE: "conclude",
                               prompts.STATE_DECISION: decision.model_dump(mode="json"),
                               prompts.STATE_RESULTS: [r.model_dump(mode="json") for r in record.results]},
                        prepare=normalise_conclusion):
                    yield event
                if (conclusion := out.get("final.conclude")) is not None:
                    decision = self._link_conclusion(conclusion, decision)
                    decision_step = "final.conclude"

        record.decision, record.plan = decision, plan
        if (registry := self._registry()) is not None:
            chosen = plan.datasets if plan else [d for p in proposals.values() for d in p.datasets]
            record.data_grounding = grounding_metrics(chosen, registry, self._shortlisted_ids())
        self._ledger.log("decision", decision_step, decision=decision.model_dump(mode="json"),
                         termination=termination.value)
        current_step.set(None)
        artifact_name = await self._save_record(ctx)
        yield Event(
            invocation_id=ctx.invocation_id, author=self.name, branch=ctx.branch,
            content=types.Content(role="model", parts=[types.Part(text=render_report(record, self._ledger))]),
            actions=EventActions(state_delta={"vl_decision_status": decision.status.value,
                                              "vl_record_artifact": artifact_name}),
        )

    # ------------------------------------------------------------------ execute mode

    async def _execute(self, ctx, plan: ExperimentPlan | None) -> AsyncGenerator[Event, None]:
        """Run the plan's fixed analyses (all of them if it names none) and publish each result
        as it lands. Dates come from the data, not from the plan."""
        record, problem = self._record, self._record.problem
        bbox = list(problem.study_area.bbox) if problem.study_area and problem.study_area.bbox else None
        period = problem.time_period
        if self._analysis_client is None or not bbox or not (
                period and inv.valid_period(period.start, period.end)):
            reason = ("no Sentinel Hub client (set CDSE_CLIENT_ID / CDSE_CLIENT_SECRET)"
                      if self._analysis_client is None else "the problem has no valid bbox and period")
            record.results.append(ExperimentResult(id="res-1", analysis="(none)", status="failed",
                                                   summary=f"Nothing executed: {reason}."))
            self._ledger.log("result", "final.execute", result=record.results[-1].model_dump(mode="json"))
            yield self._state_event(ctx, {prompts.STATE_RESULTS: _dump(record.results)})
            return
        chosen = [a for a in (plan.analyses if plan else []) if a in ANALYSES] or list(ANALYSES)
        actx = AnalysisContext(bbox=bbox, start=period.start[:10], end=period.end[:10],
                               pad_days=self.lab_config.inventory_pad_days,
                               out_dir=_results_dir())
        current_step.set("final.execute")
        args = {"bbox": bbox, "start": actx.start, "end": actx.end}
        for name in with_dependencies(chosen):
            entry, out = await self._direct_tool_call(
                "final.execute", f"analysis:{name}", args,
                lambda name=name: RUNNERS[name](actx, self._analysis_client), budget=None)
            entry["analysis"] = True
            if out is None:
                out = {"status": "failed", "summary": f"{name} raised: {entry.get('result_preview')}",
                       "metrics": {}, "details": {}, "artifacts": []}
            if out["status"] != "ok":
                entry["status"] = "error"
            result = ExperimentResult(id=f"res-{len(record.results) + 1}", analysis=name,
                                      status=out["status"], summary=out["summary"],
                                      metrics=out.get("metrics", {}),
                                      details=json.loads(json.dumps(out.get("details", {}), default=str)),
                                      artifacts=out.get("artifacts", []), tool_call_ids=[entry["id"]])
            record.results.append(result)
            self._ledger.log("result", "final.execute", result=result.model_dump(mode="json"))
            yield self._state_event(ctx, {prompts.STATE_RESULTS: _dump(record.results)})

    def _link_conclusion(self, conclusion: ScientificDecision, before: ScientificDecision) -> ScientificDecision:
        results = self._record.results
        justification = link_results(conclusion.justification, results)
        conclusion = finalize_conclusion(conclusion.model_copy(update={"justification": justification}),
                                         results)
        links = list(before.provenance) + [
            ProvenanceLink(source=f"result:{r.id}", relation="supported_by_result") for r in results
            if any(e.result_id == r.id for e in justification)]
        return conclusion.model_copy(update={"provenance": links})

    async def _review_plan(self, ctx, plan: ExperimentPlan, plan_check: ScientificCritique,
                           out: dict) -> AsyncGenerator[Event, None]:
        """The Scientific Critic checks the plan's internal logic (things code can't judge)."""
        critic = self._participants.critic
        if critic is None or not self._budget_allows(1, reserve=1):
            return
        state = {prompts.STATE_PHASE: "plan_review", prompts.STATE_PLAN: plan.model_dump(mode="json"),
                 prompts.STATE_VALIDATION: [plan_check.model_dump(mode="json")]}
        prepare = self._prepare_critiques(CRITIC, self._record.rounds_run)
        async for event in self._call(ctx, critic, "final.plan_review", CritiqueBatch, out, state=state,
                                      prepare=prepare):
            yield event

    # ------------------------------------------------------------------ literature check

    def _contested_claims(self, critiques: list[ScientificCritique], round_: int) -> list[dict]:
        """Claims one participant disputes in another's proposal (validator findings excluded:
        they come from the sensor catalog, not from an argument)."""
        issues = [(c, i) for c in critiques if c.author != "validator"
                  for i in c.issues if i.category in _CONTESTABLE and i.severity in ("blocking", "major")]
        issues.sort(key=lambda ci: ci[1].severity != "blocking")
        return [{"claim": i.claim, "objection": i.rationale, "raised_by": c.author,
                 "target": c.target, "round": round_} for c, i in issues]

    @staticmethod
    def _disagreement_claims(decision: ScientificDecision, round_: int) -> list[dict]:
        return [{"claim": f"{d.topic}: " + " | ".join(f"{a}: {p}" for a, p in d.positions.items()),
                 "objection": d.evidence_needed, "raised_by": "lab_pi", "target": None, "round": round_}
                for d in decision.disagreements]

    async def _direct_tool_call(self, step: str, tool: str, args: dict, run, via: str | None = None,
                                budget: str | None = "literature") -> tuple[dict | None, Any]:
        """Call a read-only tool from code, under the same ledger as agent tool calls.
        budget: "literature" (the literature checks' own allowance) or None (the data inventory,
        unbudgeted). Returns (ledger entry, result); (None, None) if the allowance is spent."""
        if budget == "literature":
            used, cfg = self._ledger.literature_tool_calls(), self.lab_config
            if used >= cfg.max_literature_tool_calls:
                logger.info("Literature tool allowance (%d) spent: skipping %s",
                            cfg.max_literature_tool_calls, tool)
                return None, None
            if used - self._lit_check_start >= cfg.max_literature_tool_calls_per_check:
                logger.info("Per-check literature limit (%d) reached: skipping %s",
                            cfg.max_literature_tool_calls_per_check, tool)
                return None, None
        elif budget is not None:
            raise ValueError(f"unknown tool budget {budget!r}")
        entry = self._ledger.add_tool_call(step=step, agent=self.name, tool=tool, via=via, args=args,
                                           status="pending", lab_call=True)
        if budget == "literature":
            entry["literature"] = True
        try:
            if budget == "literature":
                result = await asyncio.wait_for(run(), timeout=self.lab_config.literature_timeout_s)
            else:
                result = await run()
        except TimeoutError:
            logger.warning("Literature tool %s timed out after %.0f s", tool, self.lab_config.literature_timeout_s)
            entry.update(status="error", timed_out=True,
                         result_preview=f"timed out after {self.lab_config.literature_timeout_s:.0f} s")
            if tool == "search_papers":
                self._arxiv_unresponsive = True
            return entry, None
        except Exception as exc:  # a failed lookup leaves the claim inconclusive
            logger.warning("Literature tool %s failed: %s", tool, exc)
            entry.update(status="error", result_preview=repr(exc)[:_PREVIEW_CHARS])
            return entry, None
        entry.update(status="ok", result_preview=json.dumps(result, default=str)[:_PREVIEW_CHARS])
        return entry, result

    async def _locate(self, problem: ScientificProblem) -> ScientificProblem:
        """Geocode the study area in code when the PI named it but gave no bbox. The PI is
        offered the geocoder but may skip it, and agents then invent coordinates (run of
        2026-11-13: 'Kypseli, Athens' had no bbox; searches ran ~70 km away and the inventory
        skipped every scene search). A bbox the PI did give is kept: it may come from the
        question, and an ambiguous place name can geocode elsewhere; a mismatch is logged."""
        area = problem.study_area
        cdse = getattr(self._catalogs, "cdse", None) if self._catalogs else None
        if area is None or not area.name.strip() or cdse is None or not hasattr(cdse, "geocode"):
            return problem
        entry, result = await self._direct_tool_call(
            "r0.geocode", "geocode_place", {"place_name": area.name},
            lambda: cdse.geocode(area.name), budget=None)
        entry["inventory"] = True   # lab bookkeeping, free, and citable like the inventory
        hits = (result.get("results") or []) if isinstance(result, dict) else []
        found = hits[0].get("bbox") if hits else None
        if not found or not (found[0] <= found[2] and found[1] < found[3]):
            entry["status"] = "error"
            logger.warning("Could not geocode study area %r: %s", area.name, result)
            return problem
        if area.bbox:
            if not bboxes_overlap(list(area.bbox), found):
                entry["note"] = f"PI bbox {list(area.bbox)} does not overlap the geocoded {found}; kept the PI's"
                logger.warning("Study area %r: %s", area.name, entry["note"])
            return problem
        entry["note"] = f"study-area bbox set from geocoding: {found} ({hits[0].get('display_name', '')})"
        logger.info("Study area %r: %s", area.name, entry["note"])
        return problem.model_copy(update={"study_area": area.model_copy(update={"bbox": found})})

    def _registry(self) -> dict[str, inv.RegistryEntry] | None:
        """This run's dataset registry, rebuilt from the ledger (so it includes every search made
        so far); None when the lab has no catalogue access, which turns grounding checks off."""
        if not self._catalogs or not self.lab_config.inventory_enabled:
            return None
        return inv.data_registry(self._ledger.tool_calls)

    def _shortlisted_ids(self) -> list[str]:
        """Registry ids of the data discovery's shortlist (the inventory's per-collection checks)."""
        return [c["id"] for c in self._ledger.tool_calls
                if c.get("inventory") and c.get("data") and c["tool"] != "search_cdse_catalog"]

    async def _build_inventory(self, problem: ScientificProblem) -> str:
        """Query every provider's public metadata API for the study area and period (see
        inventory.py), each call through the ledger. Returns the DATA INVENTORY text."""
        step = "r0.inventory"

        async def call(tool: str, args: dict, run) -> tuple[str | None, Any]:
            # The inventory is the lab's own bookkeeping: it does not use the agents' budget.
            entry, result = await self._direct_tool_call(step, tool, args, run, budget=None)
            entry["inventory"] = True
            if result is None or (isinstance(result, dict) and result.get("error")):
                entry["status"] = "error"
            else:
                if tool == "search_cdse_catalog":
                    entry["catalog"] = catalog_summary(result)
                if records := data_records(tool, args, result):
                    entry["data"] = records
                if self.lab_config.cache_tool_calls:
                    self._ledger.remember(entry, result)   # specialists' repeats are free
            return entry["id"], result

        return await inv.build_inventory(problem, self._catalogs, call,
                                         self.lab_config.inventory_pad_days, self._embedder)

    async def _search_papers(self, ctx, step: str, query: str) -> tuple[str | None, list[dict]]:
        """search_papers through the ledger, reusing the result of an identical earlier query.
        Returns (ledger call id, papers); (None, []) if the tool budget is spent."""
        key = " ".join(query.lower().split())
        if key in self._query_cache:
            return self._query_cache[key]
        if self._arxiv_unresponsive:   # one timeout per check: don't wait on a dead service again
            logger.info("arXiv unresponsive: skipping search '%s'", query)
            return None, []
        n = self.lab_config.literature_results_per_query
        entry, papers = await self._direct_tool_call(
            step, "search_papers", {"query": query, "max_results": n},
            lambda: self._search(ctx, query, n), via="arxiv_rag_agent")
        if entry is None:
            return None, []
        papers = papers or []
        entry["paper_ids"] = [p["id"] for p in papers]
        if entry["status"] == "ok":
            self._query_cache[key] = (entry["id"], papers)
        return entry["id"], papers

    async def _literature_check(self, ctx, step: str, claims: list[dict], round_: int,
                                reserve: int) -> AsyncGenerator[Event, None]:
        """Search arXiv for each contested claim, keep the most-cited hits, have the
        literature reviewer judge them, and publish the results to session state."""
        cfg, reviewer = self.lab_config, self._participants.literature_reviewer
        fresh: list[dict] = []
        for c in claims:  # skip rewordings of claims already checked in this run or this batch
            if not is_duplicate_claim(c["claim"], self._checked_claims + [f["claim"] for f in fresh]):
                fresh.append(c)
        claims = fresh[:cfg.literature_max_claims]
        if not (cfg.literature_check and reviewer and self._search and claims):
            return
        if not self._budget_allows(2, reserve=reserve):
            return
        self._lit_check_start = self._ledger.literature_tool_calls()
        self._arxiv_unresponsive = False
        for n, c in enumerate(claims, 1):
            c["claim_id"] = f"{step.split('.')[0]}-c{n}"
            self._checked_claims.append(c["claim"])
        out: dict[str, BaseModel] = {}

        async for event in self._call(ctx, reviewer, f"{step}.query", LiteratureQueryBatch, out,
                                      state={prompts.STATE_PHASE: "lit_query",
                                             prompts.STATE_CONTESTED: claims}):
            yield event
        plans = {q.claim_id: q for q in out[f"{step}.query"].claims} if f"{step}.query" in out else {}

        checks: dict[str, LiteratureCheck] = {}
        papers_for: dict[str, list[dict]] = {}
        for c in claims:
            q = plans.get(c["claim_id"])
            queries = list(dict.fromkeys(limit_and(x) for x in (q.queries if q else [])))
            check = LiteratureCheck(**c, scope=q.scope if q else "general",
                                    queries=queries[:cfg.literature_queries_per_claim])
            checks[c["claim_id"]] = check
            if check.scope == "event_specific":
                check.summary = "Event-specific claim: papers cannot settle it; only data from this event can."
                continue
            found: dict[str, dict] = {}
            for query in check.queries:
                call_id, papers = await self._search_papers(ctx, step, query)
                if call_id is None:
                    break
                if call_id not in check.tool_call_ids:  # a cached query reuses its call id
                    check.tool_call_ids.append(call_id)
                if not papers and (broad := broaden_query(query)):
                    call_id, papers = await self._search_papers(ctx, step, broad)
                    if call_id is None:
                        break
                    if call_id not in check.tool_call_ids:
                        check.tool_call_ids.append(call_id)
                    check.queries.append(broad)
                for p in papers:
                    found.setdefault(p["id"], p)
            if not found:
                check.verdict, check.summary = "inconclusive", (
                    "arXiv did not respond in time; the claim could not be checked against papers."
                    if self._arxiv_unresponsive else "No papers found for the queries.")
                continue
            if cfg.literature_min_terms:  # before ranking, so citations can't promote off-topic hits
                terms = query_terms(check.queries)
                found = {i: p for i, p in found.items() if is_relevant(p, terms, cfg.literature_min_terms)}
                if not found:
                    check.verdict, check.summary = "inconclusive", (
                        "No retrieved paper mentions the claim's key terms.")
                    continue
            ranked = [{"arxiv_id": i} for i in found]  # search order if ranking fails
            if self._rank is not None:
                entry, result = await self._direct_tool_call(
                    step, "rank_papers_by_citations", {"arxiv_ids": list(found)},
                    lambda ids=list(found): self._rank(ids))
                if entry is not None:
                    check.tool_call_ids.append(entry["id"])
                    if result and result.get("ranked"):
                        ranked = result["ranked"]
            top = [r for r in ranked if r["arxiv_id"] in found][:cfg.literature_papers_per_claim]
            check.papers = [PaperRef(arxiv_id=r["arxiv_id"], title=found[r["arxiv_id"]].get("title", ""),
                                     year=r.get("year") or _year(found[r["arxiv_id"]].get("published")),
                                     citation_count=r.get("citation_count")) for r in top]
            papers_for[check.claim_id] = [
                {**p.model_dump(), "abstract": found[p.arxiv_id].get("abstract", "")[:_ABSTRACT_CHARS]}
                for p in check.papers]

        if papers_for and self._budget_allows(1, reserve=reserve):
            async for event in self._call(ctx, reviewer, f"{step}.assess", LiteratureAssessmentBatch, out,
                                          state={prompts.STATE_PHASE: "lit_assess",
                                                 prompts.STATE_CONTESTED: claims,
                                                 prompts.STATE_PAPERS: papers_for}):
                yield event
            for a in (out[f"{step}.assess"].assessments if f"{step}.assess" in out else []):
                check = checks.get(a.claim_id)
                if check is None or not check.papers:
                    continue
                listed = {p.arxiv_id for p in check.papers}
                cited = [i for i in (normalise_arxiv_id(x) for x in a.papers) if i in listed]
                check.verdict = a.verdict if cited or a.verdict == "inconclusive" else "inconclusive"
                check.summary = a.summary + ("" if cited else " (no listed paper cited)")
        for check in checks.values():
            if check.papers and check.verdict == "not_searched":
                check.verdict, check.summary = "inconclusive", "The reviewer did not assess these papers."

        self._record.literature.extend(checks.values())
        for check in checks.values():
            self._ledger.log("literature", step, check=check.model_dump(mode="json"))
        yield self._state_event(ctx, {prompts.STATE_LITERATURE: _dump(self._record.literature),
                                      prompts.STATE_CONTESTED: None, prompts.STATE_PAPERS: None})

    def _cap_and_link(self, decision, proposals, checks, plan, record) -> ScientificDecision:
        selected_ids = set(decision.selected_hypotheses)
        selected_agents = {n for n, p in proposals.items()
                           if any(h.id in selected_ids for h in p.hypotheses)} or set(proposals)
        open_blocking = any(i.severity == "blocking" for c in checks if c.target in selected_agents
                            for i in c.issues)
        plan_blocking = bool(record.plan_validation) and any(
            i.severity == "blocking" for i in record.plan_validation.issues)
        evidence = list(decision.justification) + [e for n in selected_agents for e in proposals[n].evidence]
        n_linked = sum(1 for e in evidence if e.kind == EvidenceKind.RETRIEVED and e.tool_call_ids)
        decision = finalize_status(decision, open_blocking=open_blocking, plan=plan,
                                   plan_blocking=plan_blocking, n_valid_proposals=len(proposals),
                                   n_linked_evidence=n_linked)
        links = [ProvenanceLink(source=f"proposal:{n}:rev{p.revision}", relation="based_on_proposal")
                 for n, p in proposals.items()]
        links += [ProvenanceLink(source=f"critique:{c.author}->{c.target}:r{c.round}", relation="addresses_critique")
                  for c in record.critiques if open_issues([c])]
        links += [ProvenanceLink(source=tc, relation="cites_tool_call")
                  for e in evidence for tc in e.tool_call_ids]
        return decision.model_copy(update={"provenance": links})

    async def _save_record(self, ctx: InvocationContext) -> str | None:
        """Persist the full record + ledger as an external artifact (too large for state)."""
        if ctx.artifact_service is None:
            return None
        payload = {"record": self._record.model_dump(mode="json"), "ledger": self._ledger.to_dict()}
        part = types.Part(inline_data=types.Blob(mime_type="application/json",
                                                 data=json.dumps(payload, indent=2, default=str).encode()))
        await ctx.artifact_service.save_artifact(app_name=ctx.app_name, user_id=ctx.user_id,
                                                 session_id=ctx.session.id, filename=RECORD_ARTIFACT,
                                                 artifact=part)
        return RECORD_ARTIFACT


def _data_section(record: LabRecord, ledger: LabLedger) -> list[str]:
    """The lab's data decision, traceable to the catalogue results: selected inputs with their
    registry ids and availability, alternatives considered and why not, unverified options."""
    chosen = record.plan.datasets if record.plan else [d for p in record.final_proposals.values()
                                                       for d in p.datasets]
    if not chosen:
        return []
    registry = inv.data_registry(ledger.tool_calls)
    seen: set[tuple[str, str]] = set()

    def line(ds) -> str:
        entries = [e for i in ds.source_ids for e in inv.resolve(registry, i)]
        refs = f" [{', '.join(ds.source_ids)}]" if ds.source_ids else ""
        found = f" — {entries[0].availability}" if entries and ds.decision != "rejected" else ""
        why = f" — {ds.justification}" if ds.justification else ""
        return f"- **{ds.name}** ({ds.role}){refs}{found}{why}"

    lines = ["", "## Data"]
    for decision, title in (("selected", "Selected"), ("rejected", "Considered, not selected"),
                            ("unverified", "Unverified options (availability not shown)")):
        rows = []
        for ds in chosen:
            key = (ds.decision, ds.name.lower())
            if ds.decision == decision and key not in seen:
                seen.add(key)
                rows.append(line(ds))
        if rows:
            lines += ["", f"**{title}**", *rows]
    return lines


def render_report(record: LabRecord, ledger: LabLedger) -> str:
    """Human-readable final answer; the structured record is the source of truth."""
    d, plan = record.decision, record.plan
    lines = [f"# Virtual Lab decision: {d.status.value if d else 'none'}", "", d.summary if d else ""]
    if d and d.status_note:
        lines += ["", f"_Status adjusted by the coordinator: {d.status_note}_"]
    if d and d.recommended_workflow:
        lines += ["", "## Recommended workflow", *[f"{i}. {s}" for i, s in enumerate(d.recommended_workflow, 1)]]
    if d and d.disagreements:
        lines += ["", "## Unresolved disagreements"]
        for dis in d.disagreements:
            lines.append(f"- **{dis.topic}**: " + "; ".join(f"{a}: {p}" for a, p in dis.positions.items())
                         + (f" — evidence needed: {dis.evidence_needed}" if dis.evidence_needed else ""))
    lines += _data_section(record, ledger)
    if plan:
        lines += ["", "## Experiment plan", f"**Question:** {plan.research_question}", "", "**Hypotheses**"]
        lines += [f"- {h.id}: {h.statement} — predicts: {h.prediction}; refuted if: {h.falsification}"
                  for h in plan.hypotheses]
        lines += ["", "**Workflow**"]
        lines += [f"{s.order}. [{s.agent}] {s.description}" for s in sorted(plan.workflow, key=lambda s: s.order)]
        for title, items in (("Reference data", plan.reference_data), ("Baselines", plan.baselines),
                             ("Metrics", plan.metrics), ("Statistical tests", plan.statistical_tests),
                             ("Confounders", plan.confounders), ("Stopping criteria", plan.stopping_criteria)):
            if items:
                lines += ["", f"**{title}:** " + "; ".join(items)]
    if record.plan_validation and open_issues([record.plan_validation]):
        lines += ["", "## Open validator findings on the plan"]
        lines += [f"- [{i.severity}] {i.category}: {i.rationale}" for i in open_issues([record.plan_validation])]
    if record.literature:
        lines += ["", "## Literature checks"]
        for c in record.literature:
            papers = ", ".join(f"arXiv:{p.arxiv_id} ({p.citation_count if p.citation_count is not None else '?'} cit.)"
                               for p in c.papers)
            lines.append(f"- **{c.verdict}** ({c.scope}) — {c.claim}: {c.summary}"
                         + (f" [{papers}]" if papers else ""))
    if record.results:
        lines += ["", "## Experiment results (executed)"]
        for r in record.results:
            lines.append(f"- **{r.id} {r.analysis}** [{r.status}]: {r.summary}")
            lines += [f"  - artifact: {a}" for a in r.artifacts]
    usage = ledger.usage()
    linked = sum(1 for p in record.final_proposals.values() for e in p.evidence
                 if e.kind == EvidenceKind.RETRIEVED and e.tool_call_ids)
    verified = sum(1 for e in (d.justification if d else []) if e.kind == EvidenceKind.VERIFIED)
    executed = any(r.status == "ok" for r in record.results)
    lines += ["", "## Evidence status",
              (f"The lab executed {sum(r.status == 'ok' for r in record.results)} analyses; "
               f"{verified} conclusion claim(s) cite an executed result." if executed else
               "No experiment was executed (plan-only mode): conclusions rest on retrieved metadata/"
               "literature and stated assumptions, not on verified results."),
              f"- Retrieved claims linked to tool calls: {linked}",
              *([f"- Data grounding: {g['grounded']}/{g['selected']} selected datasets cite data "
                 f"found here; {g['shortlisted_considered']}/{g['shortlisted_with_data']} discovered "
                 f"collections with data were assessed; {g['rejected']} rejected, "
                 f"{g['unverified']} unverified"] if (g := record.data_grounding) else []),
              f"- Termination: {record.termination}; deliberation rounds: {record.rounds_run}",
              f"- Tool calls: {usage['tool_calls']} ({usage['blocked_tool_calls']} blocked); "
              f"participant LLM calls: {usage['lab_llm_calls']}; tokens in/out: "
              f"{usage['input_tokens']}/{usage['output_tokens']}"]
    return "\n".join(lines)
