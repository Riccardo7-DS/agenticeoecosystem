"""
validation.py — Deterministic scientific checks and termination rules.

These run in code, not in an LLM, so they can't be talked out of a finding:
dataset availability and coverage (sensor_catalog), physical observing
constraints, methodological feasibility of a plan, evidence sufficiency, and
the caps on the final decision status. Keyword heuristics over free-text
`observable` fields are deliberately conservative and documented inline.
"""

from __future__ import annotations

import re
from datetime import date
from enum import StrEnum
from typing import Any

from .schemas import (
    CritiqueIssue,
    DatasetChoice,
    DecisionStatus,
    EvidenceItem,
    EvidenceKind,
    ExperimentPlan,
    LiteratureCheck,
    ScientificCritique,
    ScientificDecision,
    ScientificProblem,
    ScientificProposal,
)
from .literature import arxiv_ids_in
from .sensor_catalog import (
    GEO_VZA_DEGRADED_DEG,
    GEO_VZA_UNUSABLE_DEG,
    geostationary_vza_deg,
    lookup,
    sensors_in_text,
)

VALIDATOR = "validator"
OPEN_SEVERITIES = ("blocking", "major")
LOW_CAPABILITY_SCORE = 40

# Observable keywords -> wildfire capability class in sensor_catalog. Checked in
# order, so "burned area under smoke" maps to BA, not SMK.
_CAPABILITY_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("FRP", ("frp", "radiative power")),
    ("AF", ("active fire", "hotspot", "hot spot", "ignition", "fire detection", "thermal anomal")),
    ("BA", ("burned area", "burnt area", "burn scar", "dnbr", "burn severity", "fire perimeter")),
    ("SMK", ("smoke", "plume", "aerosol")),
    ("REC", ("recovery", "regrowth", "revegetation")),
)
_NIGHT_KEYWORDS = ("night", "nocturnal")
# Sub-agents that process local files rather than fetch from a provider: any sensor is fine.
_PROCESSING_AGENTS = {"gis_agent", "vlm_agent"}
# Tests whose validity depends on replicated samples, and words that name the replication unit.
_SAMPLE_TESTS = ("t-test", "t test", "anova", "wilcoxon", "mann-whitney", "chi-square", "chi2")
_REPLICATION_UNITS = ("pixel", "overpass", "scene", "sample", "n=", "n =", "event", "fires",
                      "points", "plots", "granule", "acquisition", "region", "sites", "seed", "fold")
_PENETRATION_KEYWORDS = ("through smoke", "under smoke", "beneath smoke", "through cloud",
                         "under cloud", "cloud-penetrat", "smoke-penetrat", "all-weather")


def _issue(category, severity, claim, rationale, fix="") -> CritiqueIssue:
    return CritiqueIssue(category=category, severity=severity, claim=claim,
                         rationale=rationale, suggested_fix=fix)


def _parse_date(text: str) -> date | None:
    try:
        return date.fromisoformat(text[:10])
    except (TypeError, ValueError):
        return None


def _bbox_centre(problem: ScientificProblem) -> tuple[float, float] | None:
    bbox = problem.study_area.bbox
    if not bbox:
        return None
    west, south, east, north = bbox
    if west > east:  # crosses the antimeridian
        east += 360
    lon = (west + east) / 2
    return (south + north) / 2, (lon - 360 if lon > 180 else lon)


def check_dataset(ds: DatasetChoice, problem: ScientificProblem) -> list[CritiqueIssue]:
    """Availability, coverage and physical-suitability checks for one dataset choice."""
    sensor = lookup(ds.name)
    if sensor is None:
        return [_issue("dataset_availability", "minor", ds.name,
                       "Not in the sensor catalog; availability, coverage and access are unverified.",
                       "Name a catalogued sensor or cite a tool result confirming availability.")]

    issues: list[CritiqueIssue] = []
    start, end = _parse_date(problem.time_period.start), _parse_date(problem.time_period.end)
    if start and end:
        if end < sensor.start or (sensor.end and start > sensor.end):
            issues.append(_issue("dataset_availability", "blocking", ds.name,
                                 f"{sensor.key} was not operating during {start}..{end} "
                                 f"(operational from {sensor.start})."))
        elif start < sensor.start:
            issues.append(_issue("dataset_availability", "major", ds.name,
                                 f"{sensor.key} only covers part of the period (from {sensor.start})."))

    if not sensor.sub_agents:
        severity = "major" if ds.role in ("primary", "reference") else "minor"
        issues.append(_issue("dataset_availability", severity, ds.name,
                             f"No sub-agent in this system can fetch {sensor.key}; the workflow "
                             "cannot be executed as proposed.",
                             "Use an accessible sensor or mark this as an external dependency."))

    centre = _bbox_centre(problem)
    if sensor.orbit == "geostationary" and centre is not None:
        vza = geostationary_vza_deg(*centre, sensor.sub_satellite_lon)
        if vza > GEO_VZA_UNUSABLE_DEG:
            issues.append(_issue("coverage", "blocking", ds.name,
                                 f"Study area is at viewing zenith ~{vza:.0f} deg from {sensor.key} "
                                 f"({sensor.sub_satellite_lon:+.1f} deg): UNUSABLE GEOMETRY. The data "
                                 "may exist, but the area is effectively outside the usable disk."))
        elif vza > GEO_VZA_DEGRADED_DEG:
            issues.append(_issue("coverage", "major", ds.name,
                                 f"Viewing zenith ~{vza:.0f} deg from {sensor.key}: DEGRADED GEOMETRY, "
                                 "large pixels and long slant paths degrade detection and timing."))

    observable = ds.observable.lower()
    if sensor.day_only and any(k in observable for k in _NIGHT_KEYWORDS):
        issues.append(_issue("physical_inconsistency", "blocking", ds.name,
                             f"{sensor.key} is reflective-only and cannot observe at night."))
    if not sensor.all_weather and any(k in observable for k in _PENETRATION_KEYWORDS):
        issues.append(_issue("physical_inconsistency", "major", ds.name,
                             f"{sensor.key} is passive optical/IR; smoke and cloud obscure the surface.",
                             "Pair with SAR or treat obscuration as a confounder."))
    for cls, keywords in _CAPABILITY_KEYWORDS:
        if any(k in observable for k in keywords):
            score = sensor.wildfire_scores.get(cls)
            if score is not None and score < LOW_CAPABILITY_SCORE:
                issues.append(_issue("dataset_suitability", "major", ds.name,
                                     f"{sensor.key} has low capability for {cls} ({score}/100 in the "
                                     "knowledge.tex capability matrix)."))
            break
    return issues


_CALL_ID = re.compile(r"tc-\d{4}")


def link_items(
    items: list[EvidenceItem], tool_calls: list[dict]
) -> tuple[list[EvidenceItem], list[CritiqueIssue]]:
    """Attach ledger tool-call ids to RETRIEVED evidence; downgrade claims no call backs.

    A claim is linked when its `source` names a tool, or the sub-agent that
    made the call, among `tool_calls`; a source citing arXiv ids links only to
    the literature calls that returned those papers, and a source 'inventory:<call id>' to
    that DATA INVENTORY call (bare 'inventory' to all of them). Calls the lab made itself
    (`lab_call`) are matched only by arXiv id: naming 'arxiv_rag_agent' does not
    show the agent read what the lab retrieved. Otherwise it becomes an ASSUMPTION.
    """
    issues: list[CritiqueIssue] = []
    linked: list[EvidenceItem] = []
    for item in items:
        if item.kind != EvidenceKind.RETRIEVED:
            linked.append(item)
            continue
        src = (item.source or "").lower()
        papers = arxiv_ids_in(src)
        if papers:
            ids = [c["id"] for c in tool_calls if papers & set(c.get("paper_ids", ()))]
        elif "inventory" in src:
            cited = set(_CALL_ID.findall(src))
            ids = [c["id"] for c in tool_calls if c.get("inventory") and (not cited or c["id"] in cited)]
        else:
            ids = [c["id"] for c in tool_calls
                   if src and not c.get("lab_call")
                   and (c["tool"].lower() in src or src in c["tool"].lower()
                        or (c.get("via") and c["via"].lower() in src))]
        if ids:
            linked.append(item.model_copy(update={"tool_call_ids": ids}))
        else:
            linked.append(item.model_copy(update={"kind": EvidenceKind.ASSUMPTION}))
            issues.append(_issue("unsupported_claim", "major", item.claim,
                                 f"Labelled as retrieved from '{item.source}', but no matching tool call "
                                 "was made; downgraded to assumption."))
    return linked, issues


def link_evidence(
    proposal: ScientificProposal, tool_calls: list[dict]
) -> tuple[ScientificProposal, list[CritiqueIssue]]:
    """link_items over a proposal, using the calls made while that proposal was written."""
    linked, issues = link_items(proposal.evidence, tool_calls)
    return proposal.model_copy(update={"evidence": linked}), issues


_TOOL_EVIDENCE_CHARS = 300


def bboxes_overlap(a: list[float], b: list[float]) -> bool:
    """[west, south, east, north] boxes intersect (touching counts)."""
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def _as_bbox(value: Any) -> list[float] | None:
    try:
        box = [float(x) for x in value]
    except (TypeError, ValueError):
        return None
    return box if len(box) == 4 else None


def off_area_calls(calls: list[dict], problem: ScientificProblem) -> list[CritiqueIssue]:
    """Tool calls whose bbox misses the study area: their results say nothing about it.
    (Run of 2026-11-13: with no study bbox, agents invented coordinates ~70 km and ~220 km
    from Kypseli, Athens, and the decision cited those searches as availability evidence.)"""
    area = problem.study_area.bbox if problem.study_area else None
    if not area:
        return []
    issues = []
    for c in calls:
        box = _as_bbox((c.get("args") or {}).get("bbox"))
        if box and not bboxes_overlap(box, area):
            issues.append(_issue(
                "coverage", "major", f"{c['tool']} ({c['id']}) bbox {box}",
                f"This search's bbox does not overlap the study area {list(area)}, so its result "
                "is not evidence about the study area.",
                "Repeat the search with the study-area bbox from the problem, and do not cite this call."))
    return issues


def tool_call_evidence(agent: str, calls: list[dict], items: list[EvidenceItem]) -> list[EvidenceItem]:
    """One RETRIEVED item per successful call `agent` made itself that no item already cites.

    Models often call a tool and then label everything 'assumption', which would leave the
    call out of the evidence trail. The claim is the call's result as returned (truncated),
    not an interpretation of it; leaf calls the sub-agent made are linked too."""
    cited = {i for item in items for i in item.tool_call_ids}
    out = []
    for c in calls:
        if c["agent"] != agent or c["status"] != "ok" or c["id"] in cited:
            continue
        leaves = [x["id"] for x in calls if x.get("via") == c["tool"]]
        result = " ".join(str(c.get("result_preview", "")).split())[:_TOOL_EVIDENCE_CHARS]
        out.append(EvidenceItem(claim=f"{c['tool']} returned: {result}", kind=EvidenceKind.RETRIEVED,
                                source=c["tool"], tool_call_ids=[c["id"], *leaves]))
    return out


def literature_uptake(proposal: ScientificProposal, checks: list[LiteratureCheck]) -> list[CritiqueIssue]:
    """A revision must engage with literature checks that settled a claim in its proposal:
    cite one of the papers in `evidence`, or name one in `limitations` to disagree."""
    if proposal.revision == 0:
        return []
    cited = arxiv_ids_in(" ".join([*(e.source or "" for e in proposal.evidence), *proposal.limitations]))
    return [_issue("evidence_sufficiency", "minor", c.claim,
                   f"Literature check {c.claim_id} found the papers {c.verdict} this claim "
                   f"({', '.join(p.arxiv_id for p in c.papers)}), but the revision neither cites "
                   "them nor explains in its limitations why it disagrees.",
                   "Cite a paper as evidence (source 'arxiv:<id>') or address it in limitations.")
            for c in checks
            if c.target == proposal.agent and c.verdict in ("supports", "contradicts", "mixed")
            and c.papers and not cited & {p.arxiv_id for p in c.papers}]


def validate_proposal(
    proposal: ScientificProposal,
    problem: ScientificProblem,
    grounding_available: bool,
    round_: int = 0,
    extra_issues: list[CritiqueIssue] | None = None,
) -> ScientificCritique:
    issues = list(extra_issues or [])
    for ds in proposal.datasets:
        issues.extend(check_dataset(ds, problem))
    if not proposal.hypotheses:
        issues.append(_issue("methodology", "blocking", proposal.summary,
                             "The proposal states no testable hypothesis."))
    for h in proposal.hypotheses:
        if not h.prediction.strip() or not h.falsification.strip():
            issues.append(_issue("methodology", "major", h.statement,
                                 "Hypothesis lacks an observable prediction or a falsification criterion."))
    if not any(e.kind == EvidenceKind.RETRIEVED for e in proposal.evidence):
        issues.append(_issue("evidence_sufficiency", "major" if grounding_available else "minor",
                             proposal.summary,
                             "No claim is backed by a tool result from this run; the proposal rests "
                             "on assumptions and model recall."))
    return ScientificCritique(author=VALIDATOR, target=proposal.agent, issues=issues, round=round_)


def validate_plan(
    plan: ExperimentPlan, problem: ScientificProblem, available_agents: set[str],
    catalog: list[dict] = (),
) -> ScientificCritique:
    """catalog: summaries of this run's CDSE catalogue searches (LabLedger.catalog_results)."""
    issues: list[CritiqueIssue] = []
    for ds in plan.datasets:
        issues.extend(check_dataset(ds, problem))
    issues += _step_sensor_checks(plan, problem)
    if not plan.hypotheses:
        issues.append(_issue("methodology", "blocking", plan.research_question,
                             "The plan has no hypothesis to test."))
    for step in plan.workflow:
        if step.agent not in available_agents:
            issues.append(_issue("methodology", "major", f"step {step.order}: {step.description}",
                                 f"'{step.agent}' is not an available sub-agent; the step cannot run.",
                                 f"Use one of: {', '.join(sorted(available_agents))}."))
    if not plan.workflow:
        issues.append(_issue("methodology", "blocking", plan.research_question, "Empty workflow."))
    if not plan.reference_data:
        issues.append(_issue("evidence_sufficiency", "major", plan.research_question,
                             "No independent reference data; results could not be validated."))
    if not plan.baselines:
        issues.append(_issue("methodology", "major", plan.research_question,
                             "No baseline or alternative approach to compare against."))
    if not plan.metrics:
        issues.append(_issue("methodology", "major", plan.research_question, "No evaluation metrics."))
    if len(plan.hypotheses) > 1 and not plan.statistical_tests:
        issues.append(_issue("methodology", "major", plan.research_question,
                             "Competing hypotheses but no statistical comparison to discriminate them."))
    if not plan.stopping_criteria:
        issues.append(_issue("methodology", "minor", plan.research_question, "No stopping criteria."))
    issues += _reference_independence(plan)
    issues += _hypotheses_tested(plan)
    issues += _step_access(plan, available_agents)
    issues += _sample_size(plan)
    issues += _step_dates_in_catalog(plan, catalog)
    return ScientificCritique(author=VALIDATOR, target="experiment_plan", issues=issues)


_COLLECTION_IN_TEXT = re.compile(r"\bsentinel[- _]?(1|2|3|5p)\b", re.IGNORECASE)
_ISO_DATE = re.compile(r"\b(20\d\d-\d\d-\d\d)\b")


def _filter_matches(product_type: str | None, text: str) -> bool:
    """A search under a product filter only speaks for steps that use that product
    ('SLC' search for an SLC step; 'MSIL2A' also matches 'L2A', 'SL_' matches 'SLSTR')."""
    if not product_type:
        return True
    upper = text.upper()
    return any(c and c in upper for c in {product_type.rstrip("_"), product_type.replace("MSI", "").rstrip("_")})


def _step_dates_in_catalog(plan: ExperimentPlan, catalog: list[dict]) -> list[CritiqueIssue]:
    """Flag a step that uses a Sentinel collection on a date the run's catalogue searches
    covered completely but found no acquisition for. Dates no search covered are not judged."""
    issues = []
    for step in plan.workflow:
        text = _step_text(step)
        dates = sorted(set(_ISO_DATE.findall(text)))
        for number in sorted({m.upper() for m in _COLLECTION_IN_TEXT.findall(text)}):
            collection = f"SENTINEL-{number}"
            searches = [c for c in catalog if c.get("collection") == collection]
            found = {d for c in searches for d in c.get("dates", [])}
            for day in dates:
                covered = any(c.get("complete") and (c.get("start") or "") <= day <= (c.get("end") or "")
                              and _filter_matches(c.get("product_type"), text) for c in searches)
                if not covered or day in found:
                    continue
                nearest = sorted(found, key=lambda d: abs((date.fromisoformat(d) - date.fromisoformat(day)).days))[:2]
                issues.append(_issue(
                    "dataset_availability", "major", f"step {step.order}: {collection} on {day}",
                    f"The catalogue search covering {day} lists no {collection} acquisition that day.",
                    f"Use an acquisition the search found (nearest: {', '.join(sorted(nearest)) or 'none'})."))
    return issues


_CLAUSE_SPLIT = re.compile(r"(?<!\d)[.;](?!\d)|\n")
# A clause with one of these sets a sensor aside rather than using it
# ("Exclude MTG FCI due to unusable geometry", "SEVIRI is not used").
_EXCLUSION = re.compile(r"\b(exclud\w*|except|without|instead of|rather than|not used|do not use|"
                        r"don't use|not (?:be )?usable|unusable|discard\w*|omit\w*|drop(?:ped)?|"
                        r"avoid\w*|reject\w*)\b", re.IGNORECASE)


def sensors_used(step) -> list:
    """Catalog sensors a workflow step actually uses: those named in its inputs or in a
    description clause that does not set them aside. A step saying 'Exclude MTG FCI (unusable
    geometry)' does not use MTG FCI."""
    clauses = [c for c in _CLAUSE_SPLIT.split(step.description) if not _EXCLUSION.search(c)]
    inputs = [i for i in step.inputs if not _EXCLUSION.search(i)]
    found = {s.key: s for text in [*clauses, *inputs] for s in sensors_in_text(text)}
    return list(found.values())


def _step_sensor_checks(plan: ExperimentPlan, problem: ScientificProblem) -> list[CritiqueIssue]:
    """Coverage and availability for sensors a step uses but `plan.datasets` doesn't list
    (a plan can drop a sensor from its datasets and still schedule it in the workflow)."""
    listed = {s.key for ds in plan.datasets for s in sensors_in_text(ds.name)}
    issues, seen = [], set()
    for step in plan.workflow:
        for sensor in sensors_used(step):
            if sensor.key in listed or sensor.key in seen:
                continue
            seen.add(sensor.key)
            # Empty observable: only coverage/availability apply, not keyword-based suitability.
            for issue in check_dataset(DatasetChoice(name=sensor.key), problem):
                issues.append(issue.model_copy(update={"claim": f"step {step.order}: {issue.claim}"}))
    return issues


def _step_text(step) -> str:
    return " ".join([step.description, *step.inputs, *step.outputs])


def _reference_independence(plan: ExperimentPlan) -> list[CritiqueIssue]:
    """A dataset under evaluation cannot also be the reference that validates it."""
    evaluated = {s.key for ds in plan.datasets if ds.role in ("primary", "baseline")
                 for s in sensors_in_text(ds.name)}
    refs = list(dict.fromkeys([*plan.reference_data,
                               *(ds.name for ds in plan.datasets if ds.role == "reference")]))
    overlapping = {ref: sorted({s.key for s in sensors_in_text(ref)} & evaluated) for ref in refs}
    overlapping = {ref: keys for ref, keys in overlapping.items() if keys}
    if not overlapping:
        return []
    listed = "; ".join(f"'{ref}' ({', '.join(keys)})" for ref, keys in overlapping.items())
    return [_issue("evidence_sufficiency", "blocking", ", ".join(overlapping),
                   f"Reference data overlaps the datasets under evaluation: {listed}. A dataset "
                   "cannot validate itself, so accuracy or timing could not be measured.",
                   "Use independent reference data (e.g. Copernicus EMS delineations, GNSS or "
                   "levelling, very high resolution imagery, field or agency reports, held-out "
                   "benchmark labels) or a sensor not under test.")]


def _hypotheses_tested(plan: ExperimentPlan) -> list[CritiqueIssue]:
    """Each hypothesis needs a step linked to it, and steps using the sensors it is about."""
    issues = []
    step_sensors = {s.key for step in plan.workflow for s in sensors_in_text(_step_text(step))}
    for h in plan.hypotheses:
        linked = [s for s in plan.workflow
                  if h.id in s.hypotheses or h.id.lower() in _step_text(s).lower()]
        if not linked:
            issues.append(_issue("methodology", "major", f"{h.id}: {h.statement}",
                                 f"No workflow step is linked to {h.id}; nothing in the plan tests it.",
                                 "List the hypothesis id in the 'hypotheses' field of the steps that test it."))
        missing = sorted({s.key for s in sensors_in_text(h.statement)} - step_sensors)
        if missing:
            issues.append(_issue("methodology", "major", f"{h.id}: {h.statement}",
                                 f"{h.id} is about {', '.join(missing)}, but no workflow step uses "
                                 f"{'it' if len(missing) == 1 else 'them'}, so the plan cannot "
                                 "discriminate this hypothesis.",
                                 "Add steps that use them, or drop the hypothesis and state that it "
                                 "cannot be tested with accessible data."))
    return issues


def _step_access(plan: ExperimentPlan, available_agents: set[str]) -> list[CritiqueIssue]:
    """A step can only fetch sensors its sub-agent has access to (per the sensor catalog)."""
    issues = []
    for step in plan.workflow:
        if step.agent not in available_agents or step.agent in _PROCESSING_AGENTS:
            continue
        for sensor in sensors_used(step):
            if sensor.sub_agents and step.agent not in sensor.sub_agents:
                issues.append(_issue("dataset_availability", "major",
                                     f"step {step.order}: {step.description}",
                                     f"'{step.agent}' cannot access {sensor.key}.",
                                     f"Assign this to {' or '.join(sensor.sub_agents)}, or split the step."))
    return issues


def _sample_size(plan: ExperimentPlan) -> list[CritiqueIssue]:
    """Flag sample-based tests with no stated unit of replication (a single event or site is n=1)."""
    tests = " ".join(plan.statistical_tests).lower()
    if not any(t in tests for t in _SAMPLE_TESTS):
        return []
    context = " ".join([*plan.statistical_tests, *plan.metrics, *(s.description for s in plan.workflow)]).lower()
    if any(u in context for u in _REPLICATION_UNITS):
        return []
    return [_issue("methodology", "minor", "; ".join(plan.statistical_tests),
                   "Sample-based test with no stated unit of replication; for a single event or "
                   "site the sample size is effectively 1.",
                   "State what is replicated (pixels, overpasses, scenes, events, regions, seeds) or "
                   "use a descriptive comparison.")]


def open_issues(critiques: list[ScientificCritique]) -> list[CritiqueIssue]:
    return [i for c in critiques for i in c.issues if i.severity in OPEN_SEVERITIES]


class TerminationReason(StrEnum):
    CONVERGED = "converged"            # no blocking/major issues left
    SETTLED = "settled_after_revision"  # revision left no blocking or validator issue: no new round
    MAX_ROUNDS = "max_rounds"
    BUDGET = "budget"
    NO_PROPOSALS = "no_valid_proposals"
    FORMULATION_FAILED = "formulation_failed"


def finalize_status(
    decision: ScientificDecision,
    *,
    open_blocking: bool,
    plan: ExperimentPlan | None,
    plan_blocking: bool,
    n_valid_proposals: int,
    n_linked_evidence: int,
) -> ScientificDecision:
    """Cap the PI's proposed status by what the evidence actually supports."""
    status, notes = decision.status, []
    plan_usable = plan is not None and bool(plan.hypotheses) and not plan_blocking

    if n_valid_proposals == 0:
        status, notes = DecisionStatus.INSUFFICIENT_EVIDENCE, ["no valid proposals"]
    elif status == DecisionStatus.DECIDED:
        if open_blocking:
            notes.append("blocking issues remain unresolved")
        if decision.disagreements:
            notes.append("specialists still disagree")
        if n_linked_evidence == 0:
            notes.append("no claim is backed by a tool result")
        if notes:
            status = DecisionStatus.NEEDS_EXPERIMENT if plan_usable else DecisionStatus.INSUFFICIENT_EVIDENCE
    if status == DecisionStatus.NEEDS_EXPERIMENT and not plan_usable:
        status = DecisionStatus.INSUFFICIENT_EVIDENCE
        notes.append("no executable experiment plan")

    if status == decision.status:
        return decision
    note = f"Changed from '{decision.status}' to '{status}': " + "; ".join(notes) + "."
    return decision.model_copy(update={"status": status, "status_note": note})


# ---------------------------------------------------------------------- execute mode

_RESULT_ID = re.compile(r"\bres-\d+\b")


def normalise_conclusion(data: dict) -> dict:
    """Before validation: a 'verified' claim without result_id takes one named in its source
    ('result:res-2'), else becomes an assumption (the schema rejects verified without a result)."""
    for item in data.get("justification") or []:
        if isinstance(item, dict) and item.get("kind") == "verified" and not item.get("result_id"):
            found = _RESULT_ID.search(str(item.get("source") or "") + " " + str(item.get("claim") or ""))
            if found:
                item["result_id"] = found.group(0)
            else:
                item["kind"] = "assumption"
    return data


def link_results(items: list[EvidenceItem], results: list) -> list[EvidenceItem]:
    """Keep 'verified' only for claims citing a successful result (with its tool-call ids);
    a claim citing a failed or unknown result becomes an assumption."""
    ok = {r.id: r for r in results if r.status == "ok"}
    out = []
    for item in items:
        if item.kind == EvidenceKind.VERIFIED:
            result = ok.get(item.result_id or "")
            if result is None:
                item = item.model_copy(update={"kind": EvidenceKind.ASSUMPTION, "result_id": None,
                                               "claim": item.claim + " (cited result missing or failed)"})
            else:
                item = item.model_copy(update={"tool_call_ids": list(result.tool_call_ids),
                                               "source": item.source or f"result:{result.id}"})
        out.append(item)
    return out


def finalize_conclusion(decision: ScientificDecision, results: list) -> ScientificDecision:
    """Cap the PI's post-experiment status by what was actually executed and cited."""
    status, notes = decision.status, []
    n_ok = sum(r.status == "ok" for r in results)
    n_verified = sum(e.kind == EvidenceKind.VERIFIED for e in decision.justification)
    if n_ok == 0 and status != DecisionStatus.INSUFFICIENT_EVIDENCE:
        status, notes = DecisionStatus.INSUFFICIENT_EVIDENCE, ["no analysis ran successfully"]
    elif status == DecisionStatus.DECIDED and n_verified == 0:
        status, notes = DecisionStatus.NEEDS_EXPERIMENT, ["no conclusion cites an executed result"]
    if status == decision.status:
        return decision
    note = f"Changed from '{decision.status}' to '{status}': " + "; ".join(notes) + "."
    return decision.model_copy(update={"status": status, "status_note": note})
