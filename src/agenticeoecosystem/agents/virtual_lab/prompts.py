"""
prompts.py — Instruction text for Virtual Lab participants.

Instructions are built per call from session state (ADK InstructionProviders),
so each participant sees exactly what its phase allows: in the proposal phase
a specialist sees only the problem, never a competing proposal. JSON skeletons
here mirror schemas.py; fields the coordinator owns (agent, author, revision,
round) are omitted so the model can't set them.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

from .sensor_catalog import format_catalog

STATE_PHASE = "vl_phase"
STATE_PROBLEM = "vl_problem"
STATE_PROPOSALS = "vl_proposals"
STATE_CRITIQUES = "vl_critiques"
STATE_VALIDATION = "vl_validation"
STATE_DECISION = "vl_decision"
STATE_ROUND = "vl_round"
STATE_LITERATURE = "vl_literature"   # completed literature checks, visible to everyone after
STATE_CONTESTED = "vl_contested"     # claims the literature reviewer must check
STATE_PAPERS = "vl_papers"           # claim_id -> most-cited papers with abstracts
STATE_PLAN = "vl_plan"
STATE_INVENTORY = "vl_inventory"     # data inventory text, built once after formulation
STATE_SUB_AGENTS = "vl_sub_agents"   # sub-agents the PI chose; others are not offered
STATE_RESULTS = "vl_results"         # execute mode: results of the lab's fixed analyses
# The lab's direct metadata tools (participants.catalog_tools), named in the tool guidance.
DIRECT_TOOLS = ("search_catalogues", "search_cdse_catalog")


def raw_key(agent: str) -> str:
    return f"vl_raw_{agent}"


def repair_key(agent: str) -> str:
    return f"vl_repair_{agent}"


SCIENTIFIC_NORMS = """\
SCIENTIFIC NORMS (apply to everything you write):
- Label every claim in "evidence" with its kind:
  "retrieved"  = returned by a tool/sub-agent YOU called in THIS step; put that tool or
                 sub-agent name in "source". Never label recalled knowledge as retrieved.
                 If you called a tool, at least one claim should state what it returned.
                 (The lab also records each successful call's result as retrieved evidence.)
  "assumption" = something you take as given without having checked it here.
  "hypothesis" = a testable claim you are proposing.
  Never use "verified": no experiment has been executed in this run.
- Rows of the DATA INVENTORY were retrieved by the lab. Cite one as kind "retrieved" with
  source "inventory:<call id>" (e.g. "inventory:tc-0003"). It is the authoritative record of
  which data it lists exist: a claim that contradicts it is wrong.
- Papers listed under LITERATURE CHECKS were retrieved by the lab. You may cite one as
  kind "retrieved" with source "arxiv:<id>" (e.g. "arxiv:2105.01234"). A paper supports a
  general physical or methodological claim; it does not show what happened in THIS event.
- When data cannot be used, say which of these it is (never just "unavailable"):
  NOT FOUND        = searched (inventory row or tool call) and none exist for the area/period;
  UNUSABLE GEOMETRY / UNUSABLE CONDITIONS = they exist but viewing angle, cloud, night etc.
                     make them unfit here; name the reason;
  NOT CHECKED      = no inventory row and no tool call covers them: availability is unknown.
  Data the DATA INVENTORY lists as found are never "unavailable".
- Agreement between agents is not evidence. Do not cite another agent's proposal as support.
- Prefer falsifiable hypotheses: state an observable prediction and what would refute it.
- Respect physical constraints: day/night, cloud and smoke obscuration, viewing geometry,
  spatial resolution vs. feature size, revisit time vs. the timing question, and what each
  sensor or model actually measures (e.g. InSAR sees line-of-sight motion only).
- Say what you do not know. Low confidence is acceptable; unsupported certainty is not.
"""

OUTPUT_RULE = ("Return ONLY one JSON object matching the schema below — no markdown fences, "
               "no prose before or after it. This holds even when a tool call is blocked or the "
               "tool budget is exhausted: never explain the budget, just answer in JSON with what "
               "you have.")

PROBLEM_SCHEMA = """\
{"question": str, "objectives": [str], "study_area": {"name": str, "bbox": [west, south, east, north] | null},
 "time_period": {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}, "constraints": [str], "expected_outputs": [str],
 "data_needs": [str],
 "task_types": ["dataset_selection" | "scientific_reasoning" | "hypothesis_testing" | "experimental_execution"],
 "participants": [specialist names], "sub_agents": [sub-agent names], "rationale": str}"""

# How the PI writes ScientificProblem.data_needs (the federated data discovery's queries).
# Shared with scripts/eval_discovery.py so the benchmark tests the production wording.
DATA_NEEDS_RULE = (
    "- In 'data_needs', give 2-6 entries, one per kind of data the problem needs. Write each "
    "the way a data catalogue would describe that data (10-25 words): the kind of instrument "
    "or measurement technique, the product level or form, and the quantity it gives, with "
    "resolution or revisit where they matter. Do not name specific products or missions. "
    "Catalogues describe instruments, not applications, so 'ground deformation' alone finds "
    "nothing; e.g. 'synthetic aperture radar single look complex images with phase, for "
    "interferometric ground deformation measurement', 'gridded satellite precipitation "
    "estimates, half-hourly to daily', 'thermal infrared active fire detections with fire "
    "radiative power'. The lab searches every data catalogue with them.\n")

PROPOSAL_SCHEMA = """\
{"summary": str,
 "hypotheses": [{"id": "H1", "statement": str, "prediction": str, "falsification": str}],
 "datasets": [{"name": str, "role": "primary" | "reference" | "baseline" | "ancillary",
               "observable": str, "justification": str}],
 "methods": [str], "assumptions": [str], "limitations": [str],
 "evidence": [{"claim": str, "kind": "retrieved" | "assumption" | "hypothesis", "source": str | null}],
 "confidence": "low" | "medium" | "high", "changes_from_previous": str | null}"""

CRITIQUE_SCHEMA = """\
{"critiques": [{"target": <agent name of the proposal>, "issues": [{
   "category": "unsupported_claim" | "physical_inconsistency" | "dataset_availability" |
               "dataset_suitability" | "coverage" | "methodology" | "evidence_sufficiency" |
               "limitation" | "other",
   "severity": "blocking" | "major" | "minor",
   "claim": <the exact claim or choice challenged>, "rationale": str, "suggested_fix": str}]}]}"""

CONCLUSION_SCHEMA = """\
{"status": "decided" | "needs_experiment" | "insufficient_evidence",
 "summary": str, "recommended_workflow": [str], "selected_hypotheses": [hypothesis ids],
 "disagreements": [{"topic": str, "positions": {<agent>: str}, "evidence_needed": str}],
 "justification": [{"claim": str, "kind": "verified" | "retrieved" | "assumption",
                    "source": str | null, "result_id": "res-<n>" | null}]}"""

DECISION_SCHEMA = """\
{"status": "decided" | "needs_experiment" | "insufficient_evidence",
 "summary": str, "recommended_workflow": [str], "selected_hypotheses": [hypothesis ids],
 "disagreements": [{"topic": str, "positions": {<agent>: str}, "evidence_needed": str}],
 "justification": [{"claim": str, "kind": "retrieved" | "assumption" | "hypothesis", "source": str | null}]}"""

PLAN_SCHEMA = """\
{"research_question": str,
 "hypotheses": [{"id": str, "statement": str, "prediction": str, "falsification": str}],
 "datasets": [{"name": str, "role": "primary" | "reference" | "baseline" | "ancillary",
               "observable": str, "justification": str}],
 "workflow": [{"order": int, "description": str, "agent": <existing sub-agent name>,
               "inputs": [str], "outputs": [str], "hypotheses": [ids of hypotheses this step tests]}],
 "reference_data": [str], "baselines": [str], "metrics": [str], "statistical_tests": [str],
 "assumptions": [str], "confounders": [str], "expected_outputs": [str], "stopping_criteria": [str],
 "analyses": [names from LAB ANALYSES, if listed; else []]}"""


LIT_QUERY_SCHEMA = """\
{"claims": [{"claim_id": str, "scope": "general" | "event_specific",
             "queries": [arXiv search queries; empty if event_specific]}]}"""

LIT_ASSESS_SCHEMA = """\
{"assessments": [{"claim_id": str, "verdict": "supports" | "contradicts" | "mixed" | "inconclusive",
                  "summary": <what the papers say about the claim, 1-3 sentences>,
                  "papers": [arXiv ids the verdict rests on]}]}"""


def _dump(obj: Any) -> str:
    return json.dumps(obj, indent=1, ensure_ascii=False)


def _literature_section(state: dict) -> str:
    checks = state.get(STATE_LITERATURE) or []
    if not checks:
        return ""
    return ("\nLITERATURE CHECKS (contested claims checked against arXiv papers, most-cited first; "
            "papers are retrieved evidence, not proof for this event):\n" + _dump(checks) + "\n")


def _inventory_section(state: dict) -> str:
    text = state.get(STATE_INVENTORY)
    if not text:
        return ""
    return ("\nDATA INVENTORY (retrieved by the lab from the public Copernicus Data Space catalogue "
            "before the meeting; the same for every participant):\n" + text + "\n")


def _results_section(state: dict) -> str:
    results = state.get(STATE_RESULTS)
    if not results:
        return ""
    return ("\nEXPERIMENT RESULTS (computed by the lab's fixed analyses on Sentinel Hub data, no LLM "
            "involved; cite one with kind \"verified\" and its result_id):\n" + _dump(results) + "\n")


def pi_conclude(state: dict, agent: str) -> str:
    return (
        _role_header(agent, "conclude")
        + "You are the Principal Investigator. The lab has now EXECUTED its analyses. Answer the "
        "scientific question from the EXPERIMENT RESULTS below.\n"
        "- Base every conclusion on the results' numbers. Cite each with kind \"verified\" and the "
        "result's id in 'result_id' (e.g. \"res-2\"); quote the numbers in the claim.\n"
        "- A claim no result backs is an \"assumption\". A failed analysis means that point was NOT "
        "CHECKED: say so, never treat it as evidence.\n"
        "- 'decided': the results answer the question. 'needs_experiment': they point one way but "
        "an analysis the lab cannot run (e.g. an independent reference map) is needed to settle it. "
        "'insufficient_evidence': the analyses failed or contradict each other.\n"
        "- Agreement between two maps is not accuracy: without an independent reference, say which "
        "sensor had usable data and how far the maps agree, not which is more accurate.\n"
        "- Keep any disagreement the results did not settle in 'disagreements'. Make no tool call.\n"
        + f"\nSCIENTIFIC PROBLEM:\n{_dump(state.get(STATE_PROBLEM, {}))}\n"
        + f"\nDECISION BEFORE THE EXPERIMENT:\n{_dump(state.get(STATE_DECISION, {}))}\n"
        + _results_section(state)
        + f"\n{OUTPUT_RULE}\n{CONCLUSION_SCHEMA}" + _repair_suffix(state, agent)
    )


def _repair_suffix(state: dict, agent: str) -> str:
    hint = state.get(repair_key(agent))
    if not hint:
        return ""
    return ("\n\nYOUR PREVIOUS OUTPUT WAS REJECTED by the schema validator:\n"
            f"{hint}\nReturn a corrected JSON object only.")


def today() -> str:
    return date.today().isoformat()


def _role_header(role: str, phase: str) -> str:
    # The ROLE/PHASE header also lets tests script a fake model per call.
    return (f"ROLE: {role}\nPHASE: {phase}\n"
            f"TODAY: {today()}. Dates before today are in the past: data for them may exist, so check "
            "the DATA INVENTORY or a tool instead of assuming it does not.\n")


def pi_formulate(state: dict, roster: dict[str, str], agent: str,
                 sub_agents: dict[str, str] | None = None, geocoder: str | None = "cdse_agent") -> str:
    roster_txt = "\n".join(f"- {name}: {desc}" for name, desc in roster.items())
    sub_txt = "\n".join(f"- {name}: {' '.join((desc or '').split())[:200]}"
                         for name, desc in (sub_agents or {}).items())
    return (
        _role_header(agent, "formulate")
        + "You are the Principal Investigator of a virtual Earth Observation science lab. "
        "Turn the user's request into a precise scientific problem statement.\n"
        "- Extract the question, objectives, constraints, study area, time period and expected outputs.\n"
        + (f"- If the study area is a place name without coordinates you may make ONE call to "
           f"{geocoder} to geocode it (a bbox only). If a bbox is given, make no tool call. "
           if geocoder else "- Make no tool call. ")
        + "Do NOT ask about data availability: right after this step the lab builds a DATA "
        "INVENTORY of what exists for the area and period, and the specialists use it.\n"
        + DATA_NEEDS_RULE +
        "- Select 2-4 specialists from the roster whose expertise is needed.\n"
        + ("- In 'sub_agents', list only the sub-agents (data and literature services) this problem "
           "needs; the lab offers no others during the meeting. Leave out services unrelated to the "
           "question's sensors, data or methods.\n" if sub_txt else "")
        + "- Classify the task type(s).\n\n"
        f"SPECIALIST ROSTER:\n{roster_txt}\n\n"
        + (f"SUB-AGENT ROSTER:\n{sub_txt}\n\n" if sub_txt else "")
        + f"{OUTPUT_RULE}\n{PROBLEM_SCHEMA}"
        + _repair_suffix(state, agent)
    )


def specialist(state: dict, agent: str, expertise: str, tool_names: list[str],
               max_calls: int, card: str = "") -> str:
    phase = state.get(STATE_PHASE, "propose")
    problem = _dump(state.get(STATE_PROBLEM, {}))
    direct = [t for t in tool_names if t in DIRECT_TOOLS]
    tools_txt = (
        f"You can ground claims by calling these tools and existing sub-agents: {', '.join(tool_names)}. "
        "They run in PLAN-ONLY mode: catalog search, metadata, overpass prediction and literature "
        "search work; downloads and processing are blocked. Send short, specific requests "
        f"(sensor, bbox, dates). Make at most {max_calls} calls in this step; repeating a call "
        "already made in this run is free (answered from the lab's cache).\n"
        + (f"{', '.join(direct)} query public data catalogues directly (no login) and answer in "
           "seconds: prefer them to a sub-agent for availability questions. search_catalogues "
           "searches every catalogue at once for a catalogue-style description of the data you "
           "need; search_cdse_catalog lists Copernicus scenes per day (use product_type, e.g. "
           "SLC for interferometry). Always use the study-area bbox from the problem.\n"
           if direct else "")
        + ("The DATA INVENTORY below already answers which data exist for this area and period "
           "(its DATA DISCOVERY section searched every catalogue for the problem's data needs): "
           "do NOT search for what it shows. Call a tool only for what it does not cover, or "
           "make no call at all.\n"
           if state.get(STATE_INVENTORY) else "")
        if tool_names else "You have no grounding tools in this step.\n"
    )
    head = (_role_header(agent, phase)
            + f"You are a domain scientist in a virtual EO science lab.\nEXPERTISE: {expertise}\n\n"
            + (f"YOUR DOMAIN KNOWLEDGE (curated; its references are real papers you may cite as "
               f"kind \"assumption\" with source \"arxiv:<id>\" unless a tool or literature check "
               f"retrieved them in this run):\n{card}\n\n" if card else "")
            + SCIENTIFIC_NORMS + "\n" + tools_txt
            + f"\nSENSOR CATALOG (deterministic metadata used by the lab's validator):\n{format_catalog()}\n"
            + f"\nSCIENTIFIC PROBLEM:\n{problem}\n"
            + _inventory_section(state))
    proposals: dict = state.get(STATE_PROPOSALS, {})
    critiques: list = state.get(STATE_CRITIQUES, [])

    head += _literature_section(state)
    if phase == "propose":
        body = ("\nPropose your scientific approach INDEPENDENTLY. Include competing hypotheses where "
                "the evidence admits them, the datasets you would use (with their role), methods, "
                "assumptions, limitations and evidence.\n")
        schema = PROPOSAL_SCHEMA
    elif phase == "critique":
        others = {k: v for k, v in proposals.items() if k != agent}
        body = ("\nPEER REVIEW. Critique the OTHER specialists' proposals below from your expertise: "
                "unsupported claims, physical inconsistencies, unsuitable datasets, methodological gaps. "
                "Be specific and only raise issues you can justify. Do not critique your own proposal.\n"
                f"\nPROPOSALS UNDER REVIEW:\n{_dump(others)}\n")
        schema = CRITIQUE_SCHEMA
    else:  # revise
        mine = proposals.get(agent, {})
        on_me = [c for c in critiques if c.get("target") == agent]
        others = {k: v for k, v in proposals.items() if k != agent}
        body = ("\nREVISION. Revise your proposal in light of the critiques addressed to you. For each "
                "critique either fix the issue or explain in 'limitations' why you keep your position. "
                "You may adopt ideas from other proposals but must justify them yourself. Summarise "
                "what changed in 'changes_from_previous'.\n"
                "If a LITERATURE CHECK concerns a claim in your proposal (its target is you), engage "
                "with it: cite a relevant paper in 'evidence' (kind \"retrieved\", source "
                "\"arxiv:<id>\"), or name the paper in 'limitations' and explain why you disagree.\n"
                f"\nYOUR CURRENT PROPOSAL:\n{_dump(mine)}\n"
                f"\nCRITIQUES OF YOUR PROPOSAL (from peers, the scientific critic and the deterministic "
                f"validator):\n{_dump(on_me)}\n\nOTHER PROPOSALS:\n{_dump(others)}\n")
        schema = PROPOSAL_SCHEMA
    return head + body + f"\n{OUTPUT_RULE}\n{schema}" + _repair_suffix(state, agent)


PLAN_REVIEW_CHECKLIST = """\
Check the plan against this list and raise an issue (target "experiment_plan") for each failure:
1. Each hypothesis: do statement, prediction and falsification point the SAME way? A prediction
   that contradicts its own statement makes the hypothesis untestable.
2. Each hypothesis: does at least one workflow step produce the observation that would
   discriminate it from its alternative (both sides of a comparison must be observed)?
3. Reference data: is it independent of the datasets being evaluated?
4. Metrics: can each be computed from the stated outputs against the stated reference?
5. Statistical tests: do they fit the sample (one event or site is n=1 unless pixels,
   scenes, regions or seeds are the replicated unit)?
6. Feasibility: can each step's sub-agent actually provide what the step asks for?
Use "blocking" only if the plan cannot answer the research question at all.
For every issue, "claim" must QUOTE the plan text you are criticising (a hypothesis, step,
dataset or metric), never a line of this checklist. Raise an issue only for a real failure.
Independent references include Copernicus EMS delineations, GNSS or levelling, very high
resolution imagery, agency or field reports and held-out benchmark labels. Only the EXECUTING SUB-AGENTS below can run workflow steps."""


def plan_review_checklist_lines() -> list[str]:
    """Checklist items without their numbering, for filtering echoed 'claims'."""
    return [line.split(". ", 1)[1] for line in PLAN_REVIEW_CHECKLIST.splitlines()
            if line[:1].isdigit() and ". " in line]


def critic(state: dict, agent: str, tool_names: list[str], executors: dict[str, str],
           max_calls: int = 1) -> str:
    if state.get(STATE_PHASE) == "plan_review":
        return critic_plan_review(state, agent, executors)
    tools_txt = (f"You may check a specific claim by calling: {', '.join(tool_names)}; at most "
                 f"{max_calls} call(s). Judge availability from the DATA INVENTORY first: a claim "
                 "that contradicts it is wrong without any further call.\n"
                 if tool_names and max_calls else "")
    return (
        _role_header(agent, "critique")
        + "You are the Scientific Critic of a virtual EO science lab. Your job is to find what is "
        "wrong, not to be agreeable. For EACH proposal identify: claims not backed by retrieved "
        "evidence, physical or observational inconsistencies (night/cloud/smoke, viewing geometry, "
        "resolution, revisit vs. timing), unsuitable or unavailable datasets, methodological gaps "
        "(no reference data, no baseline, unfalsifiable hypotheses), and unresolved limitations. "
        "Use 'blocking' only when the approach cannot answer the question as stated.\n\n"
        + SCIENTIFIC_NORMS + "\n" + tools_txt
        + f"\nSENSOR CATALOG:\n{format_catalog()}\n"
        + f"\nSCIENTIFIC PROBLEM:\n{_dump(state.get(STATE_PROBLEM, {}))}\n"
        + _inventory_section(state)
        + f"\nPROPOSALS:\n{_dump(state.get(STATE_PROPOSALS, {}))}\n"
        + "\nDETERMINISTIC VALIDATOR FINDINGS (already reported; do not repeat, build on them):\n"
        + f"{_dump(state.get(STATE_VALIDATION, []))}\n"
        + _literature_section(state)
        + f"\n{OUTPUT_RULE}\n{CRITIQUE_SCHEMA}" + _repair_suffix(state, agent)
    )


def critic_plan_review(state: dict, agent: str, executors: dict[str, str]) -> str:
    exec_txt = "\n".join(f"- {name}: {desc}" for name, desc in executors.items())
    return (
        _role_header(agent, "plan_review")
        + "You are the Scientific Critic reviewing the Experiment Designer's plan before the lab "
        "accepts it. Find what would make the experiment fail to answer the question.\n\n"
        + PLAN_REVIEW_CHECKLIST + "\n"
        + f"\nEXECUTING SUB-AGENTS:\n{exec_txt}\n"
        + f"\nSCIENTIFIC PROBLEM:\n{_dump(state.get(STATE_PROBLEM, {}))}\n"
        + _inventory_section(state)
        + f"\nEXPERIMENT PLAN:\n{_dump(state.get(STATE_PLAN, {}))}\n"
        + "\nDETERMINISTIC VALIDATOR FINDINGS ON THE PLAN (already reported; do not repeat):\n"
        + f"{_dump(state.get(STATE_VALIDATION, []))}\n"
        + f"\n{OUTPUT_RULE}\n{CRITIQUE_SCHEMA}" + _repair_suffix(state, agent)
    )


def literature_reviewer(state: dict, agent: str) -> str:
    phase = state.get(STATE_PHASE)
    claims = _dump(state.get(STATE_CONTESTED, []))
    if phase == "lit_query":
        return (
            _role_header(agent, "lit_query")
            + "You are the lab's literature reviewer. Specialists disagree about the claims below. "
            "For each claim decide its scope:\n"
            "- \"general\": a physical or methodological statement the literature can speak to "
            "(e.g. how smoke affects dNBR, how viewing angle affects geostationary fire detection).\n"
            "- \"event_specific\": a statement about what happened in THIS event (e.g. whether smoke "
            "covered the area on a given day). Papers cannot settle it; give no queries.\n"
            "For general claims write 1-2 arXiv search queries. arXiv matches every term joined "
            "by AND, so long AND chains return nothing. Use 2-4 key terms, OR between synonyms, "
            "at most one AND, no dates, no years, no study-area names, and quote only a short "
            "fixed term. Good: 'burned area AND (smoke OR haze)', 'InSAR AND (deep learning OR "
            "foundation model)'. "
            "Bad: '\"MODIS FRP accuracy\" AND (\"Terra/Aqua drift\" OR \"data gaps\")'.\n"
            + f"\nSCIENTIFIC PROBLEM:\n{_dump(state.get(STATE_PROBLEM, {}))}\n"
            + f"\nCONTESTED CLAIMS:\n{claims}\n"
            + f"\n{OUTPUT_RULE}\n{LIT_QUERY_SCHEMA}" + _repair_suffix(state, agent)
        )
    return (
        _role_header(agent, "lit_assess")
        + "You are the lab's literature reviewer. For each contested claim, read the retrieved "
        "papers (most-cited first; prefer the conclusions of well-cited papers when they conflict, "
        "but say so) and judge whether they support or contradict the claim. Use only what the "
        "abstracts state; if they do not address the claim, the verdict is \"inconclusive\". "
        "First check each paper's topic: a paper about another field or phenomenon (e.g. "
        "condensed matter, telecommunications) is off-topic even if it shares words with the "
        "claim; ignore it and do not cite it. If no listed paper is on-topic, the verdict is "
        "\"inconclusive\". \"contradicts\" needs an on-topic abstract stating the opposite of the "
        "claim, not merely a different finding. Never cite a paper that is not listed.\n"
        + f"\nCONTESTED CLAIMS:\n{claims}\n"
        + f"\nRETRIEVED PAPERS PER CLAIM:\n{_dump(state.get(STATE_PAPERS, {}))}\n"
        + f"\n{OUTPUT_RULE}\n{LIT_ASSESS_SCHEMA}" + _repair_suffix(state, agent)
    )


def pi_decide(state: dict, agent: str, tool_names: list[str] = (), max_calls: int = 1) -> str:
    tools_txt = (f"- Availability is answered by the DATA INVENTORY: usually make no tool call. Only "
                 f"for a fact it does not cover, you may make at most {max_calls} call(s) to: "
                 f"{', '.join(tool_names)}.\n" if tool_names and max_calls else "- Make no tool call.\n")
    return (
        _role_header(agent, "decide")
        + "You are the Principal Investigator. Synthesise the specialists' final proposals into a "
        "scientific decision.\n" + tools_txt
        + "- 'decided': one approach is defensible on the retrieved evidence and no blocking issue "
        "remains.\n"
        "- 'needs_experiment': competing hypotheses remain that only an experiment can separate.\n"
        "- 'insufficient_evidence': no reliable conclusion is possible with the available data.\n"
        "Preserve genuine disagreements in 'disagreements' instead of averaging them away. "
        "Consensus among specialists is NOT evidence of correctness.\n\n"
        + SCIENTIFIC_NORMS
        + f"\nSCIENTIFIC PROBLEM:\n{_dump(state.get(STATE_PROBLEM, {}))}\n"
        + _inventory_section(state)
        + f"\nFINAL PROPOSALS:\n{_dump(state.get(STATE_PROPOSALS, {}))}\n"
        + f"\nLATEST CRITIQUES (raised before the final revision):\n{_dump(state.get(STATE_CRITIQUES, []))}\n"
        + f"\nOPEN VALIDATOR FINDINGS ON THE FINAL PROPOSALS:\n{_dump(state.get(STATE_VALIDATION, []))}\n"
        + _literature_section(state)
        + f"\n{OUTPUT_RULE}\n{DECISION_SCHEMA}" + _repair_suffix(state, agent)
    )


def designer(state: dict, agent: str, executors: dict[str, str], lab_members: list[str] = (),
             analyses: dict[str, str] | None = None) -> str:
    exec_txt = "\n".join(f"- {name}: {desc}" for name, desc in executors.items())
    analyses_txt = ("\nLAB ANALYSES (execute mode: the lab RUNS the ones you list in 'analyses' right "
                    "after this step, on Sentinel Hub data, choosing the clearest dates itself; list "
                    "every one your hypotheses need, by exact name):\n"
                    + "\n".join(f"- {n}: {d}" for n, d in analyses.items()) + "\n") if analyses else ""
    members = ", ".join(lab_members) or "the PI, specialists and critic"
    return (
        _role_header(agent, "design")
        + "You are the Experiment Designer. Translate the lab's decision into an executable, "
        "falsifiable experiment plan. If hypotheses compete, design the experiment so its outcome "
        "DISCRIMINATES between them. Every workflow step must be executable by one of the existing "
        "sub-agents listed below (use its exact name). Include independent reference data where "
        "available, baselines/alternatives, metrics, statistical comparisons, confounders and "
        "stopping criteria. Do not invent sub-agents or datasets the catalog marks as inaccessible "
        "without flagging them as external dependencies.\n"
        "- List in each step's 'hypotheses' the ids of the hypotheses it helps test.\n"
        "- Reference data must be INDEPENDENT of the datasets under evaluation (a sensor cannot "
        "validate itself): e.g. Copernicus EMS delineations, GNSS or levelling, very high "
        "resolution imagery, agency or field reports, held-out benchmark labels, or a sensor "
        "not under test.\n"
        "- Statement, prediction and falsification of each hypothesis must point the same way.\n"
        "- A comparison hypothesis needs steps that observe BOTH sides. If one side cannot be "
        "observed with accessible data, drop the hypothesis and say why in 'assumptions'.\n"
        "- Assign each step to a sub-agent that can access its data (see 'fetch via' in the "
        "catalog).\n"
        f"- Lab members ({members}) wrote the proposals but execute nothing: a step's 'agent' "
        "must be a name from EXECUTING SUB-AGENTS, never a lab member. "
        + ("Analysis of downloaded data (spectral indices such as NBR/dNBR, clipping, zonal "
           "statistics, comparisons, maps) goes to gis_agent.\n\n" if "gis_agent" in executors else
           "Analysis without a fetching sub-agent (e.g. raster maths on downloaded data) goes to "
           "the sub-agent that supplies the data.\n\n")
        + SCIENTIFIC_NORMS
        + f"\nEXECUTING SUB-AGENTS:\n{exec_txt}\n"
        + f"\nSENSOR CATALOG:\n{format_catalog()}\n"
        + analyses_txt
        + f"\nSCIENTIFIC PROBLEM:\n{_dump(state.get(STATE_PROBLEM, {}))}\n"
        + _inventory_section(state)
        + f"\nDECISION:\n{_dump(state.get(STATE_DECISION, {}))}\n"
        + f"\nFINAL PROPOSALS:\n{_dump(state.get(STATE_PROPOSALS, {}))}\n"
        + _literature_section(state)
        + f"\n{OUTPUT_RULE}\n{PLAN_SCHEMA}" + _repair_suffix(state, agent)
    )
