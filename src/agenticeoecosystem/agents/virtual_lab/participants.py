"""
participants.py — The lab's scientists, critic, PI and experiment designer.

Lab participants are new, thin LlmAgents: they reason and argue. They do not
duplicate any data access. Each specialist is bound to existing sub-agents
(cdse_agent, eumetsat_agent, ...) exposed as AgentTools, which it calls to
ground claims; the plugin in ledger.py keeps those calls read-only in
plan_only mode.

AgentTool runs the wrapped agent as the root of a new Runner, and ADK 2.x
(2.2 through at least 2.13) refuses a `single_turn` root agent, which is what
the existing sub-agents are (the `multi` orchestrator needs that mode). So the
lab wraps chat-mode clones. Clones share the originals' toolsets (and MCP
sessions); the originals are never modified or re-parented, so `multi` keeps
working in the same process. (Attaching single_turn clones as sub_agents, ADK's
preferred pattern, needs the node runtime, which ADK 2.2 does not use for a
custom BaseAgent root like the coordinator.)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.base_toolset import BaseToolset
from google.adk.tools.function_tool import FunctionTool

from . import prompts
from .analyses import ANALYSES
from .config import VirtualLabConfig
from .discovery import PROVIDER_LABELS, as_tool_result, direct_call, discover_for
from .literature import rank_papers_by_citations

logger = logging.getLogger(__name__)

PI = "lab_pi"
CRITIC = "scientific_critic"
DESIGNER = "experiment_designer"
LIT_REVIEWER = "literature_reviewer"

LITERATURE_AGENTS = ("arxiv_rag_agent", "graph_rag_agent", "mooc_rag_agent")
# Sub-agents that read from disk only; useless for grounding before any data exists.
_NOT_FOR_GROUNDING = {"gis_agent", "vlm_agent"}


ROLE_CARDS_DIR = Path(__file__).parent / "roles"


@dataclass(frozen=True)
class SpecialistSpec:
    name: str
    expertise: str                    # one line: what the PI sees when picking specialists
    grounding_agents: tuple[str, ...]

    @property
    def card(self) -> str:
        """Curated domain knowledge (methods, data facts, pitfalls, validation, references)
        from roles/<name>.md; empty if the role has no card."""
        return _read_card(self.name)


@cache
def _read_card(name: str) -> str:
    path = ROLE_CARDS_DIR / f"{name}.md"
    return path.read_text().strip() if path.exists() else ""


SPECIALISTS: dict[str, SpecialistSpec] = {s.name: s for s in (
    SpecialistSpec(
        "optical_specialist",
        "Optical/multispectral remote sensing: Sentinel-2 MSI, Landsat OLI/TIRS, Sentinel-3 OLCI; "
        "burned-area mapping with NBR/dNBR/RdNBR, spectral change detection, cloud and smoke screening.",
        ("cdse_agent", "gee_agent", "eo_stac_agent"),
    ),
    SpecialistSpec(
        "thermal_specialist",
        "Active-fire and thermal remote sensing: geostationary (MTG FCI, MSG SEVIRI, GOES ABI) and "
        "polar (VIIRS, MODIS, Sentinel-3 SLSTR) fire detection, FRP, detection-time uncertainty, "
        "overpass timing.",
        ("eumetsat_agent", "gee_agent", "tle_agent"),
    ),
    SpecialistSpec(
        "sar_specialist",
        "Synthetic aperture radar and InSAR: Sentinel-1 backscatter and coherence change (floods, "
        "burned area, disturbance under cloud/smoke), DInSAR and PS/SBAS time series for ground "
        "deformation, GNSS validation.",
        ("cdse_agent", "tle_agent"),
    ),
    SpecialistSpec(
        "atmosphere_specialist",
        "Weather, precipitation and atmosphere: ERA5 reanalysis (wind, humidity, temperature, "
        "precipitation), satellite precipitation (GPM IMERG, EUMETSAT H SAF / MSG-MTG rain "
        "rates), fire-weather and drought indices, smoke transport, Sentinel-5P trace gases.",
        ("climate_agent", "cdse_agent"),
    ),
    SpecialistSpec(
        "foundation_model_specialist",
        "Geospatial machine learning and EO foundation models (Prithvi, SatMAE, CROMA, DOFA, "
        "TerraMind, ...): pretraining modalities, fine-tuning protocols, benchmarks (PANGAEA, "
        "GEO-Bench), fair model comparison.",
        ("gee_agent", "eo_stac_agent"),
    ),
)}


class SelectedSubAgents(BaseToolset):
    """A participant's tools, minus the sub-agents the PI did not choose for this problem
    (session state, set after formulation). Unchosen sub-agents are never offered to the
    model, so their MCP servers never start. Other tools are always offered."""

    def __init__(self, tools: list[BaseTool], sub_agent_names: set[str]) -> None:
        super().__init__()
        self.all_tools = tools
        self._sub_agents = sub_agent_names
        self._use_invocation_cache = False   # the choice is made mid-invocation

    async def get_tools(self, readonly_context: ReadonlyContext | None = None) -> list[BaseTool]:
        return [t for t in self.all_tools if t.name in selected_tool_names(
            readonly_context.state if readonly_context else {}, [t.name], self._sub_agents)]


class PhaseTools(BaseToolset):
    """Tools that depend on the lab phase in session state (e.g. the PI geocodes while
    formulating and checks catalogues while deciding); `default` for other phases."""

    def __init__(self, by_phase: dict[str, list[BaseTool]], default: list[BaseTool]) -> None:
        super().__init__()
        self.by_phase, self.default = by_phase, default
        self.all_tools = list({id(t): t for group in [default, *by_phase.values()] for t in group}.values())
        self._use_invocation_cache = False

    async def get_tools(self, readonly_context: ReadonlyContext | None = None) -> list[BaseTool]:
        phase = readonly_context.state.get(prompts.STATE_PHASE) if readonly_context else None
        return list(self.by_phase.get(phase, self.default))


def selected_tool_names(state, names: list[str], sub_agent_names: set[str]) -> list[str]:
    chosen = state.get(prompts.STATE_SUB_AGENTS)
    if not chosen:
        return list(names)
    return [n for n in names if n not in sub_agent_names or n in chosen]


GEOCODER = "geocode"   # catalog_tools key of the PI's geocoding tool (not a data catalogue)
DISCOVERY = "discovery"   # catalog_tools key of search_catalogues (every provider)


def catalog_tools(catalogs, embedder=None) -> dict[str, list[BaseTool]]:
    """The lab's direct metadata tools (plain HTTP to public APIs, no sub-agent LLM loop), the
    same functions as the MCP servers and the data inventory use:
      - search_catalogues: federated discovery over every provider (discovery.py), ranked by
        meaning, with availability in the bbox and period;
      - search_cdse_catalog: Copernicus scene lists per day (the plan's date check reads them);
      - geocode_place (the PI's, while formulating)."""
    tools: dict[str, list[BaseTool]] = {}
    cdse = getattr(catalogs, "cdse", None)
    if catalogs:
        async def search_catalogues(query: str, bbox: list[float] | None = None,
                                    start_date: str | None = None, end_date: str | None = None,
                                    providers: list[str] | None = None) -> dict:
            """Search every data catalogue at once (Copernicus Data Space, EUMETSAT, NASA
            Earthdata, Copernicus Climate Data Store) for collections matching `query`, ranked by
            meaning, and check which have data in bbox [west, south, east, north] between
            start_date and end_date (YYYY-MM-DD). Write `query` the way a catalogue describes
            data (instrument or technique, product form, quantity), e.g. 'synthetic aperture
            radar single look complex images with phase' or 'gridded satellite precipitation
            estimates, daily'. providers: optional subset of cdse, eumetsat, earthdata, cds."""
            found = await discover_for([query], catalogs, bbox, start_date, end_date, direct_call,
                                       embedder, [p for p in (providers or []) if p in PROVIDER_LABELS])
            return as_tool_result(found)

        tools[DISCOVERY] = [FunctionTool(search_catalogues)]
    if cdse is not None and hasattr(cdse, "search_scenes"):
        async def search_cdse_catalog(collection: str, bbox: list[float], start_date: str, end_date: str,
                                      product_type: str | None = None,
                                      max_cloud_cover: float | None = None) -> dict:
            """List the Copernicus Data Space catalogue scenes intersecting bbox [west, south, east,
            north] between start_date and end_date (YYYY-MM-DD, inclusive), per acquisition day.
            collection: SENTINEL-1, SENTINEL-2, SENTINEL-3, SENTINEL-5P, LANDSAT-8 or LANDSAT-9.
            product_type: substring of the product name, e.g. SLC (Sentinel-1 with phase, needed
            for interferometry), GRD (amplitude only), MSIL2A, SL_2_FRP, SL_1_RBT. max_cloud_cover
            in percent. Metadata only: dates, product types, cloud cover, orbit."""
            return await cdse.search_scenes(collection, bbox, start_date, end_date, product_type,
                                            max_cloud_cover, limit=50)

        tools["cdse"] = [FunctionTool(search_cdse_catalog)]
    if cdse is not None and hasattr(cdse, "geocode"):
        async def geocode_place(place_name: str) -> dict:
            """Bounding box [west, south, east, north] of a named place (OpenStreetMap)."""
            return await cdse.geocode(place_name)

        tools[GEOCODER] = [FunctionTool(geocode_place)]
    return tools


@dataclass
class LabParticipants:
    pi: LlmAgent
    critic: LlmAgent | None
    designer: LlmAgent
    specialists: dict[str, LlmAgent]
    literature_reviewer: LlmAgent | None = None
    grounding: dict[str, list[str]] = field(default_factory=dict)  # agent -> tool names
    executors: dict[str, str] = field(default_factory=dict)        # sub-agent name -> description
    sub_agents: dict[str, str] = field(default_factory=dict)       # PI may choose among these

    @property
    def all_agents(self) -> list[LlmAgent]:
        agents = [self.pi, self.designer, *self.specialists.values()]
        return agents + [a for a in (self.critic, self.literature_reviewer) if a]

    @property
    def names(self) -> set[str]:
        return {a.name for a in self.all_agents}


def load_existing_agents() -> dict[str, BaseAgent]:
    """The sub-agents the `multi` pipeline uses, keyed by name; empty if none load."""
    try:
        from agenticeoecosystem.agents.subagents import ALL_SUB_AGENTS
    except Exception as exc:  # e.g. graph_rag raising on missing env vars at import
        logger.warning("Existing sub-agents unavailable, lab runs without grounding: %s", exc)
        return {}
    return {a.name: a for a in ALL_SUB_AGENTS}


def _lab_agent(name: str, description: str, instruction, llm, tools: list[BaseTool]) -> LlmAgent:
    return LlmAgent(
        name=name,
        model=llm,
        description=description,
        instruction=instruction,
        tools=tools,
        include_contents="none",       # history comes only from the instruction, built from state
        output_key=prompts.raw_key(name),
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
    )


def build_participants(
    config: VirtualLabConfig,
    llm,
    existing_agents: dict[str, BaseAgent],
    extra_tools: dict[str, list[BaseTool]] | None = None,
    catalogs=None,
    embedder=None,
) -> LabParticipants:
    """Build lab agents. `extra_tools` (agent name -> tools) lets tests inject fakes.
    With `catalogs` (inventory.Catalogs), every specialist, the critic and the deciding PI get
    the same direct metadata tools (search_catalogues over all providers, plus Copernicus scene
    lists); with CDSE available, cdse_agent is left to the PI for geocoding."""
    extra_tools = extra_tools or {}
    direct_by_provider = catalog_tools(catalogs, embedder) if catalogs else {}
    all_direct = [t for p, group in direct_by_provider.items() if p != GEOCODER for t in group]

    wrapped: dict[str, AgentTool] = {}

    def tools_for(names) -> list[BaseTool]:
        out = []
        for n in names:
            if n in existing_agents and n not in _NOT_FOR_GROUNDING:
                # Chat-mode clone: AgentTool's sub-run rejects a single_turn root agent.
                wrapped.setdefault(n, AgentTool(agent=existing_agents[n].clone(update={"mode": "chat"})))
                out.append(wrapped[n])
        return out

    literature = LITERATURE_AGENTS if config.literature_enabled else ()
    # The PI picks the sub-agents a problem needs from these (STATE_SUB_AGENTS).
    offered = {n for n in existing_agents if n not in _NOT_FOR_GROUNDING}
    # Agents that can search arXiv can also rank what they find by citations.
    citation_tool = [FunctionTool(rank_papers_by_citations)] if (
        config.literature_enabled and "arxiv_rag_agent" in existing_agents) else []
    grounding: dict[str, list[str]] = {}
    specialists: dict[str, LlmAgent] = {}
    for name in config.participants:
        spec = SPECIALISTS.get(name)
        if spec is None:
            logger.warning("Unknown specialist '%s' ignored (known: %s)", name, list(SPECIALISTS))
            continue
        direct = list(all_direct)
        agents = tuple(a for a in spec.grounding_agents
                       if not ("cdse" in direct_by_provider and a == "cdse_agent"))
        tools = direct + tools_for(agents + literature) + citation_tool + extra_tools.get(name, [])
        grounding[name] = [t.name for t in tools]

        def instruction(ctx: ReadonlyContext, _spec=spec, _tools=tuple(grounding[name])) -> str:
            return prompts.specialist(ctx.state, _spec.name, _spec.expertise,
                                      selected_tool_names(ctx.state, list(_tools), offered),
                                      max_calls=_phase_limit(config, ctx.state), card=_spec.card)

        specialists[name] = _lab_agent(name, spec.expertise, instruction, llm,
                                       [SelectedSubAgents(tools, offered)])

    roster = {n: SPECIALISTS[n].expertise for n in specialists}
    geocoder = direct_by_provider.get(GEOCODER)
    if geocoder:
        # No sub-agent (and no MCP server start-up) for the PI: it geocodes while formulating
        # and may check a catalogue directly while deciding.
        pi_formulate_tools = geocoder + extra_tools.get(PI, [])
        pi_decide_tools = all_direct + extra_tools.get(PI, [])
        pi_tools = [PhaseTools({"decide": pi_decide_tools, "conclude": []}, pi_formulate_tools)]
        grounding[PI] = [t.name for t in pi_tools[0].all_tools]
    else:
        pi_formulate_tools = pi_decide_tools = tools_for(("cdse_agent",)) + extra_tools.get(PI, [])
        pi_tools = pi_formulate_tools
        grounding[PI] = [t.name for t in pi_tools]

    sub_agent_roster = {n: existing_agents[n].description for n in sorted(offered)}

    def pi_instruction(ctx: ReadonlyContext) -> str:
        if ctx.state.get(prompts.STATE_PHASE) == "conclude":
            return prompts.pi_conclude(ctx.state, PI)
        if ctx.state.get(prompts.STATE_PHASE) == "decide":
            return prompts.pi_decide(ctx.state, PI, [t.name for t in pi_decide_tools],
                                     config.max_decide_tool_calls)
        return prompts.pi_formulate(ctx.state, roster, PI, sub_agent_roster,
                                    geocoder=pi_formulate_tools[0].name if pi_formulate_tools else None)

    pi = _lab_agent(PI, "Principal Investigator: frames the problem and makes the decision.",
                    pi_instruction, llm, pi_tools)

    executors = {n: a.description for n, a in existing_agents.items()}
    critic = None
    if config.critic_enabled:
        critic_tools = all_direct + tools_for(literature) + citation_tool + extra_tools.get(CRITIC, [])
        grounding[CRITIC] = [t.name for t in critic_tools]
        critic = _lab_agent(
            CRITIC, "Scientific Critic: challenges unsupported claims and inconsistencies.",
            lambda ctx: prompts.critic(ctx.state, CRITIC,
                                       selected_tool_names(ctx.state, grounding[CRITIC], offered),
                                       executors, max_calls=config.max_critique_tool_calls),
            llm, [SelectedSubAgents(critic_tools, offered)])

    designer = _lab_agent(
        DESIGNER, "Experiment Designer: turns the decision into an executable experiment plan.",
        lambda ctx: prompts.designer(ctx.state, DESIGNER, executors,
                                     [PI, *specialists, CRITIC, LIT_REVIEWER],
                                     analyses=ANALYSES if config.mode == "execute" else None), llm,
        extra_tools.get(DESIGNER, []))

    # Searching is done by the coordinator, so the reviewer only reads and judges.
    reviewer = _lab_agent(
        LIT_REVIEWER, "Literature reviewer: turns contested claims into arXiv queries and judges "
        "what the most-cited retrieved papers say.",
        lambda ctx: prompts.literature_reviewer(ctx.state, LIT_REVIEWER), llm, []
    ) if config.literature_check else None

    return LabParticipants(pi=pi, critic=critic, designer=designer, specialists=specialists,
                           literature_reviewer=reviewer, grounding=grounding, executors=executors,
                           sub_agents=sub_agent_roster)


def _phase_limit(config: VirtualLabConfig, state) -> int:
    """Tool calls a specialist may make in the current phase (the guard enforces it)."""
    return {"critique": config.max_critique_tool_calls,
            "revise": config.max_revise_tool_calls}.get(state.get(prompts.STATE_PHASE),
                                                        config.max_proposal_tool_calls)
