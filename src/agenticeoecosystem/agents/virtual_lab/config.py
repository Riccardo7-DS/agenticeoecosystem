"""
config.py — Virtual Lab settings and the EVE-backed model used by its participants.

Every knob can be set from the environment (VIRTUAL_LAB_*) or passed directly,
so the same lab can be swept over rounds/participants in a benchmark run.
"""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, Field

# Leaf MCP tools that only read catalogs/metadata/literature. In plan_only mode
# every other leaf tool (fetch_*, download_*, GIS processing, ...) is blocked,
# so grounding can never trigger a data download.
READ_ONLY_TOOLS: frozenset[str] = frozenset({
    # cdse_agent
    "geocode_place", "reverse_geocode_place", "resolve_named_region", "search_cdse_catalog",
    "list_openeo_collections",
    # lab direct metadata tools (participants.catalog_tools) and the inventory's discovery
    "search_catalogues", "list_stac_collections", "search_stac_items", "list_cds_datasets",
    "list_eumetsat_collections", "count_eumetsat_products", "search_earthdata_collections",
    "search_earthdata_granules", "cds_dataset_info",
    # eumetsat_agent
    "list_eumetsat_products", "search_eumetsat_products",
    # gee_agent
    "search_gee_catalog", "get_lat_lon",
    # climate_agent
    "list_era5_single_level_variables", "list_era5_pressure_level_variables", "list_cds_datasets",
    # eo_stac_agent
    "list_narratives", "search_narratives", "get_narrative_content", "list_themes",
    "get_narrative_indicators", "search_collections", "get_collection_items",
    "find_narratives_for_indicator",
    # tle_agent
    "get_satellite_position", "get_positions_for_set", "get_target_overpasses",
    "get_acquisition_schedule",
    # arxiv_rag_agent
    "search_papers", "get_abstract", "list_papers", "semantic_search",
    # virtual_lab.literature (Semantic Scholar citation counts)
    "rank_papers_by_citations",
    # graph_rag_agent
    "search_eoss", "query_ontology", "list_ontology_nodes",
    # mooc_rag_agent
    "search_moocs",
})


def _env_list(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    if not raw:
        return default
    return [s.strip() for s in raw.split(",") if s.strip()]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    return default if raw is None else raw.strip().lower() in ("1", "true", "yes", "on")


class VirtualLabConfig(BaseModel):
    max_rounds: int = Field(default=2, ge=0, description="Deliberation rounds (critique + revise).")
    participants: list[str] = Field(
        default_factory=lambda: ["optical_specialist", "thermal_specialist", "sar_specialist",
                                 "atmosphere_specialist", "foundation_model_specialist"],
        description="Specialists the PI may select from (it picks 2-4 per question).",
    )
    critic_enabled: bool = True
    peer_review: bool = True
    literature_enabled: bool = True
    literature_check: bool = Field(
        default=True,
        description="When claims are contested, search arXiv, rank hits by citations and have a "
                    "literature reviewer assess them before revision and before the final decision.",
    )
    literature_backend: Literal["arxiv", "scholar"] = Field(
        default="arxiv",
        description="Search behind the literature check: 'arxiv' (arxiv_rag_agent's MCP server) or "
                    "'scholar' (scholar-search-mcp: Semantic Scholar + arXiv; papers without an "
                    "arXiv id are dropped; SEMANTIC_SCHOLAR_API_KEY raises its rate limit).")
    literature_max_claims: int = Field(default=3, ge=1, description="Contested claims checked per check.")
    literature_queries_per_claim: int = Field(default=2, ge=1)
    literature_results_per_query: int = Field(default=10, ge=1, le=50)
    literature_papers_per_claim: int = Field(default=3, ge=1, description="Most-cited papers kept per claim.")
    literature_categories: list[str] = Field(
        default_factory=lambda: ["physics.ao-ph", "physics.geo-ph", "eess.IV", "eess.SP", "cs.CV"],
        description="arXiv categories searched (any of); empty searches all of arXiv. "
                    "Ignored by the 'scholar' backend.",
    )
    literature_min_terms: int = Field(
        default=2, ge=0, description="Query terms a paper's title/abstract must mention to be kept "
                                     "before citation ranking (0 disables the filter).")
    mode: Literal["plan_only", "execute"] = Field(
        default="plan_only",
        description="plan_only: the meeting ends with a plan. execute: the lab then runs its fixed "
                    "analyses (analyses.py, Sentinel Hub on CDSE) and the PI concludes from the "
                    "results. Lab agents are read-only in both modes.",
    )
    inventory_enabled: bool = Field(
        default=True,
        description="After problem formulation, the coordinator lists the openEO collections and "
                    "searches the CDSE catalogue for the study area itself (no LLM, no credits) and "
                    "shows every participant the result as the lab's DATA INVENTORY.")
    inventory_pad_days: int = Field(default=14, ge=0, description="Days added before and after "
                                    "the problem's time period for Sentinel-1/2 and Landsat scene "
                                    "searches (pre-/post-event images).")
    parallel: bool = Field(default=True, description="Run specialists concurrently within a phase.")
    max_tool_calls: int = Field(default=30, ge=0, description="Grounding tool calls the lab's agents "
                                "(and their sub-agents) make across the run. The data inventory "
                                "and the literature check have their own allowance; blocked and "
                                "cached calls are not counted.")
    max_literature_tool_calls: int = Field(default=18, ge=0, description="arXiv searches and "
                                           "citation lookups the literature checks may make across "
                                           "the run, separate from max_tool_calls so they cannot "
                                           "starve the PI's decision.")
    literature_timeout_s: float = Field(default=45.0, gt=0, description="Seconds one arXiv search "
                                        "or citation lookup may take. A timed-out search finds no "
                                        "papers, and the rest of that check's searches are skipped "
                                        "(arXiv is likely down), so an outage cannot stall the meeting.")
    max_literature_tool_calls_per_check: int = Field(default=6, ge=0, description="Literature "
                                                     "tool calls one check may make, so the first "
                                                     "check cannot use the whole allowance.")
    max_formulate_tool_calls: int = Field(default=1, ge=0, description="Tool calls the PI may make "
                                          "while formulating the problem (geocoding a place name; "
                                          "data availability is the inventory's job).")
    max_decide_tool_calls: int = Field(default=1, ge=0, description="Tool calls the PI may make "
                                       "per decision step (the inventory already answers availability).")
    max_proposal_tool_calls: int = Field(default=3, ge=0, description="Tool calls one specialist "
                                         "may make for its first proposal (leaf calls inside "
                                         "sub-agents are capped separately).")
    max_critique_tool_calls: int = Field(default=1, ge=0, description="Tool calls one participant "
                                         "may make while critiquing.")
    max_revise_tool_calls: int = Field(default=2, ge=0, description="Tool calls one specialist "
                                       "may make while revising.")
    max_sub_agent_tool_calls: int = Field(default=3, ge=0, description="Tool calls one sub-agent "
                                          "run (one AgentTool call) may make.")
    cache_tool_calls: bool = Field(default=True, description="Answer a repeated read-only call "
                                   "from the earlier result (catalogue searches also from a "
                                   "complete earlier search covering the same window).")
    next_round_only_if_open: bool = Field(
        default=True,
        description="Run another deliberation round only if the last critique raised a blocking "
                    "issue or the validator still finds blocking/major issues in the revised "
                    "proposals; otherwise go straight to the decision.")
    max_llm_calls: int = Field(default=100, ge=1)
    max_repair_attempts: int = Field(default=1, ge=0)
    allowed_tools: frozenset[str] = READ_ONLY_TOOLS
    temperature: float = 0.0
    max_tokens: int = 4096
    model: str = Field(default_factory=lambda: os.getenv("VIRTUAL_LAB_MODEL")
                       or os.getenv("MULTI_AGENT_MODEL", "EVE-Instruct"))

    @classmethod
    def from_env(cls, **overrides) -> VirtualLabConfig:
        base = cls()
        values = {
            "max_rounds": int(os.getenv("VIRTUAL_LAB_MAX_ROUNDS", base.max_rounds)),
            "participants": _env_list("VIRTUAL_LAB_PARTICIPANTS", base.participants),
            "critic_enabled": _env_bool("VIRTUAL_LAB_CRITIC", base.critic_enabled),
            "peer_review": _env_bool("VIRTUAL_LAB_PEER_REVIEW", base.peer_review),
            "literature_enabled": _env_bool("VIRTUAL_LAB_LITERATURE", base.literature_enabled),
            "literature_check": _env_bool("VIRTUAL_LAB_LITERATURE_CHECK", base.literature_check),
            "literature_backend": os.getenv("VIRTUAL_LAB_LITERATURE_BACKEND", base.literature_backend).strip().lower(),
            "parallel": _env_bool("VIRTUAL_LAB_PARALLEL", base.parallel),
            "inventory_enabled": _env_bool("VIRTUAL_LAB_INVENTORY", base.inventory_enabled),
            "mode": os.getenv("VIRTUAL_LAB_MODE", base.mode),
            "max_tool_calls": int(os.getenv("VIRTUAL_LAB_MAX_TOOL_CALLS", base.max_tool_calls)),
            "max_literature_tool_calls": int(os.getenv("VIRTUAL_LAB_MAX_LITERATURE_TOOL_CALLS",
                                                       base.max_literature_tool_calls)),
            "literature_timeout_s": float(os.getenv("VIRTUAL_LAB_LITERATURE_TIMEOUT_S",
                                                    base.literature_timeout_s)),
            "next_round_only_if_open": _env_bool("VIRTUAL_LAB_NEXT_ROUND_ONLY_IF_OPEN",
                                                 base.next_round_only_if_open),
            "max_llm_calls": int(os.getenv("VIRTUAL_LAB_MAX_LLM_CALLS", base.max_llm_calls)),
            "max_tokens": int(os.getenv("VIRTUAL_LAB_MAX_TOKENS", base.max_tokens)),
        }
        values.update(overrides)
        return cls(**values)


def build_eve_llm(config: VirtualLabConfig):
    """LiteLlm pointed at the EVE gateway (EVE_ENDPOINT, Basic Auth), independent of
    MULTI_AGENT_PROVIDER so the lab always runs on EVE."""
    from google.adk.models.lite_llm import LiteLlm
    from agenticeoecosystem.agents.subagents.llm_init import _basic_auth_header

    endpoint = os.getenv("EVE_ENDPOINT")
    if not endpoint:
        raise RuntimeError("EVE_ENDPOINT is not set; the Virtual Lab runs on the EVE gateway.")
    return LiteLlm(
        model=f"openai/{config.model}",
        base_url=endpoint,
        headers=_basic_auth_header(),
        api_key="EMPTY",
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
