"""
Offline fakes for Virtual Lab tests: a scripted LLM keyed on the ROLE/PHASE
header every lab instruction starts with, and fake "existing" sub-agents with
plain function tools, wrapped by the lab through real AgentTools. No network,
no MCP servers, no EVE endpoint.
"""

import json
import os
import re
from collections.abc import AsyncGenerator, Callable

os.environ.setdefault("AGENTICEO_SUBAGENTS", "none")  # don't load real sub-agents

from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

QUERY = (
    "For the EMSR859_1 wildfire in south-central Chile (bbox [-72.73, -37.41, -72.59, -37.22], "
    "17-18 January 2026), which sensor combination best constrains ignition time and final burned "
    "area, and what experiment would discriminate geostationary-first from VIIRS/S3-first ignition "
    "detection and Sentinel-2 dNBR from SAR-based burned area under smoke?"
)

PROBLEM = {
    "question": "Which sensors best constrain ignition time and burned area for EMSR859_1?",
    "objectives": ["Constrain ignition time", "Map final burned area"],
    "study_area": {"name": "EMSR859_1, Biobio, Chile", "bbox": [-72.73, -37.41, -72.59, -37.22]},
    "time_period": {"start": "2026-01-17", "end": "2026-01-18"},
    "constraints": ["plan only"],
    "expected_outputs": ["ignition time window", "burned-area map"],
    "task_types": ["dataset_selection", "hypothesis_testing"],
    "participants": ["thermal_specialist", "optical_specialist"],
    "rationale": "timing is a thermal question, extent an optical/SAR one",
}

THERMAL_V0 = {
    "summary": "THERMAL-V0: geostationary FCI first detection fixes ignition to 10 minutes",
    "hypotheses": [{"id": "H1", "statement": "FCI detects the fire before any polar sensor",
                    "prediction": "first FCI anomaly precedes first VIIRS detection",
                    "falsification": "VIIRS detection precedes any FCI anomaly"}],
    "datasets": [{"name": "MTG FCI", "role": "primary", "observable": "active fire ignition timing"},
                 {"name": "VIIRS", "role": "reference", "observable": "active fire detection"}],
    "methods": ["contextual fire detection on FCI 3.8 um"],
    "evidence": [{"claim": "FCI products exist for 17 Jan 2026", "kind": "retrieved",
                  "source": "eumetsat_agent"}],
    "confidence": "medium",
}

THERMAL_V1 = {
    **THERMAL_V0,
    "summary": "THERMAL-V1: VIIRS night + day passes bracket ignition; FCI dropped (disk edge)",
    "datasets": [{"name": "VIIRS", "role": "primary", "observable": "active fire detection"},
                 {"name": "Sentinel-3 SLSTR", "role": "reference", "observable": "active fire"}],
    "changes_from_previous": "Dropped FCI after the coverage finding.",
}

OPTICAL_V0 = {
    "summary": "OPTICAL-V0: Sentinel-2 dNBR maps the final burned area",
    "hypotheses": [{"id": "H1", "statement": "S2 dNBR captures the perimeter within 1 pixel",
                    "prediction": "dNBR perimeter matches EMS delineation",
                    "falsification": "IoU with EMS delineation below 0.7"}],
    "datasets": [{"name": "Sentinel-2 L2A", "role": "primary", "observable": "burned area dNBR"}],
    "methods": ["pre/post dNBR with Otsu threshold"],
    # Claims retrieval from cdse_agent, but the optical script never calls it.
    "evidence": [{"claim": "cloud-free S2 scene on 20 Jan", "kind": "retrieved", "source": "cdse_agent"}],
    "confidence": "medium",
}

CRITIC_R1 = {"critiques": [{"target": "thermal_specialist", "issues": [{
    "category": "coverage", "severity": "blocking", "claim": "FCI detects ignition",
    "rationale": "Chile is at the edge of the 0 deg disk", "suggested_fix": "use VIIRS/GOES"}]}]}

DECISION = {
    "status": "decided",
    "summary": "Use VIIRS/SLSTR for ignition timing and S2 dNBR for extent.",
    "recommended_workflow": ["VIIRS active fire", "S2 dNBR"],
    "selected_hypotheses": ["thermal-H1"],
    "disagreements": [],
    "justification": [{"claim": "FCI products exist", "kind": "retrieved", "source": "eumetsat_agent"}],
}

PLAN = {
    "research_question": "Which sensor first detects EMSR859_1 ignition?",
    "hypotheses": [{"id": "thermal-H1", "statement": "VIIRS detects first", "prediction": "p",
                    "falsification": "f"}],
    "datasets": [{"name": "VIIRS", "role": "primary", "observable": "active fire detection"}],
    "workflow": [{"order": 1, "description": "search VIIRS active fire", "agent": "gee_agent",
                  "hypotheses": ["thermal-H1"]},
                 {"order": 2, "description": "search SLSTR fire product", "agent": "eumetsat_agent",
                  "hypotheses": ["thermal-H1"]}],
    "reference_data": ["EMS EMSR859 delineation"],
    "baselines": ["MODIS MCD14"],
    "metrics": ["first-detection time difference"],
    "statistical_tests": [],
    "stopping_criteria": ["both products retrieved"],
}

_HEADER = re.compile(r"ROLE: (\S+)\nPHASE: (\S+)")


def _text(r: str | dict) -> LlmResponse:
    body = r if isinstance(r, str) else json.dumps(r)
    return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=body)]))


def _call(name: str, args: dict) -> LlmResponse:
    return LlmResponse(content=types.Content(role="model", parts=[
        types.Part(function_call=types.FunctionCall(name=name, args=args))]))


def _has_tool_result(req: LlmRequest) -> bool:
    return any(p.function_response for c in req.contents for p in (c.parts or []))


def _request_text(req: LlmRequest) -> str:
    si = req.config.system_instruction if req.config else ""
    si = si if isinstance(si, str) else json.dumps(str(si))
    parts = [p.text or "" for c in req.contents for p in (c.parts or [])]
    return si + "\n" + "\n".join(parts)


Script = Callable[[str, str, LlmRequest], LlmResponse]


def default_script(role: str, phase: str, req: LlmRequest) -> LlmResponse:
    revised = phase == "revise"
    if role == "lab_pi":
        return _text(PROBLEM if phase == "formulate" else DECISION)
    if role == "thermal_specialist":
        if phase in ("propose", "revise") and not _has_tool_result(req):
            return _call("eumetsat_agent", {"request": "FCI L1C over Chile 2026-01-17"})
        if phase == "critique":
            return _text({"critiques": [{"target": "optical_specialist", "issues": [{
                "category": "physical_inconsistency", "severity": "major",
                "claim": "S2 dNBR", "rationale": "smoke may obscure the post-fire scene"}]}]})
        return _text(THERMAL_V1 if revised else THERMAL_V0)
    if role == "optical_specialist":
        if phase == "critique":
            return _text({"critiques": []})
        return _text({**OPTICAL_V0, "summary": "OPTICAL-V1: revised"} if revised else OPTICAL_V0)
    if role == "scientific_critic":
        rnd = 1 if "THERMAL-V0" in _request_text(req) else 2
        return _text(CRITIC_R1 if rnd == 1 else {"critiques": []})
    if role == "experiment_designer":
        return _text(PLAN)
    if role == "eumetsat_agent":  # fake existing sub-agent
        if not _has_tool_result(req):
            return _call("search_eumetsat_products", {"collection": "FCI", "date": "2026-01-17"})
        return _text("Found 144 FCI L1C products for 2026-01-17.")
    if role == "cdse_agent":
        if not _has_tool_result(req):
            return _call("fetch_sentinel2", {"bbox": [-72.73, -37.41, -72.59, -37.22]})
        return _text("fetch failed")
    return _text("unscripted")


class ScriptedLlm(BaseLlm):
    """Routes each request to a script by its ROLE/PHASE header and records requests."""

    model: str = "scripted"
    script: Script = default_script
    seen: list = []

    async def generate_content_async(self, llm_request: LlmRequest, stream: bool = False
                                     ) -> AsyncGenerator[LlmResponse, None]:
        text = _request_text(llm_request)
        m = _HEADER.search(text)
        role, phase = (m.group(1), m.group(2)) if m else ("?", "?")
        self.seen.append((role, phase, text))
        yield self.script(role, phase, llm_request)


def search_eumetsat_products(collection: str, date: str) -> dict:
    """Fake read-only catalog search."""
    return {"count": 144, "collection": collection, "date": date}


def fetch_sentinel2(bbox: list[float]) -> dict:
    """Fake downloading tool; plan_only must block it."""
    raise AssertionError("fetch_sentinel2 must never execute in plan_only mode")


def fake_existing_agents(llm: ScriptedLlm) -> dict:
    """Stand-ins for the real sub-agents, named like them and, like them, single_turn
    (the mode the `multi` orchestrator gives them, which AgentTool can't run directly)."""
    def agent(name, tools):
        return LlmAgent(name=name, model=llm, description=f"fake {name}", mode="single_turn",
                        instruction=f"ROLE: {name}\nPHASE: tool\n", tools=tools)
    return {
        "eumetsat_agent": agent("eumetsat_agent", [search_eumetsat_products]),
        "cdse_agent": agent("cdse_agent", [fetch_sentinel2]),
        "gee_agent": agent("gee_agent", []),
    }


class FakeSentinelHub:
    """Offline stand-in for analyses.SentinelHubClient. The left half of the grid burns:
    NIR drops in Sentinel-2 and VH drops ~4 dB in Sentinel-1. Acquisitions every 3 days;
    the scene on `cloudy_day` is 60% cloud."""

    def __init__(self, cloudy_day: str = "2026-01-18", fail: bool = False) -> None:
        self.cloudy_day, self.fail = cloudy_day, fail
        self.requests: list[dict] = []

    async def statistics(self, body: dict) -> dict:
        import datetime as dt
        self.requests.append(body)
        if self.fail:
            raise RuntimeError("Sentinel Hub unavailable")
        rng = body["aggregation"]["timeRange"]
        day, last = dt.date.fromisoformat(rng["from"][:10]), dt.date.fromisoformat(rng["to"][:10])
        data = []
        while day <= last:
            cloud = 0.6 if day.isoformat() == self.cloudy_day else 0.0
            stats = lambda mean: {"bands": {"B0": {"stats": {"mean": mean, "sampleCount": 100, "noDataCount": 0}}}}
            data.append({"interval": {"from": f"{day}T00:00:00Z"},
                         "outputs": {"cloud": stats(cloud), "aot": stats(0.3), "dataMask": stats(1.0)}})
            day += dt.timedelta(days=3)
        return {"data": data}

    async def process(self, body: dict):
        import numpy as np
        self.requests.append(body)
        if self.fail:
            raise RuntimeError("Sentinel Hub unavailable")
        w, h = body["output"]["width"], body["output"]["height"]
        data = body["input"]["data"][0]
        post = data["dataFilter"]["timeRange"]["from"][:10] > "2026-01-18"
        burned = np.zeros((h, w), bool)
        burned[:, : w // 2] = post
        if data["type"] == "sentinel-2-l2a":
            nir = np.where(burned, 0.10, 0.35)
            return np.stack([nir, np.full((h, w), 0.15), np.full((h, w), 4.0), np.ones((h, w))]).astype("float32")
        vh = np.where(burned, 0.02, 0.05)
        return np.stack([vh, np.full((h, w), 0.2), np.full((h, w), 2.0)]).astype("float32")
