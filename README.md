
# Agentic EO Ecosystems

This project explores the integration of **EVE (Earth Virtual Expert)**, a Large Language Model specialised in **Earth Observation (EO)** and Earth Sciences, developed by Pi School in collaboration with ESA.

The primary goal is to enable EVE to connect ESA's EO platforms and data clusters, creating an **integrated ecosystem for the EO community**. This approach leverages **agentic capabilities**, allowing EVE to dynamically interact with EO tools, databases, and APIs to reason and act autonomously.

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Layout](#repository-layout)
3. [Installation](#installation)
4. [Running the Client](#running-the-client)
5. [MCP Servers](#mcp-servers)
6. [Benchmarking](#benchmarking)
7. [Adding Custom Sub-agents](#adding-custom-sub-agents)

---

## Overview

This project implements a **multi-agent orchestrator architecture** (built on [google-adk](https://github.com/google/adk-python), see [`agents/multi_agent.py`](src/agenticeoecosystem/agents/multi_agent.py)) in which an orchestrator agent routes each user query to one or more specialised sub-agents. This is the current, default system (`--pipeline multi`, used by both the interactive client and the benchmark). Each sub-agent connects to its own MCP server:

- **CDSE agent** (`cdse_agent`) — the primary source for satellite quicklooks/GeoTIFFs: Sentinel-1/2/3, Sentinel-5P, Landsat, and DEM via the Copernicus Data Space Ecosystem (Sentinel Hub), plus place name ↔ bbox/lat-lon geocoding (OpenStreetMap Nominatim, free)
- **EUMETSAT agent** (`eumetsat_agent`) — searches and downloads geostationary products from EUMETSAT's Data Store: MTG FCI and MSG SEVIRI only
- **GEE agent** (`gee_agent`) — Google Earth Engine, used only for collections CDSE/EUMETSAT don't cover (e.g. NASA VIIRS/MODIS, JAXA GPM/GSMaP or ALOS); includes a live GEE STAC catalog search so it doesn't rely on memorised, possibly stale dataset IDs
- **Climate agent** (`climate_agent`) — ERA5/ERA5-Land reanalysis data (monthly, hourly, daily-statistics) via the Copernicus Climate Data Store, plus local NetCDF processing (temporal aggregation, unit conversion, spatial clipping)
- **EO Dashboard agent** (`eo_stac_agent`) — browses narratives, themes, and STAC collections from the ESA/NASA/JAXA [EO Dashboard](https://eodashboard.org/)
- **EOSS GraphRAG agent** (`graph_rag_agent`) — retrieves from the ESA EO Science Strategy via GraphRAG: semantic vector search plus a Memgraph knowledge graph (science themes, science questions, strategic objectives, enabling tools, tipping points, treaties, KAOs). Backed by a remote MCP server (the `graphrag` sibling repo)
- **MOOCs agent** (`mooc_rag_agent`) — semantic search over Earth Observation MOOC (online course) transcripts, backed by a local Qdrant store
- **ArXiv agent** (`arxiv_rag_agent`) — searches and retrieves scientific papers from ArXiv (semantic search, abstracts, full-text download)
- **GIS agent** (`gis_agent`) — spatial analysis, raster processing, coordinate transforms, map visualization, and spectral indices (NDVI, EVI, NDWI, NDMI, NBR) on rasters already on disk (it doesn't fetch imagery itself — that's CDSE/GEE/EUMETSAT's job)
- **VLM agent** (`vlm_agent`) — vision-language reasoning over image files already on disk (e.g. "does this scene show cloud cover", "what changed between these two images"). Runs its own fixed model (Qwen3-VL) via a separately-hosted vLLM server, independent of whichever model is driving the orchestrator and the other sub-agents — see [VLM (Vision-Language Model)](#vlm-vision-language-model) below.

> An older architecture (LangChain supervisor + sub-agents, [`client/client.py`](src/agenticeoecosystem/client/client.py)) is still used by `--pipeline single`/`multi_legacy` and by the interactive client fallback. It has a smaller, different agent set (7 agents, including an **EVE retrieval agent** for the EVE knowledge base over AWS Bedrock, which the current `multi` pipeline does not have) and is superseded by the architecture described above for benchmarking purposes.

Using the **[MCP protocol](https://modelcontextprotocol.io/)**, sub-agents can perform multi-step reasoning: for example, geocoding a place name, fetching the matching Sentinel-2 scene, and computing NDVI over it.

```mermaid
graph TD

    Start([Start])
    End([End])

    Start --> Orchestrator

    Orchestrator --> CDSE[cdse_agent]
    Orchestrator --> EUMETSAT[eumetsat_agent]
    Orchestrator --> GEE[gee_agent]
    Orchestrator --> Climate[climate_agent]
    Orchestrator --> EO_STAC[eo_stac_agent]
    Orchestrator --> RAG[graph_rag_agent]
    Orchestrator --> MOOC[mooc_rag_agent]
    Orchestrator --> ArXiv[arxiv_rag_agent]
    Orchestrator --> GIS[gis_agent]
    Orchestrator --> VLM[vlm_agent]

    CDSE --> CDSE_MCP["CDSE MCP\n(Sentinel Hub imagery + geocoding)"]
    CDSE_MCP --> CDSE

    EUMETSAT --> EUMETSAT_MCP["EUMETSAT MCP\n(MTG FCI / MSG SEVIRI products)"]
    EUMETSAT_MCP --> EUMETSAT

    GEE --> GEE_MCP["GEE MCP\n(satellite image download)"]
    GEE_MCP --> GEE

    Climate --> CLIMATE_MCP["CDS MCP\n(ERA5 reanalysis)"]
    CLIMATE_MCP --> Climate

    EO_STAC --> EO_MCP["EO Dashboard MCP\n(narratives, themes, STAC collections)"]
    EO_MCP --> EO_STAC

    RAG --> GRAPHRAG_MCP["EOSS GraphRAG MCP\n(vector search + Memgraph knowledge graph)"]
    GRAPHRAG_MCP --> RAG

    MOOC --> MOOC_MCP["MOOCs MCP\n(local Qdrant, course transcripts)"]
    MOOC_MCP --> MOOC

    ArXiv --> ARXIV_MCP["ArXiv MCP\n(papers, abstracts, semantic search)"]
    ARXIV_MCP --> ArXiv

    GIS --> GIS_MCP["GIS MCP\n(geometry, raster, visualization)"]
    GIS_MCP --> GIS

    VLM --> VLM_MCP["VLM MCP\n(image loading) + Qwen3-VL"]
    VLM_MCP --> VLM

    CDSE --> Orchestrator
    EUMETSAT --> Orchestrator
    GEE --> Orchestrator
    Climate --> Orchestrator
    EO_STAC --> Orchestrator
    RAG --> Orchestrator
    MOOC --> Orchestrator
    ArXiv --> Orchestrator
    GIS --> Orchestrator
    VLM --> Orchestrator

    Orchestrator --> End
```

The orchestrator routes each user query to one or more specialised sub-agents. Each sub-agent calls its MCP server(s) and returns a synthesised result. The orchestrator passes the result through to the user.

---

## Repository Layout

```
AgenticEOEcosystem/
├── agenticeoecosystem/     ← this repo
└── graphrag/               ← EOSS GraphRAG pipeline + MCP server (sibling)
```

The `graphrag` repository hosts the EOSS GraphRAG MCP server, which the client reaches over HTTP (not as a local subprocess). If `GRAPHRAG_MCP_URL` / `GRAPHRAG_AUTH_HEADER` are unset or the server is unreachable, the EOSS GraphRAG server is skipped gracefully at startup — every other server continues to work. Each optional server is registered independently, so a missing credential only disables that one tool.

```
agenticeoecosystem/
├── src/agenticeoecosystem/
│   ├── agents/         ← sub-agent definitions and system prompts
│   ├── client/         ← supervisor + interactive client
│   ├── config/         ← MCP config, credentials
│   ├── evaluation/     ← benchmarking framework
│   ├── servers/        ← MCP server implementations
│   └── utils/          ← registry, logging, helpers
├── tests/
│   └── benchmarking.py ← benchmark entry point
├── data/tasks/         ← task JSON files
├── output/ESAEOBench/  ← benchmark results (CSV, HTML, JSON) — the actual deliverable
└── tool_output/        ← raw MCP tool downloads (CDSE, EUMETSAT, ERA5, GEE, ArXiv, GIS)
```

`output/` and `tool_output/` are deliberately separate: `output/` holds the benchmark's own results plus whatever a task's reference solution explicitly asks a tool to save under `output/<task_ID>/...` (see [`utils/output_paths.py`](src/agenticeoecosystem/utils/output_paths.py) — this convention is unchanged and still anchored to `output/`, including its `output/misc/` catch-all for stray paths). `tool_output/` holds each MCP server's *default* download location — where files land when a model doesn't specify a path at all (e.g. `CDSE_OUTPUT_DIR`, `EUMETSAT_OUTPUT_DIR`). Keeping these separate means `output/` doesn't get buried under tens of gigabytes of raw satellite/climate downloads.

---

## Installation

For model reruns using the EVE API or OpenRouter, see
[the colleague setup and rerun guide](docs/candidate_reruns.md). It includes
portable task inputs, credentials, one-model commands and checkpoint resuming.

**Requirements:** Python 3.12+

Clone the repository:
```bash
git clone https://gitlab.esa.int/PhiLab/internal/agenticeoecosystem.git
cd agenticeoecosystem
```

### Option 1: Using uv (recommended)

Install uv if not already installed:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install dependencies and the local package:
```bash
uv sync
```

### Option 2: Standard pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Option 3: Conda

```bash
conda create -n multi_agent python=3.12
conda activate multi_agent
pip install -e .
```

---

## Running the Client

The main entry point is `src/agenticeoecosystem/client/client.py`. Startup takes a few minutes — the MCP servers need to initialise and tool discovery runs at import time. Subsequent queries in the same session are fast.

> Note: this interactive client runs the older LangChain-based architecture described in the box under [Overview](#overview) (7 agents, including EVE retrieval, no CDSE/EUMETSAT/Climate/VLM). For the current, larger sub-agent set used by the benchmark, run the `multi` pipeline via `scripts/run_generate.py` or `tests/benchmarking.py` instead (see [Benchmarking](#benchmarking)).

```bash
python src/agenticeoecosystem/client/client.py
```

By default the client uses **EVE** as the LLM backend. To override:

```bash
python src/agenticeoecosystem/client/client.py --model <model_name> --provider <provider>
```

Once loaded, the client enters an interactive loop. Type your query and press Enter. Type `quit` to exit.

```
Query (or 'quit' to exit): Is there a narrative about COVID-19?
```

Each session writes a timestamped log to `logs/client_YYYYMMDD_HHMMSS.log`.

---

## MCP Servers

The sections below cover every MCP server in the repo; which ones are actually reachable depends on which pipeline you run. In the legacy `single`/`multi_legacy` pipelines, `client.py` registers up to nine servers at startup via [`mcp_config.py`](src/agenticeoecosystem/config/mcp_config.py) — seven back a dedicated sub-agent (EO Dashboard, GEE, EVE retrieval, EOSS GraphRAG, MOOCs, ArXiv, GIS); two more (EUMETSAT Data Store and Climate/CDS) load their tools when credentials are present but aren't wired to a sub-agent there, so they're unreachable from that pipeline's supervisor. The current `multi` pipeline (`agents/multi_agent.py`) wires up ten servers instead, each behind its own dedicated sub-agent, including EUMETSAT and Climate/CDS (now reachable) plus two servers the legacy pipeline doesn't have at all — CDSE and VLM. Either way, each server is registered independently — if its dependency or credential is missing, that one server/sub-agent is skipped with a warning and the client still starts.

### EO Dashboard

No setup required. The server fetches data from public ESA EO Dashboard endpoints and exposes tools for browsing narratives, themes, and STAC collections. It starts automatically as a stdio subprocess with the client.

**Exposing it externally** (e.g. as an MCP connector on the ASK ESA / Mistral platform): run it in streamable-HTTP mode instead, via the NEOHPC gateway — same pattern as EOSS GraphRAG below.

```bash
cd agenticeoecosystem
screen -dmS eodashboard-http bash -c 'source .venv/bin/activate && python src/agenticeoecosystem/servers/EOdashboard/eodashboard_server.py --transport streamable-http --port 13000; exec bash'
```

Connector URL: `https://phi-gateway.tce.ples.dedyn.io/neohpc-phi-gpu-01/13000/mcp`, same gateway Basic Auth credential as GraphRAG below. `--transport` defaults to `stdio`, so the local client's usage (`mcp_config.py`) is unaffected by running this in parallel.

### CDSE (Copernicus Data Space Ecosystem)

The primary source for satellite quicklooks/GeoTIFFs in the current (`multi`) pipeline — used by `cdse_agent`. Backed by `src/agenticeoecosystem/servers/cdse_geocode/server.py`, which wraps the Sentinel Hub API. Covers Sentinel-1 GRD (SAR), Sentinel-2 (L1C/L2A optical), Sentinel-3 (OLCI, SLSTR, SYN L2), Sentinel-5P (trace gases), Landsat (OT L1), and DEM elevation — each via its own `fetch_*` tool with sensor-appropriate presets (e.g. `true_color`, `ndvi`, `elevation_color`) plus a custom-bands/evalscript escape hatch. Also exposes `geocode_place` / `reverse_geocode_place` (place name ↔ bbox/lat-lon via OpenStreetMap Nominatim — free, no API key) and `resolve_named_region` for macro-regions (e.g. "the Sahel") that Nominatim doesn't resolve as a single place.

Requires a Sentinel Hub OAuth client from the [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/) (free registration → Dashboard → create an OAuth client):

```env
CDSE_CLIENT_ID=your-client-id
CDSE_CLIENT_SECRET=your-client-secret
```

Downloads land under `tool_output/cdse_geocode/` by default (override with `CDSE_OUTPUT_DIR`). It starts automatically as a stdio subprocess with the client.

### Google Earth Engine (GEE)

In the current (`multi`) pipeline, `gee_agent` is the **secondary** imagery source — the orchestrator only routes to it for collections CDSE/EUMETSAT don't cover (e.g. NASA VIIRS/MODIS, JAXA GPM/GSMaP or ALOS). It includes a `search_gee_catalog` tool that queries the live GEE STAC catalog for the exact, current Earth Engine asset ID (dataset IDs are frequently re-versioned, so a memorised ID is often stale) rather than relying only on a small hardcoded list of known-safe dataset IDs (Sentinel-2 SR, Landsat 8 SR, Sentinel-5P NO2).

Enables satellite image downloads via the [Earth Engine Python API](https://developers.google.com/earth-engine/guides/python_install). One-time setup:

1. **Create a Google account** and sign up for [Google Earth Engine](https://earthengine.google.com/). Academic/research access is free.
2. **Create a Google Cloud project** at [console.cloud.google.com](https://console.cloud.google.com) and note the **Project ID**.
3. **Enable the Earth Engine API**: APIs & Services → Enable APIs → search *Earth Engine API*.
4. **Create a service account**: IAM & Admin → Service Accounts → Create. Assign the role *Earth Engine Resource Writer*.
5. **Register with Earth Engine**: [code.earthengine.google.com/register](https://code.earthengine.google.com/register), select your Cloud project, complete registration.
6. **Download a JSON key**: Service Accounts → your account → Keys → Add Key → JSON.
7. Add to `.env`:

```env
GEE_PROJECT=your-gcp-project-id
GEE_SERVICE_ACCOUNT=your-service-account@your-project.iam.gserviceaccount.com
GEE_PRIVATE_KEY_PATH=/absolute/path/to/your-key.json
GOOGLE_MAPS_API_KEY=your-google-maps-api-key   # used by get_lat_lon to geocode place names
```

### EVE Knowledge Retrieval

Provides retrieval-augmented access to the EVE knowledge base. **Only used by the `single`/`multi_legacy` pipelines** (`client/client.py`'s `get_eve_knowledge` tool) — the current default `multi` pipeline (`agents/multi_agent.py`) has no equivalent sub-agent. Requires OAuth2 credentials:

```env
EVE_RETRIEVAL_CLIENT_ID=your-client-id
EVE_RETRIEVAL_CLIENT_SECRET=your-client-secret
```

Contact the EVE platform team to obtain these. If not set, the server is skipped and `get_eve_knowledge` will be unavailable.

### EO Science Strategy (EOSS GraphRAG)

Provides GraphRAG retrieval over the **ESA Earth Observation Science Strategy** — combining semantic vector search with a **Memgraph** knowledge graph that links document chunks to domain concepts (science themes, science questions, strategic objectives, enabling tools, tipping points, treaties, and KAOs). It exposes three tools: `search_eoss` (natural-language GraphRAG search), `query_ontology` (read-only Cypher), and `list_ontology_nodes` (enumerate nodes by label).

Unlike the other servers, this one runs as a **standalone HTTP MCP server** in the `graphrag` sibling repo, and the client connects to it remotely. Point the client at it via `.env`:

```env
GRAPHRAG_MCP_URL=https://phi-gateway.tce.ples.dedyn.io/neohpc-phi-gpu-01/12000/mcp
GRAPHRAG_AUTH_HEADER=Basic <base64(user:pass)>
```

`GRAPHRAG_MCP_URL` is the gateway URL of the running GraphRAG server (must end in `/mcp`). `GRAPHRAG_AUTH_HEADER` is the full `Authorization` header value; use `graphrag/gen_auth_header.py` to generate it.

**Start the server** (from the `graphrag` repo, e.g. inside a `screen` session), after loading the Memgraph graph via the numbered pipeline scripts (`00_schema.py` … `03_link_chunks.py`):

```bash
uv run python 05_mcp_server.py            # listens on 0.0.0.0:12000
```

If the server is unreachable or the env vars are unset, the EOSS GraphRAG tools are skipped and the client starts without them.

### EO MOOCs

Provides semantic search over Earth Observation MOOC (online course) transcripts via the `search_moocs` tool. The server uses a **local, file-based Qdrant store** bundled in `src/agenticeoecosystem/servers/moocs/qdrant_storage/` — no cloud service or credentials required. On first launch it embeds and indexes the bundled `data/MOOC.jsonl`; subsequent starts reuse the existing collection. It starts automatically as a stdio subprocess with the client.

**Exposing it externally** (e.g. as an MCP connector on the ASK ESA / Mistral platform): run it in HTTP mode via the NEOHPC gateway. The local Qdrant store takes an exclusive file lock, so a second process can't share the same `qdrant_storage/` directory as the stdio instance above — point it at a separate `--storage-path` instead (auto-ingested on first run, ~30s; instant afterwards):

```bash
cd agenticeoecosystem
screen -dmS moocs-http bash -c 'source .venv/bin/activate && python src/agenticeoecosystem/servers/moocs/server.py --transport http --port 14000 --storage-path src/agenticeoecosystem/servers/moocs/qdrant_storage_http; exec bash'
```

Connector URL: `https://phi-gateway.tce.ples.dedyn.io/neohpc-phi-gpu-01/14000/mcp`, same gateway Basic Auth credential as EOSS GraphRAG above.

### ArXiv

Provides access to scientific papers via the [arxiv-mcp-server](https://github.com/blazickjp/arxiv-mcp-server). No API key required. The server runs in an isolated uv tool environment.

Pre-install (one-time, speeds up the first connection):
```bash
uv tool install arxiv-mcp-server
```

Papers fetched during a session are cached in `tool_output/arxiv/` (both the current `multi` pipeline and the legacy `single`/`multi_legacy` one point at this same location). Available tools include `search_papers`, `get_abstract`, `download_paper`, `semantic_search`, `citation_graph`, and `watch_topic`. `arxiv_rag_agent`'s system prompt caps usage to avoid context blowups: at most one `download_paper` and at most five `get_abstract` calls per task, preferring the lighter discovery tools otherwise.

### GIS Analysis

Provides spatial analysis tools via [gis-mcp](https://github.com/mahdin75/gis-mcp): geometry operations (Shapely), vector file processing (GeoPandas), raster analysis (Rasterio), spatial statistics (PySAL), and map visualisation (Folium/matplotlib). **`gis_agent` only performs analysis on files already on disk — it cannot fetch imagery itself.** gis-mcp's own data-download tools (`download_worldcover`, `compute_s2_ndvi`, `download_boundaries`, `download_climate_data`, `download_street_network`, `download_species_occurrences`, etc.) are explicitly filtered out of its toolset (`_GIS_DOWNLOAD_TOOL_NAMES` in [`sub_agent_gis.py`](src/agenticeoecosystem/agents/subagents/sub_agent_gis.py)) since that responsibility now belongs to the CDSE/GEE/EUMETSAT/Climate sub-agents — routing a "download X" request here will fail.

On top of gis-mcp's own `compute_ndvi`, the sub-agent adds four spectral-index tools implemented directly in `sub_agent_gis.py` (not part of the third-party package): `compute_evi`, `compute_ndwi`, `compute_ndmi`, `compute_nbr` — each takes a local multi-band raster + 1-based band indices and writes a single-band float32 GeoTIFF. It also adds `clip_raster_with_bbox`, for clipping a raster to a `[west, south, east, north]` bbox without needing a shapefile (use gis-mcp's own `clip_raster_with_shapefile` when one is available).

The server runs in an isolated uv tool environment (`uv tool run --python 3.12 gis-mcp`, cwd pinned to `tool_output/gis_analysis/` so relative output paths land there instead of wherever the process was launched from) to avoid numpy version conflicts with other dependencies. Install once:
```bash
uv tool install "gis-mcp[visualize]"
```

No `.env` variables are required — none of gis-mcp's download extras (`administrative-boundaries`, `climate`) are relevant since those tools are filtered out for this sub-agent.

### EUMETSAT Data Store

Wired to `eumetsat_agent` in the current (`multi`) pipeline — searches and downloads geostationary products from EUMETSAT's Data Store: **MTG FCI and MSG SEVIRI only** (not Sentinel/Landsat — that's CDSE's job). Backed by `src/agenticeoecosystem/servers/eumetsat/server.py`, which exposes four tools:

| Tool | Purpose |
|---|---|
| `list_eumetsat_products()` | Look up valid product keys (e.g. `MTG-FCI-L1C-FDHSI`) with their `product_id`, `satpy_reader`, and bands |
| `search_eumetsat_products(product, start_date, end_date, bbox)` | Dry-run count of matching products before downloading (results capped at 20) |
| `download_eumetsat_product(args)` | Downloads matching files to `EUMETSAT_OUTPUT_DIR`. `method="datastore"` (default) is a fast raw archived file; `method="datatailor"` reprojects/clips/converts format server-side (slower, polls until done). Refuses any request matching more than 20 products in one call. |
| `read_eumetsat_product(args)` | Opens downloaded files with `satpy`, returns per-band stats with correct calibration (reflectance for VIS/NIR, brightness temperature for IR/WV); `as_geotiff=true` also writes GeoTIFFs so `gis_agent` can process them (the raw `.nat`/`.nc` files aren't directly GIS-readable) |

```env
EUMETSAT_CONSUMER_KEY=your-consumer-key
EUMETSAT_CONSUMER_SECRET=your-consumer-secret
```

> **Known issue:** `method="datatailor"` downloads can land in the repo root instead of `EUMETSAT_OUTPUT_DIR`, due to a bug in the third-party `eumetsearch` package's success-path file write (bare `open(filename)` instead of joining it to the configured output directory). See `docs/gemma4_benchmark_run_issues_2026-08-07.md` (issue #12) — unfixed as of that doc. `method="datastore"` is unaffected.

### Climate / CDS

Wired to `climate_agent` in the current (`multi`) pipeline — ERA5/ERA5-Land reanalysis data via the Copernicus Climate Data Store. Backed by `src/agenticeoecosystem/servers/climate/server.py`, which exposes discovery (`list_era5_single_level_variables`, `list_era5_pressure_level_variables`, `list_cds_datasets`), download (`fetch_era5_single_levels`/`fetch_era5_pressure_levels` for monthly means; `fetch_era5_hourly_single_levels`/`fetch_era5_hourly_pressure_levels`/`fetch_era5_land` for hourly; `fetch_era5_daily_statistics` for pre-computed daily stats; `fetch_cds_dataset` as a generic escape hatch for any other CDS dataset), and local NetCDF processing tools (`inspect_netcdf`, `convert_hourly_to_daily`/`convert_daily_to_monthly`/`resample_temporal` for temporal aggregation, `convert_units`, `clip_to_bbox`).

```env
CDS_URL=https://cds.climate.copernicus.eu/api
CDS_KEY=your-cds-api-key
```

### VLM (Vision-Language Model)

`vlm_agent` reasons over image files already on disk (a downloaded quicklook, a single scene, or a directory of up to 10 images) — e.g. "does this scene show visible cloud cover", "what changed between these two images". It's the only sub-agent in the current pipeline that runs its **own separate, fixed model** (`Qwen3-VL`, via [`vlm_init.py`](src/agenticeoecosystem/agents/subagents/vlm_init.py)) rather than whichever candidate model `MULTI_AGENT_MODEL`/`MULTI_AGENT_PROVIDER` selects for the orchestrator and every other sub-agent.

The VLM is served by a separate vLLM instance and reached over an SSH local port-forward — this tunnel must be established before starting the client/benchmark, or `vlm_agent` calls will fail to connect:

```bash
ssh -L 18001:localhost:18001 <user>@<llava-vm-host>
```

Backed by `src/agenticeoecosystem/servers/vlm_tools/server.py`, which exposes `resolve_images(path, limit)` (list what images exist at a path) and `load_images(path, limit)` (load them into context — downscaled before base64-encoding to control per-image token cost). Its own system prompt enforces calling `load_images` at most once per distinct path, to avoid burning context on duplicate loads.

---

## Benchmarking

The evaluation framework lives in `src/agenticeoecosystem/evaluation/` and the entry point is `tests/benchmarking.py`. It runs a set of EO tasks through the agent pipeline, scores the results with one or more LLM-as-judge models, and produces a comparison table across generator models.

### Pipeline

For each generator model:
1. **Generate** — run every task through the agent pipeline and record the tool-call sequence (Solution)
2. **Evaluate** — for each evaluator model, compare the generated solution against the reference solution and assign a score (0 = no match, 1 = partial, 2 = full match)
3. **Save** — write per-model results to `output/ESAEOBench/` and a summary table as CSV and HTML

### Task format

Tasks are stored as JSON files in `data/tasks/` and loaded as a `TaskSet`:

```python
from agenticeoecosystem.evaluation.dataclasses import TaskSet

tasks = TaskSet.read_from_file("benchmark_tasks.json")
```

Each `Task` has:
- `task_text` — the natural-language query
- `task_labels` — one or more of: `Tool Grounding`, `Sequential Reasoning`, `Cross-agent Synthesis`, `Scope Awareness`
- `reference_solutions` — list of acceptable tool-call sequences
- `generated_solution` — populated after the pipeline runs
- `evaluator_scores` / `evaluator_reasoning` — populated after LLM-as-judge runs

Useful selection helpers:
```python
tasks.select_by_ids(["TASK_...", "TASK_..."])   # by ID
tasks.select_by_labels(["Tool Grounding"])       # by category
tasks.subset("Sequential Reasoning", n=5)        # single category, capped
tasks.sample_stratified(20)                      # stratified random sample
```

### Model registry

Generator and evaluator models are defined in [`src/agenticeoecosystem/utils/registry.py`](src/agenticeoecosystem/utils/registry.py). Models that need a local vLLM server are listed in `MODEL_REGISTRY`:

| Key | Model | Gated |
|---|---|---|
| `qwen3-14b` | Qwen/Qwen3-14B | free |
| `qwen2.5-14b` | Qwen/Qwen2.5-14B-Instruct | free |
| `qwen3-30b-moe` | Qwen/Qwen3-30B-A3B | free |
| `deepseek-r1-14b` | deepseek-ai/DeepSeek-R1-Distill-Qwen-14B | free |
| `phi4-mini` | microsoft/Phi-4-mini-instruct | free |
| `nemotron-2-12b` | nvidia/NVIDIA-Nemotron-Nano-12B-v2 | free |
| `gemma3-12b` | google/gemma-3-12b-it | gated |
| `gemma4-31b` | google/gemma-4-31B-it | free |
| `qwen3.6-27b` | Qwen/Qwen3.6-27B | free |
| `qwen3.6-35b-a3b` | Qwen/Qwen3.6-35B-A3B (MoE, 3B active) | free |
| `mistral-small-24b` | mistralai/Mistral-Small-3.1-24B-Instruct-2503 | gated |
| `llama3.1-8b` | meta-llama/Llama-3.1-8B-Instruct | gated |
| `llama3.2-3b` | meta-llama/Llama-3.2-3B-Instruct | gated |

Gated models require accepting the licence on huggingface.co and setting `HF_TOKEN` in `.env`.

Download models before running (skips if already cached):

```python
from agenticeoecosystem.utils.registry import download_model, list_downloaded

list_downloaded()                          # show download status for all models
download_model("qwen3-14b")               # free model
download_model("gemma3-12b")              # gated — needs HF_TOKEN
```

### Running a single generator model (CLI)

For a one-off run against a single model — e.g. comparing a new candidate LLM against EVE — use `scripts/run_generate.py` directly instead of editing `tests/benchmarking.py`. Each invocation is a fresh process: for `--provider vllm` it launches its own vLLM server (reusing one already serving the same model if present) and tears it down on exit unless `--keep-vllm-alive` is passed; for `--provider eve` it talks to `EVE_ENDPOINT` (`.env`) directly and never touches vLLM.

```bash
# EVE-Instruct — provider="eve", no vLLM server launched by this process
uv run python scripts/run_generate.py \
  --label       "EVE-Instruct" \
  --model       "EVE-Instruct" \
  --provider    "eve" \
  --pipeline    "multi" \
  --tasks-file  benchmark_tasks.json \
  --task-labels "Tool Grounding,Sequential Reasoning,Cross-agent Synthesis,Scope Awareness" \
  --output-dir  "output/ESAEOBench/run_$(date +%Y%m%d_%H%M%S)" \
  --task-timeout 2400

# A local candidate model — provider="vllm", auto-launches on port 8000
uv run python scripts/run_generate.py \
  --label       "Qwen3.6-27B" \
  --model       "qwen3.6-27b" \
  --provider    "vllm" \
  --pipeline    "multi" \
  --tasks-file  benchmark_tasks.json \
  --task-labels "Tool Grounding,Sequential Reasoning,Cross-agent Synthesis,Scope Awareness" \
  --output-dir  "output/ESAEOBench/run_$(date +%Y%m%d_%H%M%S)" \
  --vllm-startup-timeout 1800 \
  --task-timeout 2400
```

Key flags:

| Flag | Description |
|---|---|
| `--label` | Human-readable name, used in the output filename (`results_<label>.json`) |
| `--model` | Registry key (for `vllm`) or model identifier (for `eve`) |
| `--provider` | `eve`, `vllm`, `openai`, … |
| `--pipeline` | `multi` (google-adk orchestrator), `single` (one LangGraph agent), or `multi_legacy` |
| `--tasks-file` | One or more task JSON filenames from `data/tasks/` (merged if multiple) |
| `--task-labels` / `--task-ids` | Filter to specific categories or task IDs |
| `--output-dir` | Directory for `results_<label>.json` |
| `--vllm-startup-timeout` | Seconds to wait for the vLLM server's `/health` (only relevant for `--provider vllm`) |
| `--task-timeout` | Max wall-clock seconds per task before it's abandoned and the run moves on |
| `--skip-solved` | Skip tasks that already have a `generated_solution` in the loaded file |
| `--keep-vllm-alive` | Don't tear down the vLLM server this process launched, so a following call for the same model can reuse it |

**Note:** the `MODEL_REGISTRY` in `registry.py` has no entry for EVE — always use `--provider eve` for it, never `--provider vllm`.

### Running a comparison

Edit `tests/benchmarking.py` to configure which models and tasks to run:

```python
from agenticeoecosystem.evaluation.model_comparison import ModelRun, run_comparison
from agenticeoecosystem.evaluation.dataclasses import TaskSet
from agenticeoecosystem.utils.logging import setup_logging

setup_logging(log_file="benchmarking")   # → logs/benchmarking_YYYYMMDD_HHMMSS.log

tasks = TaskSet.read_from_file("benchmark_tasks.json")
selected = tasks.select_by_ids(["TASK_...", "TASK_..."])

runs = [
    # External API — always available, no vLLM needed
    ModelRun(label="EVE-Instruct", model="EVE-Instruct", provider="eve"),

    # Local models via vLLM — launched and shut down automatically
    ModelRun(label="Qwen3-14B",  model="qwen3-14b",  provider="vllm",
             launch_vllm=True, vllm_startup_timeout=600),
    ModelRun(label="Gemma3-12B", model="gemma3-12b", provider="vllm",
             launch_vllm=True, vllm_startup_timeout=600),
]

evaluators = [
    ModelRun(label="Nemotron-2-12B", model="nemotron-2-12b", provider="vllm",
             launch_vllm=True, vllm_startup_timeout=600),
]

df = run_comparison(runs, selected, evaluators=evaluators)
print(df.to_string())
```

`ModelRun` parameters:

| Parameter | Description |
|---|---|
| `label` | Human-readable name used in output columns |
| `model` | Registry key (vLLM) or model identifier (eve, ollama, …) |
| `provider` | `"eve"`, `"vllm"`, `"ollama"`, `"openai"`, `"custom"` |
| `temperature` | Sampling temperature (default `0.0`) |
| `launch_vllm` | Start/stop a local vLLM server automatically |
| `vllm_startup_timeout` | Seconds to wait for `/health` (default `600`) |

If a vLLM server is already running and serving the correct model, `launch_vllm=True` will reuse it instead of restarting.

Run from the project root:
```bash
python tests/benchmarking.py
```

### Outputs

Each full `benchmark.sh` run writes to a timestamped subdirectory so results from different runs never overwrite each other:

| File | Description |
|---|---|
| `output/ESAEOBench/run_YYYYMMDD_HHMMSS/results_<Label>.json` | Per-model task results with scores |
| `output/ESAEOBench/run_YYYYMMDD_HHMMSS/model_comparison.csv` | Summary table (generators × evaluators) |
| `output/ESAEOBench/run_YYYYMMDD_HHMMSS/model_comparison.html` | Same table as HTML |
| `logs/benchmarking_YYYYMMDD_HHMMSS.log` | Full run log |

To regenerate the comparison table from any archived run:
```bash
python scripts/build_table.py --output-dir output/ESAEOBench/run_YYYYMMDD_HHMMSS
```

### Multi-model bash orchestration

For full multi-model runs with per-process GPU isolation (required on a single GPU to avoid vLLM CUDA memory leaks), use the bash orchestrator:

```bash
bash scripts/benchmark.sh            # all runnable task categories (default)
bash scripts/benchmark.sh --subset   # curated 9-task subset only
```

Each generator model runs in its own Python subprocess. Between runs, any orphaned vLLM workers are hard-killed via `pkill`. Evaluator models (Nemotron-2-12B, Gemma3-12B, Qwen3-14B) score every generator's results afterwards.

Edit `scripts/benchmark.sh` to add or remove generator and evaluator models.

### Smoke test

To run a quick sanity check on a small subset before the full run:

```python
df = run_comparison(
    runs, tasks, evaluators=evaluators,
    smoke_test_labels=["Tool Grounding"],
    smoke_test_n=2,
)
```

This filters to 2 tasks with the `Tool Grounding` label and appends `_smoke` to all output filenames.

---

## Adding Custom Sub-agents

The two subsections below target different pipelines — pick the one matching what you're actually running. Most new work should go through **the current (`multi`) pipeline** below; the LangChain instructions further down are for `--pipeline single`/`multi_legacy` only.

### For the current (`multi`) pipeline

Every sub-agent lives in its own `src/agenticeoecosystem/agents/subagents/sub_agent_<name>.py` file. [`subagents/__init__.py`](src/agenticeoecosystem/agents/subagents/__init__.py) auto-discovers them: it scans the package for any module starting with `sub_agent_` and collects every `google.adk.agents.llm_agent.Agent` instance it defines into `ALL_SUB_AGENTS` — so a new file is picked up automatically, no manual registration list to edit. Follow [`sub_agent_mooc.py`](src/agenticeoecosystem/agents/subagents/sub_agent_mooc.py) as the simplest template (a single stdio MCP server, no extra local tools):

```python
from agenticeoecosystem.agents.subagents.llm_init import llm
from google.adk.agents.llm_agent import Agent
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters
from typing import Optional
from agenticeoecosystem.definitions import ROOT_DIR

_MY_SYSTEM_PROMPT = "You are a retrieval assistant for <domain>. Use the available tools to ..."

toolset = [McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command='python',
            args=[f"{ROOT_DIR}/src/agenticeoecosystem/servers/my_server/server.py"]),
        timeout=300))]

def _fetch_agent(toolset) -> Optional[Agent]:
    if toolset:
        return Agent(
            model=llm,               # shared candidate model (MULTI_AGENT_MODEL/PROVIDER); use vlm_init.vlm instead for a fixed vision model, as vlm_agent does
            name='my_agent',
            description='A helpful assistant for answering user questions related to <domain>.',
            instruction=_MY_SYSTEM_PROMPT,
            tools=toolset,
            mode='single_turn'
        )
    return None

my_agent = _fetch_agent(toolset)
```

Build the MCP server itself the same way as for the legacy pipeline (see [Creating the MCP tools](#creating-the-mcp-tools) below) — the server side doesn't change between pipelines, only how the client wires it up.

Two more steps, both in [`agents/multi_agent.py`](src/agenticeoecosystem/agents/multi_agent.py):
1. Add a paragraph to `_SUPERVISOR_SYSTEM_PROMPT` describing when to route to the new sub-agent, and any scope boundaries it should respect relative to existing sub-agents (see how the CDSE/GEE/EUMETSAT split is worded there as an example of avoiding overlapping responsibilities).
2. Nothing else — `root = Agent(..., sub_agents=ALL_SUB_AGENTS, ...)` already picks up every agent discovered in step 1 automatically.

### For the `single`/`multi_legacy` pipelines (LangChain, `client/client.py`)

#### Creating the MCP tools

Follow the example server at [`src/agenticeoecosystem/servers/example/stdio_localserver/example_server.py`](src/agenticeoecosystem/servers/example/stdio_localserver/example_server.py) or any other server in the [`servers/`](src/agenticeoecosystem/servers) folder. Once created, add an entry to `get_mcp_servers()` in [`mcp_config.py`](src/agenticeoecosystem/config/mcp_config.py):

```python
"example": {
    "transport": "stdio",
    "command": "python",
    "args": [f"{ROOT_DIR}/src/agenticeoecosystem/servers/example/stdio_localserver/example_server.py"],
},
```

See the [MCP documentation](https://modelcontextprotocol.io/docs/develop/build-server) for more on building MCP servers.

#### Creating the sub-agent

Declare the agent:

```python
my_custom_agent = create_agent(
    model,
    tools=[t for t in mcp_tools if t.name in ("my_tool_a", "my_tool_b")],
    system_prompt="..."
)
```

Wrap it as a tool for the supervisor:

```python
@tool
async def my_agent_as_tool(request: str) -> str:
    """Describe the functionality of the agent."""
    logger.info("[TOOL CALL] my_agent_as_tool: %s", request)
    result = await my_custom_agent.ainvoke({"messages": [{"role": "user", "content": request}]})
    text_output = result["messages"][-1].text
    if isinstance(text_output, tuple):
        text_output = " ".join(str(t) for t in text_output)
    return str(text_output)
```

Register it with the supervisor:

```python
supervisor_agent = create_agent(
    model,
    tools=[
        get_eo_stac_data, get_gee_data, get_eve_knowledge,
        get_eossgraphrag_context, get_arxiv_data, get_gis_analysis,
        my_agent_as_tool,
    ],
    system_prompt=(
        "You are a routing agent. "
        "You can get data from EO Dashboards, Google Earth Engine, "
        "the EVE knowledge base, the ESA EO Science Strategy knowledge base, "
        "ArXiv scientific papers, the GIS analysis server, "
        "and <describe your custom agent here>. "
        "Break down user requests into appropriate tool calls and coordinate the results. "
        "Return tool results directly to the user without summarising."
    )
)
```

If you run into any trouble, feel free to open an issue.
