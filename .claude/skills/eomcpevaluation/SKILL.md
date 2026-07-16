---
name: eomcpevaluation
description: Check all available MCP servers in the EO ecosystem, compare against benchmark data, and generate new synthetic benchmark tasks for any servers that are new or uncovered. Use whenever the user wants to update the EO benchmark dataset, add tasks for a new MCP server, or audit benchmark coverage.
---

# EO MCP Evaluation Skill

When this skill is invoked, follow the steps below **in order**. Do not skip steps.

The project root is `/shared/home/rdercole/Riccardo/projects/agenticeoecosystem`.
The primary benchmark file is `data/tasks/benchmark_tasks.json` (merged from `eo_benchmark_tasks.json` + `arxiv_gis_benchmark_tasks.json`).
The skill helper script lives at `~/.claude/skills/eomcpevaluation/scripts/analyze_coverage.py`.

---

## Step 1 — Inventory all MCP servers

Read `src/agenticeoecosystem/config/mcp_config.py`. Extract every key registered inside `get_mcp_servers()` — but this function is **not the complete picture**: some servers are wired up per-agent instead (see note below). Cross-check against `src/agenticeoecosystem/agents/subagents/*.py` for any MCP server a subagent connects to that isn't in `get_mcp_servers()`.

For each server that has a local Python file, read that file and extract the tools:

| Server key       | Server file (relative to project root)                                          | Notes |
|------------------|---------------------------------------------------------------------------------|-------|
| eodashboard      | `src/agenticeoecosystem/servers/EOdashboard/eodashboard_server.py`             | registered in `get_mcp_servers()` |
| gee-mcp          | `src/agenticeoecosystem/servers/gee/server.py`                                 | registered in `get_mcp_servers()` |
| moocs            | `src/agenticeoecosystem/servers/moocs/server.py`                               | registered in `get_mcp_servers()` |
| geos_ops         | `src/agenticeoecosystem/servers/eumetsat/server.py`                            | registered in `get_mcp_servers()`; folder is `eumetsat`, FastMCP app name is `"eumetsat-mcp"`, config key is `geos_ops` |
| climate          | `src/agenticeoecosystem/servers/climate/server.py`                             | registered in `get_mcp_servers()` |
| eoss_graphrag    | `src/eoss_graphrag/05_mcp_server.py`                                           | registered in `get_mcp_servers()` as `transport: streamable_http` (remote). Source lives in a separate GitLab repo checked out at `src/eoss_graphrag/` — it has its own nested `.git` and is excluded via this repo's `.gitignore` (`src/eoss_graphrag/`), but is present on disk and readable for inventory purposes. |
| cdse             | `src/agenticeoecosystem/servers/cdse_geocode/server.py`                        | **not present in `get_mcp_servers()`** — wired independently in `src/agenticeoecosystem/agents/subagents/sub_agent_cdse.py`, which builds its own `McpToolset` pointing at this file. Still an active, benchmarked server — just registered outside the shared config. |

For third-party servers (`arxiv-mcp-server`, `gis-mcp`, `eve_retrieval`) note the server name but do not try to read their tool list — just record them as "external".

From each server file, collect every `@mcp.tool(...)`-decorated function's **MCP-registered name** — this is the Python function name *unless* the decorator passes a `name=` kwarg, in which case that overrides it (e.g. `cdse_geocode/server.py` defines `def tool_sentinel1_grd(...)` under `@mcp.tool(name="fetch_sentinel1_grd", ...)`; the registered tool name is `fetch_sentinel1_grd`, not `tool_sentinel1_grd`). Also collect parameter names/types and the first line of the docstring/description.

Build an inventory table:
```
server_key | tool_name | signature | description
```

---

## Step 2 — Run coverage analysis

Run the helper script against the project root and capture the JSON output:

```bash
python ~/.claude/skills/eomcpevaluation/scripts/analyze_coverage.py /shared/home/rdercole/Riccardo/projects/agenticeoecosystem
```

The output has two top-level keys:

**`task_coverage`** — keyed by `function_name`:
- `count` — total tasks that reference this function (Placeholder + active)
- `active_count` — tasks NOT labeled Placeholder
- `has_placeholder` — true if any Placeholder task uses this function
- `categories` — list of task_labels seen
- `task_ids` — all task IDs that reference this function

**`mcp_tool_coverage`** — keyed by server name:
- `source` — `"local"` (we can read the server file) or `"external"` (third-party)
- `tools` — per-tool dict: `active_task_count`, `total_task_count`, `categories`, `covered`
- `all_covered` — true if every tool in this server has at least one active task
- `uncovered_tools` — list of tool names with zero active tasks

---

## Step 3 — Compute gaps and mismatches

Use the `mcp_tool_coverage` section from Step 2 directly — it already cross-references the server inventory against task coverage. Also scan `task_coverage` for function names that don't appear in any known server.

**Gap types to detect:**

1. **Uncovered tool** — `covered: false` in `mcp_tool_coverage` (zero active tasks). Needs tasks from scratch.
2. **Placeholder-only tool** — `active_count == 0` but `has_placeholder == True` in `task_coverage`. Server is now live but tasks were never activated.
3. **Function name mismatch** — a `function_name` in `task_coverage` does not match any tool in any server inventory. Flag it with the likely intended tool.
4. **Under-covered tool** — `active_task_count < 3`. Mention it but do not generate unless asked.
5. **External server** — `source: "external"` servers cannot be auto-inventoried; note them and ask the user if coverage should be added manually.

Print a summary table:
```
Tool              | Server      | Active tasks | Placeholder tasks | Gap type
search_moocs      | moocs       | 0            | 0                 | New tool
mooc_search       | (no server) | 0            | 13                | Name mismatch → search_moocs
list_themes       | eodashboard | 2            | 0                 | OK
...
```

---

## Step 4 — Generate new benchmark tasks

For each tool with gap type **New tool** or **Placeholder-only tool**, generate new Task JSON objects. If the user asked to focus on a specific server (e.g., "generate data to evaluate the MOOC MCP"), prioritise that server.

### Task schema

```json
{
  "task_ID": "TASK_YYMMDD_HHMMSS_XXXXXX",
  "task_text": "<natural language query>",
  "task_labels": ["<one of the four active categories>"],
  "reference_solution_description": "<plain English explanation of the correct approach>",
  "reference_solutions": [
    {
      "steps": [
        {
          "function_name": "<exact tool name from server>",
          "arguments": { "<param>": "<realistic value>" },
          "comment": "<grading hint>"
        }
      ]
    }
  ],
  "generated_solution": null,
  "generated_solution_input_tokens": null,
  "generated_solution_output_tokens": null,
  "match_reasoning_LLM": null,
  "match_score_LLM": null,
  "match_reasoning_Human": null,
  "match_score_Human": null,
  "evaluator_scores": {},
  "evaluator_reasoning": {}
}
```

### task_ID format

Use today's date and a sequential counter. Example: `TASK_260619_120000_200001`. The last six digits are a unique counter (start from `200001` to avoid collisions with existing IDs).

### Category patterns

Generate tasks across all four active categories:

**Tool Grounding** — single call to the tool, concrete arguments, no chaining:
- task_text: asks directly for what the tool returns
- steps: exactly one step with the target tool
- arguments: specific, realistic values (not `<derived>` placeholders)

**Sequential Reasoning** — two or three steps where the output of one feeds the next:
- Either: preceding step produces data used as the argument of the target tool
- Or: target tool produces data consumed by a following step
- Use `"<derived from previous step>"` only for arguments that genuinely come from a prior result

**Cross-agent Synthesis** — at least two different MCP servers involved:
- Combine the target tool with tools from other servers
- Example for `search_moocs`: hybridrag retrieves a strategic objective → search_moocs finds an educational resource on that topic

**Scope Awareness** — negative test:
- task_text asks for something the tool cannot provide
- reference_solution either has zero steps (correct answer: don't call any tool) or calls the tool with the most plausible query and explains it will return no useful result
- comment must state why this is a negative test

### Minimum per gap tool

| Category            | Minimum tasks |
|---------------------|--------------|
| Tool Grounding      | 3            |
| Sequential Reasoning| 2            |
| Cross-agent Synthesis| 2           |
| Scope Awareness     | 1            |

### Quality rules

- `function_name` must match the **exact** Python function name from the server file
- `arguments` must only include parameters that the function actually accepts
- `arguments` values must be realistic (real place names, real date ranges, real EO topics)
- Do not use placeholder strings like `"<some value>"` in Tool Grounding tasks — those are for dynamic steps only
- Each `task_text` must be a self-contained natural-language request a real user could ask

---

## Step 5 — Present results

Show the user:

1. The gap summary table from Step 3
2. Any function name mismatches (and your recommendation: update existing tasks vs leave them)
3. The generated tasks as a formatted JSON block
4. A count: "Will append N new tasks to `data/tasks/eo_benchmark_tasks.json`"

Ask the user to confirm before writing anything.

---

## Step 6 — Append on confirmation

If the user confirms:

1. Read `data/tasks/eo_benchmark_tasks.json`
2. Append the new Task objects to the `tasks` array
3. Write the file back
4. Report the new total task count

Do **not** modify existing tasks (do not fix `mooc_search` → `search_moocs` in old Placeholder tasks unless the user explicitly asks for that separately).

---

## Reference: known server-to-tool mapping

| Server         | Tools                                                                                              |
|----------------|----------------------------------------------------------------------------------------------------|
| eodashboard    | list_narratives, search_narratives, get_narrative_content, list_themes, get_narrative_indicators, search_collections, get_collection_items, find_narratives_for_indicator |
| gee-mcp        | get_lat_lon, download_gee_product                                                                   |
| moocs          | search_moocs                                                                                       |
| geos_ops       | list_eumetsat_products, search_eumetsat_products, download_eumetsat_product, read_eumetsat_product |
| climate        | list_era5_single_level_variables, list_era5_pressure_level_variables, list_cds_datasets, inspect_netcdf, fetch_era5_single_levels, fetch_era5_pressure_levels, fetch_era5_hourly_single_levels, fetch_era5_hourly_pressure_levels, fetch_era5_land, fetch_era5_daily_statistics, fetch_cds_dataset, convert_hourly_to_daily, convert_daily_to_monthly, resample_temporal, convert_units, clip_to_bbox, compute_wind_speed |
| eoss_graphrag  | search_eoss, query_ontology, list_ontology_nodes (remote; source in separate `src/eoss_graphrag/` repo) |
| cdse           | fetch_sentinel1_grd, fetch_sentinel2, fetch_sentinel3_olci, fetch_sentinel3_slstr, fetch_sentinel3_syn_l2, fetch_sentinel5p_l2, fetch_landsat_ot_l1, fetch_dem, geocode_place, reverse_geocode_place, resolve_named_region (not in `get_mcp_servers()` — see Step 1 note) |
| arxiv-mcp-server | search_papers, download_paper, read_paper, get_abstract, citation_graph, semantic_search, list_papers (external) |
| gis-mcp        | metadata_raster, raster_band_statistics, compute_s2_ndvi, compute_ndvi, reproject_raster, clip_raster_with_shapefile, clip_raster_with_bbox, transform_coordinates, get_utm_crs, get_utm_zone, calculate_geodetic_distance, buffer, calculate_geodetic_area, sjoin_gpd, point_in_polygon, raster_histogram, create_map, zonal_statistics (external) |
| eve_retrieval  | (external — HTTP — EVE knowledge base retrieval; tools not yet inventoried)                       |
