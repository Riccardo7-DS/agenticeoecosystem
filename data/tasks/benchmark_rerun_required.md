# Benchmark tasks requiring reruns

Recorded: 2026-09-30. Updated: 2026-10-01. Status: pending.

The task wording and reference workflows changed for these 28 tasks. Existing
candidate answers and judge scores belong to the previous task versions and
should be retained separately, rather than compared as answers to the new tasks.

| Task ID | Reason for rerun |
| --- | --- |
| TASK_260713_000071_600071 | Now explicitly requires a saved equal-weight, cell-by-cell mean SST field from the same month in three years, using the common valid-ocean mask. The reference now calls mean_netcdf_fields instead of treating metadata inspection as computation. |
| TASK_260713_000075_600075 | Corrected monthly-mean accumulated-energy conversion from division by 3600 to division by 86400. The reference now passes accumulation_seconds=86400 and verifies the output. |
| TASK_260713_000074_500053 | Replaced winter jet-stream anomalies with a Walker-circulation task grounded in the EarthCARE narrative. Simplified to one June 2025 vertical-velocity field at 500 hPa over representative western/eastern tropical Pacific regions; interpret ascent/descent from actual evidence. Updated task, reference workflow and rubric; old candidate answers and scores must be rerun. |
| TASK_260713_000054_600054 | Simplified to ERA5 daily mean temperature and warmest/coldest dates using an unweighted valid-cell regional mean. Reference uses daily-statistics retrieval and summarize_regional_timeseries; no final numerical temperatures or hourly download required. Rerun candidate generation and date/coverage evaluation. |
| TASK_260713_000015_600015 | Download three separate CDSE Sentinel-5P GeoTIFFs (CO, NO2, CH4) over Madrid and its surroundings for 1–30 June 2025. Final answer requires only product–data-band identifiers; no fixed bbox or reported file paths. Rerun candidate generation, download-contract checks and product–band mapping evaluation. |
| TASK_260713_000085_600085 | Removed unrequested Celsius conversion, removed unsupported area argument, and added explicit clipping to sub-Saharan Africa. User requested rerun of generation and evaluation against this corrected reference; task wording is unchanged. |
| TASK_260713_000063_500042 | Added Tigray, Ethiopia as required region for January 2021 precipitation. Reference resolves region, downloads, clips and inspects regional NetCDF; removed unsupported area argument. Rerun generation and regional/download/metadata evaluation. |
| TASK_260713_000064_500043 | Made Europe explicit in July 2021 monthly mean temperature download. Reference downloads, clips to Europe and inspects regional NetCDF; removed unsupported area argument. Rerun generation and regional/download/metadata evaluation. |
| TASK_260713_000065_500044 | Made Europe explicit in the download request; reference removes unsupported area argument and adds regional clipping before inspection. Rerun candidate generation and AAJ regional/artifact/metadata evaluation. |
| TASK_260713_000080_500059 | Made Europe explicit in the download request; reference removes unsupported area argument and adds regional clipping before inspection. Rerun candidate generation and AAJ regional/artifact/metadata evaluation. |
| TASK_260713_000021_400021 | Explicitly names ESA Science Strategy; buffer centre is Paradise, matching the geocoding workflow, rather than an unsupported ignition point. Metric/geodesic 20km buffer remains illustrative. Rerun generation and geometry/source-grounded AAJ evaluation. |
| TASK_260713_000072_500051 | Makes Europe explicit in ERA5 download request. Reference/evaluation require regional data and visualization, pressure-coordinate inspection and scientifically plausible interpretation. Rerun generation and AAJ regional/visualization evaluation. |
| TASK_260713_000026_500005 | Replaced EnablingTool–ScienceQuestion direct-link query with listing all ESA Science Strategy enabling tools. Reference uses list_ontology_nodes; five-tool inventory GT and set metrics assigned. Rerun candidate generation and set scoring; prior empty-link answers belong to old task. |
| TASK_260713_000014_600014 | Simplified to the climate tool’s curated catalogue: identify future climate projections and seasonal forecasts, reporting the two identifier–classification pairs. Removed EO Dashboard narrative retrieval. Rerun candidate generation, tool-use metrics and deterministic set/pair scoring against the new GT. |

| TASK_260713_000004_600004 | Made summer year window 2023–2025 explicit; corrected soil-moisture reference and global-then-clip workflow. Rerun candidate generation and corresponding evaluation; preserve old task versions separately. |
| TASK_260713_000053_600053 | Made temporal-first, unweighted valid-cell spatial reduction and reported units explicit. Rerun candidate generation and corresponding evaluation; preserve old task versions separately. |
| TASK_260713_000061_500040 | Added a fixed compact city/surroundings region: Girona, Spain. Rerun candidate generation and corresponding evaluation; preserve old task versions separately. |
| TASK_260713_000061_600061 | Added a fixed compact city/surroundings region: Braga, Portugal. Made temporal-first, unweighted valid-cell spatial reduction and reported units explicit. Rerun candidate generation and corresponding evaluation; preserve old task versions separately. |
| TASK_260713_000062_500041 | Added a fixed compact city/surroundings region: Trento, Italy. Rerun candidate generation and corresponding evaluation; preserve old task versions separately. |
| TASK_260713_000062_600062 | Added a fixed compact city/surroundings region: Annecy, France. Made temporal-first, unweighted valid-cell spatial reduction and reported units explicit. Rerun candidate generation and corresponding evaluation; preserve old task versions separately. |
| TASK_260713_000063_600063 | Made temporal-first, unweighted valid-cell spatial reduction and reported units explicit. Rerun candidate generation and corresponding evaluation; preserve old task versions separately. |
| TASK_260713_000067_600067 | Made temporal-first, unweighted valid-cell spatial reduction and reported units explicit. Rerun candidate generation and corresponding evaluation; preserve old task versions separately. |
| TASK_260713_000068_600068 | Made temporal-first, unweighted valid-cell spatial reduction and reported units explicit. Rerun candidate generation and corresponding evaluation; preserve old task versions separately. |
| TASK_260713_000070_600070 | Made temporal-first, unweighted valid-cell spatial reduction and reported units explicit. Rerun candidate generation and corresponding evaluation; preserve old task versions separately. |
| TASK_260713_000073_600073 | Made temporal-first, unweighted valid-cell spatial reduction and reported units explicit. Rerun candidate generation and corresponding evaluation; preserve old task versions separately. |
| TASK_260713_000081_600081 | Defined grid-cell-centre W,S,E,N extent in WGS84 degrees and -180..180 longitude. Rerun candidate generation and corresponding evaluation; preserve old task versions separately. |
| TASK_260713_000082_600082 | Defined grid-cell-centre W,S,E,N extent in WGS84 degrees and -180..180 longitude. Rerun candidate generation and corresponding evaluation; preserve old task versions separately. |
| TASK_260713_000083_600083 | Defined grid-cell-centre W,S,E,N extent in WGS84 degrees and -180..180 longitude. Rerun candidate generation and corresponding evaluation; preserve old task versions separately. |

## Checklist

- [ ] Restart the climate MCP server with the updated tool code and refresh tool discovery in candidate clients.
- [ ] Confirm mean_netcdf_fields and summarize_regional_timeseries are exposed, convert_units accepts accumulation_seconds, and the updated clipping/resampling helpers are loaded.
- [ ] Archive candidate evidence and declare/calibrate task-specific numerical tolerances before scoring.
- [ ] Regenerate or review task-specific rubrics and other evaluation expectations for the revised wording.
- [ ] Rerun candidate generation for these 28 IDs for every model included in the comparison.
- [ ] Preserve input/output artifacts under candidate- and task-specific paths.
- [ ] Rerun tool-use metrics, numerical/artifact checks, and applicable judge evaluations for the fresh candidate runs.
- [ ] Recompute rankings after replacement results are complete; do not mix old and new task versions.

The reference/evaluation-only updates below require re-evaluation with the corrected protocol; valid existing candidate runs can be reused when their original question is unchanged.
Old results remain useful for reproducing the previous benchmark version.

## Scientific reference

ERA5 monthly means of daily means use an effective daily accumulation interval
for accumulated fluxes. Convert energy per unit area to flux by dividing by
86400 seconds, preserving the ECMWF sign convention:
[ECMWF ERA5 documentation](https://confluence.ecmwf.int/pages/viewpage.action?pageId=179741903).

## Walker-circulation task

Source narrative:
[Exploring our atmosphere with EarthCARE](https://esa-eodashboards.github.io/eodashboard-narratives/science-hub-team-1-story.md).
Search the current title/theme catalogue with `EarthCARE`, then retrieve full
content. `Walker` and `circulation` appear in the narrative body, which the
current search tool does not search.

June 2025 and the 500 hPa vertical-velocity variable are fixed. Western/eastern
tropical Pacific boxes are flexible; the reference boxes are illustrative.
Only one monthly field is downloaded. Inspect actual numerical values or maps
and accept supported agreement or disagreement with the narrative description.

## Reference/evaluation-only updates

These 20 tasks have unchanged questions. Re-evaluate existing valid candidate runs with corrected references and runtime artifact evidence. Regenerate only when the existing run lacks necessary evidence or failed the task; do not require imitation of the new reference sequence.

| Task ID | Update |
| --- | --- |
| TASK_260713_000003_600003 | Remove unsupported monthly area argument; fetch globally and clip before regional inspection/calculation. |
| TASK_260713_000008_600008 | Remove unsupported monthly area argument; fetch globally and clip before regional inspection/calculation. |
| TASK_260713_000009_600009 | Remove unsupported monthly area argument; fetch globally and clip before regional inspection/calculation. |
| TASK_260713_000058_600058 | Define independent numerical reference from archived candidate inputs; no fixed values required. |
| TASK_260713_000059_600059 | Define independent numerical reference from archived candidate inputs; no fixed values required. |
| TASK_260713_000065_600065 | Define independent numerical reference from archived candidate inputs; no fixed values required. |
| TASK_260713_000066_500045 | Remove unsupported monthly area argument; fetch globally and clip before regional inspection/calculation. |
| TASK_260713_000069_500048 | Remove unsupported monthly area argument; fetch globally and clip before regional inspection/calculation. |
| TASK_260713_000075_500054 | Remove unsupported monthly area argument; fetch globally and clip before regional inspection/calculation. |
| TASK_260713_000076_600076 | Remove unsupported monthly area argument; fetch globally and clip before regional inspection/calculation. |
| TASK_260713_000077_500056 | Remove unsupported monthly area argument; fetch globally and clip before regional inspection/calculation. |
| TASK_260713_000078_500057 | Remove unsupported monthly area argument; fetch globally and clip before regional inspection/calculation. |
| TASK_260713_000078_600078 | Remove unsupported monthly area argument; fetch globally and clip before regional inspection/calculation. |
| TASK_260713_000079_500058 | Remove unsupported monthly area argument; fetch globally and clip before regional inspection/calculation. |
| TASK_260713_000079_600079 | Remove unsupported monthly area argument; fetch globally and clip before regional inspection/calculation. |
| TASK_260713_000081_500060 | Remove unsupported monthly area argument; fetch globally and clip before regional inspection/calculation. |
| TASK_260713_000082_500061 | Remove unsupported monthly area argument; fetch globally and clip before regional inspection/calculation. |
| TASK_260713_000083_500062 | Remove unsupported monthly area argument; fetch globally and clip before regional inspection/calculation. |
| TASK_260715_000001_700010 | Remove unsupported monthly area argument; fetch globally and clip before regional inspection/calculation. |
| TASK_260715_000002_700011 | Remove unsupported monthly area argument; fetch globally and clip before regional inspection/calculation. |
