# Benchmark tasks requiring reruns

Recorded: 2026-09-30. Status: pending.

The task wording and reference workflows changed for these six tasks. Existing
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

## Checklist

- [ ] Restart the climate MCP server with the updated tool code and refresh tool discovery in candidate clients.
- [ ] Confirm mean_netcdf_fields and summarize_regional_timeseries are exposed, and convert_units accepts accumulation_seconds.
- [ ] Regenerate or review task-specific rubrics and other evaluation expectations for the revised wording.
- [ ] Rerun candidate generation for these six IDs for every model included in the comparison.
- [ ] Preserve input/output artifacts under candidate- and task-specific paths.
- [ ] Rerun tool-use metrics, numerical/artifact checks, and applicable judge evaluations for the fresh candidate runs.
- [ ] Recompute rankings after replacement results are complete; do not mix old and new task versions.

Other benchmark tasks do not require reruns because of these six task/reference changes.
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
