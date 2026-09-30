# AAJ GIS and NetCDF verification

AAJ can delegate numerical evidence gathering through `ask_gis_agent`. This is
an evaluation-specific read-only specialist, not the candidate's unrestricted
ADK GIS agent. It uses Rasterio/NumPy for raster inspection and xarray for NetCDF
verification. No third-party GIS MCP server is started and no candidate files
are modified.

Supported checks:

- Raster metadata, masked finite statistics, histogram counts and bin edges.
- NetCDF dimensions, coordinate endpoints, units and numerical summaries.
- Affine unit conversions, temporal resampling, wind magnitude and equal-weight
  single-time field means. Comparison requires matching coordinates and masks;
  reports MAE/RMSE, maximum error, units and the tolerances used.
- Existing AAJ image previews remain available for visual/scientific interpretation.

The specialist gathers evidence; the parent AAJ assigns the final score. Both
levels' inspection calls and numerical results are saved in
`evaluator_tool_calls`, and included in judge transcripts. Scientifically valid
alternative workflows can receive full credit. Missing archived evidence must
be described as unverified, rather than automatically attributed to the agent.

## Running selected tasks

```bash
python scripts/run_agent_judge.py \
  --results-file /path/to/results_candidate.json \
  --evaluator-model YOUR_JUDGE_MODEL \
  --evaluator-provider eve \
  --evaluator-label agent_judge_gis_v1 \
  --artifact-manifest /path/to/candidate_artifacts.json \
  --task-ids TASK_260427_000019_200019 TASK_260713_000020_600020 \
  --output-dir /path/to/new_judge_results
```

The manifest is candidate-specific and maps full task IDs to absolute paths:

```json
{
  "TASK_260427_000019_200019": [
    "/archive/candidate/task/sentinel2.tif",
    "/archive/candidate/task/ndvi.tif"
  ],
  "TASK_260713_000020_600020": [
    "/archive/candidate/task/temperature_K.nc",
    "/archive/candidate/task/temperature_C.nc"
  ]
}
```

Paths explicitly present in candidate tool arguments under the normal output
root are also allowed. Files mentioned only in prose or returned tool comments
must be supplied through the manifest. The judge does not browse other
candidates' shared output directories. Manifest ownership must be established
by the evaluation runner; inferred trace paths are not an independent proof of
provenance. Include source and derived artifacts, not just the final output.

Use a new evaluator label to avoid skipping previous evaluations. `--task-ids`
selects which tasks to judge while retaining the full candidate result file.
The default specialist budget is five turns per delegation, configurable with
`--gis-max-turns`.

## Interpretation and limitations

Histogram frequencies must be compared to the candidate's valid selected raster,
not an unrelated reference scene. Returned counts are exact numerical evidence;
a visual plot alone does not verify them. The inspector excludes nonfinite values
in addition to the raster mask, and reports that policy explicitly.

The specialist must justify conversion factors and aggregation from the task and
product. ERA5 monthly means of daily means energy-to-flux conversion uses 86400
seconds. Tolerances are recorded; defaults are not calibrated benchmark thresholds.
Temporal sums require review of accumulation conventions to avoid double counting.
Numerical agreement does not establish correct scene selection or scientific claims.

All comparisons currently load the relevant arrays into memory. Very large
artifacts require suitable memory or future chunked validators. The parent judge
and specialist use the same configured model/provider. No live API evaluation is
launched by these code changes.
