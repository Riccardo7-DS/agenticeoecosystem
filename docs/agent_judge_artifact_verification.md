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

## Candidate evidence and rubric scoring

New candidate runs preserve successful tool responses in `Step.tool_result`,
including retrieved narrative/paper text when the tool returns it. Existing
results without these responses cannot reconstruct them from final-answer prose.

Use `--archive-evidence` with `scripts/run_generate.py` to copy exact recorded
artifact paths into `<output-dir>/evidence/<results-name>/<task-ID>/...` after
each candidate task. The result JSON stores archive paths and SHA-256 hashes.
This option consumes disk space but avoids another download and protects files
from subsequent candidates overwriting the shared output directory. Paths not
explicitly recorded in tool arguments or structured responses are not discovered
by scanning folders; supply those through a candidate-owned manifest. Without
this flag, artifact preservation remains the caller's responsibility.

AAJ uses archived files in preference to mutable originals. Hash mismatches do
not silently fall back to the originals. The runner/manifest must establish
ownership; a candidate-written file path alone is not independent provenance.
Archived manifests are trusted evaluation-runner records, not candidate input.

The AAJ runner reads `task_evaluation_strategies.json` to enforce GIS consultation
for every task mapped to `gis_verification_agent`. `--evaluation-strategies`
selects another mapping. Calling the specialist is not itself proof of successful
verification: inspect its recorded tool results and unavailable checks.

For rubric evaluation, add `--inspect-evidence` to the existing rubric batch
command. This performs offline AAJ inspection before scoring, using the selected
judge model/provider and additional judge API calls; no retrieval/download tool
is exposed. To reuse an inspection already saved in the input result JSON, pass
`--inspection-evaluator-label <AAJ-label>` instead. A manifest can be supplied
with `--artifact-manifest` when evaluating exactly one candidate results file.

The rubric receives captured responses and candidate-bound inspection results.
AAJ overall scores are not passed as evidence. GIS tool outputs are measurements;
model summaries remain interpretations. `inspect_timeseries` reads saved values
and can compute monthly means from saved daily data, with explicit unweighted
spatial reduction and a bounded number of reported points. Verifying a saved
aggregation requires the original daily input; a monthly output alone supports
interpretation but does not prove the aggregation was correct.

Structured results are saved in `evaluator_criteria` and `evaluator_evidence`.
When grounding is unavailable, its criterion score is null/unverified while
scientific correctness and internal consistency may still be scored. A complete
weighted total is then null; weights are not silently redistributed. Missing or
truncated material evidence can also produce an unverified grounding criterion.
Genuinely absent written answers still score zero for the written-answer rubric.
Use a new evaluator label when comparing this evidence-aware protocol to older
judgments, and report evidence coverage alongside any ranking.

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

Approved climate-reference fixes (2026-10-01): monthly fetch tools do not accept
`area`; references retrieve the monthly field and clip it before regional
inspection. Clipping normalizes either longitude convention to -180–180 and
supports ascending or descending latitude. Aggregation accepts `time` and
`valid_time`; precipitation sums retain wholly missing cells as missing.
For the eight regional numeric questions, aggregate time per cell first, then
compute the unweighted mean across finite valid cells.

The six formerly pending numerical references now have runtime recomputation
recipes in the strategy mapping. Three compare derived fields against archived
source inputs; three measure actual grid-cell-centre W,S,E,N bounds using
`inspect_extent`. This does not populate fixed numeric GT values or launch
scoring. Source/region validity remains a separate AAJ check, and calibrated
numerical tolerances and candidate evidence are still required.
