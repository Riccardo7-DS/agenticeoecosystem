# Scalar numerical scoring

All ordinary scalar quantities use the same metric, including geodetic distance,
area, reflectance statistics, climate values, percentages, coordinates and counts.
Ground-truth computation, source suitability and unit conversion happen before
this scorer. There is no separate scoring function for distance.

For ground truth `y`, prediction `y_pred` and field tolerances:

```text
effective_tol = max(abs_tol, rel_tol * abs(y), floor_scale)
absolute_error = abs(y_pred - y)
normalized_error = absolute_error / effective_tol
Hit@tol = 1 if normalized_error <= 1 else 0
NumScore = 1 if normalized_error <= 1 else 2 ** (-(normalized_error - 1))
```

`score_item` aligns predictions by ground-truth field key. It averages both scores
across every required field; missing, invalid and unparseable predictions get zero
and remain in the denominator. Extra predictions do not affect the mean. Numeric
strings are accepted if they parse to a finite scalar; units are not guessed from
strings such as `"100 km"`. NaN, infinity and booleans are invalid predictions.

Each field retains its effective tolerance and unrounded absolute/normalized
errors. Missing and invalid predictions have null errors because no finite error
can be calculated. Invalid GT or tolerance configuration is an evaluation error,
not a candidate zero. All tolerances must be finite and nonnegative, with a positive
effective tolerance for each GT value. Relative-only tolerance at `y=0` needs an
explicit positive `abs_tol` or `floor_scale`.

```python
# Illustrative tolerance values, not calibrated benchmark defaults.
from agenticeoecosystem.evaluation.numerical_metrics import ToleranceSpec, score_item

result = score_item(
    prediction={"distance_m": 1004.0, "area_km2": None},
    ground_truth={"distance_m": 1000.0, "area_km2": 2.0},
    tolerances={
        "distance_m": ToleranceSpec(abs_tol=2.0, rel_tol=0.0, floor_scale=0.0),
        "area_km2": ToleranceSpec(abs_tol=0.01, rel_tol=0.01, floor_scale=0.0),
    },
)
# mean Hit@tol = 0; mean NumScore = 0.25
# distance: absolute_error=4, effective_tol=2, normalized_error=2, NumScore=0.5
print(result.to_dict())
```

## Selecting fields in the numerical judge

Use `numerical_field_tolerances` in a task's evaluation-strategy entry, keyed by
required scalar GT paths. Alternatively store `field_tolerances` in its GT entry.
Strategy configuration takes precedence. Only configured fields are evaluated;
this avoids scoring unrequested statistics or a distance task's endpoint
coordinates. Paths can be nested, such as `transform.x`; literal dictionary keys
containing dots, such as `pct_pixels_ndvi_gt_0.5`, are also supported.

```json
{
  "numerical_field_tolerances": {
    "distance_m": {"abs_tol": 2.0, "rel_tol": 0.0, "floor_scale": 0.0}
  }
}
```

These numbers illustrate the schema; choose and record task-appropriate tolerances
before running benchmark scoring. No new task tolerances are inferred automatically.
Numerical components in hybrid tasks can use the same scorer. Their numeric score
is one component, not proof of complete task fulfillment.

Without explicit field configuration, existing band-statistic/latitude-longitude
extraction and tolerance presets remain available for compatibility. That legacy
fallback does not establish full output coverage. `tau_abs`, `tau_rel` and `floor`
remain accepted as aliases for the new canonical parameter names. Existing exact
matching for non-numeric fields and explicit positional alignment remain legacy
utilities; neither changes the scalar numeric formula.

The numerical judge stores mean NumScore in `evaluator_scores_raw`, mean Hit@tol
and NumScore in `evaluator_evidence`, and per-field diagnostics in
`evaluator_criteria`. Its existing 0/1/2 compatibility bucket remains in
`evaluator_scores`; use the two continuous/binary mean metrics for analysis.
GT records marked pending, source-dependent, error or skipped are not scored as
fixed GTs. For `200020`, AAJ must first verify the paper's study-area centroid and
an independent GIS calculation must supply the runtime distance GT. The generic
metric can then compare the reported scalar distance using declared tolerances.

Requested fields are now explicit in `task_evaluation_strategies.json`, including
projected coordinates, area, distance and NDVI percentages. Unrequested
statistics and incidental coordinates are excluded. New quantity-specific
tolerances are initial defaults requiring empirical calibration; they do not
introduce different scoring formulas. Prediction extraction targets the final
written answer and accepts explicit compatible unit conversion, without using
tool outputs to fill omitted final-answer values.

See `docs/numerical_gt_generation.md` for the 20-task reference-generation
report. Source-dependent observations must match validated source/support,
timestamps, masks, units and calibration, or be independently recomputed from
candidate-owned valid inputs. Field mappings alone do not implement automatic
source matching for every artifact-dependent task.

For source-dependent runtime distances, AAJ verifies both endpoints and the GIS
inspector records `measure_geodetic_distance` with a captured source quotation.
`CandidateEvidence.runtime_distance_reference` replays the WGS84 calculation,
checks source identity and candidate fingerprint, and rejects conflicting evidence.
Pass `inspection_label` to select the inspection run. Missing evidence remains
unscored; configure distance tolerances before applying the standard metric.
