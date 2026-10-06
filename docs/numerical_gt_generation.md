# Numerical ground-truth generation and coverage

Generated 2026-10-02. The 20 previously missing numerical references now all have captured or computed values for their current task versions. FCI uses the approved revised date, 1 October 2024. This is reference generation, not a candidate/judge evaluation run.

| Task ID | Reference outputs | Status |
|---|---|---|
| `TASK_260427_000017_200017` | Area: 78.540722959 km² | computed geometry |
| `TASK_260713_000019_400019` | Area: 28.265827321 km² | computed geometry |
| `TASK_260713_000032_500011` | 11 label counts | captured graph snapshot |
| `TASK_260713_000034_500013` | 23 direct-link counts; all zero (schema caveat below) | captured graph snapshot |
| `TASK_260713_000049_500028` | 24 slots | captured catalogue snapshot |
| `TASK_260713_000053_600053` | 31 daily regional means (m); all values stored in GT | computed source series |
| `TASK_260713_000061_600061` | Regional mean: 0.01173761487 m | computed source observation |
| `TASK_260713_000062_600062` | Regional mean: 293.7860718 K | computed source observation |
| `TASK_260713_000063_600063` | Regional mean: 0.08864705265 m | computed source observation |
| `TASK_260713_000067_600067` | Regional mean: 300.2471924 K | computed source observation |
| `TASK_260713_000068_600068` | 31 daily regional means (Pa); all values stored in GT | computed source series |
| `TASK_260713_000070_600070` | Regional mean: 0.04649442434 m | computed source observation |
| `TASK_260713_000073_600073` | Regional mean: 32.28779221 degC | computed source observation |
| `TASK_260713_000081_600081` | [-20.0, -35.0, 55.0, 38.0] degrees | computed reference extent |
| `TASK_260713_000082_600082` | [24.0, -5.0, 40.0, 32.0] degrees | computed reference extent |
| `TASK_260713_000083_600083` | [60.0, 5.0, 100.0, 35.0] degrees | computed reference extent |
| `TASK_260716_000004_700016` | Five-channel min/max/mean reflectance statistics for the captured 1 October 2024 Alps observation; all stored in GT. | computed imagery observation |
| `TASK_260716_000005_700017` | HRV: min=3.001728, max=61.235260 (percent reflectance) | computed imagery observation |
| `TASK_260716_000006_700018` | IR_108: min=264.966187, max=314.908020; IR_120: min=264.795898, max=312.536957 (K) | computed imagery observation |
| `TASK_260716_000007_700019` | VIS006: min=7.738078, max=24.259922; VIS008: min=20.449223, max=35.321384 (percent reflectance) | computed imagery observation |

## Important scope

- Stored scene, region and series observations apply only to the same validated inputs, support, masks, calibration and units. A different permitted input needs independent recomputation; these are not universal answer keys.
- The three extent observations use documented reference clipping choices; existing candidate-specific extent recipes remain applicable.
- `500013` has 23 zero **direct graph-link** counts because the graph has no ScienceQuestion/StrategicObjective edges, in either direction. This does not imply zero scientific influence. The original Cypher reference omitted zero-link nodes; it now uses OPTIONAL MATCH. The task may need revision if indirect scientific synthesis was intended.
- `700016` was revised, with user approval, to 1 October 2024 because 10 March 2024 returned no products. Reflectance GT was regenerated successfully from all 40 chunks of one observation; rerun candidate generation and evaluation for the revised date and corrected path handoff.
- `700018` reference band identifiers were corrected to Satpy's IR_108 and IR_120. The question text is unchanged.
- EUMETSAT reading crops requested channels separately and slices stacked HRV geometry into its native component windows. Shifted windows are padded with NaN rather than interpolated; regressions verify preserved pixel statistics, georeferencing and explicit non-overlap handling. Real-imagery validation passed on the downloaded 20 February 2024 Nairobi scene: VIS006 (15×11), VIS008 (15×11), HRV (43×33), all finite with geometry matching data shape. VIS006/VIS008 statistics match the stored reference exactly. The validation record is `output/ground_truth/eum_numeric/700019_mixed_area_verification.json`.

## Coverage repair

All 18 previously stored ordinary numerical tasks now select their requested fields, as does the stored hybrid NDVI task. After generation, 40 stored numerical tasks/components have explicit field mappings (39 ordinary tasks plus the hybrid task). Runtime distance 200020 is configured separately; EVI 400026 now has a regenerated valid-source scene reference.

Projected x/y, geodetic area, distance in metres and NDVI percentages use the same standardized scalar metric. Additional initial tolerances are explicitly configurable and remain pending empirical calibration. Existing band-statistic and lat/lon defaults are retained.

Unrequested NDVI min/std/mean fields are excluded where the question asks for only a subset. ESRIN/ESAC coordinates and the duplicate kilometre distance are excluded from distance scoring. Reflectance range questions score only min/max, not incidental means.

Numeric prediction extraction uses the final written answer; trace/artifact values cannot rescue omitted final-answer fields. Units are supplied to the extractor; incompatible or ambiguous units produce missing values. Artifact validity and intermediate-step evaluation remain separate.

## Files

- `data/tasks/ground_truths.json`: consolidated references, provenance and explicit diagnostics.
- `data/tasks/task_evaluation_strategies.json`: requested fields, units and tolerance parameters.
- `docs/numerical_gt_coverage.json`: exact requested-field mappings and removed legacy fields.
- `scripts/regenerate_numeric_ground_truths.py`: executable NetCDF reference computation; exact local snapshot paths are documented in configuration.
- `output/ground_truth`: captured source responses, downloaded city inputs and constructed buffer geometries in this workspace.

## FCI download-to-reader handoff repair

The revised October 2024 FCI download returned 40 readable NetCDF chunks. Parent-identifier filename matching found none because chunk names differ from the catalogue identifier. The reference now passes exact `new_files` paths for its single-product download. Single-product download responses also populate per-product `paths` with the returned readable files; multi-product calls retain conservative attribution. Three regression tests cover single-product chunk attribution, multi-product separation and rejection of explicitly empty paths. The corrected retry succeeded: all five channels returned finite min/max/mean statistics, now stored in the consolidated GT file.

The earlier W National Park EVI retry was rejected with `invalid_client`. After the user replaced the OAuth client pair, direct authentication returned HTTP 200 and imagery retrieval succeeded. The stored reference mean is 0.4070399488270284 and maximum is 0.7570676449806785 over 1,001,000 valid pixels. Source imagery and EVI raster hashes, the compact approximately 10 m sampling region and the reflectance/mask calculation are recorded in the GT provenance.

Current primary numerical-answer coverage: 39 stored references, one source-dependent runtime distance reference (200020), and no pending static numerical GT. These counts exclude hybrid tasks.
