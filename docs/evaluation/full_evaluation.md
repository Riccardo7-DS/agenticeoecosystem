# Run the task-specific judges

One command runs the available evaluation passes for the supplied candidate
snapshots. It never invokes the retired reference-workflow comparison judge.

```bash
python scripts/judging/run_full_evaluation.py \
  --results-file data/candidate_reruns/missing_and_required_20261003/multi/results_Mistral-Small-2506_missing_rerun.json \
  --results-file data/candidate_reruns/missing_and_required_20261003/single/results_Mistral-Small-3.2-24B-single_missing_rerun.json \
  --evaluator-model Mistral-Small-3.2-24B-Instruct-2506 \
  --evaluator-provider eve \
  --evaluator-label mistral_current_v1 \
  --output-dir data/judge_runs/current/mistral_current_v1
```

Add `--dry-run` to produce a dispatch plan without model calls. The provider
must already be configured; a `vllm` provider uses an already-running server
and its served model ID. The runner does not launch candidate agents or models.
Each file is evaluated exactly as supplied. Partial rerun checkpoints contain
only their selected questions; use a merged current candidate snapshot for
whole-benchmark totals.

## Dispatch and evidence

The defaults use `task_evaluation_strategies.json`, `ground_truths.json`,
`benchmark_tasks.json` and `rubrics_108q_v4_evidence.json` under `data/tasks/`.
Use `--evaluation-strategies`, `--ground-truths`, `--benchmark-file` and
`--rubrics-file` to select a versioned alternate configuration. Questions must
match the selected benchmark and rubric wording. Version mismatches are blocked
and reported; missing answers are reported separately rather than invented.

1. AAJ inspects candidate-owned evidence for assigned inspection methods and
   evidence-profile rubrics. GIS consultation follows the strategy mapping.
2. Numerical scoring extracts explicitly stated final-answer fields and runs
   the existing Hit@tol/NumScore metric, using the AAJ label for runtime distance
   references. It stores per-field errors and tolerances. For explicitly scored
   latitude/longitude fields it also reports WGS84 geodetic position error in metres.
3. Deterministic answer checks extract literal results without supplying GT
   values to the extraction model. Metric computation supports canonical set
   precision/recall/F1/exact match, explicitly aliased string pairs, CRS
   equivalence, separately reported UTM zone/hemisphere and exact API IDs when
   the selected configuration supplies an executable contract.
4. Rubrics use their explicit trace/evidence protocol. The legacy `llm_judge`
   mapping is interpreted as the rubric pass only when `rubrics` is also
   configured; it never revives generic reference comparison.

Consultation or a filename does not establish validation. The runner retains
successful GIS measurements and whether comparisons supplied an actual
`verified` result. Methods without an executable contract are `unsupported`;
missing evidence is `unverified`. Scalar coordinate errors and geodetic
position error are reported as separate metrics.
Some strategy entries still contain placeholder validators, so a full dispatch
can finish with incomplete coverage. Inspect `method_coverage` before ranking.

Use `--artifact-manifest` only with one candidate file, mapping task IDs to
lists of independently established candidate-owned absolute paths. Recorded
archived artifacts are used automatically by the existing inspectors.

## Numerical/source validity

A computed error against stored GT is not proof that the candidate used the
same valid source, mask, calibration, dates, region or construction. Such
comparisons are retained as `diagnostic` and excluded from validated means.
Fixed identifier contracts and unchanged versioned server-catalogue references
can be checked automatically. Source-dependent distance uses the existing
verified/replayed GIS runtime reference.

To record an independently reviewed scalar/set reference alignment, supply
`--reference-validations PATH` with task-ID-keyed records:

```json
{
  "TASK_ID": {
    "status": "verified",
    "reason": "Record the actual independent source/support validation here.",
    "candidate_fingerprint": "SHA256 from CandidateEvidence.fingerprint(task)",
    "ground_truth_sha256": "SHA256 from full_evaluation.digest(the_task_GT_entry)",
    "artifact_state_sha256": "Actual artifact-state hash from the task's .protocol record"
  }
}
```

The artifact-state hash is required when the task has referenced local files;
it binds the validity review to their actual contents and availability.
These are evaluator-owned attestations, not candidate input. Never mark them
verified merely to obtain a score. Changing an answer, GT or artifact invalidates
resume. An unattested diagnostic result stays available for investigation.

## Checkpoints and outputs

The original input is never overwritten. Each candidate produces:

- `<candidate-stem>_<label>.json`: clean current-protocol results with separate
  `.aaj`, `.numeric`, `.deterministic`, `.rubric` and `.protocol` labels.
- `<candidate-stem>_<label>/summary.json`: per-method coverage, per-task statuses,
  validated means and diagnostic measurements.
- `<candidate-stem>_<label>/transcripts/`: AAJ/GIS tool records.
- Batch `summary.json` and `judging.log` in the selected output directory.

Dry runs write `plan.json` instead of scored candidate files. The evaluator
copies no old reference scores from the inputs. It does not blend unlike scales
into a single ranking number. Summary means cover validated/scored cases only;
report their denominators and the separate missing/unverified/diagnostic counts.

Resume is enabled by default. Run the same command again to reuse checkpoints
only when candidate, actual artifact state, configuration, implementation and
judge settings match. Errors retry; a retried inspection invalidates dependent
numerical/rubric scores. `--no-resume` reruns all selected passes. `--task-ids`
selects a subset while retaining the full candidate file. Local random seeds
are recorded; API model outputs need not be deterministic at temperature zero.

Exit status is 0 for a completed plan or complete coverage, 2 for a saved run
with missing/unverified/unsupported/diagnostic methods, and 1 for fatal input or
output errors. Partial results remain checkpointed. One output directory is
locked to prevent concurrent evaluators from overwriting each other.
