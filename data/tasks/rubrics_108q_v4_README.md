# Rubric version 4.0

These profiles keep the 108 task IDs, original questions, three criterion types,
and 0.5/0.3/0.2 weights. Version 3 remains unchanged in `../archive/tasks_20261013/rubrics_108q_v3.json`.

## Historical runs: trace profile

Use `rubrics_108q_v4_trace.json`. The judge receives the original question,
recorded tool names and arguments, and final answer. All candidates use this
same evidence boundary, even when some also have captured responses or files.
The main score measures task fulfillment supported by the recorded trace and
reported results. It does not verify execution, source fidelity, observed
features, numerical outputs, or the existence of saved files.

Task-specific expectations remain in the criteria. Evidence-dependent wording
now refers to specific reported observations and consistency with recorded
calls. Generic scientific background, a bare success claim, or troubleshooting
alone does not earn credit for the requested result. Explicit failures with no
requested result and clearly wrong targets with no valid requested result score
zero on all criteria. Partial requested results are assessed individually.

The profile uses numeric 0/1/2 scores and the original weighted sum. Available
artifact/source verification can be reported separately; it must not be added
selectively to some candidates' trace-profile prompts.

## Future runs: evidence profile

Use `rubrics_108q_v4_evidence.json`. All task-specific descriptions are identical
to version 3. Its shared instructions explicitly require candidate-bound source
responses and relevant visual/numerical artifacts or recorded inspections for
material evidence-dependent claims. A filename or tool call does not establish
successful execution. Evidence can be nonvisual for source/text/numerical tasks;
visual observations require the corresponding visual evidence.

Missing material evidence is unverified, not proof of candidate failure.
Evidence-dependent criteria remain null when they cannot be assessed; any null
criterion leaves the weighted total null. Explicit failures with no requested
result still receive zero under the shared task-fulfillment rule.

## Running and comparing

`scripts/run_rubric_judges_batch.py` reads `protocol_version`, `evidence_mode`
and `scoring_instructions` from each rubric and enforces one profile per batch.
Trace runs reject inspection flags/manifests to maintain uniform judge inputs.
Evidence runs can use `--inspect-evidence`, `--inspection-evaluator-label`, or
candidate-owned manifests as documented in `docs/evaluation/agent_judge_artifact_verification.md`.
Direct callers of `rubric_evaluate_solutions` must also pass `rubric_protocols`;
direct calls of `rubric_score_task_solution` must pass `rubric_protocol`.

Use distinct evaluator labels and output directories, for example
`Mistral-Small-3.2_rubric_v4_trace` and `Mistral-Small-3.2_rubric_v4_evidence`.
Version/profile and a hash of criteria/instructions are stored with each score,
preventing reuse of scores from a different rubric under the same label. Do not
combine profile scores or historical-v3 scores into a single-method ranking.

For the historical comparison, exclude these three revised questions from
every candidate until answers to the revised tasks exist:

- `TASK_260713_000061_500040`
- `TASK_260713_000071_600071`
- `TASK_260713_000074_500053`

All 108 questions remain in the master rubric files for future runs. To score
the current historical comparison, use `rubrics_105q_v4_trace_historical.json`,
which preserves each entry's protocol fields. No evaluation is started by creating
or selecting these profiles.

`rubrics_108q_v4_manifest.json` records version lineage and source hash.
`rubrics_108q_v4_changes.json` lists each trace criterion before and after.
