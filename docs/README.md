# Documentation

For MCP server setup, interactive usage, wildfire workflows and adding agents,
see the [detailed setup and development guide](project_guide.md).

## Evaluation and judge status

Start with [judge run organization](evaluation/judge_run_organization.md) to
distinguish superseded scores from answers awaiting current-protocol judging.
The local [judge inventory](../data/judge_runs/README.md) lists the actual runs
and per-question rejudging status; it is generated and excluded from Git.

- [AAJ, GIS and artifact verification](evaluation/agent_judge_artifact_verification.md): candidate evidence, inspections, provenance and unverified cases.
- [One-command evaluation](evaluation/full_evaluation.md): all configured passes, resume and coverage reporting.
- [Numerical scoring](evaluation/numerical_scoring.md): Hit@tol, NumScore, units, field selection and tolerances.
- [Numerical reference generation](evaluation/numerical_gt_generation.md): generated references, source limitations and coverage.
- [Rubric v4 profiles](../data/tasks/rubrics_108q_v4_README.md): historical trace versus evidence-aware scoring.

Generic reference-workflow comparison is superseded. The old scoring entry
points still exist, so their output must not be treated as current-protocol
evaluation. Strategy mappings describe required checks, not completed runs.

## Benchmark operations

- [Candidate reruns](benchmarking/candidate_reruns.md): setup, model-specific inputs, checkpoints and evidence archiving.
- [Questions requiring reruns](../data/archive/tasks_20261013/benchmark_rerun_required.md): revised question contracts.

Task definitions, strategies, rubrics and ground truths stay under `data/tasks/`.
Executable entry points are grouped in the [script index](../scripts/README.md).

## Audits

These are dated findings or coverage records, not current judge scores.

- [Evaluation consistency audit](audits/evaluation_consistency_audit.json).
- [Numerical field coverage](audits/numerical_gt_coverage.json) (local, ignored).
- [MTG FCI acquisition-date audit](audits/mtg_fci_date_audit.md) and [JSON companion](audits/mtg_fci_date_audit.json) (local, ignored).

## Reference material

- [EO Dashboard narratives and themes](reference/eo_dashboard_narratives.md): captured catalogue, not a live inventory.
- `reference/MOOC.jsonl`: local transcript reference copy, excluded from Git.
- `reference/EO Science Strategy 2040 V6 - 20240911 (Final) (1).pdf`: local source document, excluded from Git.

## Historical descriptions

[Archive](archive/README.md) contains earlier benchmark descriptions that use
the old reference-comparison protocol. Preserve them as history rather than
using them as instructions for new judging.
