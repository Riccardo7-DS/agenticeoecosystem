# Judge outputs and rejudging status

The local entry point is `data/judge_runs/README.md`, with machine-readable
candidate summaries in `data/judge_runs/index.json` and per-question checklists
in `data/judge_runs/pending/`. These generated files stay out of Git.

Refresh the inventory from the repository root:

```bash
bash scripts/reporting/organize_judge_runs.sh
```

## Historical outputs

Generic reference-comparison results (`*_reference_all`, `run_evaluate.py` and
`run_judges_batch.py`) are superseded and excluded from current rankings.
Their original directories remain intact because candidate preparation still
reads them. The organized historical view contains links and status records.
The old scoring commands are archived under `scripts/archive/reference_comparison/`;
use [the unified evaluation runner](full_evaluation.md) for the current protocol.

Historical rubric scores retain their original trace/evidence protocol. Trace
scores do not validate artifacts and must not be pooled with evidence scores.
Rerun answers and revised questions need fresh judgments. Numerical pilot and
audit outputs remain partial diagnostics; reuse requires matching provenance,
question versions, answers, references and evidence.

Judge logs have categorized links under `data/judge_runs/historical/`. Candidate
generation logs and checkpoints remain in place, including active runs.

## What needs judging again

Each pending candidate inventory overlays the latest available rerun checkpoint
on the historical candidate by task ID. It records the answer source, whether
an answer exists, whether the question matches the current benchmark, and the
configured evaluation methods. It intentionally excludes old scores.

`needs_rejudge` identifies recorded answers awaiting the current protocol.
`missing_answer` identifies questions with no recorded answer in the selected
source. Question-version mismatches require review or regeneration; do not
silently apply revised rubrics to older questions. Inventories are snapshots;
refresh after GLM or another candidate finishes.

For future evaluations, use distinct labels and directories per protocol and
candidate snapshot. Run AAJ/GIS where assigned, then applicable numerical and
evidence-aware rubric passes. Report verification coverage and unverified
cases along with scores. Do not substitute generic workflow matching for an
unimplemented configured validator.

`data/judge_runs/current/` is reserved for evaluated runs with explicit protocol,
source hashes, question versions and coverage. An empty directory means no run
has been registered there; it does not certify the absence of results elsewhere.
No judge API calls are performed by the organization script.
