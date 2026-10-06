# Script entry points

Run commands from the repository root with the project environment active.
Shared evaluation and agent implementation lives under `src/agenticeoecosystem/`.

| Folder | Entry point | Purpose |
| --- | --- | --- |
| `generation/` | `run_generate.py` | Generate candidate answers; resume checkpoints with `--skip-solved`. |
| `generation/` | `prepare_missing_candidate_reruns.py` | Prepare validated inputs and queue candidate reruns. |
| `generation/` | `verify_tools.py` | Manual EVE tool verification; moved out of `tests/`, makes live calls. |
| `judging/` | `run_agent_judge.py` | AAJ with candidate-bound GIS/artifact evidence. |
| `judging/` | `run_full_evaluation.py` | One-command task-specific evaluation, resumable checkpoints and coverage. |
| `judging/` | `run_rubric_judges_batch.py` | Explicit trace/evidence rubric profile scoring. |
| `references/` | `regenerate_numeric_ground_truths.py` | Recompute numerical references from the configured local snapshots. |
| `reporting/` | `organize_judge_runs.sh` | Refresh historical-run index and pending rejudging inventory; no model calls. |
| `serving/` | Local model launchers | Machine-specific vLLM/VLM launchers, excluded from Git. |
| `local/installers/` | Downloaded `.deb` files | Local installation artifacts, excluded from Git. |
| `archive/reference_comparison/` | Retired scripts | Historical workflow-matching judge, orchestration and comparison tables. |

Only the supported generation/rerun entry points, judging, numerical references,
judge inventory and this index are versioned. Manual verification, archived
scripts, serving launchers, installers and root compatibility links are local
and ignored; they may be absent in a fresh checkout.

## Current operations

```bash
python scripts/generation/run_generate.py --help
python scripts/generation/prepare_missing_candidate_reruns.py --help
python scripts/judging/run_agent_judge.py --help
python scripts/judging/run_full_evaluation.py --help
python scripts/judging/run_rubric_judges_batch.py --help
bash scripts/reporting/organize_judge_runs.sh
```

The numerical reference generator uses known local paths; it is not a numerical
candidate-scoring CLI. Numerical candidate scoring lives in
`src/agenticeoecosystem/evaluation/judges/numerical_judge.py`. The unified
`run_full_evaluation.py` connects the available passes and explicitly reports
missing validators/evidence. See the [full evaluation guide](../docs/evaluation/full_evaluation.md).

## Retired entry points

`run_evaluate.py`, `run_judges_batch.py`, `run_pipeline_comparison.py`,
`benchmark.sh` and `build_table.py` now live exclusively in the archive. They
use reference-workflow matching or its 0/1/2 summaries and must not produce
current rankings. Keep them only for reproducing historical experiments.

Compatibility links at the root retain `run_generate.py` and
`prepare_missing_candidate_reruns.py` for existing queue commands, plus the
local `qwen36_serve.sh` launcher path for external service configuration. New
commands should use the categorized paths. The manual verification script has
no compatibility entry under `tests/` because it is not an automated test.
