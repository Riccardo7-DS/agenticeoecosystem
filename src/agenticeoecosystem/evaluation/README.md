# Evaluation code

The current judging pipeline combines AAJ/GIS verification, rubric profiles,
numerical metrics, and configured deterministic checks. Run it through
`scripts/judging/run_full_evaluation.py`; see the
[runner guide](../../../docs/evaluation/full_evaluation.md) for arguments and coverage limits.

| Folder | Responsibility |
| --- | --- |
| `judges/` | Agent, rubric, numerical, and deterministic task scoring |
| `metrics/` | Pure numerical metrics and tolerance calculations |
| `evidence/` | Candidate evidence, artifact inspection, GIS verification, prompt formatting |
| `runners/` | Full judging orchestration and candidate generation pipelines |
| `reporting/` | Conversation exports |
| `legacy/` | Superseded 0/1/2 reference comparison and model comparison tooling |

Shared task schemas remain in `dataclasses.py`; shared constants remain in
`constants.py`. Legacy evaluators are excluded from the full judging runner.
Reference solutions may still provide judge evidence; this does not invoke the
retired reference-comparison scorer.

Use the canonical imports in new code:

```python
from agenticeoecosystem.evaluation.runners.full_evaluation import FullEvaluationConfig, FullEvaluationRunner
from agenticeoecosystem.evaluation.metrics.numerical_metrics import ToleranceSpec, score_item
from agenticeoecosystem.evaluation.dataclasses import Task, TaskSet
```

Package exports load lazily. `compatibility.py` preserves former flat module
paths as aliases to the canonical modules, including legacy imports for old
scripts. Importing the evaluation package alone does not initialize judges or
load the legacy evaluator.
