# Pipeline Eval Templates

Reusable scaffolding for manually benchmarking your agentic pipelines.
Modeled after [GeoBenchX](../README.md) but domain-agnostic.

## Structure

```
pipeline_eval/
├── constants.py        — Score values, task labels, model names, folder paths
├── dataclasses.py      — Pydantic schemas: Step, Solution, Task, TaskSet
├── pipeline_runner.py  — Stub for running your pipeline; batch generation loop
├── evaluator.py        — LLM-as-judge evaluator with prompts and batch runner
├── save_results.py     — HTML report generator for task conversations
├── Benchmarking.ipynb  — End-to-end workflow notebook
└── tasks/
    └── example_tasks.json  — Example task JSON (copy & edit for your own tasks)
```

## Quick start

### 1. Adapt the templates to your domain

| File | What to change |
|------|---------------|
| `constants.py` | `TaskLabels` enum, model names |
| `pipeline_runner.py` | `run_pipeline()` — call your agent here |
| `evaluator.py` | Prompts (`EVALUATION_TAXONOMY`, `SCORING_INSTRUCTIONS`, `EXAMPLES`) and the LLM client call |
| `save_results.py` | `conversation_turn_to_html()` — match your history format |

### 2. Add your tasks

Copy `tasks/example_tasks.json` and fill in:
- `task_text` — the prompt your pipeline receives
- `task_labels` — category tags matching your `TaskLabels` enum
- `reference_solutions` — one or more correct step sequences to compare against

### 3. Run the notebook

Open `Benchmarking.ipynb` and follow the cells:
1. Load tasks
2. Generate solutions (calls your pipeline)
3. Save HTML conversation logs (optional)
4. Score with LLM-as-judge
5. View summary stats and manually review individual tasks

## Task JSON schema

```json
{
  "task_ID": "TASK_YYMMDD_HHMMSS_RANDOM",
  "task_text": "Natural-language task prompt",
  "task_labels": ["Category A"],
  "reference_solution_description": "Grading notes (optional)",
  "reference_solutions": [
    {
      "steps": [
        { "function_name": "tool_name", "arguments": {}, "comment": null }
      ]
    }
  ],
  "generated_solution": null,
  "generated_solution_input_tokens": null,
  "generated_solution_output_tokens": null,
  "match_reasoning_LLM": null,
  "match_score_LLM": null,
  "match_reasoning_Human": null,
  "match_score_Human": null
}
```

## Score values

| Value | Label | Meaning |
|-------|-------|---------|
| 2 | MATCH | Correct, equivalent to reference |
| 1 | PARTIAL_MATCH | Right approach, minor differences |
| 0 | NO_MATCH | Wrong tools, data, or result |
