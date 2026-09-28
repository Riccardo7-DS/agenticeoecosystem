# Pipeline Eval Templates

Reusable scaffolding for manually benchmarking your agentic pipelines.
Modeled after [GeoBenchX](../README.md) but domain-agnostic.

## Structure

```
pipeline_eval/
├── constants.py        — Score values, task labels, model names, folder paths
├── dataclasses.py      — Pydantic schemas: Step, Solution, Task, TaskSet
├── pipeline_runner.py  — Stub for running your pipeline; batch generation loop
├── evaluator.py        — one-shot LLM-as-judge: reads task + reference + trace as text
├── agent_judge.py      — tool-using agent-as-judge: lists output/<server>/, opens files
│                         the trace produced, and looks at image/GeoTIFF previews before scoring
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

### 4. Sample run + agent-as-judge (script-driven)

Try the tool-using judge on a small stratified sample before scaling to the
full task set — `logs/` gets the human-readable run log (via
`setup_logging()`), `--output-dir` gets `results_<label>.json` (the trace +
scores), and the actual produced artifacts already land under
`output/<server>/` as usual:

```bash
# 1. Generate on a 10-task stratified sample
uv run python scripts/run_generate.py \
    --label Sample --model EVE-Instruct --provider eve --pipeline multi \
    --tasks-file benchmark_tasks.json --sample-n 10 \
    --output-dir output/sample_run

# 2. Score that same sample with the tool-using agent-as-judge
uv run python scripts/run_agent_judge.py \
    --results-file output/sample_run/results_Sample.json \
    --evaluator-label agent_judge --evaluator-model claude-sonnet-4-6 --evaluator-provider eve
```

`agent_judge.py`'s judge can call `list_output_files`/`inspect_file` to
check what a step actually produced (and see a real image/GeoTIFF preview,
since the judge model is vision-capable) before scoring — see its module
docstring for how it differs from `evaluator.py`'s one-shot LLM-as-judge.

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
