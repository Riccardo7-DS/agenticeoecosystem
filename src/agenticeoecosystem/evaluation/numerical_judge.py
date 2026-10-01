"""
numerical_judge.py — Dispatches `evaluation_strategy == "numerical_answer"`
tasks to tolerance-based scoring (numerical_metrics.py's Hit@tol/NumScore)
against this project's curated static ground truths, instead of an LLM
judge comparing tool-call traces or a rubric.

Unlike evaluator.py / agent_judge.py / rubric_evaluator.py, the "judge" here
does almost no judgment: it (1) checks whether a task is in scope, (2)
extracts the candidate's claimed numeric field values via one structured
LLM completion, and (3) scores them deterministically against the fixed
ground truth. The LLM is only ever used for extraction (pulling numbers
already present in the candidate's own answer/trace), never for deciding
whether those numbers are "close enough" — that's numerical_metrics.py's
job, which involves no LLM at all.

A task is in scope only if ALL of:
  1. strategies_by_id[task_ID]["evaluation_strategy"] == "numerical_answer"
  2. ground_truths_by_id[task_ID] exists and isn't an "_status": error/skipped
     placeholder (see data/tasks/ground_truths/README.md)
  3. numerical_metrics.extract_scorable_fields() recognizes at least one
     field in it (today: band-stat groups and lat/lon pairs — see that
     module's BAND_STAT_TOLERANCE / LATLON_TOLERANCE). Other GT shapes this
     project has curated but not yet fixed a tolerance for (distance_km,
     area_km2, projected x/y, UTM zone/CRS codes) are NOT silently guessed
     at — those fields are just excluded from the averaged field set,
     honestly reported in `field_coverage` rather than masked.

Tasks failing any of these are left untouched, same convention as
rubric_evaluate_solutions() — "not an error", so a caller can run this over
every evaluation strategy's tasks in one TaskSet and let each judge pick up
only what applies to it.
"""

import json
import logging
from typing import Any, Dict, Optional

from tqdm import tqdm

from .constants import MODEL_CLAUDE_SONNET
from .dataclasses import Task, TaskSet
from .candidate_evidence import CandidateEvidence
from . import numerical_metrics as nm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loading the two task_ID-keyed data sources this judge needs
# ---------------------------------------------------------------------------

def load_strategies_by_id(path) -> Dict[str, dict]:
    """task_ID -> its full entry in task_evaluation_strategies.json."""
    data = json.loads(open(path, encoding="utf-8").read())
    return {entry["task_ID"]: entry for entry in data["tasks"]}


def load_ground_truths_by_id(path) -> Dict[str, dict]:
    """task_ID -> its entry in numerical_ground_truths_lean.json (either a
    real GT dict, or an {"_status": "error"/"skipped", ...} placeholder)."""
    return json.loads(open(path, encoding="utf-8").read())


def _is_usable_gt(entry: Optional[dict]) -> bool:
    return bool(entry) and "_status" not in entry


# ---------------------------------------------------------------------------
# Extraction — one structured completion, no multi-turn tool loop needed
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT = """
You extract already-stated numeric values from an AI agent's answer to an
Earth Observation / GIS task — you do not compute, estimate, or look
anything up yourself. For each requested field, find the value the
candidate actually reported (in its final answer or in a tool result shown
in its trace) and submit it. If a field was never stated anywhere, submit
null for it rather than guessing or computing it yourself.
""".strip()


def _extraction_tool_schema(field_names: list[str]) -> list[dict]:
    properties = {
        name: {"type": ["number", "null"], "description": f"The candidate's stated value for '{name}', or null if never stated."}
        for name in field_names
    }
    return [{
        "type": "function",
        "function": {
            "name": "submit_extracted_values",
            "description": "Submit the candidate's stated value for each requested field.",
            "parameters": {"type": "object", "properties": properties, "required": field_names},
        },
    }]


def extract_candidate_values(
    task: Task, field_names: list[str], model: str, provider: str, temperature: float = 0.0,
) -> Dict[str, Optional[float]]:
    """One structured completion: what did the candidate actually claim for
    each of `field_names`? Returns {field_name: float_or_None}."""
    from agenticeoecosystem.config.credentials import get_openai_client

    final_answer = (task.generated_solution.final_answer if task.generated_solution is not None else None) or ""
    evidence = CandidateEvidence.context(task)
    prompt = (
        f"<TASK>{task.task_text}</TASK>\n\n"
        f"<CANDIDATE FINAL ANSWER>\n{final_answer}\n</CANDIDATE FINAL ANSWER>\n\n"
        f"<CANDIDATE EVIDENCE (recorded tool outputs)>\n{json.dumps(evidence, ensure_ascii=False)[:24000]}\n</CANDIDATE EVIDENCE>\n\n"
        f"Requested fields (dotted paths, e.g. 'band_statistics.B02.mean'): {field_names}"
    )
    messages = [
        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    tools = _extraction_tool_schema(field_names)
    client = get_openai_client(provider=provider)
    resp = client.chat.completions.create(
        model=model, temperature=temperature, max_tokens=1500,
        messages=messages, tools=tools, tool_choice={"type": "function", "function": {"name": "submit_extracted_values"}},
    )
    message = resp.choices[0].message
    if not message.tool_calls:
        raise RuntimeError("Extraction model did not call submit_extracted_values.")
    args = json.loads(message.tool_calls[0].function.arguments)
    return {name: args.get(name) for name in field_names}


# ---------------------------------------------------------------------------
# Single-task scoring
# ---------------------------------------------------------------------------

def numeric_score_task_solution(
    task: Task,
    strategies_by_id: Dict[str, dict],
    ground_truths_by_id: Dict[str, dict],
    model: str = MODEL_CLAUDE_SONNET,
    provider: str = "eve",
    temperature: float = 0.0,
    evaluator_label: str = "numeric_tolerance",
) -> Optional[Task]:
    """Score one task numerically if (and only if) it's in scope — see
    module docstring. Returns the same Task with fields populated if scored,
    or None if out of scope for this judge (left completely untouched, not
    even a None placeholder written — matches rubric_evaluate_solutions()'s
    "not an error" convention for tasks it doesn't own)."""
    strategy_entry = strategies_by_id.get(task.task_ID)
    if not strategy_entry or strategy_entry.get("evaluation_strategy") != "numerical_answer":
        return None

    gt_entry = ground_truths_by_id.get(task.task_ID)
    if not _is_usable_gt(gt_entry):
        logger.info("Task %s: numerical_answer strategy but no usable ground truth yet (status=%s) — skipping.",
                    task.task_ID, (gt_entry or {}).get("_status", "missing"))
        return None

    gt_values, tolerances = nm.extract_scorable_fields(gt_entry)
    if not gt_values:
        logger.info("Task %s: ground truth has no fields this judge has a tolerance preset for — skipping.",
                     task.task_ID)
        return None

    if task.generated_solution is None:
        task.evaluator_scores[evaluator_label] = 0
        task.evaluator_scores_raw[evaluator_label] = 0.0
        task.evaluator_reasoning[evaluator_label] = "No generated solution — scored 0."
        return task

    field_names = list(gt_values.keys())
    predicted = extract_candidate_values(task, field_names, model=model, provider=provider, temperature=temperature)
    # Fields the candidate never stated become genuinely missing (not merely
    # null) so score_item() applies the paper's zero-credit-for-missing rule.
    predicted = {k: v for k, v in predicted.items() if v is not None}

    item = nm.score_item(predicted, gt_values, tolerances)

    task.evaluator_scores_raw[evaluator_label] = item.num_score
    task.evaluator_scores[evaluator_label] = (
        2 if item.num_score >= 0.75 else 1 if item.num_score >= 0.25 else 0
    )
    task.evaluator_criteria[evaluator_label] = [
        {"field": f.field, "y_hat": f.y_hat, "y": f.y, "hit": f.hit,
         "num_score": round(f.num_score, 4), "missing": f.missing}
        for f in item.fields
    ]
    task.evaluator_evidence[evaluator_label] = {
        "method": "numerical_metrics (TerraBench Hit@tol / NumScore)",
        "candidate_fingerprint": CandidateEvidence.fingerprint(task),
        "fields_scored": field_names,
        "fields_in_gt_not_scored": [k for k in _flatten_keys(gt_entry) if k not in gt_values],
    }
    missing_n = sum(1 for f in item.fields if f.missing)
    task.evaluator_reasoning[evaluator_label] = (
        f"Hit@tol={item.hit_at_tol:.2f}  NumScore={item.num_score:.2f}  "
        f"({len(item.fields) - missing_n}/{len(item.fields)} fields stated by candidate). "
        f"Scored fields: {field_names}."
    )
    return task


def _flatten_keys(d: Any, prefix: str = "") -> list[str]:
    """All leaf-field dotted paths in a GT blob, scored or not — used only
    to report field_coverage honestly in evaluator_evidence."""
    keys = []
    if isinstance(d, dict):
        for k, v in d.items():
            if k == "task_ID":
                continue
            new_prefix = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                keys.extend(_flatten_keys(v, new_prefix))
            else:
                keys.append(new_prefix)
    return keys


# ---------------------------------------------------------------------------
# Batch evaluator
# ---------------------------------------------------------------------------

def numeric_evaluate_solutions(
    tasks: TaskSet,
    strategies_by_id: Dict[str, dict],
    ground_truths_by_id: Dict[str, dict],
    model: str = MODEL_CLAUDE_SONNET,
    provider: str = "eve",
    temperature: float = 0.0,
    skip_evaluated: bool = True,
    evaluator_label: str = "numeric_tolerance",
) -> TaskSet:
    """Run the numeric tolerance judge over every in-scope task in `tasks`.
    Tasks whose strategy isn't numerical_answer, or that have no usable
    ground truth, or whose ground truth has no scorable fields, are left
    untouched (not an error — same convention as rubric_evaluate_solutions())."""
    n_total = len(tasks)
    n_candidates = sum(
        1 for t in tasks
        if strategies_by_id.get(t.task_ID, {}).get("evaluation_strategy") == "numerical_answer"
    )
    logger.info(
        "Numeric-tolerance evaluation — model=%s label=%s  %d/%d task(s) are numerical_answer strategy",
        model, evaluator_label, n_candidates, n_total,
    )

    n_scored = n_skipped = n_errors = 0
    for i, task in enumerate(tqdm(tasks, desc=f"Numeric-scoring [{evaluator_label}]"), 1):
        if (skip_evaluated and task.evaluator_scores.get(evaluator_label) is not None):
            continue
        try:
            result = numeric_score_task_solution(
                task, strategies_by_id, ground_truths_by_id,
                model=model, provider=provider, temperature=temperature,
                evaluator_label=evaluator_label,
            )
        except Exception:
            logger.error("Task %s: numeric scoring failed — leaving unscored, continuing batch.",
                         task.task_ID, exc_info=True)
            n_errors += 1
            continue
        if result is None:
            n_skipped += 1
            continue
        n_scored += 1
        logger.info("[%d/%d] Task %s: scored (num_score=%.2f)",
                    i, n_total, task.task_ID, task.evaluator_scores_raw.get(evaluator_label, 0.0))

    logger.info(
        "Numeric-tolerance evaluation complete — scored %d, skipped (out of scope) %d, errors %d",
        n_scored, n_skipped, n_errors,
    )
    return tasks
