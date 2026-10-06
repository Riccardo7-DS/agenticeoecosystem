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
  1. The strategy defines a numerical-answer component or explicit scalar fields.
  2. ground_truths_by_id[task_ID] exists and isn't a status placeholder.
  3. At least one scalar field has an explicitly configured tolerance or
     a legacy band-stat/lat-lon preset. Configure numerical_field_tolerances
     in the strategy, or field_tolerances in the GT, keyed by dotted field
     paths with abs_tol, rel_tol and floor_scale. This selects exactly the
     required numeric outputs; distance uses the same metric as other scalars.
     Unselected GT fields are reported as unscored, not silently included.

Tasks failing any of these are left untouched, same convention as
rubric_evaluate_solutions() — "not an error", so a caller can run this over
every evaluation strategy's tasks in one TaskSet and let each judge pick up
only what applies to it.
"""

import json
import logging
from typing import Any, Dict, Optional

from tqdm import tqdm

from ..constants import MODEL_CLAUDE_SONNET
from ..dataclasses import Task, TaskSet
from ..evidence.candidate_evidence import CandidateEvidence
from ..metrics import numerical_metrics as nm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loading the two task_ID-keyed data sources this judge needs
# ---------------------------------------------------------------------------

def load_strategies_by_id(path) -> Dict[str, dict]:
    """task_ID -> its full entry in task_evaluation_strategies.json."""
    data = json.loads(open(path, encoding="utf-8").read())
    return {entry["task_ID"]: entry for entry in data["tasks"]}


def load_ground_truths_by_id(path) -> Dict[str, dict]:
    """Load the task-ID-keyed unified GT inventory.

    Entries include numerical GTs, source-backed sets/dictionaries and explicit
    error/skipped placeholders. Numerical scoring only uses recognized fields.
    """
    return json.loads(open(path, encoding="utf-8").read())


def _is_usable_gt(entry: Optional[dict]) -> bool:
    return bool(entry) and "_status" not in entry


def _has_numeric_component(entry: Optional[dict]) -> bool:
    """Recognize scalar numerical components, including those in hybrid tasks."""
    return bool(entry) and (
        entry.get("evaluation_strategy") == "numerical_answer"
        or entry.get("numerical_field_tolerances") is not None
        or any(component.get("type") == "numerical_answer"
               for component in entry.get("evaluation_components", []))
    )


# ---------------------------------------------------------------------------
# Extraction — one structured completion, no multi-turn tool loop needed
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT = """
You extract already-stated numeric values from an AI agent's answer to an
Earth Observation / GIS task — you do not compute, estimate, or look
anything up yourself. For each requested field, find the value the
candidate actually reported in its final written answer and submit it.
Tool outputs cannot replace a missing final-answer field. If a field was never stated, submit
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
    field_units: Optional[Dict[str, str]] = None,
) -> Dict[str, Optional[float]]:
    """One structured completion: what did the candidate actually claim for
    each of `field_names`? Returns {field_name: float_or_None}."""
    final_answer = (task.generated_solution.final_answer if task.generated_solution is not None else None) or ""
    if not final_answer.strip():
        return {name: None for name in field_names}
    from agenticeoecosystem.config.credentials import get_openai_client
    prompt = (
        f"<TASK>{task.task_text}</TASK>\n\n"
        f"<CANDIDATE FINAL ANSWER>\n{final_answer}\n</CANDIDATE FINAL ANSWER>\n\n"
        f"Requested fields (dotted paths, e.g. 'band_statistics.B02.mean'): {field_names}\n"
        f"Reference units: {json.dumps(field_units or {})}. Only convert explicitly stated compatible units; "
        "do not infer an unstated unit, calculate an unstated statistic, or use the reference as an answer. "
        "Use null for incompatible or ambiguous units."
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
    inspection_label: Optional[str] = None,
) -> Optional[Task]:
    """Score one task numerically if (and only if) it's in scope — see
    module docstring. Returns the same Task with fields populated if scored,
    or None if out of scope for this judge (left completely untouched, not
    even a None placeholder written — matches rubric_evaluate_solutions()'s
    "not an error" convention for tasks it doesn't own)."""
    strategy_entry = strategies_by_id.get(task.task_ID)
    if not _has_numeric_component(strategy_entry):
        return None

    gt_entry = ground_truths_by_id.get(task.task_ID)
    runtime_reference = None
    if gt_entry and gt_entry.get("reference_type") == "source_dependent_runtime":
        runtime_reference = CandidateEvidence.runtime_distance_reference(task, inspection_label)
        if runtime_reference is None:
            logger.info("Task %s: no verified candidate-bound runtime distance reference — unverified.", task.task_ID)
            return None
        gt_entry = {"distance_m": runtime_reference["distance_m"],
                    "field_tolerances": gt_entry.get("field_tolerances", {})}
    if not _is_usable_gt(gt_entry):
        logger.info("Task %s: numerical_answer strategy but no usable ground truth yet (status=%s) — skipping.",
                    task.task_ID, (gt_entry or {}).get("_status", "missing"))
        return None

    configured = strategy_entry.get("numerical_field_tolerances")
    if configured is None:
        configured = gt_entry.get("field_tolerances")
    specifications = None if configured is None else {
        field: nm.ToleranceSpec(**parameters) for field, parameters in configured.items()
    }
    gt_values, tolerances = nm.extract_scorable_fields(gt_entry, specifications)
    if not gt_values:
        logger.info("Task %s: ground truth has no fields this judge has a tolerance preset for — skipping.",
                     task.task_ID)
        return None

    field_names = list(gt_values.keys())
    predicted = {} if task.generated_solution is None else extract_candidate_values(
        task, field_names, model=model, provider=provider, temperature=temperature,
        field_units=strategy_entry.get("numerical_field_units"))
    # Fields the candidate never stated become genuinely missing (not merely
    # null) so score_item() applies the paper's zero-credit-for-missing rule.
    predicted = {k: v for k, v in predicted.items() if v is not None}

    item = nm.score_item(predicted, gt_values, tolerances)

    task.evaluator_scores_raw[evaluator_label] = item.num_score
    task.evaluator_scores[evaluator_label] = (
        2 if item.num_score >= 0.75 else 1 if item.num_score >= 0.25 else 0
    )
    task.evaluator_criteria[evaluator_label] = [field.to_dict() for field in item.fields]
    task.evaluator_evidence[evaluator_label] = {
        "method": "standardized_scalar_hit_at_tol_and_num_score",
        "hit_at_tol": item.hit_at_tol,
        "num_score": item.num_score,
        "field_tolerances": {
            field: {"abs_tol": spec.abs_tol, "rel_tol": spec.rel_tol,
                    "floor_scale": spec.floor_scale}
            for field, spec in tolerances.items()
        },
        "candidate_fingerprint": CandidateEvidence.fingerprint(task),
        "fields_scored": field_names,
        "fields_in_gt_not_scored": [k for k in _flatten_keys(gt_entry) if k not in gt_values],
        "runtime_reference": runtime_reference,
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
    inspection_label: Optional[str] = None,
) -> TaskSet:
    """Run the numeric tolerance judge over every in-scope task in `tasks`.
    Tasks without a numerical component, or that have no usable
    ground truth, or whose ground truth has no scorable fields, are left
    untouched (not an error — same convention as rubric_evaluate_solutions())."""
    n_total = len(tasks)
    n_candidates = sum(
        1 for t in tasks
        if _has_numeric_component(strategies_by_id.get(t.task_ID))
    )
    logger.info(
        "Numeric-tolerance evaluation — model=%s label=%s  %d/%d task(s) have numerical components",
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
                inspection_label=inspection_label,
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
