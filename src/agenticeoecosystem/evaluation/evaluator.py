"""
evaluator.py — LLM-as-judge evaluator for agentic pipeline outputs.

Mirrors GeoBenchX evaluation.py. Adapt the prompts and scoring rules
to your domain before use.
"""

import logging
import re
import time
from typing import Optional

from tqdm import tqdm

from .constants import MODEL_CLAUDE_SONNET, ScoreValues
from .dataclasses import Task, TaskSet
from agenticeoecosystem.definitions import GEOBENCH_RESULTS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompts — edit these to match your evaluation criteria
# ---------------------------------------------------------------------------

EVALUATOR_PERSONA = """
You are an expert evaluator assessing the quality of an AI agent's solution
to a given task.
""".strip()


EVALUATION_TAXONOMY = """
Score the candidate solution against the reference solution(s) using this scale:

  2 — MATCH        : The candidate solution is correct and equivalent to the
                     reference. All key steps, tools, and outputs align.

  1 — PARTIAL MATCH: The candidate solution is partially correct. It takes the
                     right approach but differs in minor details (e.g. slightly
                     different parameters, a missing optional step, or an
                     alternative but acceptable method).

  0 — NO MATCH     : The candidate solution is incorrect, uses wrong tools or
                     data sources, or produces a different result.
""".strip()


SCORING_INSTRUCTIONS = """
Step-by-step instructions:
1. Read the TASK carefully.
2. Read each REFERENCE SOLUTION — there may be multiple valid approaches.
3. Read the CANDIDATE SOLUTION.
4. Identify whether the candidate matches any reference solution (fully or
   partially), or does not match at all.
5. Write a short MATCH REASONING (2-4 sentences).
6. State the MATCH SCORE as a single integer: 0, 1, or 2.

Output format (use these exact tags):
<MATCH REASONING>your reasoning here</MATCH REASONING>
<MATCH SCORE>0 or 1 or 2</MATCH SCORE>
""".strip()


# Few-shot examples — add at least 2-3 representative examples for your domain
EXAMPLES = """
--- EXAMPLE 1 ---
<TASK>...</TASK>
<REFERENCE SOLUTIONS>
<REFERENCE SOLUTION>
step_a(param="x")
step_b(param="y")
</REFERENCE SOLUTION>
</REFERENCE SOLUTIONS>
<CANDIDATE SOLUTION>
step_a(param="x")
step_b(param="y")
</CANDIDATE SOLUTION>
<MATCH REASONING>The candidate solution matches the reference exactly.</MATCH REASONING>
<MATCH SCORE>2</MATCH SCORE>

--- EXAMPLE 2 ---
<TASK>...</TASK>
<REFERENCE SOLUTIONS>
<REFERENCE SOLUTION>
step_a(param="x")
step_b(param="y")
</REFERENCE SOLUTION>
</REFERENCE SOLUTIONS>
<CANDIDATE SOLUTION>
step_a(param="z")
step_b(param="y")
</CANDIDATE SOLUTION>
<MATCH REASONING>The candidate uses the wrong parameter for step_a.</MATCH REASONING>
<MATCH SCORE>0</MATCH SCORE>
""".strip()


EVALUATION_PROMPT_TEMPLATE = """
{persona}

{taxonomy}

{instructions}

{examples}

Now evaluate the following:

<TASK>{task_text}</TASK>

<REFERENCE SOLUTIONS>
{reference_solutions}
</REFERENCE SOLUTIONS>

<CANDIDATE SOLUTION>
{candidate_solution}
</CANDIDATE SOLUTION>
""".strip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_reference_solutions(task: Task) -> str:
    parts = []
    for sol in (task.reference_solutions or []):
        parts.append(f"<REFERENCE SOLUTION>\n{sol}\n</REFERENCE SOLUTION>")
    return "\n".join(parts) if parts else "<REFERENCE SOLUTION>N/A</REFERENCE SOLUTION>"


def _parse_score(response_text: str) -> tuple[Optional[str], Optional[ScoreValues]]:
    reasoning_match = re.search(
        r"<MATCH REASONING>(.*?)</MATCH REASONING>", response_text, re.DOTALL
    )
    score_match = re.search(r"<MATCH SCORE>\s*([012])\s*</MATCH SCORE>", response_text)

    reasoning = reasoning_match.group(1).strip() if reasoning_match else None
    score = ScoreValues(int(score_match.group(1))) if score_match else None
    return reasoning, score


# ---------------------------------------------------------------------------
# Single-task evaluator
# ---------------------------------------------------------------------------

def score_task_solution(
    task: Task,
    model: str = MODEL_CLAUDE_SONNET,
    temperature: float = 0.0,
    evaluator_label: Optional[str] = None,
) -> Task:
    """
    Ask an LLM to score `task.generated_solution` against `task.reference_solutions`.
    Populates `task.match_reasoning_LLM` and `task.match_score_LLM`.

    Args:
        task        : Task object with generated_solution already set.
        model       : Evaluator model identifier.
        temperature : Sampling temperature (0 for deterministic scoring).

    Returns:
        The same Task object with LLM score fields populated.
    """
    if task.generated_solution is None:
        logger.info("Task %s: no generated solution — skipping evaluation.", task.task_ID)
        return task

    prompt = EVALUATION_PROMPT_TEMPLATE.format(
        persona=EVALUATOR_PERSONA,
        taxonomy=EVALUATION_TAXONOMY,
        instructions=SCORING_INSTRUCTIONS,
        examples=EXAMPLES,
        task_text=task.task_text,
        reference_solutions=_format_reference_solutions(task),
        candidate_solution=str(task.generated_solution),
    )

    # ---------------------------------------------------------------------------
    # Active: OpenAI-compatible endpoint via shared credentials helper.
    # provider controls which backend is used (set EVALUATOR_PROVIDER in .env).
    #
    #   "eve"    — EVE endpoint with Basic Auth (EVE_ENDPOINT, EVE_USERNAME, EVE_PASSWORD)
    #   "openai" — standard OpenAI API       (OPENAI_API_KEY)
    #   "custom" — any OpenAI-compatible URL  (OPENAI_ENDPOINT, OPENAI_API_KEY)
    #   "ollama" — local Ollama               (no credentials needed)
    # ---------------------------------------------------------------------------
    import os
    from agenticeoecosystem.config.credentials import get_openai_client
    from langchain.chat_models import init_chat_model

    _provider = os.getenv("EVALUATOR_PROVIDER", "eve")

    if _provider == "ollama":
        llm = init_chat_model(model, model_provider="ollama", temperature=temperature)
        response_text = llm.invoke(prompt).content
    else:
        _oai = get_openai_client(provider=_provider)
        resp = _oai.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        response_text = resp.choices[0].message.content

    reasoning, score = _parse_score(response_text)
    if evaluator_label:
        task.evaluator_scores[evaluator_label] = int(score) if score is not None else None
        task.evaluator_reasoning[evaluator_label] = reasoning
    else:
        task.match_reasoning_LLM = reasoning
        task.match_score_LLM = score
    logger.info("  → Score: %s | Reasoning: %s", score, (reasoning or "")[:120])
    return task


# ---------------------------------------------------------------------------
# Batch evaluator
# ---------------------------------------------------------------------------

def evaluate_solutions(
    tasks: TaskSet,
    model: str = MODEL_CLAUDE_SONNET,
    temperature: float = 0.0,
    output_filename: Optional[str] = None,
    skip_evaluated: bool = True,
    evaluator_label: Optional[str] = None,
) -> TaskSet:
    """
    Run the LLM evaluator over all tasks that have a generated solution.

    Args:
        tasks           : TaskSet with generated_solution fields populated.
        model           : Evaluator model identifier.
        temperature     : Sampling temperature.
        output_filename : Save intermediate results after each task if set.
        skip_evaluated  : Skip tasks that already have an LLM score.

    Returns:
        TaskSet with match_score_LLM / match_reasoning_LLM fields populated.
    """
    n_total = len(tasks)
    already_scored = (
        lambda t: t.evaluator_scores.get(evaluator_label) is not None
        if evaluator_label else t.match_score_LLM is not None
    )
    n_skip = sum(1 for t in tasks if already_scored(t) and skip_evaluated)
    n_no_solution = sum(1 for t in tasks if t.generated_solution is None)
    logger.info(
        "Evaluation phase — model=%s  label=%s  %d task(s) total, %d skipped (already scored), %d have no solution",
        model, evaluator_label or "default", n_total, n_skip, n_no_solution,
    )

    for i, task in enumerate(tqdm(tasks, desc=f"Evaluating [{evaluator_label or model}]"), 1):
        if task.generated_solution is None:
            continue
        if already_scored(task) and skip_evaluated:
            logger.info("[%d/%d] Task %s: skipped (already scored by %s)", i, n_total, task.task_ID, evaluator_label or "LLM")
            continue

        logger.info("[%d/%d] Task %s: %s", i, n_total, task.task_ID, task.task_text[:80])
        t0 = time.time()
        score_task_solution(task, model=model, temperature=temperature, evaluator_label=evaluator_label)
        logger.info("  → Scored in %.1fs", time.time() - t0)

        if output_filename:
            tasks.save_to_file(output_filename, folder=GEOBENCH_RESULTS)
            logger.debug("  → Intermediate results saved to %s", output_filename)

    summary = (
        tasks.score_summary_for_evaluator(evaluator_label)
        if evaluator_label else tasks.score_summary().get("LLM", {})
    )
    if summary.get("n", 0):
        logger.info(
            "Evaluation complete — match: %d  partial: %d  no_match: %d  accuracy: %.1f%%",
            summary.get("match", 0), summary.get("partial", 0), summary.get("no_match", 0),
            summary.get("accuracy_full", 0.0) * 100,
        )
    return tasks
