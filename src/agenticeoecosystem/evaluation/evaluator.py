"""
evaluator.py — LLM-as-judge evaluator for agentic pipeline outputs.

Mirrors GeoBenchX evaluation.py. Adapt the prompts and scoring rules
to your domain before use.
"""

import re
from typing import Optional

from tqdm import tqdm

from .constants import MODEL_CLAUDE_SONNET, ScoreValues
from .dataclasses import Task, TaskSet
from agenticeoecosystem.definitions import GEOBENCH_RESULTS


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
        print(f"  Task {task.task_ID}: no generated solution — skipping evaluation.")
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

    # TODO: replace with your preferred LLM client
    # Example using the Anthropic SDK:
    # import anthropic
    # client = anthropic.Anthropic()
    # message = client.messages.create(
    #     model=model,
    #     max_tokens=1024,
    #     messages=[{"role": "user", "content": prompt}],
    # )
    # response_text = message.content[0].text

    raise NotImplementedError(
        "Plug in your LLM client here and assign the response text to `response_text`."
    )

    reasoning, score = _parse_score(response_text)  # noqa: F821
    task.match_reasoning_LLM = reasoning
    task.match_score_LLM = score
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
    for task in tqdm(tasks, desc="Evaluating solutions"):
        if task.generated_solution is None:
            continue
        if task.match_score_LLM is not None and skip_evaluated:
            continue

        score_task_solution(task, model=model, temperature=temperature)

        if output_filename:
            tasks.save_to_file(output_filename, folder=GEOBENCH_RESULTS)

    return tasks
