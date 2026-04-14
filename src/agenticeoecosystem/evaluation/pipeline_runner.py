"""
pipeline_runner.py — Runs your agentic pipeline on a TaskSet and records results.

Replace the `run_pipeline` stub with your actual pipeline call.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from agenticeoecosystem.definitions import GEOBENCH_RESULTS
from .dataclasses import Solution, TaskSet


# ---------------------------------------------------------------------------
# TODO: implement this function to call your pipeline
# ---------------------------------------------------------------------------

def run_pipeline(task_text: str) -> tuple[Solution, int, int]:
    """
    Run your agentic pipeline on a single task.

    Args:
        task_text: The natural-language task prompt.

    Returns:
        solution         : Solution object with the agent's steps
        input_tokens     : Tokens consumed in the prompt
        output_tokens    : Tokens generated in the response
    """
    raise NotImplementedError(
        "Replace this stub with a call to your actual pipeline.\n"
        "Return (Solution, input_token_count, output_token_count)."
    )

    # Example skeleton:
    # response = my_agent.run(task_text)
    # steps = [Step(function_name=a.tool, arguments=a.args) for a in response.actions]
    # return Solution(steps=steps), response.usage.input, response.usage.output


# ---------------------------------------------------------------------------
# Batch runner — mirrors GeoBenchX generate_solutions()
# ---------------------------------------------------------------------------

def generate_solutions(
    tasks: TaskSet,
    model: str,
    temperature: float,
    output_filename: Optional[str] = None,
    skip_solved: bool = True,
    wait_between_tasks: float = 0.0,
) -> tuple[TaskSet, int, int]:
    """
    Iterate over all tasks in `tasks`, call `run_pipeline`, and store results.

    Args:
        tasks            : TaskSet to process.
        model            : Model identifier (stored in metadata).
        temperature      : Sampling temperature (stored in metadata).
        output_filename  : If set, save intermediate results after every task.
        skip_solved      : Skip tasks that already have a generated_solution.
        wait_between_tasks: Seconds to wait between API calls (rate-limit guard).

    Returns:
        tasks            : Same TaskSet with generated_solution fields populated.
        total_input      : Sum of input tokens across all tasks.
        total_output     : Sum of output tokens across all tasks.
    """
    tasks.metadata["model"] = model
    tasks.metadata["temperature"] = temperature

    total_input = 0
    total_output = 0

    for task in tqdm(tasks, desc="Generating solutions"):
        print(f"\nTask {task.task_ID}: {task.task_text}")

        if task.generated_solution is not None and skip_solved:
            print("  → Skipping (already solved).")
            continue

        try:
            solution, inp_tok, out_tok = run_pipeline(task.task_text)
            task.generated_solution = solution
            task.generated_solution_input_tokens = inp_tok
            task.generated_solution_output_tokens = out_tok
            total_input += inp_tok
            total_output += out_tok
            print(f"  → Done. Tokens: {inp_tok} in / {out_tok} out")
            print(str(solution))

        except Exception as e:
            print(f"  → ERROR: {e}")

        if output_filename:
            tasks.save_to_file(output_filename, folder=GEOBENCH_RESULTS)

        if wait_between_tasks:
            time.sleep(wait_between_tasks)

    tasks.metadata["total_input_tokens"] = total_input
    tasks.metadata["total_output_tokens"] = total_output

    if output_filename:
        tasks.save_to_file(output_filename, folder=GEOBENCH_RESULTS)

    print(f"\nDone. Total tokens — input: {total_input}, output: {total_output}")
    return tasks, total_input, total_output
