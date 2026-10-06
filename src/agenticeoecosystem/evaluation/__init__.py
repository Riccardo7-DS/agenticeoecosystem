"""Benchmark evaluation components, loaded only when requested.

Use judges, metrics, evidence, runners, and reporting for current evaluation.
Superseded reference-comparison tools live in legacy.
"""
from importlib import import_module
from typing import Any

from .compatibility import MODULE_ALIASES

_EXPORTS: dict[str, str] = {
    "Task": "dataclasses",
    "TaskSet": "dataclasses",
    "Solution": "dataclasses",
    "Step": "dataclasses",
    "score_task_solution": "legacy.evaluator",
    "evaluate_solutions": "legacy.evaluator",
    "ModelRun": "legacy.model_comparison",
    "run_comparison": "legacy.model_comparison",
    "ended_in_error": "runners.pipeline_runner",
    "run_pipeline": "runners.pipeline_runner",
    "generate_solutions": "runners.pipeline_runner",
    "conversation_turn_to_html": "reporting.save_results",
    "save_conversation_to_html": "reporting.save_results",
    "FullEvaluationConfig": "runners.full_evaluation",
    "FullEvaluationRunner": "runners.full_evaluation",
    "ROLE_SYSTEM": "constants",
    "ROLE_USER": "constants",
    "ROLE_ASSISTANT": "constants",
    "NO_LABEL": "constants",
    "ScoreValues": "constants",
    "TaskLabels": "constants",
    "MODEL_CLAUDE_SONNET": "constants",
    "MODEL_CLAUDE_HAIKU": "constants",
    "MODEL_GPT_41": "constants",
    "MODEL_GPT_MINI": "constants",
    "MODEL_GEMINI_FLASH": "constants",
    "MODEL_GEMINI_PRO": "constants",
}
__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve public exports without eagerly loading judge dependencies."""
    if name in MODULE_ALIASES:
        value = import_module(f"{__name__}.{MODULE_ALIASES[name]}")
    elif name in _EXPORTS:
        value = getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
