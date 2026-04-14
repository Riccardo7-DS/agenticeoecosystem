"""
constants.py — Enums, model names, score values, and folder paths.

Adapt the TaskLabels and model names to match your pipeline's domain.
"""

from enum import Enum, IntEnum
# ---------------------------------------------------------------------------
# Roles (used when building message lists for the LLM evaluator)
# ---------------------------------------------------------------------------
ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"


# ---------------------------------------------------------------------------
# Models
# Add or remove entries as your evaluation grows.
# ---------------------------------------------------------------------------
MODEL_CLAUDE_SONNET = "claude-sonnet-4-6"
MODEL_CLAUDE_HAIKU = "claude-haiku-4-5-20251001"
MODEL_GPT_41 = "gpt-4.1-2025-04-14"
MODEL_GPT_MINI = "gpt-4.1-mini-2025-04-14"
MODEL_GEMINI_FLASH = "gemini-2.0-flash-001"
MODEL_GEMINI_PRO = "gemini-2.5-pro-preview-05-06"

# ---------------------------------------------------------------------------
# Score values used by the LLM-as-judge evaluator and for human scoring
# ---------------------------------------------------------------------------
class ScoreValues(IntEnum):
    NO_MATCH = 0
    PARTIAL_MATCH = 1
    MATCH = 2

    @classmethod
    def values(cls):
        return [m.value for m in cls]

    @classmethod
    def names(cls):
        return [m.name for m in cls]


# ---------------------------------------------------------------------------
# Task labels — edit these to reflect the categories in your benchmark.
# ---------------------------------------------------------------------------
class TaskLabels(str, Enum):
    # TODO: replace / extend with your own categories
    CATEGORY_A = "Category A"
    CATEGORY_B = "Category B"
    CATEGORY_C = "Category C"
    HARD = "Hard"
    CONTROL = "Control question"


NO_LABEL = "<NO_LABEL>"
