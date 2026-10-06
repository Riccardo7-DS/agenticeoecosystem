"""Shared formatting for reference evidence included in judge prompts."""
from ..dataclasses import Task


def format_reference_solutions(task: Task) -> str:
    """Render reference solutions using the existing prompt delimiters."""
    parts = [f"<REFERENCE SOLUTION>\n{solution}\n</REFERENCE SOLUTION>"
             for solution in (task.reference_solutions or [])]
    return "\n".join(parts) if parts else "<REFERENCE SOLUTION>N/A</REFERENCE SOLUTION>"
