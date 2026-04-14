"""
save_results.py — Export conversation histories and evaluation results to HTML.

Adapt `conversation_turn_to_html` to match whatever your pipeline returns
as a conversation history (list of dicts, LangChain messages, etc.).
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any, List, Optional

from .dataclasses import Task


# ---------------------------------------------------------------------------
# HTML skeleton
# ---------------------------------------------------------------------------

HTML_HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: sans-serif; max-width: 960px; margin: 2em auto; padding: 0 1em; }}
  h1   {{ font-size: 1.4em; }}
  h2   {{ font-size: 1.1em; margin-top: 2em; color: #444; }}
  .user      {{ background: #f0f4ff; border-left: 4px solid #3b82f6; padding: .5em 1em; margin: .5em 0; }}
  .assistant {{ background: #f0fff4; border-left: 4px solid #22c55e; padding: .5em 1em; margin: .5em 0; }}
  .tool      {{ background: #fffbe6; border-left: 4px solid #f59e0b; padding: .5em 1em; margin: .5em 0; }}
  .meta      {{ color: #888; font-size: .85em; }}
  pre        {{ background: #f8f8f8; padding: .5em; overflow-x: auto; border-radius: 4px; }}
  img        {{ max-width: 100%; margin: .5em 0; }}
</style>
</head>
<body>
"""

HTML_FOOT = "\n</body>\n</html>\n"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_image(path: str) -> str:
    """Base64-encode a local image file for inline embedding."""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    ext = os.path.splitext(path)[-1].lstrip(".").lower()
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "svg": "svg+xml"}.get(ext, ext)
    return f'<img src="data:image/{mime};base64,{data}" />'


def conversation_turn_to_html(turn: Any) -> str:
    """
    Convert a single conversation turn to an HTML block.

    TODO: adapt this to match the shape of your pipeline's conversation history.
    Currently handles simple dict turns with keys: role, content.
    """
    # Example: turn = {"role": "user"|"assistant"|"tool", "content": "..."}
    role = turn.get("role", "unknown")
    content = turn.get("content", "")
    css_class = role if role in ("user", "assistant", "tool") else "meta"
    label = role.upper()
    return f'<div class="{css_class}"><strong>{label}</strong><pre>{content}</pre></div>\n'


# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------

def save_conversation_to_html(
    task: Task,
    conversation_history: Optional[List[Any]],
    output_folder: Path,
) -> None:
    """
    Save a task's conversation history and evaluation results to an HTML file.

    Args:
        task                : Task object (should have generated_solution set).
        conversation_history: List of conversation turns from your pipeline.
                              Pass None if not captured.
        output_folder       : Directory where the HTML file will be written.
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    title = f"Task {task.task_ID}"
    html_parts = [HTML_HEAD.format(title=title)]

    # Header
    html_parts.append(f"<h1>{title}</h1>\n")
    html_parts.append(f"<p><strong>Task:</strong> {task.task_text}</p>\n")

    # Labels
    if task.task_labels:
        labels_str = ", ".join(str(l) for l in task.task_labels)
        html_parts.append(f'<p class="meta">Labels: {labels_str}</p>\n')

    # Conversation
    if conversation_history:
        html_parts.append("<h2>Conversation</h2>\n")
        for turn in conversation_history:
            html_parts.append(conversation_turn_to_html(turn))

    # Generated solution
    html_parts.append("<h2>Generated Solution</h2>\n")
    if task.generated_solution:
        html_parts.append(f"<pre>{task.generated_solution}</pre>\n")
    else:
        html_parts.append("<p><em>No solution generated.</em></p>\n")

    # Evaluation
    html_parts.append("<h2>Evaluation</h2>\n")
    html_parts.append(f"<p><strong>LLM Score:</strong> {task.match_score_LLM}</p>\n")
    if task.match_reasoning_LLM:
        html_parts.append(f"<p>{task.match_reasoning_LLM}</p>\n")
    html_parts.append(f"<p><strong>Human Score:</strong> {task.match_score_Human}</p>\n")
    if task.match_reasoning_Human:
        html_parts.append(f"<p>{task.match_reasoning_Human}</p>\n")

    # Token usage
    if task.generated_solution_input_tokens is not None:
        html_parts.append(
            f'<p class="meta">Tokens — input: {task.generated_solution_input_tokens}, '
            f"output: {task.generated_solution_output_tokens}</p>\n"
        )

    html_parts.append(HTML_FOOT)

    out_path = output_folder / f"task_{task.task_ID}.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(html_parts)
