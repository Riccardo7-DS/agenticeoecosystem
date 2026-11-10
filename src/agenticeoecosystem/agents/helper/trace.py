import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

# DECISION: how much of each tool response the evaluator sees.
MAX_RESPONSE_CHARS = 4000


@dataclass
class TraceStep:
    index: int
    tool: str
    args: dict
    response: str


@dataclass
class Interaction:
    agent_name: str
    question: Optional[str] = None
    question_source: str = "unknown"   # tells you how much to trust `question`
    answer: Optional[str] = None
    steps: list[TraceStep] = field(default_factory=list)
    last_model_text: str = ""          # fallback answer if the tool response is empty
    failed: bool = False
    interaction_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def trace_text(self) -> str:
        return "\n\n".join(
            f"[step {s.index}] {s.tool}({json.dumps(s.args, default=str)})\n-> {s.response}"
            for s in self.steps
        )


def to_text(x: Any, limit: int = MAX_RESPONSE_CHARS) -> str:
    s = x if isinstance(x, str) else json.dumps(x, default=str)
    return s if len(s) <= limit else s[:limit] + f"... [truncated {len(s) - limit} chars]"


def content_text(content) -> str:
    if not content or not getattr(content, "parts", None):
        return ""
    return "".join(getattr(p, "text", None) or "" for p in content.parts)


def numbered_lines(text: str, max_line: int = 400) -> list[str]:
    """Splits an answer into short lines so the LLM can label by index instead of
    copying text back (cheaper in tokens, and lengths are computed exactly)."""
    out = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if len(line) <= max_line:
            out.append(line)
        else:
            out += [s for s in re.split(r"(?<=[.!?])\s+", line) if s]
    return out