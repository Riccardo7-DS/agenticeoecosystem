"""Structured LLM calls for the reviewer, on its own model (EVAL_PROVIDER / EVAL_MODEL)."""
import json
import logging
import os
import re
from typing import Optional, Type, TypeVar

import litellm
from pydantic import BaseModel, ValidationError

from agenticeoecosystem.agents.subagents.llm_init import _default_max_tokens, make_llm_config

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

# DECISION: reviewer model. Defaults to the agents' model, which is the worst option
# (a model judging itself). Set EVAL_PROVIDER / EVAL_MODEL (and EVAL_VLLM_PORT for a
# second local vLLM server).
_MAIN = (os.getenv("MULTI_AGENT_PROVIDER", "eve"), os.getenv("MULTI_AGENT_MODEL", "EVE-Instruct"))
EVAL_PROVIDER = os.getenv("EVAL_PROVIDER", _MAIN[0])
EVAL_MODEL_NAME = os.getenv("EVAL_MODEL", _MAIN[1])
EVAL_MAX_TOKENS = int(os.getenv("EVAL_MAX_TOKENS",
                                str(max(4096, _default_max_tokens(EVAL_MODEL_NAME)))))
eval_llm_kwargs, _, _ = make_llm_config(EVAL_PROVIDER, EVAL_MODEL_NAME, EVAL_MAX_TOKENS,
                                        vllm_port=os.getenv("EVAL_VLLM_PORT"))
if (EVAL_PROVIDER, EVAL_MODEL_NAME) == _MAIN:
    logger.warning("Reviewer uses the same model as the agents; scores will share their blind spots.")

# DECISION: set True if the reviewer backend supports response_format with a JSON schema.
USE_RESPONSE_FORMAT = False

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def _extract_json(text: str) -> str:
    text = _THINK.sub("", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in output")
    return text[start:end + 1]


async def run_structured(instruction: str, prompt: str, schema: Type[T],
                         usage: Optional[dict] = None, retries: int = 1) -> Optional[T]:
    """One structured call. Token usage is added to `usage` (kept apart from the
    system's own token counts, so benchmark numbers stay comparable)."""
    system = (instruction
              + "\n\nReply with ONE JSON object that validates against this JSON schema. "
                "No prose, no markdown fences.\n"
              + json.dumps(schema.model_json_schema()))
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    extra = {"response_format": schema} if USE_RESPONSE_FORMAT else {}

    for attempt in range(retries + 1):
        try:
            resp = await litellm.acompletion(messages=messages, **eval_llm_kwargs, **extra)
        except Exception:
            logger.exception("Reviewer LLM call failed")
            return None
        u = getattr(resp, "usage", None)
        if usage is not None and u is not None:
            usage["input"] += getattr(u, "prompt_tokens", 0) or 0
            usage["output"] += getattr(u, "completion_tokens", 0) or 0
            usage["calls"] += 1
        choice = resp.choices[0]
        text = choice.message.content or ""
        if choice.finish_reason == "length":
            logger.warning("Reviewer output hit max_tokens; raise EVAL_MAX_TOKENS.")
        try:
            return schema.model_validate_json(_extract_json(text))
        except (ValueError, ValidationError) as e:
            logger.warning("Invalid structured output (attempt %d): %s", attempt + 1, str(e)[:300])
            messages += [{"role": "assistant", "content": text},
                         {"role": "user", "content": f"Invalid: {str(e)[:500]}. "
                                                     "Return only the corrected JSON object."}]
    return None