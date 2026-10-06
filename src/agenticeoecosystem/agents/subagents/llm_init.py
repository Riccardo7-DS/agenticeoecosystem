from google.adk.models.lite_llm import LiteLlm
import base64
import logging
import os

from dotenv import load_dotenv


load_dotenv()
logger = logging.getLogger(__name__)

def mask_secret(value: str, visible: int = 4) -> str:
    """Return a masked version of a secret safe for logging (e.g. 'abcd****')."""
    if not value:
        return "<not set>"
    return value[:visible] + "*" * max(0, len(value) - visible)

def _basic_auth_header(username_env: str = "EVE_USERNAME",
                       password_env: str = "EVE_PASSWORD") -> dict:
    """Build an Authorization: Basic header from env vars. Never logs the values."""
    username = os.getenv(username_env, "")
    password = os.getenv(password_env, "")
    if not username or not password:
        logger.warning(
            "Credentials not found: %s=%s, %s=%s",
            username_env, mask_secret(username),
            password_env, mask_secret(password),
        )
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


_MODEL_NAME = os.getenv("MULTI_AGENT_MODEL", "EVE-Instruct")
_PROVIDER = os.getenv("MULTI_AGENT_PROVIDER", "eve")

# Reasoning models (e.g. DeepSeek-R1 distillations) spend a large chunk of
# their output budget on chain-of-thought before ever reaching a tool call
# or final answer, on every turn — the 4096 default that's fine for
# non-reasoning models risks truncating mid-thought for these. Bump the
# default for any model with "deepseek" in its name; an operator-set
# MULTI_AGENT_MAX_TOKENS always wins over either default.
_DEFAULT_MAX_TOKENS = 8192 if "deepseek" in _MODEL_NAME.lower() else 4096
_MAX_TOKENS = int(os.getenv("MULTI_AGENT_MAX_TOKENS", str(_DEFAULT_MAX_TOKENS)))

if _PROVIDER == "vllm":
    # Local vLLM server (see utils.registry.launch_vllm) — same OpenAI-compatible
    # endpoint the "single" and "multi_legacy" pipelines use, no EVE auth needed.
    from agenticeoecosystem.utils.registry import MODEL_REGISTRY
    _host = os.getenv("VLLM_HOST", "127.0.0.1")
    _port = os.getenv("VLLM_PORT", "8000")
    _model_id = MODEL_REGISTRY[_MODEL_NAME].model_id if _MODEL_NAME in MODEL_REGISTRY else _MODEL_NAME
    LLM_BASE_URL = f"http://{_host}:{_port}/v1"
    LLM_AUTH_HEADERS: dict | None = None
    llm = LiteLlm(model=f'openai/{_model_id}', base_url=LLM_BASE_URL, api_key="EMPTY", temperature=0, max_tokens=_MAX_TOKENS)
elif _PROVIDER == "openai":
    # Real OpenAI API — no gateway, no Basic Auth, real billing.
    LLM_BASE_URL = "https://api.openai.com/v1"
    LLM_AUTH_HEADERS = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', '')}"}
    llm = LiteLlm(model=f'openai/{_MODEL_NAME}', api_key=os.getenv("OPENAI_API_KEY"), temperature=0, max_tokens=_MAX_TOKENS)
elif _PROVIDER == "openrouter":
    # OpenRouter — _MODEL_NAME must include the vendor prefix it expects,
    # e.g. "openai/gpt-5.4". Real billing.
    LLM_BASE_URL = "https://openrouter.ai/api/v1"
    LLM_AUTH_HEADERS = {"Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY', '')}"}
    llm = LiteLlm(model=f'openrouter/{_MODEL_NAME}', api_key=os.getenv("OPENROUTER_API_KEY"), temperature=0, max_tokens=_MAX_TOKENS)
else:
    # Self-hosted vLLM behind the EVE gateway (Basic Auth), e.g. EVE-Instruct or Mistral.
    LLM_BASE_URL = os.getenv("EVE_ENDPOINT")
    LLM_AUTH_HEADERS = _basic_auth_header()
    llm = LiteLlm(model=f'openai/{_MODEL_NAME}', base_url=LLM_BASE_URL, headers=LLM_AUTH_HEADERS, api_key="EMPTY", temperature=0, max_tokens=_MAX_TOKENS)
