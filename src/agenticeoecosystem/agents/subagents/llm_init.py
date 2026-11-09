import base64
import logging
import os

from dotenv import load_dotenv
from google.adk.models.lite_llm import LiteLlm

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
        logger.warning("Credentials not found: %s=%s, %s=%s",
                       username_env, mask_secret(username), password_env, mask_secret(password))
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def _default_max_tokens(model_name: str) -> int:
    # Reasoning models (e.g. DeepSeek-R1 distillations) spend much of their budget on
    # chain-of-thought; 4096 risks truncating them mid-thought.
    return 8192 if "deepseek" in model_name.lower() else 4096


def make_llm_config(provider: str, model_name: str, max_tokens: int,
                    vllm_port: str | None = None) -> tuple[dict, str | None, dict | None]:
    """Returns (kwargs, base_url, auth_headers).
    kwargs work both as LiteLlm(**kwargs) and as litellm.acompletion(**kwargs)."""
    common = dict(temperature=0, max_tokens=max_tokens)

    if provider == "vllm":
        from agenticeoecosystem.utils.registry import MODEL_REGISTRY
        host = os.getenv("VLLM_HOST", "127.0.0.1")
        port = vllm_port or os.getenv("VLLM_PORT", "8000")
        model_id = MODEL_REGISTRY[model_name].model_id if model_name in MODEL_REGISTRY else model_name
        base = f"http://{host}:{port}/v1"
        return dict(model=f"openai/{model_id}", base_url=base, api_key="EMPTY", **common), base, None

    if provider == "openai":
        key = os.getenv("OPENAI_API_KEY", "")
        return (dict(model=f"openai/{model_name}", api_key=key, **common),
                "https://api.openai.com/v1", {"Authorization": f"Bearer {key}"})

    if provider == "openrouter":
        key = os.getenv("OPENROUTER_API_KEY", "")
        return (dict(model=f"openrouter/{model_name}", api_key=key, **common),
                "https://openrouter.ai/api/v1", {"Authorization": f"Bearer {key}"})

    if provider == "mistral":
        key = os.getenv("MISTRAL_API_KEY", "")
        return (dict(model=f"mistral/{model_name}", api_key=key, **common),
                os.getenv("MISTRAL_API_BASE", "https://api.mistral.ai/v1"),
                {"Authorization": f"Bearer {key}"})

    # Default: self-hosted vLLM behind the EVE gateway (Basic Auth).
    base, headers = os.getenv("EVE_ENDPOINT"), _basic_auth_header()
    return (dict(model=f"openai/{model_name}", base_url=base, headers=headers,
                 api_key="EMPTY", **common), base, headers)


_MODEL_NAME = os.getenv("MULTI_AGENT_MODEL", "EVE-Instruct")
_PROVIDER = os.getenv("MULTI_AGENT_PROVIDER", "eve")
_MAX_TOKENS = int(os.getenv("MULTI_AGENT_MAX_TOKENS", str(_default_max_tokens(_MODEL_NAME))))

llm_kwargs, LLM_BASE_URL, LLM_AUTH_HEADERS = make_llm_config(_PROVIDER, _MODEL_NAME, _MAX_TOKENS)
llm = LiteLlm(**llm_kwargs)