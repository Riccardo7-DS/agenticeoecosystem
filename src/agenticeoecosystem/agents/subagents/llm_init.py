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

if _PROVIDER == "vllm":
    # Local vLLM server (see utils.registry.launch_vllm) — same OpenAI-compatible
    # endpoint the "single" and "multi_legacy" pipelines use, no EVE auth needed.
    from agenticeoecosystem.utils.registry import MODEL_REGISTRY
    _host = os.getenv("VLLM_HOST", "127.0.0.1")
    _port = os.getenv("VLLM_PORT", "8000")
    _model_id = MODEL_REGISTRY[_MODEL_NAME].model_id if _MODEL_NAME in MODEL_REGISTRY else _MODEL_NAME
    llm = LiteLlm(model=f'openai/{_model_id}', base_url=f"http://{_host}:{_port}/v1", api_key="EMPTY", temperature=0, max_tokens=1024)
else:
    llm = LiteLlm(model=f'openai/{_MODEL_NAME}', base_url=os.getenv("EVE_ENDPOINT"), headers=_basic_auth_header(), api_key="EMPTY", temperature=0, max_tokens=1024)
