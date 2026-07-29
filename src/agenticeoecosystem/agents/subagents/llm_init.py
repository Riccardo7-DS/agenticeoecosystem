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

llm=LiteLlm(model=f'openai/{_MODEL_NAME}', base_url="http://localhost:18000/v1", api_key="EMPTY", temperature=0)
