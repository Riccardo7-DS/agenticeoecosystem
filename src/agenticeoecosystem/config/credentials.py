"""
credentials.py — Helpers for building LLM clients without exposing secrets.

Credentials are read from environment variables (loaded via .env).
They are never logged or printed; use mask_secret() if you need to
display a credential value for debugging.
"""

import base64
import logging
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from openai import OpenAI

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


def get_openai_client(
    provider: str = "eve",
    base_url_env: str = "EVE_ENDPOINT",
) -> OpenAI:
    """
    Return a configured OpenAI-compatible client.

    Args:
        provider    : "eve"    — EVE endpoint with Basic Auth (default)
                      "openai" — standard OpenAI API using OPENAI_API_KEY
                      "custom" — any OpenAI-compatible endpoint; set
                                 OPENAI_ENDPOINT and OPENAI_API_KEY in .env
        base_url_env: env-var name that holds the base URL (only for "eve"
                      and "custom" providers).

    Returns:
        Configured openai.OpenAI instance.
    """
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY is not set.")
        return OpenAI(api_key=api_key)

    if provider == "custom":
        return OpenAI(
            base_url=os.getenv(base_url_env),
            api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
        )

    if provider == "vllm":
        host = os.getenv("VLLM_HOST", "127.0.0.1")
        port = os.getenv("VLLM_PORT", "8000")
        return OpenAI(base_url=f"http://{host}:{port}/v1", api_key="EMPTY")

    # Default: EVE with Basic Auth
    return OpenAI(
        base_url=os.getenv(base_url_env),
        api_key="EMPTY",
        default_headers=_basic_auth_header(),
    )


class _EVEChatOpenAI(ChatOpenAI):
    """ChatOpenAI that normalises tool-message content to plain strings.

    The EVE API rejects list-typed content and LangChain's internal lc_* id
    fields in tool messages.
    """

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        for msg in payload.get("messages", []):
            if msg.get("role") == "tool" and isinstance(msg.get("content"), list):
                msg["content"] = " ".join(
                    item.get("text", str(item)) if isinstance(item, dict) else str(item)
                    for item in msg["content"]
                )
        return payload


def get_langchain_chat_openai(
    model_name: str,
    temperature: float = 0.0,
    provider: str = "eve",
    base_url_env: str = "EVE_ENDPOINT",
):
    """
    Return a LangChain ChatOpenAI configured for the given provider.

    Args:
        model_name  : Model identifier string.
        temperature : Sampling temperature.
        provider    : "eve" | "openai" | "custom" (same semantics as get_openai_client).
        base_url_env: env-var name holding the base URL (for "eve" / "custom").

    Returns:
        langchain_openai.ChatOpenAI instance.
    """
    if provider == "openai":
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    if provider == "custom":
        return ChatOpenAI(
            base_url=os.getenv(base_url_env),
            api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
            model=model_name,
            temperature=temperature,
        )

    # EVE: use subclass that strips list content / extra fields from tool messages
    return _EVEChatOpenAI(
        base_url=os.getenv(base_url_env),
        api_key="EMPTY",
        model=model_name,
        temperature=temperature,
        default_headers=_basic_auth_header(),
    )
