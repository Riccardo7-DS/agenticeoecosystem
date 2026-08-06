import asyncio
import logging

from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

load_dotenv()
logger = logging.getLogger(__name__)

# --- 1. Model ---
# Reached via an SSH local port-forward from this machine, e.g.:
#   ssh -L 18001:localhost:18001 <user>@<llava-vm-host>
# `-L` always lands on localhost on THIS side regardless of the remote host,
# so this must stay "localhost", not a specific IP.
VLM_BASE_URL = "http://localhost:18001/v1"
vlm = LiteLlm(
    model="openai/Qwen3-VL",
    base_url=VLM_BASE_URL,
    api_key="EMPTY",   # vLLM doesn't check this, but LiteLLM requires something non-empty
    temperature=0,
)