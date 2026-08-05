import asyncio
import logging
import os

from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

load_dotenv()
logger = logging.getLogger(__name__)

# --- 1. Model ---
# Using an SSH tunnel: ssh -L 18000:localhost:18000 <user>@<llava-vm-host>
vlm = LiteLlm(
    model="openai/Qwen3-VL",
    base_url=os.getenv("LLAVA_ENDPOINT","http://10.0.0.118:18001/v1" ),
    api_key="EMPTY",   # vLLM doesn't check this, but LiteLLM requires something non-empty
    temperature=0,
)
#"http://localhost:18000/v1"