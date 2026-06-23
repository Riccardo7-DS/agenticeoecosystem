"""Quick connectivity test for the EVE LLM endpoint."""

import base64
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

endpoint = os.getenv("EVE_ENDPOINT")
username = os.getenv("EVE_USERNAME")
password = os.getenv("EVE_PASSWORD")
model    = os.getenv("MAIN_MODEL_NAME", "EVE-Instruct")

print(f"EVE_ENDPOINT  : {endpoint or '*** NOT SET ***'}")
print(f"EVE_USERNAME  : {username or '*** NOT SET ***'}")
print(f"EVE_PASSWORD  : {'***' if password else '*** NOT SET ***'}")
print(f"MAIN_MODEL_NAME: {model}")
print()

if not endpoint:
    sys.exit("ERROR: EVE_ENDPOINT is not set. Add it to your .env file.")
if not username or not password:
    sys.exit("ERROR: EVE_USERNAME or EVE_PASSWORD is not set.")

credentials = base64.b64encode(f"{username}:{password}".encode()).decode()

client = OpenAI(
    base_url=endpoint,
    api_key="EMPTY",
    default_headers={"Authorization": f"Basic {credentials}"},
)

print("Sending test prompt to EVE...")
try:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        max_tokens=10,
    )
    print(f"SUCCESS — EVE replied: {response.choices[0].message.content!r}")
except Exception as e:
    print(f"FAILED — {e}")
