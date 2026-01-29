from dotenv import load_dotenv
import os 
import httpx
import socket
import ssl
import urllib.request

load_dotenv()

# r = httpx.get(API_URL, headers={"Authorization": f"Bearer {API_KEY}"}, timeout=60.0, verify=False)
# print(r.status_code)


# # Extract hostname from URL
# from urllib.parse import urlparse
# hostname = urlparse(API_URL).hostname

# try:
#     ip = socket.gethostbyname(hostname)
#     print(f"Hostname {hostname} resolves to {ip}")
# except Exception as e:
#     print(f"DNS resolution failed: {e}")


# port = 443
# context = ssl.create_default_context()

# try:
#     with socket.create_connection((hostname, port), timeout=5) as sock:
#         with context.wrap_socket(sock, server_hostname=hostname) as ssock:
#             print(f"Connected to {hostname}:{port} with SSL")
# except Exception as e:
#     print(f"Connection failed: {e}")


# req = urllib.request.Request(
#     API_URL,
#     headers={
#         "Authorization": f"Bearer {API_KEY}",
#         "User-Agent": "curl/7.85.0",  # mimic curl
#         "Accept": "*/*"
#     }
# )

# try:
#     with urllib.request.urlopen(req, timeout=30) as response:
#         print(f"✅ HTTP status: {response.status}")
#         print(response.read().decode()[:200])
# except Exception as e:
#     print(f"❌ HTTP request failed: {e}")



# with httpx.Client(http2=True, timeout=30.0) as client:
#     r = client.get(API_URL, headers={"Authorization": f"Bearer {API_KEY}"})
#     print(r.status_code)
#     print(r.text[:200])


# import ssl
# print(ssl.OPENSSL_VERSION)

import os
import asyncio
import httpx

API_URL = os.getenv("MAIN_MODEL_URL")
API_KEY = os.getenv("MAIN_MODEL_API_KEY")
MODEL_NAME = os.getenv("MAIN_MODEL_NAME", "eve")

# data = {
#     "model": "eve-esa/eve_v0.1",
#     "messages": [{"role": "user", "content": "Explain async def"}]
# }

# with httpx.Client(timeout=60.0) as client:
#     r = client.post(API_URL,
#                     headers={
#                         "Authorization": f"Bearer {API_KEY}",
#                         "Content-Type": "application/json"
#                     },
#                     json=data)
#     print(r.status_code)
#     print(r.text)




class SimpleLLM:
    def __init__(self, api_url=None, api_key=None, model="eve", timeout=60, max_retries=3):
        self.api_url = api_url or os.getenv("MAIN_MODEL_URL")
        self.api_key = api_key or os.getenv("MAIN_MODEL_API_KEY")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

        if not self.api_url or not self.api_key:
            raise ValueError("API_URL and API_KEY must be provided either as arguments or environment variables.")

    async def chat(self, messages):
        """
        messages: list of dicts, e.g.
        [{"role": "user", "content": "Explain async def"}]
        """
        payload = {
            "model": self.model,
            "messages": messages
        }

        for attempt in range(1, self.max_retries + 2):  # max_retries + 1 tries
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        self.api_url,
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json"
                        },
                        json=payload
                    )
                    response.raise_for_status()
                    return response.json()  # returns full response JSON
            except httpx.RequestError as e:
                print(f"[Attempt {attempt}] Request error: {e}")
            except httpx.HTTPStatusError as e:
                print(f"[Attempt {attempt}] HTTP error: {e.response.status_code} - {e.response.text}")

        raise RuntimeError(f"Failed to get response after {self.max_retries + 1} attempts.")

# -----------------------------
# Example usage in an async context (works in Jupyter or scripts)
# -----------------------------
async def main():
    llm = SimpleLLM(model=MODEL_NAME)  # uses API_URL and API_KEY from .env
    messages = [{"role": "user", "content": "Explain async def"}]
    response = await llm.chat(messages)
    print(response)

# For scripts:
# asyncio.run(main())

# In Jupyter, just use:
# await main()
if __name__ == "__main__":
    asyncio.run(main())