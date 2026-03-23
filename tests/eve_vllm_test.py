from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

resp = client.chat.completions.create(
    model="eve_v0.4",
    messages=[{"role": "user", "content": "Explain NDVI"}]
)

print(resp.choices[0].message.content)