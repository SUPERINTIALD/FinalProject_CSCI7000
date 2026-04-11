from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio",
)

response = client.chat.completions.create(
    model="qwen3.5-0.8b",
    temperature=0.2,
    messages=[
        {"role": "system", "content": "You are a robotics planning assistant."},
        {"role": "user", "content": "Reply with exactly: local test successful"},
    ],
)

print(response.choices[0].message.content)