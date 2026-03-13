#!/usr/bin/env python3
"""
Use ai_assistant_server as a drop-in replacement for OpenAI.

This demonstrates that any tool expecting the OpenAI API format — such as
Continue.dev, Cursor, LangChain, LlamaIndex, etc. — works with ai_assistant_server
out of the box by pointing base_url to http://localhost:8090/v1.

Requirements:
  pip install openai

Usage:
  # Start the server:
  #   ai_assistant_server.exe --port 8090
  #
  python openai_compat.py
"""

import sys

try:
    from openai import OpenAI
except ImportError:
    print("Install the OpenAI SDK:  pip install openai")
    sys.exit(1)

# Point the OpenAI client at our local server
client = OpenAI(
    base_url="http://localhost:8090/v1",
    api_key="not-needed",  # required by the SDK but our server doesn't enforce it by default
)

# ---- Example 1: Simple completion -----------------------------------------
print("=== Simple completion ===")
response = client.chat.completions.create(
    model="llama3.2:1b",  # or whatever model Ollama has loaded
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ],
    temperature=0.3,
)
print(f"Response: {response.choices[0].message.content}\n")

# ---- Example 2: Streaming -------------------------------------------------
print("=== Streaming ===")
stream = client.chat.completions.create(
    model="llama3.2:1b",
    messages=[{"role": "user", "content": "Write a short poem about Rust programming."}],
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.content:
        print(delta.content, end="", flush=True)
print("\n")

# ---- Example 3: Multi-turn conversation -----------------------------------
print("=== Multi-turn conversation ===")
messages = [
    {"role": "system", "content": "You are a concise assistant. Reply in one sentence."},
]

questions = [
    "What is Rust?",
    "What makes it different from C++?",
    "Should a beginner learn it?",
]

for q in questions:
    messages.append({"role": "user", "content": q})
    r = client.chat.completions.create(model="llama3.2:1b", messages=messages, temperature=0.5)
    answer = r.choices[0].message.content
    print(f"Q: {q}")
    print(f"A: {answer}\n")
    messages.append({"role": "assistant", "content": answer})

# ---- Example 4: List models -----------------------------------------------
print("=== Available models ===")
models = client.models.list()
for m in models.data:
    print(f"  - {m.id}")
