#!/usr/bin/env python3
"""
Interactive chat client for ai_assistant_server.

Features demonstrated:
  - Basic chat (POST /chat)
  - SSE streaming (POST /chat/stream)
  - System prompts & temperature control
  - Model listing and switching
  - Server health check

Requirements:
  pip install requests sseclient-py

Usage:
  # Start the server first:
  #   ai_assistant_server.exe --port 8090
  #
  # Then run this script:
  python python_chat.py
  python python_chat.py --url http://localhost:8090 --stream
  python python_chat.py --model llama3.2:1b --system "You are a pirate"
"""

import argparse
import json
import sys

try:
    import requests
except ImportError:
    print("Install requests:  pip install requests")
    sys.exit(1)


def check_health(base: str) -> bool:
    try:
        r = requests.get(f"{base}/health", timeout=3)
        return r.status_code == 200
    except requests.ConnectionError:
        return False


def list_models(base: str):
    r = requests.get(f"{base}/models", timeout=10)
    r.raise_for_status()
    data = r.json()
    models = data if isinstance(data, list) else data.get("models", [])
    print("\nAvailable models:")
    for m in models:
        name = m if isinstance(m, str) else m.get("name", m.get("id", str(m)))
        print(f"  - {name}")
    print()


def chat_simple(base: str, messages: list, model: str | None, temperature: float):
    """Send a message and get the full response at once."""
    body = {"messages": messages, "temperature": temperature}
    if model:
        body["model"] = model
    r = requests.post(f"{base}/chat", json=body, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data.get("response", data.get("content", json.dumps(data)))


def chat_stream(base: str, messages: list, model: str | None, temperature: float):
    """Send a message and stream the response via SSE."""
    body = {"messages": messages, "temperature": temperature}
    if model:
        body["model"] = model
    r = requests.post(f"{base}/chat/stream", json=body, stream=True, timeout=120)
    r.raise_for_status()

    full_response = []
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        payload = line[6:]  # strip "data: "
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
            token = chunk.get("token", chunk.get("content", ""))
            if token:
                print(token, end="", flush=True)
                full_response.append(token)
        except json.JSONDecodeError:
            pass
    print()
    return "".join(full_response)


def main():
    parser = argparse.ArgumentParser(description="Chat with ai_assistant_server")
    parser.add_argument("--url", default="http://localhost:8090", help="Server URL")
    parser.add_argument("--model", default=None, help="Model name (e.g. llama3.2:1b)")
    parser.add_argument("--system", default=None, help="System prompt")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature (0.0-2.0)")
    parser.add_argument("--stream", action="store_true", help="Enable SSE streaming")
    args = parser.parse_args()

    base = args.url.rstrip("/")

    # Health check
    if not check_health(base):
        print(f"Cannot connect to server at {base}")
        print("Start it with:  ai_assistant_server.exe --port 8090")
        sys.exit(1)

    print(f"Connected to {base}")
    list_models(base)

    # Build conversation history
    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
        print(f"System prompt: {args.system}\n")

    chat_fn = chat_stream if args.stream else chat_simple
    mode = "streaming" if args.stream else "sync"
    print(f"Mode: {mode} | Temperature: {args.temperature}")
    if args.model:
        print(f"Model: {args.model}")
    print("Type your message (Ctrl+C to quit)\n")

    try:
        while True:
            user_input = input("You> ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("/quit", "/exit", "exit", "quit"):
                break
            if user_input.lower() == "/models":
                list_models(base)
                continue

            messages.append({"role": "user", "content": user_input})

            if not args.stream:
                print("Assistant> ", end="", flush=True)
            else:
                print("Assistant> ", end="", flush=True)

            response = chat_fn(base, messages, args.model, args.temperature)

            if not args.stream:
                print(response)

            messages.append({"role": "assistant", "content": response})
            print()
    except (KeyboardInterrupt, EOFError):
        print("\nBye!")


if __name__ == "__main__":
    main()
