#!/usr/bin/env bash
# ============================================================================
# curl examples for ai_assistant_server
#
# Start the server first:
#   ai_assistant_server.exe --port 8090
#
# If you set --api-key, add:  -H "Authorization: Bearer YOUR_KEY"
# ============================================================================

BASE="http://localhost:8090"

# --- Health check -----------------------------------------------------------
echo "=== Health Check ==="
curl -s "$BASE/health" | python3 -m json.tool 2>/dev/null || curl -s "$BASE/health"
echo -e "\n"

# --- List models ------------------------------------------------------------
echo "=== List Models ==="
curl -s "$BASE/models" | python3 -m json.tool 2>/dev/null || curl -s "$BASE/models"
echo -e "\n"

# --- Simple chat ------------------------------------------------------------
echo "=== Simple Chat ==="
curl -s -X POST "$BASE/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello! What can you do?"}]
  }' | python3 -m json.tool 2>/dev/null || echo "(raw response above)"
echo -e "\n"

# --- Chat with system prompt and temperature --------------------------------
echo "=== Chat with System Prompt & Temperature ==="
curl -s -X POST "$BASE/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a concise assistant. Answer in one sentence."},
      {"role": "user", "content": "Explain what Rust is."}
    ],
    "temperature": 0.3
  }' | python3 -m json.tool 2>/dev/null || echo "(raw response above)"
echo -e "\n"

# --- Chat with model selection ----------------------------------------------
echo "=== Chat with Specific Model ==="
curl -s -X POST "$BASE/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Count from 1 to 5."}],
    "model": "llama3.2:1b",
    "temperature": 0.0
  }' | python3 -m json.tool 2>/dev/null || echo "(raw response above)"
echo -e "\n"

# --- Multi-turn conversation ------------------------------------------------
echo "=== Multi-turn Conversation ==="
curl -s -X POST "$BASE/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "My name is Alice."},
      {"role": "assistant", "content": "Hello Alice! Nice to meet you."},
      {"role": "user", "content": "What is my name?"}
    ]
  }' | python3 -m json.tool 2>/dev/null || echo "(raw response above)"
echo -e "\n"

# --- SSE Streaming ----------------------------------------------------------
echo "=== SSE Streaming ==="
echo "(tokens appear one by one)"
curl -s -N -X POST "$BASE/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Write a haiku about programming."}]
  }'
echo -e "\n\n"

# --- OpenAI-compatible endpoint ---------------------------------------------
echo "=== OpenAI-Compatible Endpoint ==="
curl -s -X POST "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:1b",
    "messages": [{"role": "user", "content": "Say hi in 3 words."}],
    "temperature": 0.5
  }' | python3 -m json.tool 2>/dev/null || echo "(raw response above)"
echo -e "\n"

# --- OpenAI-compatible with streaming (same format as OpenAI SSE) -----------
echo "=== OpenAI-Compatible Streaming ==="
curl -s -N -X POST "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:1b",
    "messages": [{"role": "user", "content": "Count to 3."}],
    "stream": true
  }'
echo -e "\n\n"

# --- Get current config -----------------------------------------------------
echo "=== Server Config ==="
curl -s "$BASE/config" | python3 -m json.tool 2>/dev/null || curl -s "$BASE/config"
echo -e "\n"

# --- Prometheus metrics ------------------------------------------------------
echo "=== Prometheus Metrics (first 20 lines) ==="
curl -s "$BASE/metrics" | head -20
echo -e "\n"

# --- OpenAPI spec -----------------------------------------------------------
echo "=== OpenAPI Spec (first 30 lines) ==="
curl -s "$BASE/openapi.json" | python3 -m json.tool 2>/dev/null | head -30
echo -e "\n"

# --- With API key (if the server was started with --api-key) ----------------
# echo "=== Authenticated Request ==="
# curl -s -X POST "$BASE/chat" \
#   -H "Content-Type: application/json" \
#   -H "Authorization: Bearer your-api-key-here" \
#   -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
