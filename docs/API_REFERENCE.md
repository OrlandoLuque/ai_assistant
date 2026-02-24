# API Reference — ai_assistant HTTP Server

> Version: v10 (2026-02-24)
>
> Base URL: `http://localhost:8090` (default)
>
> All endpoints are available with and without the `/api/v1/` prefix.
> Both `/health` and `/api/v1/health` route to the same handler.

---

## Table of Contents

- [Authentication](#authentication)
- [Common Headers](#common-headers)
- [Error Format](#error-format)
- [Endpoints](#endpoints)
  - [GET /health](#get-health)
  - [GET /models](#get-models)
  - [POST /chat](#post-chat)
  - [POST /chat/stream](#post-chatstream)
  - [GET /config](#get-config)
  - [POST /config](#post-config)
  - [GET /metrics](#get-metrics)
  - [GET /sessions](#get-sessions)
  - [GET /sessions/{id}](#get-sessionsid)
  - [DELETE /sessions/{id}](#delete-sessionsid)
  - [GET /ws](#get-ws)
  - [GET /openapi.json](#get-openapijson)

---

## Authentication

When `auth.enabled = true`, all requests (except exempt paths) must include one of:

| Method | Header | Example |
|--------|--------|---------|
| Bearer Token | `Authorization: Bearer <token>` | `Authorization: Bearer my-secret` |
| API Key | `X-API-Key: <key>` | `X-API-Key: abc123` |

**Exempt paths** (default): `/health`

**Failure response**: `401 Unauthorized`

```json
{
  "error": "Authentication required"
}
```

---

## Common Headers

### Request Headers

| Header | Description | Required |
|--------|-----------|---------|
| `Content-Type` | Must be `application/json` for POST requests | Yes (POST) |
| `Authorization` | Bearer token authentication | When auth enabled |
| `X-API-Key` | API key authentication | When auth enabled |
| `X-Request-Id` | Correlation ID (reused if provided, generated otherwise) | No |
| `Accept-Encoding` | Set to `gzip` to receive compressed responses | No |

### Response Headers

| Header | Description |
|--------|-----------|
| `Content-Type` | `application/json` for most endpoints |
| `X-Request-Id` | 32-character hex correlation ID |
| `X-API-Version` | API version (e.g., `v1`) |
| `Access-Control-Allow-Origin` | CORS origin (configurable) |
| `Access-Control-Allow-Methods` | CORS methods |
| `Access-Control-Allow-Headers` | CORS headers |
| `Content-Encoding` | `gzip` when response is compressed |
| `Retry-After` | Seconds to wait (on 429 responses) |

---

## Error Format

All errors return a structured JSON body:

```json
{
  "error_code": "INVALID_JSON",
  "message": "Failed to parse request body",
  "details": "expected ',' or '}' at line 3 column 1",
  "retry_after_secs": null
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|------------|-----------|
| `INVALID_JSON` | 400 | Malformed JSON in request body |
| `AUTH_FAILED` | 401 | Missing or invalid authentication |
| `RATE_LIMITED` | 429 | Too many requests; check `Retry-After` header |
| `VALIDATION_ERROR` | 422 | Request validation failed (message too long, etc.) |
| `MODEL_ERROR` | 500 | LLM provider returned an error |
| `INTERNAL_ERROR` | 500 | Unexpected server error |
| `NOT_FOUND` | 404 | Unknown endpoint |

---

## Endpoints

### GET /health

Health check endpoint. Exempt from authentication by default.

**Response**: `200 OK`

```json
{
  "status": "ok",
  "version": "0.1.0",
  "model": "llama2",
  "provider": "Ollama",
  "uptime_secs": 3600,
  "active_sessions": 5,
  "conversation_messages": 42
}
```

| Field | Type | Description |
|-------|------|-----------|
| `status` | string | Always `"ok"` when server is running |
| `version` | string | Crate version from Cargo.toml |
| `model` | string | Currently selected LLM model |
| `provider` | string | Active LLM provider name |
| `uptime_secs` | integer | Server uptime in seconds |
| `active_sessions` | integer | Number of stored sessions |
| `conversation_messages` | integer | Messages in current conversation |

---

### GET /models

List available LLM models and current configuration.

**Response**: `200 OK`

```json
[
  {
    "id": "llama2",
    "provider": "Ollama",
    "selected": true
  },
  {
    "id": "gpt-4",
    "provider": "OpenAI",
    "selected": false
  }
]
```

---

### POST /chat

Send a message to the AI assistant (non-streaming).

**Request Body**:

```json
{
  "message": "Hello, how are you?",
  "system_prompt": "You are a helpful assistant.",
  "knowledge_context": "Optional RAG context to include."
}
```

| Field | Type | Required | Description |
|-------|------|---------|-----------|
| `message` | string | Yes | The user message (max: `max_message_length` chars) |
| `system_prompt` | string | No | System prompt override |
| `knowledge_context` | string | No | Additional context for RAG |

**Response**: `200 OK`

```json
{
  "content": "Hello! I'm doing well. How can I help you today?",
  "model": "llama2"
}
```

**Errors**:
- `422` — Message exceeds `max_message_length`
- `500` — LLM provider error

---

### POST /chat/stream

Send a message with Server-Sent Events streaming response.

**Request Body**: Same as `POST /chat`

**Response**: `200 OK` with `Content-Type: text/event-stream`

```
data: {"token":"Hello"}

data: {"token":" how"}

data: {"token":" are"}

data: {"token":" you?"}

data: [DONE]

```

Each SSE event contains a JSON object with a `token` field. The stream ends with
`data: [DONE]` sentinel.

**Headers**:
- `Content-Type: text/event-stream`
- `Cache-Control: no-cache`
- `Connection: keep-alive`

---

### GET /config

Retrieve the current assistant configuration.

**Response**: `200 OK`

```json
{
  "provider": "ollama",
  "model": "llama2",
  "temperature": 0.7,
  "max_history": 20
}
```

---

### POST /config

Update the assistant configuration.

**Request Body**:

```json
{
  "model": "llama3",
  "temperature": 0.5
}
```

Supported fields:
- `model` (string) — Switch the active model
- `temperature` (float) — Adjust generation temperature
- `provider` (string) — Switch the active provider

**Response**: `200 OK`

```json
{
  "status": "updated"
}
```

---

### GET /metrics

Prometheus-format metrics endpoint.

**Response**: `200 OK` with `Content-Type: text/plain`

```
# HELP ai_server_requests_total Total number of HTTP requests.
# TYPE ai_server_requests_total counter
ai_server_requests_total 1234
# HELP ai_server_requests_by_status HTTP requests by status class.
# TYPE ai_server_requests_by_status counter
ai_server_requests_by_status{status="2xx"} 1100
ai_server_requests_by_status{status="4xx"} 120
ai_server_requests_by_status{status="5xx"} 14
# HELP ai_server_request_duration_seconds_total Total time spent processing requests.
# TYPE ai_server_request_duration_seconds_total counter
ai_server_request_duration_seconds_total 45.123456
```

---

### GET /sessions

List all stored sessions.

**Response**: `200 OK`

```json
{
  "sessions": ["default", "session-abc", "session-xyz"],
  "count": 3
}
```

---

### GET /sessions/{id}

Get details of a specific session.

**Response**: `200 OK`

```json
{
  "session_id": "session-abc",
  "message_count": 15
}
```

**Errors**:
- `404` — Session not found

---

### DELETE /sessions/{id}

Delete a specific session.

**Response**: `200 OK`

```json
{
  "deleted": "session-abc"
}
```

**Errors**:
- `404` — Session not found

---

## CORS

The server handles CORS preflight (`OPTIONS`) requests automatically based on `CorsConfig`:

```bash
# Preflight request
curl -X OPTIONS \
  -H "Origin: https://app.example.com" \
  -H "Access-Control-Request-Method: POST" \
  http://localhost:8090/chat

# Response headers:
# Access-Control-Allow-Origin: *
# Access-Control-Allow-Methods: GET, POST, OPTIONS
# Access-Control-Allow-Headers: Content-Type, Authorization, X-API-Key
# Access-Control-Max-Age: 86400
```

---

## API Versioning

All endpoints are available under both root (`/`) and versioned (`/api/v1/`) paths:

| Root Path | Versioned Path |
|-----------|---------------|
| `GET /health` | `GET /api/v1/health` |
| `GET /models` | `GET /api/v1/models` |
| `POST /chat` | `POST /api/v1/chat` |
| `POST /chat/stream` | `POST /api/v1/chat/stream` |
| `GET /config` | `GET /api/v1/config` |
| `POST /config` | `POST /api/v1/config` |
| `GET /metrics` | `GET /api/v1/metrics` |
| `GET /sessions` | `GET /api/v1/sessions` |
| `GET /sessions/{id}` | `GET /api/v1/sessions/{id}` |
| `DELETE /sessions/{id}` | `DELETE /api/v1/sessions/{id}` |

The `X-API-Version` response header indicates the API version (`v1`).

---

## Rate Limiting

When rate limiting is configured, the server tracks requests per time window.
Exceeding the limit returns:

**Response**: `429 Too Many Requests`

```json
{
  "error_code": "RATE_LIMITED",
  "message": "Rate limit exceeded",
  "retry_after_secs": 30
}
```

The `Retry-After` header indicates how many seconds to wait before retrying.

---

## Usage Examples

### curl

```bash
# Health check
curl http://localhost:8090/health

# Send a chat message
curl -X POST http://localhost:8090/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is Rust?"}'

# Streaming chat
curl -X POST http://localhost:8090/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain async/await"}'

# With authentication
curl -X POST http://localhost:8090/api/v1/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer my-token" \
  -d '{"message": "Hello"}'

# List sessions
curl http://localhost:8090/sessions

# Delete a session
curl -X DELETE http://localhost:8090/sessions/old-session

# Get metrics
curl http://localhost:8090/metrics
```

### Python

```python
import requests

BASE = "http://localhost:8090/api/v1"
HEADERS = {"Authorization": "Bearer my-token"}

# Chat
resp = requests.post(f"{BASE}/chat",
    json={"message": "Hello!"},
    headers=HEADERS)
print(resp.json()["content"])

# Streaming
resp = requests.post(f"{BASE}/chat/stream",
    json={"message": "Tell me a story"},
    headers=HEADERS,
    stream=True)
for line in resp.iter_lines():
    if line:
        print(line.decode())
```

### JavaScript (fetch)

```javascript
// Chat
const resp = await fetch('http://localhost:8090/api/v1/chat', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer my-token'
  },
  body: JSON.stringify({ message: 'Hello!' })
});
const data = await resp.json();
console.log(data.content);

// SSE Streaming
const eventSource = new EventSource('http://localhost:8090/chat/stream');
eventSource.onmessage = (event) => {
  if (event.data === '[DONE]') {
    eventSource.close();
    return;
  }
  const { token } = JSON.parse(event.data);
  process.stdout.write(token);
};
```

---

## GET /ws

**WebSocket chat endpoint** (RFC 6455). Send JSON messages, receive streaming responses.

### Upgrade Request

```
GET /ws HTTP/1.1
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==
Sec-WebSocket-Version: 13
```

### Message Format (client → server)

```json
{
  "message": "Hello, assistant!",
  "system_prompt": "You are a helpful assistant"
}
```

### Response Frames (server → client)

```json
{"type": "chunk", "data": "Hello"}
{"type": "chunk", "data": "! How"}
{"type": "chunk", "data": " can I help?"}
{"type": "done"}
```

### JavaScript Example

```javascript
const ws = new WebSocket('ws://localhost:8090/ws');
ws.onopen = () => {
  ws.send(JSON.stringify({ message: 'Hello!' }));
};
ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  if (msg.type === 'done') return;
  process.stdout.write(msg.data);
};
```

Also available at `/api/v1/ws`.

---

## GET /openapi.json

Returns the OpenAPI 3.0.0 specification for all server endpoints.

```bash
curl http://localhost:8090/api/v1/openapi.json | jq .info
```

```json
{
  "title": "AI Assistant API",
  "version": "1.0.0",
  "description": "Embedded HTTP server for the ai_assistant crate"
}
```

Also available at `/openapi.json`.
