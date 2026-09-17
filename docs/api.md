# API Reference

SmarterRouter provides OpenAI-compatible endpoints for seamless integration with existing tools and applications.

## Base URL

- **Development:** `http://localhost:11436`
- **Production:** Configure based on your deployment

## Authentication

| Endpoint Type | Authentication Required |
|---------------|------------------------|
| `/v1/*` (chat, embeddings, models) | No |
| `/admin/*` | Yes - `ROUTER_ADMIN_API_KEY` required |
| `/health`, `/metrics` | No |

Admin endpoints require header: `Authorization: Bearer your-admin-api-key`

## Endpoints

### Interactive Web UI Demo & Telemetry

#### `GET /` or `GET /demo`

Serves the built-in, real-time visual web dashboard with live GPU VRAM telemetry, Dynamic Context Allocation (DCA) state, and before/after prompt compression inspections.

- **Content-Type:** `text/html; charset=utf-8`
- **Authentication:** None (public demo UI)

#### `GET /api/demo/telemetry`

Fetches real-time hardware telemetry and pipeline statistics for the dashboard.

**Response:**
```json
{
  "vram": {
    "used_gb": 4.1,
    "total_gb": 6.0,
    "free_gb": 1.9,
    "utilization_pct": 68.4,
    "device": "NVIDIA GeForce RTX 4050",
    "models": ["qwen2.5:3b (Pinned)"]
  },
  "dca": {
    "bucket": "4,096",
    "mode": "📉 Nerf Mode (Hysteresis Active)"
  },
  "stats": {
    "total_requests": 142,
    "avg_tokens_saved_pct": 57.9,
    "avg_latency_ms": 1.45
  },
  "memory": {
    "total_memories_stored": 28,
    "total_memories_recalled": 19
  }
}
```

#### `POST /api/demo/inspect`

Pre-flight test harness to preview prompt compression transformations and token reduction without forwarding to inference models.

**Request:**
```json
{
  "prompt": "Here is a large context block with repeated imports and logs..."
}
```

**Response:**
```json
{
  "original_tokens": 1250,
  "compressed_tokens": 520,
  "token_savings_pct": 58.4,
  "latency_ms": 1.82,
  "category": "coding",
  "compressed_prompt": "Cleaned and pruned context..."
}
```

---

### Core Compatibility Endpoints

### `GET /health`

Health check endpoint with subsystem diagnostics.

**Response:**
```json
{
  "status": "healthy",
  "checks": {
    "database": "ok",
    "backend": "initialized",
    "gpu": {
      "total_gb": 6.0,
      "used_gb": 4.1,
      "free_gb": 1.9,
      "vendor": "NVIDIA"
    },
    "cache": "memory (ok)",
    "background_tasks": 3,
    "dlq": {
      "failed": 0,
      "retrying": 0,
      "dead": 0
    }
  },
  "version": "2.2.7",
  "request_id": "req_abc123"
}
```

**Notes:** health data is intentionally concise and is produced by the live runtime state in the router. The `gpu` value is either a metrics object or a status string, and the `database` and `backend` checks are simple status strings rather than nested detailed objects.

### `GET /metrics`

Prometheus metrics endpoint for monitoring integration.

**Content-Type:** `text/plain; version=0.0.4`

**Metrics included:**
- `smarterrouter_requests_total` - Request count by endpoint and method
- `smarterrouter_request_duration_seconds` - Request duration histogram
- `smarterrouter_errors_total` - Error count by endpoint and type
- `smarterrouter_model_selections_total` - Model selection distribution
- `smarterrouter_cache_hits_total` / `cache_misses_total` - Cache statistics
- `smarterrouter_vram_total_gb`, `vram_used_gb`, `vram_utilization_pct` - GPU memory

### `GET /v1/models`

Returns the router itself as a virtual model plus any registered or discovered models.

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "smarterrouter/main",
      "object": "model",
      "created": 1708162374.0,
      "owned_by": "local",
      "description": "An intelligent LLM router that automatically selects the best model..."
    }
  ]
}
```

### `GET /v1/skills`

Returns the names of all registered skills available for tool/function calling.

**Response:**
```json
{
  "skills": [
    "calculate",
    "search",
    "fetch"
  ]
}
```

### `POST /v1/chat/completions`

Main chat completion endpoint. Fully compatible with OpenAI API format.

**Request:**
```json
{
  "model": "smarterrouter/main",
  "messages": [
    {"role": "user", "content": "Write a Python function..."}
  ],
  "temperature": 0.7,
  "max_tokens": 500,
  "stream": false
}
```

**Features:**
- **Dynamic Context Compression:** Automatically intercepts, scans, partitions (DCA), and prunes low-entropy prompt tokens.
- **Long-Term Vector Memory:** When persistent conversation memory is active, relevant past turns from `data/conversation_memory.db` are recalled via cosine similarity and injected into context.
- **Streaming:** Set `"stream": true` for Server-Sent Events (SSE) format.
- **Rate limiting:** When enabled, chat uses a dedicated per-IP limit via `ROUTER_RATE_LIMIT_CHAT_REQUESTS_PER_MINUTE` and returns HTTP `429` (`Rate limit exceeded`) when exceeded.
- **Request timeout:** End-to-end request processing is bounded by `ROUTER_REQUEST_TIMEOUT_SECONDS` (default: 300s). Exceeded requests return HTTP `504` with a `timeout_error` payload.
- **Error observability:** Structured error logs include correlation fields (`request_id`, `user_ip`, `model_name`, `prompt_hash`).

**Aliases:** `/chat/completions`, `/v1/responses`, `/responses`, `/v1/completions`, `/completions`.

**Response:**
```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1708162374,
  "model": "smarterrouter/main",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Generated response..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 150,
    "total_tokens": 165
  }
}
```

**Note:** The `model` field returns `smarterrouter/main` (or configured `ROUTER_EXTERNAL_MODEL_NAME`). The actual selected model is appended to the response signature when `ROUTER_SIGNATURE_ENABLED=true`.

### `POST /v1/embeddings`

Generate vector embeddings for text.

**Request:**
```json
{
  "model": "nomic-embed-text",
  "input": "The quick brown fox jumps over the lazy dog"
}
```

### `POST /v1/feedback`

Submit user feedback to refine future routing decisions.

**Request:**
```json
{
  "response_id": "chatcmpl-...",
  "score": 1.0,
  "comment": "Great answer! Very helpful."
}
```

**Parameters:**
- `response_id` (required): Response ID from chat completion
- `score` (required): Float in the validated range `[-1.0, 1.0]` (`-1.0` = poor, `0.0` = neutral, `1.0` = excellent)
- `comment` (optional): Text feedback

---

## Admin Endpoints (Require Authentication)

All admin endpoints require `Authorization: Bearer <ROUTER_ADMIN_API_KEY>` header.

### `GET /admin/stats`

Overall system diagnostics summary: uptime, total requests, active cache size, loaded models, and DLQ status.

### `GET /admin/profiles`

View local performance profiles of all models with pagination (`limit`, `offset`, `cursor`).

### `GET /admin/benchmarks`

View aggregated benchmark data (MMLU, HumanEval, Math, GPQA) from HuggingFace, LMSYS, or provider.db. Query params: `model`, `limit`, `offset`, `cursor`.

### `POST /admin/reprofile` or `POST /admin/models/reprofile`

Trigger manual model reprofiling. Query params: `force=true`, `models=llama3:8b,codellama:34b`.

### `POST /admin/models/refresh`

Trigger immediate background model discovery and hot-swap against inference backends.

### `POST /admin/sync-benchmarks`

Trigger immediate background benchmark synchronization with configured sources.

### `GET /admin/compression/stats`

Real-time telemetry and metrics summary from the Pre-Flight Dynamic Context Compression Pipeline:
```json
{
  "enabled": true,
  "mode": "full",
  "total_requests": 142,
  "original_tokens_sum": 384000,
  "compressed_tokens_sum": 161600,
  "avg_token_savings_pct": 57.9,
  "avg_latency_ms": 1.45,
  "dca_current_limit": 8192,
  "car_cache_size": 240
}
```

### `GET /admin/vram`

View real-time GPU VRAM usage, detected GPUs, per-model memory allocations, and eviction warnings.

### `GET /admin/explain`

Detailed scoring breakdown and model selection explanation for a given prompt. Query params: `prompt` (required), `category` (optional override).

### `GET /admin/cache/stats/detailed`

Detailed cache statistics with hit/miss breakdowns across exact hash, semantic similarity, and response caches.

### `POST /admin/cache/invalidate`

Invalidate cache entries (`?type=routing`, `?type=response`, or `?all=true`).

### `POST /admin/cache/clear`

Completely flush all in-memory, persistent, and Redis cache tiers.

### `POST /admin/cache/warm`

Pre-warm cache for popular queries and benchmark prompts.

### `POST /admin/cache/evict`

Force cache eviction based on LRU policy.

### `GET /admin/dlq`

List Dead Letter Queue entries for failed background operations (`?status=failed|retrying|dead|resolved`, `limit`, `offset`).

### `POST /admin/dlq/retry/{entry_id}`

Manually re-dispatch and retry a specific failed DLQ entry.

### `GET /admin/audit-log`

Query administrative action audit logs recorded in SQLite (`?limit=50`, `?action=...`, `?user_ip=...`).

---

## Error Codes

| HTTP Status | Meaning | Common Causes |
|-------------|---------|---------------|
| 200 | Success | Request succeeded |
| 400 | Bad Request | Invalid JSON, missing parameters, payload exceeding limits |
| 401 | Unauthorized | Missing or invalid `ROUTER_ADMIN_API_KEY` |
| 403 | Forbidden | Client IP not allowed by `ROUTER_ADMIN_ALLOWED_IPS` |
| 404 | Not Found | Unknown endpoint or missing resource |
| 429 | Too Many Requests | Rate limit exceeded (`Retry-After` header included) |
| 500 | Internal Server Error | Unexpected failure in backend or router |
| 503 | Service Unavailable | Backend offline, or circuit breaker is OPEN |
| 504 | Gateway Timeout | Total processing exceeded `ROUTER_REQUEST_TIMEOUT_SECONDS` |

## Rate Limits

- **General endpoints:** `ROUTER_RATE_LIMIT_REQUESTS_PER_MINUTE` (default: 60/min)
- **Chat completions:** `ROUTER_RATE_LIMIT_CHAT_REQUESTS_PER_MINUTE` (default: 100/min)
- **Admin endpoints:** `ROUTER_RATE_LIMIT_ADMIN_REQUESTS_PER_MINUTE` (default: 10/min)

## CORS

CORS is configured via `ROUTER_CORS_ORIGINS` in `.env`:

```env
ROUTER_CORS_ORIGINS=http://localhost:3000,https://myapp.example.com
ROUTER_CORS_ALLOW_CREDENTIALS=false
ROUTER_CORS_ALLOW_METHODS=GET,POST,PUT,DELETE,OPTIONS
ROUTER_CORS_ALLOW_HEADERS=*
ROUTER_CORS_MAX_AGE=600
```

---

## Client Compatibility

SmarterRouter is compatible with any OpenAI client library:
- **OpenAI Python SDK** (`base_url="http://localhost:11436/v1"`)
- **OpenAI Node / TypeScript SDK** (`@ai-sdk/openai-compatible`)
- **OpenCode IDE** (`opencode.json` configuration)
- **OpenWebUI** (v0.2+)
- **Continue** & **Cursor** extensions
- **SillyTavern** & general chat UIs

The virtual model name is `smarterrouter/main`. Model selection is transparent to the client.

