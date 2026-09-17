# SmarterRouter

**An intelligent, OpenAI-compatible FastAPI gateway that automatically selects the best language model for every request.**

SmarterRouter sits in front of local inference backends (Ollama, llama.cpp) or any OpenAI-compatible endpoint—including cloud providers like OpenAI, Anthropic, Google, Cohere, and Mistral—and routes each request to the most appropriate model. Clients send requests to a single stable endpoint instead of choosing and managing models themselves.

The router combines **prompt analysis**, **model capability detection**, **local performance profiles**, **benchmark data from HuggingFace/LMSYS/ArtificialAnalysis**, **user feedback**, **VRAM availability**, **multi-layered caching**, and **backend health monitoring** to make intelligent routing decisions in real time.

---

## Table of Contents

- [Key Features](#key-features)
- [Architecture Overview](#architecture-overview)
- [How It Works — The Routing Pipeline](#how-it-works--the-routing-pipeline)
- [Requirements](#requirements)
- [Quick Start with Docker](#quick-start-with-docker)
- [Install Locally](#install-locally)
- [Configuration](#configuration)
  - [Minimum Configuration](#minimum-configuration)
  - [Important Settings](#important-settings)
  - [Profiling & Benchmarks](#profiling--benchmarks)
  - [External Providers](#external-providers)
  - [Caching](#caching)
  - [Security Settings](#security-settings)
- [Choose a Backend](#choose-a-backend)
- [API Reference](#api-reference)
  - [Core Endpoints](#core-endpoints)
  - [Admin Endpoints](#admin-endpoints)
  - [Chat Completion Parameters](#chat-completion-parameters)
  - [Response Structure](#response-structure)
- [Client Integration Examples](#client-integration-examples)
- [CLI](#cli)
- [Dynamic Context Compression](#dynamic-context-compression)
- [GPU & VRAM Management](#gpu--vram-management)
- [Operations & Deployment](#operations--deployment)
  - [Production Checklist](#production-checklist)
  - [Docker Persistence & Logs](#docker-persistence--logs)
  - [Kubernetes](#kubernetes)
- [Observability & Monitoring](#observability--monitoring)
- [Security](#security)
- [Development & Testing](#development--testing)
- [Project Structure](#project-structure)
- [Documentation Index](#documentation-index)
- [Contributing](#contributing)
- [License](#license)

---

## Key Features

| Category | Capabilities |
| :--- | :--- |
| **Intelligent Routing** | Prompt complexity analysis · task categorization (coding, reasoning, creativity, factual) · model scoring with quality-vs-speed tuning (`ROUTER_QUALITY_PREFERENCE`) · modality-aware routing (vision, tool calling, embeddings) · fallback cascades on failure |
| **Context Compression** | Pre-flight compression pipeline · selective cross-lingual token arbitrage · AST-aware code chunking · BM25 + entropy statistical pruning · dynamic context-window allocation (DCA) · prefix cache alignment (RadixAttention/APC compatible) · −57.9% tokens |
| **Vector Memory** | Long-term conversation memory backed by SQLite vector store (`conversation_memory.db`) · cosine similarity recall · automated prompt context injection · sub-millisecond in-process hash vectorizer fallback |
| **Visual Web Demo** | Interactive real-time web dashboard (`GET /` and `/demo`) · live GPU VRAM telemetry · DCA hysteresis state · pre-flight before/after prompt inspection harness |
| **Backend Support** | Ollama (model load/unload, discovery) · llama.cpp · any OpenAI-compatible API · external cloud providers (OpenAI, Anthropic, Google, Cohere, Mistral) |
| **Model Profiling** | Automated local profiling with MT-Bench-inspired prompts · optional LLM-as-Judge grading · capability detection (vision, tool calling) · VRAM footprint measurement |
| **Benchmark Integration** | HuggingFace leaderboard · LMSYS Chatbot Arena · ArtificialAnalysis · `provider.db` with 400+ models from 28+ benchmark sources |
| **Caching** | Exact-hash routing cache · semantic similarity cache (cosine similarity with adaptive thresholds) · full response cache · embedding cache · persistent SQLite cache · optional Redis backend |
| **GPU & VRAM** | Multi-vendor monitoring (NVIDIA, AMD/ROCm, Intel i915/xe, Apple Silicon) · multi-GPU aggregation · VRAM-aware routing · automatic model load/unload with LRU or largest-first eviction · AMD APU unified memory support |
| **Security** | Rate limiting (per-IP, per-endpoint) · prompt-injection detection · input sanitization · admin API key auth · IP whitelist · CORS · request size limits · encrypted API key storage (Fernet) · SQL injection prevention · admin audit logging |
| **Observability** | Prometheus metrics (`/metrics`) · structured JSON logging · request correlation (`X-Request-ID`) · health endpoint with DB/GPU/backend/DLQ status · VRAM dashboard |
| **API Compatibility** | Full OpenAI-compatible API · streaming (SSE) · tool/function calling · `/v1/chat/completions`, `/v1/embeddings`, `/v1/models`, `/v1/skills`, `/v1/responses` aliases · drop-in replacement for OpenAI SDK · OpenCode IDE provider |


> **Note:** SmarterRouter does **not** run an LLM by itself. You must provide at least one reachable inference backend.

---

## Architecture Overview

```text
OpenAI-compatible client / OpenCode / Web UI
                       |
                       v
FastAPI Gateway & Security Middleware
   - IP whitelist, validation, rate limits, CORS, content moderation
                       |
                       v
Pre-Flight Dynamic Context Compression Pipeline
   - Language scanner (bypasses English) -> Dynamic Context Allocation (DCA)
   - Cross-lingual token arbitrage -> AST chunking & BM25/entropy lexical pruner
   - Critical syntax force-keep -> Prefix cache aligner & CAR recovery handles
                       |
                       v
Long-Term Conversation Vector Memory
   - SQLite vector store -> semantic turn recall -> context injection
                       |
                       v
Router Engine & VRAM Manager
   - Prompt analysis & capability detection
   - Profiles + benchmarks + user feedback scoring
   - VRAM budget enforcement & automated LRU model eviction
   - Circuit breakers & retry policies
                       |
                       +--> local backend: Ollama, llama.cpp
                       +--> OpenAI-compatible backend (vLLM, LiteLLM)
                       +--> external cloud: OpenAI, Anthropic, Google, Cohere, Mistral
                       |
                       +--> SQLite router.db (profiles, benchmarks, feedback, audit)
                       +--> SQLite conversation_memory.db (persistent vector memory)
                       +--> optional Redis distributed cache
                       +--> provider.db benchmark database (400+ models)
                       +--> GPU/VRAM monitor (NVIDIA, AMD ROCm/APU, Intel, Apple)
```

For full architecture diagrams (Mermaid), data-flow sequences, and scoring algorithm visualization, see [docs/architecture.md](docs/architecture.md).


---

## How It Works — The Routing Pipeline

Every incoming request passes through a well-defined pipeline:

1. **Validate & Normalize** — Parse the incoming OpenAI-style request, validate model names, enforce request size limits.
2. **Security & Rate Limiting** — Apply rate limits, prompt-injection detection, content moderation, and CORS checks.
3. **Check Caches** — Exact-hash routing cache → semantic similarity cache → response cache. On cache hit, return immediately.
4. **Context Compression** *(optional)* — Run the pre-flight compression pipeline: language scanner → dynamic context-window allocation → cross-lingual arbitrage → AST-aware statistical pruning → prefix cache alignment.
5. **Prompt Analysis** — Classify the prompt by task (coding, reasoning, creativity, factual), estimate complexity (easy/medium/hard), detect required capabilities (vision, tool calling).
6. **Score & Select Model** — Pull all profiled models. Filter by required capabilities and VRAM budget. Calculate a weighted score for each model combining:
   - **Static benchmarks** (HuggingFace, LMSYS, provider.db) — how the model performs in general.
   - **Local runtime profiles** — how the model performs on *your* hardware.
   - **User feedback** — historical ratings submitted via `/v1/feedback`.
   - **Quality preference** — `ROUTER_QUALITY_PREFERENCE` (0.0 = speed, 1.0 = quality).
   - **Category-minimum size** — prevents small models from handling complex tasks (e.g., coding hard → 8B+).
   - **Name affinity** — heuristic boost (e.g., route coding tasks to `*coder*` models).
   - **Diversity penalty** — avoid routing everything to a single model.
7. **Execute & Cascade** — Check VRAM, load/unload models if needed, forward to the selected backend. If the selected model fails, cascade to the next-best model automatically.
8. **Post-Process** — Append model signature (optional), cache the routing decision and response, log the audit trail, return the OpenAI-compatible response.

---

## Requirements

### Runtime

- **Python 3.11+** for local installation
- **Docker & Docker Compose v2** for the recommended deployment
- A reachable model backend:
  - [Ollama](https://ollama.com/) — default and easiest local option
  - llama.cpp server
  - Any OpenAI-compatible API
  - Configured external cloud providers
- *(Optional)* GPU drivers for VRAM monitoring and accelerated inference

### Backend Examples

**Ollama:**
```bash
ollama serve
ollama pull llama3.2
ollama pull codellama
```

**llama.cpp:**
```bash
./llama-server -m models/model.gguf -c 4096 --port 8080
```

> **Docker note:** When the backend runs on the host, use `host.docker.internal`, `172.17.0.1`, or a Compose service name instead of `localhost` (which refers to the container itself).

---

## Quick Start with Docker

```bash
git clone https://github.com/peva3/SmarterRouter.git
cd SmarterRouter
cp ENV_DEFAULT .env          # On Windows: copy ENV_DEFAULT .env
docker compose up -d
```

Edit `.env` if Ollama is not reachable at the configured URL. The Compose file publishes the router at `http://localhost:11436` and persists state in `./data`.

**Verify:**
```bash
curl http://localhost:11436/health
curl http://localhost:11436/v1/models
docker compose logs -f smarterrouter
```

### GPU-Specific Docker Setup

| Hardware | Template |
| :--- | :--- |
| NVIDIA | [docker-compose.nvidia.yml](docs/docker-compose.nvidia.yml) |
| AMD / ROCm | [docker-compose.amd.yml](docs/docker-compose.amd.yml) |
| Intel | [docker-compose.intel.yml](docs/docker-compose.intel.yml) |
| Apple Silicon | [docker-compose.apple.md](docs/docker-compose.apple.md) |
| Mixed / Multi-GPU | [docker-compose.multi-gpu.yml](docs/docker-compose.multi-gpu.yml) |

For a guided Docker launch, see [docker-run.sh](docker-run.sh). For the full deployment guide, see [docs/DOCKER.md](docs/DOCKER.md).

---

## Install Locally

**1. Create and activate a virtual environment:**

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Create configuration:**
```bash
cp ENV_DEFAULT .env          # On Windows: copy ENV_DEFAULT .env
# Edit .env with your settings
```

**4. Start the server:**
```bash
# Development (auto-reload)
python -m uvicorn main:app --reload --host 0.0.0.0 --port 11436

# Production
python -m uvicorn main:app --host 0.0.0.0 --port 11436

# Or use the CLI
smarterrouter --help
```

---

## Configuration

Configuration uses **Pydantic Settings**. All variables use the `ROUTER_` prefix and can be set via environment variables or a `.env` file. Copy [ENV_DEFAULT](ENV_DEFAULT) as the starting point — it is the complete annotated template with all available settings.

### Minimum Configuration

```env
ROUTER_PROVIDER=ollama
ROUTER_OLLAMA_URL=http://localhost:11434
ROUTER_PORT=11436
ROUTER_ADMIN_API_KEY=replace-with-a-long-random-secret
```

### Important Settings

| Setting | Purpose | Default |
| :--- | :--- | :--- |
| `ROUTER_HARDWARE_PRESET` | Preset hardware profile: `6GB_VRAM`, `12GB_VRAM`, `24GB_VRAM`, `CUSTOM` | `6GB_VRAM` |
| `ROUTER_COMPRESSION_ENABLED` | Pre-flight dynamic context compression toggle | `true` |
| `ROUTER_COMPRESSION_MODE` | Strategy: `full`, `statistical_only`, `arbitrage_only`, `cache_align_only` | `full` |
| `ROUTER_PROVIDER` | Backend type: `ollama`, `llama.cpp`, or `openai` | `ollama` |
| `ROUTER_OLLAMA_URL` | Ollama/compatible backend URL | `http://localhost:11434` |
| `ROUTER_LLAMA_CPP_URL` | llama.cpp server URL | — |
| `ROUTER_OPENAI_BASE_URL` | OpenAI-compatible base URL | — |
| `ROUTER_OPENAI_API_KEY` | API key for that endpoint | — |
| `ROUTER_HOST` / `ROUTER_PORT` | Bind address and port | `0.0.0.0` / `11436` |
| `ROUTER_QUALITY_PREFERENCE` | `0.0` = speed, `1.0` = quality | `0.5` |
| `ROUTER_MODEL` | Small model for LLM-based dispatch | — |
| `ROUTER_PINNED_MODEL` | Model to keep loaded permanently (auto-pinned by preset) | `qwen2.5:3b` |
| `ROUTER_MODEL_KEEP_ALIVE` | Backend keep-alive seconds; `-1` = indefinitely | `-1` |
| `ROUTER_CASCADING_ENABLED` | Retry with next-best model on failure | `true` |
| `ROUTER_GENERATION_TIMEOUT` | Backend generation timeout (seconds) | `120` |
| `ROUTER_REQUEST_TIMEOUT_SECONDS` | Overall request budget (seconds) | `300` |
| `ROUTER_DATABASE_URL` | SQLAlchemy DB URL | `sqlite:///data/router.db` |
| `ROUTER_CACHE_BACKEND` | `memory` or `redis` | `memory` |
| `ROUTER_REDIS_URL` | Redis URL when cache backend is `redis` | — |
| `ROUTER_LOG_LEVEL` / `ROUTER_LOG_FORMAT` | Logging verbosity and format (`text`/`json`) | `INFO` / `text` |
| `ROUTER_SIGNATURE_ENABLED` | Append selected model name to responses | `true` |


### Profiling & Benchmarks

```env
ROUTER_MODEL_POLLING_ENABLED=true
ROUTER_MODEL_AUTO_PROFILE_ENABLED=true
ROUTER_PROFILE_TIMEOUT=90
ROUTER_PROFILE_PROMPTS_PER_CATEGORY=3
ROUTER_BENCHMARK_SOURCES=huggingface,lmsys
```

Available benchmark sources: `huggingface`, `lmsys`, `artificial_analysis`. ArtificialAnalysis requires `ROUTER_ARTIFICIAL_ANALYSIS_API_KEY`.

**LLM-as-Judge** (optional, for high-quality profiling):
```env
ROUTER_JUDGE_ENABLED=true
ROUTER_JUDGE_MODEL=gpt-4o
ROUTER_JUDGE_BASE_URL=https://api.openai.com/v1
ROUTER_JUDGE_API_KEY=sk-...
```

### External Providers

Mix cloud models with local models:

```env
ROUTER_PROVIDER=ollama
ROUTER_EXTERNAL_PROVIDERS_ENABLED=true
ROUTER_EXTERNAL_PROVIDERS=openai,anthropic,google

ROUTER_OPENAI_API_KEY=sk-...
ROUTER_ANTHROPIC_API_KEY=sk-ant-...
ROUTER_GOOGLE_API_KEY=...
```

External models use provider prefixes: `openai/gpt-4o`, `anthropic/claude-3-opus`, `google/gemini-1.5-pro`. The router uses `provider.db` (auto-downloaded every 4 hours) to look up benchmark scores for 400+ external models.

See [docs/external-providers.md](docs/external-providers.md) for full provider-specific settings.

### Caching

SmarterRouter implements a **multi-layered caching system**:

| Cache Layer | Purpose | Key | Default Size |
| :--- | :--- | :--- | :--- |
| **Routing Cache** | Cache model-selection decisions | SHA-256 hash + semantic similarity | 500 entries, 1h TTL |
| **Response Cache** | Cache full LLM responses | (model, prompt_hash) | 200 entries |
| **Embedding Cache** | Cache embedding vectors | prompt hash | 2500 entries, 24h TTL |
| **Persistent Cache** | Survive restarts (SQLite) | All of the above | 7-day max age |

```env
ROUTER_CACHE_ENABLED=true
ROUTER_CACHE_MAX_SIZE=500
ROUTER_CACHE_TTL_SECONDS=3600
ROUTER_CACHE_SIMILARITY_THRESHOLD=0.85
ROUTER_CACHE_RESPONSE_MAX_SIZE=200
ROUTER_EMBED_MODEL=nomic-embed-text      # Required for semantic similarity
ROUTER_PERSISTENT_CACHE_ENABLED=true

# Redis (for multi-instance deployments)
ROUTER_CACHE_BACKEND=redis
ROUTER_REDIS_URL=redis://localhost:6379/0
```

Features: adaptive similarity thresholds · LRU eviction · query pattern learning · top-K popular query pre-caching · vectorized numpy similarity search · per-model cache analytics.

### Security Settings

```env
ROUTER_ADMIN_API_KEY=$(openssl rand -hex 32)
ROUTER_RATE_LIMIT_ENABLED=true
ROUTER_RATE_LIMIT_REQUESTS_PER_MINUTE=60
ROUTER_RATE_LIMIT_CHAT_REQUESTS_PER_MINUTE=30
ROUTER_CORS_ORIGINS=http://localhost:3000
ROUTER_VERIFY_TLS=true
ROUTER_HOST=127.0.0.1                    # Bind to localhost only
```

See [docs/configuration.md](docs/configuration.md) for the complete configuration reference.

---

## Choose a Backend

| Backend | Best For | Model Management |
| :--- | :--- | :--- |
| **Ollama** | Local inference with automatic model discovery | SmarterRouter can load/unload models |
| **llama.cpp** | Direct GGUF serving, high-performance local inference | Manage server processes yourself |
| **OpenAI-compatible** | OpenAI, vLLM, TGI, LiteLLM, LocalAI, OpenRouter, etc. | Managed by the remote service |
| **External providers** | Direct OpenAI, Anthropic, Google, Cohere, Mistral routing | Provider-managed |

Set `ROUTER_PROVIDER=ollama`, `llama.cpp`, or `openai` and provide the corresponding URL. Detailed setup is in [docs/backends.md](docs/backends.md).

---

## API Reference

**Base URL:** `http://localhost:11436`

The router exposes OpenAI-compatible endpoints. The virtual model name shown to clients is `smarterrouter/main` by default; the actual selected model is included in the response signature when enabled.

### Core Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/` or `/demo` | GET | Interactive Visual Web Dashboard with live GPU VRAM telemetry and compression inspection. |
| `/api/demo/telemetry` | GET | Real-time telemetry feed (GPU memory, DCA bucket, compression savings, vector memory stats). |
| `/api/demo/inspect` | POST | Test harness to inspect verbatim before/after compression payloads. |
| `/v1/chat/completions` | POST | Main chat endpoint. Intelligently routes to the best model. Supports streaming and tool calling. |
| `/v1/embeddings` | POST | Generate vector embeddings (forwarded to the specified embedding model). |
| `/v1/models` | GET | List virtual router model and all discovered/registered models. |
| `/v1/skills` | GET | List available tools/skills schemas for agentic workflows. |
| `/v1/feedback` | POST | Submit user feedback to improve future routing. |
| `/health` | GET | Subsystem health check with DB, GPU, backend, cache, and DLQ status. |
| `/metrics` | GET | Prometheus metrics scraper. |
| `/v1/responses` | POST | Alias for `/v1/chat/completions`. |

Additional compatibility aliases: `/chat/completions`, `/responses`, `/v1/completions`, `/completions`.

### Admin Endpoints

All admin endpoints require `Authorization: Bearer <ROUTER_ADMIN_API_KEY>`.

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/admin/stats` | GET | Overall system diagnostics, uptime, and throughput summary. |
| `/admin/profiles` | GET | View local performance profiles for all models. |
| `/admin/benchmarks` | GET | View aggregated benchmark data. |
| `/admin/reprofile` | POST | Trigger manual model reprofiling. |
| `/admin/models/refresh` | POST | Trigger immediate background model discovery and hot-swap. |
| `/admin/sync-benchmarks` | POST | Trigger immediate background benchmark sync. |
| `/admin/compression/stats` | GET | Pre-flight dynamic context compression metrics and token savings summary. |
| `/admin/explain` | GET/POST | Detailed scoring breakdown and selection rationale for a prompt. |
| `/admin/vram` | GET | Real-time GPU VRAM status, loaded models, and eviction warnings. |
| `/admin/cache/stats` | GET | Cache statistics with time-series data. |
| `/admin/cache/stats/detailed` | GET | Detailed breakdown across exact, semantic, and response cache tiers. |
| `/admin/cache/invalidate` | POST | Invalidate cache entries (optionally per-model). |
| `/admin/cache/clear` | POST | Completely flush all cache tiers. |
| `/admin/cache/warm` | POST | Pre-warm cache for popular queries. |
| `/admin/cache/evict` | POST | Force LRU cache eviction. |
| `/admin/dlq` | GET | Inspect Dead Letter Queue for failed background operations. |
| `/admin/dlq/retry/{id}` | POST | Retry a failed DLQ entry. |
| `/admin/audit-log` | GET | Query admin audit logs. |


Full API documentation is also available at `/docs` (Swagger UI) and `/redoc`.

See [docs/api.md](docs/api.md) for the complete endpoint reference.

### Chat Completion Parameters

All standard OpenAI generation parameters are supported:

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `messages` | array | Message objects. **Required.** |
| `model` | string | Model override (optional; router selects automatically if omitted or `smarterrouter/main`). |
| `temperature` | float | Sampling temperature (0.0–2.0). |
| `top_p` | float | Nucleus sampling (0.0–1.0). |
| `max_tokens` | integer | Maximum tokens to generate. |
| `stream` | boolean | Enable server-sent events streaming. |
| `tools` | array | Tools the model may call. |
| `tool_choice` | string/object | Force specific tool or auto. |
| `response_format` | object | Require JSON output. |
| `n`, `seed`, `logprobs`, `top_logprobs`, `presence_penalty`, `frequency_penalty`, `logit_bias`, `user` | — | Standard OpenAI parameters. |

### Response Structure

**Chat Completion:**
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "llama3:8b-instruct",
  "choices": [{
    "index": 0,
    "message": { "role": "assistant", "content": "..." },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 50,
    "total_tokens": 60
  }
}
```

**Embeddings:**
```json
{
  "object": "list",
  "data": [{
    "object": "embedding",
    "embedding": [0.123, -0.456, ...],
    "index": 0
  }],
  "model": "nomic-embed-text",
  "usage": { "prompt_tokens": 8, "completion_tokens": 0, "total_tokens": 8 }
}
```

---

## Client Integration Examples

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11436/v1",
    api_key="unused-but-required-by-some-clients",
)

response = client.chat.completions.create(
    model="smarterrouter/main",
    messages=[{"role": "user", "content": "Write a short Python function."}],
)
print(response.choices[0].message.content)
```

### cURL

```bash
curl http://localhost:11436/v1/chat/completions \
   -H "Content-Type: application/json" \
   -d '{
      "model": "smarterrouter/main",
      "messages": [{"role": "user", "content": "Explain binary search briefly."}],
      "stream": false
   }'
```

### Streaming

```bash
curl http://localhost:11436/v1/chat/completions \
   -H "Content-Type: application/json" \
   -d '{
      "model": "smarterrouter/main",
      "messages": [{"role": "user", "content": "Tell me a joke."}],
      "stream": true
   }'
```

### Embeddings

```bash
curl http://localhost:11436/v1/embeddings \
   -H "Content-Type: application/json" \
   -d '{"model":"nomic-embed-text","input":"A short document"}'
```

### Submitting Feedback

```bash
curl http://localhost:11436/v1/feedback \
   -H "Content-Type: application/json" \
   -d '{"response_id":"chatcmpl-12345678","score":1.0,"comment":"Useful answer"}'
```

### OpenCode IDE Integration

SmarterRouter natively integrates with OpenCode via [opencode.json](opencode.json):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "smarterrouter": {
      "name": "SmarterRouter",
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "http://localhost:11436/v1",
        "apiKey": "admin123"
      },
      "models": {
        "smarterrouter/main": {
          "name": "SmarterRouter (VRAM Adaptive & Compressed)"
        }
      }
    }
  }
}
```

### Academic Benchmark Evaluation

SmarterRouter includes an academic benchmarking suite [benchmark.py](benchmark.py) to measure token reduction, prefill latency, and VRAM savings:

```bash
python benchmark.py --url http://localhost:11436 --iterations 3
```

More integration examples for **JavaScript/TypeScript**, **OpenWebUI**, **Continue**, and **Cursor** are in [docs/examples/client-integration.md](docs/examples/client-integration.md).


---

## CLI

After installing dependencies, the `smarterrouter` command provides:

```bash
smarterrouter setup              # Interactive setup wizard (detects hardware & models)
smarterrouter check --config .env # Validate config, backend reachability, GPU detection
smarterrouter generate-env --output .env --overwrite  # Generate suggested config
```

- **`setup`** — Detects Ollama installation, GPU hardware (NVIDIA, AMD, Intel, Apple Silicon), and available models. Interactively writes an optimized configuration.
- **`check`** — Validates configuration, Ollama reachability, available models, and GPU detection. Exits non-zero on failure.
- **`generate-env`** — Creates a suggested configuration without the interactive wizard.

The equivalent module command is `python -m router.cli ...`.

---

## Dynamic Context Compression

SmarterRouter includes a **pre-flight context compression pipeline** designed to solve the "100:1 agentic payload problem" — where multi-turn agent loops pass 50K–100K tokens of redundant context for just 50–300 tokens of output.

### Three Pillars of Compression

| Pillar | Component | Technique | Overhead |
| :--- | :--- | :--- | :--- |
| **1. Cross-Lingual Arbitrage** | [scanner.py](router/compression/scanner.py) + [arbitrage.py](router/compression/arbitrage.py) | Sub-millisecond non-ASCII density scanner bypasses SLM for English prompts. For multilingual input, a micro-SLM (e.g., Qwen 0.5B) rewrites into compact Bi/Tri-block format. | 0ms (English) / ~500ms (multilingual) |
| **2. Statistical Pruner** | [chunker.py](router/compression/chunker.py) + [statistical.py](router/compression/statistical.py) + [force_keep.py](router/compression/force_keep.py) | AST-aware code chunking (preserves `import os.path`, function boundaries). Composite BM25 + overlap + position + entropy + filler scorer. Force-keep gate protects auth tokens, paths, and critical signatures. | <2ms on 50KB |
| **3. Cache Alignment** | [dca.py](router/compression/dca.py) + [cache_aligner.py](router/compression/cache_aligner.py) + [car.py](router/compression/car.py) | Dynamic context-window allocation by query type. Byte-exact prefix isolation + alphabetical tool schema sorting (RadixAttention/APC safe). Content-Addressable Recovery via SHA-256 reference handles. | <1ms |

### Configuration

```env
ROUTER_COMPRESSION_ENABLED=true
ROUTER_COMPRESSION_MODE=full              # "full", "statistical_only", "cache_align_only"
ROUTER_HARDWARE_PRESET=6GB_VRAM           # "6GB_VRAM", "12GB_VRAM", "24GB_VRAM", "CUSTOM"
ROUTER_ARBITRAGE_ENABLED=true
ROUTER_ARBITRAGE_MULTILINGUAL_THRESHOLD=0.15
ROUTER_PRUNER_ENABLED=true
ROUTER_DCA_ENABLED=true
ROUTER_DCA_TARGET_CONTEXT_LIMIT=8192
```

### Expected Results (RTX 4050 6GB Benchmark)

| Metric | Baseline | Compressed | Improvement |
| :--- | :--- | :--- | :--- |
| Avg Input Tokens/Turn | 8,420 | 3,540 | **−57.9%** |
| Time-To-First-Token | 1,840ms | 420ms | **4.38× faster** |
| Peak VRAM | 5.75 GB | 3.62 GB | **−37.0%** |
| Prefix Cache Hit Rate | 12.4% | 91.8% | **+79.4%** |
| Task Pass@1 Accuracy | 94.0% | 94.0% | **0% regression** |

See [implementation.md](implementation.md) for the full specification and [DEEPDIVE.md](DEEPDIVE.md) for the architectural deep dive.

---

## GPU & VRAM Management

SmarterRouter includes comprehensive VRAM monitoring and management across all major GPU vendors:

### Supported Hardware

| Vendor | Detection Method | Multi-GPU | Special |
| :--- | :--- | :--- | :--- |
| **NVIDIA** | `nvidia-smi` / pyNVML | ✅ | Per-GPU metrics in Prometheus |
| **AMD** | `rocm-smi` + sysfs | ✅ | APU unified memory (GTT pool) |
| **Intel** | sysfs (`i915` / `xe` drivers) | ✅ | Battlemage Xe2 support |
| **Apple Silicon** | Unified memory reporting | — | M-series chips |

### How It Works

- **Background Monitor** — `VRAMMonitor` polls GPU memory at configurable intervals (default 30s), maintaining a rolling buffer of metrics.
- **Profiling Integration** — During model profiling, actual VRAM footprint is measured and stored as `vram_required_gb`.
- **VRAM-Aware Routing** — `VRAMManager` tracks loaded models and their VRAM usage. Before loading a new model, it checks the budget (`max_vram − buffer − loaded`). If over budget, it evicts models (LRU or largest-first), respecting pinned models.
- **Auto-Detection** — If `ROUTER_VRAM_MAX_TOTAL_GB` is not set, the router auto-detects total GPU VRAM and defaults to 90% utilization.

```env
ROUTER_VRAM_MAX_TOTAL_GB=              # Auto-detected if empty (90% of total)
ROUTER_VRAM_DEFAULT_ESTIMATE_GB=2.5    # Fallback when profiling data unavailable
ROUTER_PROFILE_MEASURE_VRAM=true       # Measure VRAM during profiling
```

---

## Operations & Deployment

### Production Checklist

1. ✅ Set a strong `ROUTER_ADMIN_API_KEY` (e.g., `openssl rand -hex 32`).
2. ✅ Keep `.env` and provider API keys out of version control.
3. ✅ Enable rate limiting and restrict CORS origins.
4. ✅ Put the service behind HTTPS termination and a reverse proxy (nginx, Caddy).
5. ✅ Persist `./data` or the configured database volume.
6. ✅ Scrape `/metrics` with Prometheus and monitor `/health`.
7. ✅ Size timeouts for the largest model and expected cold-load time.
8. ✅ Restrict network access to the router and backend ports.
9. ✅ Review [SECURITY.md](SECURITY.md) before exposing outside a trusted network.

### Docker Persistence & Logs

```bash
docker compose ps
docker compose logs -f smarterrouter
docker compose restart smarterrouter
docker compose down
```

> ⚠️ Do **not** remove `./data` unless you intend to delete the SQLite database, profiles, cache state, and all persisted data.

### Kubernetes

The repository includes deployment guides with Helm-style and raw-manifest examples. See [docs/kubernetes.md](docs/kubernetes.md).

Validate image names, chart availability, ports, secrets, and GPU device plugins against your cluster before production use.

---

## Observability & Monitoring

### Prometheus Metrics

Scraped from `GET /metrics`:

| Metric | Labels | Description |
| :--- | :--- | :--- |
| `smarterrouter_requests_total` | endpoint, method | Total request count |
| `smarterrouter_request_duration_seconds` | endpoint | Request latency histogram |
| `smarterrouter_errors_total` | endpoint, error_type | Error count |
| `smarterrouter_model_selections_total` | selected_model, category | Model selection distribution |
| `smarterrouter_cache_hits_total` / `misses_total` | cache_type | Cache performance |
| `smarterrouter_vram_total_gb` / `used_gb` / `utilization_pct` | — | Global GPU metrics |
| `smarterrouter_gpu_total_gb` / `used_gb` / `free_gb` | gpu_index | Per-GPU metrics |

### Structured Logging

Set `ROUTER_LOG_FORMAT=json` for structured JSON logs. Each entry includes timestamp, level, logger, message, and contextual fields. A unique `request_id` is injected per HTTP request for end-to-end tracing.

### Health Checks

`GET /health` returns:
- Database connectivity
- Backend status
- GPU metrics (total/used/free VRAM)
- Cache status
- Background task counts
- DLQ counts
- Request ID

### VRAM Dashboard

`GET /admin/vram` (admin auth required) returns real-time GPU memory, loaded models with estimates, and historical trend data.

---

## Security

### Built-in Protections

| Layer | Feature |
| :--- | :--- |
| **Authentication** | Admin API key (timing-safe comparison via `hmac.compare_digest`) |
| **Rate Limiting** | Per-IP, per-endpoint, chat-specific rate limits (in-memory) |
| **Input Validation** | Prompt sanitization (control character stripping), length limits (10K chars, 100 messages), model name whitelist validation |
| **SQL Injection** | SQLAlchemy ORM parameterized queries, whitelist validation on bulk operations |
| **API Keys** | Fernet-encrypted storage for external provider keys, per-provider isolation, never logged |
| **CORS** | Configurable origins, methods, headers, credentials |
| **Network** | IP whitelist for admin endpoints (CIDR + exact), TLS verification toggle |
| **Docker** | Non-root user, capability restrictions, read-only filesystem support |
| **Audit** | Persistent admin audit log with query endpoint |

### Key Security Recommendations

```bash
# Generate a strong admin key
ROUTER_ADMIN_API_KEY=$(openssl rand -hex 32)

# Bind to localhost if not exposing externally
ROUTER_HOST=127.0.0.1

# Enable rate limiting
ROUTER_RATE_LIMIT_ENABLED=true
ROUTER_RATE_LIMIT_REQUESTS_PER_MINUTE=60
```

See [SECURITY.md](SECURITY.md) for the full security policy, vulnerability reporting process, and best practices.

---

## Development & Testing

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=. --cov-report=term-missing

# Specific test file
pytest tests/test_router.py -v

# Specific test
pytest tests/test_router.py::test_select_model_coding_prompt -v

# By marker
pytest -m "not slow" -v
```

The test suite includes **54+ test files** covering:
- Unit tests for router, profiler, config, schemas, modality, model filtering
- Backend contract tests (Ollama, llama.cpp, OpenAI)
- Security tests (injection, edge cases, admin IP whitelist, audit)
- Cache tests (persistence, Redis, statistics, semantic)
- Integration tests (full API, extended scenarios)
- Property-based tests
- Concurrency stress tests
- Compression pipeline tests (scanner, chunker, statistical, arbitrage, DCA, pipeline)
- VRAM manager and monitor tests
- Snapshot routing tests

### Linting & Type Checking

```bash
ruff check .           # Linting
ruff format --check .  # Formatting
mypy .                 # Type checking
```

### Code Standards

- **Line length:** 100 characters
- **Python:** 3.11+, type annotations, snake_case / PascalCase / SCREAMING_SNAKE
- **Testing:** `pytest-asyncio`, descriptive names (`test_<method>_<expected_behavior>`)
- **Commits:** [Conventional Commits](https://www.conventionalcommits.org/) (`feat(router): ...`, `fix(profiler): ...`)

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full contributing guide.

---

## Project Structure

```
SmarterRouter/
├── main.py                          # FastAPI app creation, middleware, router registration
├── benchmark.py                     # Academic benchmark suite (token savings, TTFT, VRAM)
├── Dockerfile                       # Production container (Python 3.11-slim, non-root user)
├── docker-compose.yml               # Default Compose with NVIDIA GPU support
├── docker-compose.external.yml      # External-providers-only deployment
├── docker-run.sh                    # Guided Docker launch with GPU auto-detection
├── ENV_DEFAULT                      # Complete annotated configuration template
├── pyproject.toml                   # Build config, dependencies, tool settings
├── requirements.txt                 # Python dependencies
│
├── router/                          # Core application package
│   ├── config.py                    #   Pydantic Settings, environment configuration
│   ├── router.py                    #   Prompt analysis, model scoring, routing, semantic cache
│   ├── state.py                     #   Shared application state, admin auth helpers
│   ├── lifecycle.py                 #   Startup, background tasks, shutdown management
│   ├── middleware.py                #   Request logging, rate limiting, security middleware
│   ├── schemas.py                   #   Pydantic request/response schemas
│   ├── models.py                    #   SQLAlchemy ORM models (profiles, benchmarks, feedback)
│   ├── database.py                  #   Database engine, session management, migrations
│   ├── profiler.py                  #   Local model profiling with MT-Bench prompts
│   ├── judge.py                     #   LLM-as-Judge evaluation pipeline
│   ├── benchmark_db.py              #   Local benchmark storage & sync
│   ├── benchmark_sync.py            #   HuggingFace/LMSYS benchmark synchronization
│   ├── provider_db.py               #   External provider benchmark database (400+ models)
│   ├── modality.py                  #   Vision, tool-calling, embedding modality detection
│   ├── model_filter.py              #   Glob-pattern model include/exclude filtering
│   ├── model_metadata.py            #   Dynamic model metadata registry with TTL caching
│   ├── vram_monitor.py              #   GPU VRAM monitoring (background task)
│   ├── vram_manager.py              #   VRAM budget tracking, model load/unload coordination
│   ├── cache.py                     #   Unified cache manager (TTL, LRU)
│   ├── cache_redis.py               #   Redis cache backend
│   ├── cache_stats.py               #   Cache analytics & time-series tracking
│   ├── persistent_cache.py          #   SQLite-backed persistent cache
│   ├── security.py                  #   Prompt-injection detection, content moderation
│   ├── circuit_breaker.py           #   Circuit breaker pattern for backend resilience
│   ├── encryption.py                #   Fernet encryption for API keys
│   ├── audit.py                     #   Admin audit logging
│   ├── dlq.py                       #   Dead Letter Queue for failed background tasks
│   ├── metrics.py                   #   Prometheus metric definitions
│   ├── skills.py                    #   Tool/skill registry for agentic workflows
│   ├── prompts.py                   #   Profiling prompt templates
│   ├── exceptions.py                #   Custom exception hierarchy
│   ├── logging_config.py            #   Structured/JSON logging configuration
│   ├── entrypoint.py                #   Docker setup, validation, server startup
│   ├── cli.py                       #   CLI commands (setup, check, generate-env)
│   │
│   ├── api/                         #   HTTP endpoint routers
│   │   ├── chat.py                  #     /v1/chat/completions, streaming, tool loop
│   │   ├── models.py                #     /v1/models, /v1/embeddings, /v1/feedback, /v1/skills
│   │   ├── admin.py                 #     All /admin/* endpoints
│   │   ├── health.py                #     /health endpoint
│   │   └── demo.py                  #     Demo/testing endpoints
│   │
│   ├── backends/                    #   LLM backend abstraction layer
│   │   ├── base.py                  #     LLMBackend Protocol definition
│   │   ├── ollama.py                #     Ollama backend (model management, VRAM)
│   │   ├── llama_cpp.py             #     llama.cpp server backend
│   │   ├── openai.py                #     OpenAI-compatible backend
│   │   ├── external.py              #     External provider factory
│   │   ├── registry.py              #     BackendRegistry (multi-backend routing)
│   │   ├── retry.py                 #     Retry logic, quota detection
│   │   └── resilience.py            #     Resilience configuration
│   │
│   ├── compression/                 #   Pre-flight context compression pipeline
│   │   ├── pipeline.py              #     Master ContextCompressionPipeline orchestrator
│   │   ├── scanner.py               #     Sub-millisecond non-ASCII language detector
│   │   ├── dca.py                   #     Dynamic Context-Window Allocation
│   │   ├── arbitrage.py             #     Cross-Lingual SLM Rewriter (Bi/Tri-Block)
│   │   ├── chunker.py              #     AST & code-aware indentation segmenter
│   │   ├── statistical.py           #     BM25 + Entropy + Position composite scorer
│   │   ├── force_keep.py            #     Security/credential/syntax preservation rules
│   │   ├── cache_aligner.py         #     Prefix cache aligner & tool schema sorter
│   │   └── car.py                   #     Content-Addressable Recovery index
│   │
│   ├── gpu_backends/                #   GPU vendor-specific VRAM detection
│   │   ├── base.py                  #     GPUBackend Protocol
│   │   ├── nvidia.py                #     NVIDIA (nvidia-smi)
│   │   ├── amdgpu.py                #     AMD (rocm-smi + sysfs, APU unified memory)
│   │   ├── intel.py                 #     Intel (i915/xe sysfs)
│   │   └── apple.py                 #     Apple Silicon (unified memory)
│   │
│   ├── memory/                      #   Vector memory / conversation memory
│   └── providers/                   #   Provider-specific configurations
│
├── tests/                           #   Test suite (54+ test files)
│   ├── conftest.py                  #     Fixtures, markers (slow, integration)
│   ├── integration/                 #     Integration test suites
│   ├── load/                        #     Load testing utilities
│   └── test_*.py                    #     Unit & integration tests
│
├── scripts/                         #   Development & deployment scripts
│   ├── apply_optimizations.py
│   ├── apply_router_optimizations.py
│   ├── download_provider_db.sh
│   ├── fix_schema.py
│   └── optimize_performance.py
│
├── data/                            #   Persistent data (SQLite databases)
│   ├── router.db                    #     Profiles, benchmarks, feedback, cache
│   ├── provider.db                  #     External provider benchmarks (400+ models)
│   └── conversation_memory.db       #     Conversation memory storage
│
├── docs/                            #   Extended documentation
│   ├── architecture.md              #     System architecture with Mermaid diagrams
│   ├── api.md                       #     Complete API reference
│   ├── configuration.md             #     Full configuration reference
│   ├── backends.md                  #     Backend provider details
│   ├── external-providers.md        #     External provider setup
│   ├── benchmarks.md                #     Benchmark sources & configuration
│   ├── installation.md              #     Detailed installation guide
│   ├── DOCKER.md                    #     Docker deployment guide
│   ├── kubernetes.md                #     Kubernetes deployment guide
│   ├── performance.md               #     Performance tuning guide
│   ├── troubleshooting.md           #     Troubleshooting guide
│   ├── contributing.md              #     Extended contributing guide
│   ├── docker-compose.*.yml/md      #     GPU-specific Compose templates
│   └── examples/                    #     Client integration & production examples
│
├── DEEPDIVE.md                      #   Architectural deep dive
├── implementation.md                #   Context compression specification
├── CHANGELOG.md                     #   Full version history (v1.0 → v2.2.7)
├── CONTRIBUTING.md                  #   Contributing guide
├── SECURITY.md                      #   Security policy & vulnerability reporting
└── LICENSE                          #   MIT License
```

---

## Documentation Index

| Document | Description |
| :--- | :--- |
| [Installation Guide](docs/installation.md) | Step-by-step installation for all platforms |
| [Configuration Reference](docs/configuration.md) | Complete list of every `ROUTER_*` setting |
| [API Reference](docs/api.md) | Full endpoint documentation |
| [Backend Providers](docs/backends.md) | Setup for Ollama, llama.cpp, OpenAI-compatible |
| [External Providers](docs/external-providers.md) | OpenAI, Anthropic, Google, Cohere, Mistral setup |
| [Benchmarks](docs/benchmarks.md) | Benchmark sources and configuration |
| [Docker Deployment](docs/DOCKER.md) | Docker and Docker Compose guide |
| [Kubernetes Deployment](docs/kubernetes.md) | K8s manifests and Helm examples |
| [Performance Tuning](docs/performance.md) | Optimization guide |
| [Troubleshooting](docs/troubleshooting.md) | Common issues and solutions |
| [Architecture](docs/architecture.md) | System architecture with diagrams |
| [Client Integration](docs/examples/client-integration.md) | Python, JS, curl, OpenWebUI, Continue, Cursor examples |
| [Production Deployment](docs/examples/production-deployment.md) | Production hardening guide |
| [OpenWebUI Integration](docs/examples/openwebui-integration.md) | OpenWebUI setup guide |
| [Deep Dive](DEEPDIVE.md) | Architectural deep dive and design rationale |
| [Implementation Spec](implementation.md) | Context compression technical specification |
| [Changelog](CHANGELOG.md) | Full version history |
| [Contributing](CONTRIBUTING.md) | How to contribute |
| [Security Policy](SECURITY.md) | Vulnerability reporting and security best practices |

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:

- Development setup
- Coding standards (Ruff, mypy, 100-char lines)
- Testing requirements (all features must include tests)
- Pull request process
- Commit message format ([Conventional Commits](https://www.conventionalcommits.org/))

**Quick start for contributors:**

```bash
git clone https://github.com/YOUR_USERNAME/SmarterRouter.git
cd SmarterRouter
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp ENV_DEFAULT .env
pytest                    # Run all tests
ruff check . && mypy .    # Lint & type check
```

---

## License

SmarterRouter is released under the **MIT License**. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <b>SmarterRouter</b> — Stop choosing models. Let the router choose for you.<br>
  <a href="https://github.com/peva3/SmarterRouter">GitHub</a> ·
  <a href="docs/api.md">API Docs</a> ·
  <a href="CHANGELOG.md">Changelog</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>
