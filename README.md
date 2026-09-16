# SmarterRouter

SmarterRouter is an OpenAI-compatible FastAPI gateway that chooses an appropriate language model for each request. It can sit in front of local Ollama or llama.cpp servers, an OpenAI-compatible endpoint, or a mixture of local and external providers.

The router combines prompt analysis, model capabilities, local performance profiles, benchmark data, feedback, VRAM availability, caching, and backend health. Clients send requests to one stable endpoint instead of choosing and managing models themselves.

## Contents

- [What it does](#what-it-does)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Quick start with Docker](#quick-start-with-docker)
- [Install locally](#install-locally)
- [Configure the router](#configure-the-router)
- [Choose a backend](#choose-a-backend)
- [Use the API](#use-the-api)
- [CLI](#cli)
- [Operations and deployment](#operations-and-deployment)
- [Development and testing](#development-and-testing)
- [Repository guide](#repository-guide)
- [Security](#security)
- [Documentation](#documentation)
- [License](#license)

## What it does

- Discovers models from the configured backend and polls for changes.
- Profiles local models with reasoning, coding, creativity, factual, speed, latency, and VRAM measurements.
- Combines profiles with benchmark data from Hugging Face, LMSYS, and optionally ArtificialAnalysis.
- Classifies prompts by task and complexity, then scores available models.
- Balances quality against speed with `ROUTER_QUALITY_PREFERENCE`.
- Detects modality and model capabilities such as vision and tool calling.
- Uses fallback cascades, retries, and an optional circuit breaker when a backend fails.
- Manages local model loading and unloading when the backend supports it.
- Tracks GPU memory on NVIDIA, AMD, Intel, Apple Silicon, and multi-GPU systems.
- Caches routing decisions, embeddings, and responses. Memory and Redis cache backends are supported.
- Accepts streaming responses and common OpenAI-compatible request shapes, including `/v1/responses` aliases.
- Exposes health, Prometheus metrics, model, feedback, profile, benchmark, cache, and routing explanation endpoints.
- Provides prompt-injection detection, optional content moderation, request validation, CORS, rate limiting, and admin authentication.

SmarterRouter does not run an LLM by itself. You must provide at least one reachable inference backend, unless you intentionally configure external providers only.

## Architecture

```text
OpenAI-compatible client
               |
               v
FastAPI API and middleware
   - validation, security, rate limits, CORS
               |
               v
Router engine
   - prompt analysis and modality detection
   - profiles + benchmarks + feedback
   - quality/speed scoring and fallback cascade
               |
               +--> local backend: Ollama, llama.cpp
               +--> OpenAI-compatible backend
               +--> external providers: OpenAI, Anthropic, Google, Cohere, Mistral
               |
               +--> SQLite router database
               +--> optional Redis cache
               +--> optional provider.db benchmark database
               +--> GPU/VRAM monitor
```

The normal request path is:

1. Validate and normalize the incoming OpenAI-style request.
2. Run configured security and rate-limit checks.
3. Check routing and response caches.
4. Analyze the prompt and determine the required capabilities.
5. Score eligible models from profiles, benchmarks, feedback, size, speed, and quality preference.
6. Reserve/load resources when supported, call the selected backend, and cascade to another model if needed.
7. Store the result and return an OpenAI-compatible response.

The application is created in [main.py](main.py). The router and cache logic live primarily in [router/router.py](router/router.py), while startup and background tasks are managed by [router/lifecycle.py](router/lifecycle.py).

## Requirements

### Runtime

- Python 3.11 or newer for a local installation.
- Docker and Docker Compose v2 for the recommended deployment.
- A reachable model backend:
   - [Ollama](https://ollama.com/) (default and easiest local option)
   - llama.cpp server
   - Any OpenAI-compatible API
   - Configured external providers
- Optional GPU drivers and runtime support for VRAM monitoring and accelerated inference.

### Backend examples

For Ollama:

```bash
ollama serve
ollama pull llama3.2
ollama pull codellama
```

For llama.cpp:

```bash
./llama-server -m models/model.gguf -c 4096 --port 8080
```

The backend URL must be reachable from the SmarterRouter process. In Docker, `localhost` means the container itself; use `host.docker.internal`, `172.17.0.1`, or a Compose service name when the backend runs elsewhere.

## Quick start with Docker

From the repository root:

```bash
git clone https://github.com/peva3/SmarterRouter.git
cd SmarterRouter
copy ENV_DEFAULT .env
docker compose up -d
```

On Linux/macOS, use `cp ENV_DEFAULT .env` instead of `copy`. Edit `.env` if Ollama is not reachable at the configured URL. The Compose file publishes the router at `http://localhost:11436` and persists the SQLite database and related state in `./data`.

Verify the service:

```bash
curl http://localhost:11436/
curl http://localhost:11436/health
docker compose logs -f smarterrouter
```

The container entrypoint performs automatic setup when no configuration exists, validates the environment, and starts Uvicorn. It does not replace the need to install or start Ollama.

### GPU-specific Docker setup

The default Compose file contains the NVIDIA reservation. For other hardware, use the matching template and follow its device-mount instructions:

- [NVIDIA](docs/docker-compose.nvidia.yml)
- [AMD/ROCm](docs/docker-compose.amd.yml)
- [Intel](docs/docker-compose.intel.yml)
- [Apple Silicon](docs/docker-compose.apple.md)
- [Mixed GPUs](docs/docker-compose.multi-gpu.yml)

For a guided Docker launch, inspect [docker-run.sh](docker-run.sh). For the full deployment guide, see [docs/DOCKER.md](docs/DOCKER.md).

## Install locally

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

On Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies and create configuration:

```bash
python -m pip install -r requirements.txt
copy ENV_DEFAULT .env
```

Start the development server:

```bash
python -m uvicorn main:app --reload --host 0.0.0.0 --port 11436
```

Or start without reload:

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 11436
```

The installed console script is also available:

```bash
smarterrouter --help
```

## Configure the router

Configuration uses Pydantic Settings. Environment variables use the `ROUTER_` prefix and override `.env` values. Copy [ENV_DEFAULT](ENV_DEFAULT) as the starting point; it is the complete annotated template.

### Minimum local configuration

```env
ROUTER_PROVIDER=ollama
ROUTER_OLLAMA_URL=http://localhost:11434
ROUTER_PORT=11436
ROUTER_ADMIN_API_KEY=replace-with-a-long-random-secret
```

### Important settings

| Setting | Purpose | Default |
| --- | --- | --- |
| `ROUTER_PROVIDER` | `ollama`, `llama.cpp`, or `openai` | `ollama` |
| `ROUTER_OLLAMA_URL` | Ollama or compatible backend URL | `http://localhost:11434` |
| `ROUTER_LLAMA_CPP_URL` | llama.cpp server URL | unset |
| `ROUTER_OPENAI_BASE_URL` | OpenAI-compatible base URL | unset |
| `ROUTER_OPENAI_API_KEY` | API key for that endpoint | unset |
| `ROUTER_HOST` / `ROUTER_PORT` | Bind address and HTTP port | `0.0.0.0` / `11436` |
| `ROUTER_QUALITY_PREFERENCE` | `0.0` favors speed, `1.0` favors quality | `0.5` |
| `ROUTER_MODEL` | Optional small model used for LLM-based dispatch | unset |
| `ROUTER_PINNED_MODEL` | Model to keep loaded when supported | unset |
| `ROUTER_MODEL_KEEP_ALIVE` | Backend keep-alive seconds; `-1` means indefinitely | `-1` |
| `ROUTER_CASCADING_ENABLED` | Try another capable model after failure | `true` |
| `ROUTER_GENERATION_TIMEOUT` | Backend generation timeout in seconds | `120` |
| `ROUTER_REQUEST_TIMEOUT_SECONDS` | Overall request budget | `300` |
| `ROUTER_DATABASE_URL` | SQLAlchemy database URL | `sqlite:///data/router.db` |
| `ROUTER_CACHE_BACKEND` | `memory` or `redis` | `memory` |
| `ROUTER_REDIS_URL` | Redis connection URL when enabled | unset |
| `ROUTER_LOG_LEVEL` / `ROUTER_LOG_FORMAT` | Logging verbosity and `text`/`json` format | `INFO` / `text` |
| `ROUTER_SIGNATURE_ENABLED` | Append the selected backend model to responses | `true` |

### Profiling and benchmarks

Profiling can take a long time, especially for large models or cold model loads. Relevant settings include:

```env
ROUTER_MODEL_POLLING_ENABLED=true
ROUTER_MODEL_AUTO_PROFILE_ENABLED=true
ROUTER_PROFILE_TIMEOUT=90
ROUTER_PROFILE_PROMPTS_PER_CATEGORY=3
ROUTER_BENCHMARK_SOURCES=huggingface,lmsys
```

Available benchmark sources are `huggingface`, `lmsys`, and `artificial_analysis`. ArtificialAnalysis requires `ROUTER_ARTIFICIAL_ANALYSIS_API_KEY`; model name mappings can be supplied with `ROUTER_ARTIFICIAL_ANALYSIS_MODEL_MAPPING_FILE`.

### External providers

To mix cloud models with local models:

```env
ROUTER_PROVIDER=ollama
ROUTER_EXTERNAL_PROVIDERS_ENABLED=true
ROUTER_EXTERNAL_PROVIDERS=openai,anthropic,google
ROUTER_OPENAI_API_KEY=sk-...
ROUTER_ANTHROPIC_API_KEY=sk-ant-...
ROUTER_GOOGLE_API_KEY=...
```

Models use provider prefixes such as `openai/gpt-4o`, `anthropic/claude-3-opus`, and `google/gemini-1.5-pro`. See [docs/external-providers.md](docs/external-providers.md) for provider-specific settings and external-only Docker deployment.

### Caching

SmarterRouter has separate caches for routing decisions, embeddings, responses, and general application data. Memory caching requires no service. Redis is useful when several router processes need a shared cache:

```env
ROUTER_CACHE_BACKEND=redis
ROUTER_REDIS_URL=redis://localhost:6379/0
```

Persistent routing/response cache settings and analytics options are documented in [docs/configuration.md](docs/configuration.md).

## Choose a backend

| Backend | Best for | Model management |
| --- | --- | --- |
| Ollama | Local inference with automatic discovery | SmarterRouter can load/unload models |
| llama.cpp | Direct GGUF serving and high-performance local inference | Manage server processes yourself |
| OpenAI-compatible | OpenAI, vLLM, TGI, LiteLLM, LocalAI, OpenRouter, and similar services | Managed by the remote service |
| External providers | Direct OpenAI, Anthropic, Google, Cohere, or Mistral routing | Provider-managed |

Set `ROUTER_PROVIDER=ollama`, `llama.cpp`, or `openai` and provide the corresponding URL. Detailed setup, limitations, and resilience settings are in [docs/backends.md](docs/backends.md).

## Use the API

Base URL: `http://localhost:11436`

The `/v1` endpoints are intentionally compatible with common OpenAI clients. The router model shown to clients is `smarterrouter/main` by default; the actual selected model is included in the response signature when enabled.

### Health and models

```bash
curl http://localhost:11436/health
curl http://localhost:11436/v1/models
curl http://localhost:11436/metrics
```

### Chat completion

```bash
curl http://localhost:11436/v1/chat/completions \
   -H "Content-Type: application/json" \
   -d '{
      "model": "smarterrouter/main",
      "messages": [{"role": "user", "content": "Explain binary search briefly."}],
      "stream": false
   }'
```

Set `"stream": true` for Server-Sent Events. The implementation also exposes compatibility aliases at `/chat/completions`, `/v1/responses`, `/responses`, `/v1/completions`, and `/completions`.

### Embeddings, feedback, and skills

```bash
curl http://localhost:11436/v1/embeddings \
   -H "Content-Type: application/json" \
   -d '{"model":"nomic-embed-text","input":"A short document"}'

curl http://localhost:11436/v1/skills

curl http://localhost:11436/v1/feedback \
   -H "Content-Type: application/json" \
   -d '{"response_id":"chatcmpl-12345678","score":1.0,"comment":"Useful answer"}'
```

Feedback can be disabled with `ROUTER_FEEDBACK_ENABLED=false`. If a response ID cannot be found, submit `model_name` with the feedback instead.

### Admin API

Admin endpoints require:

```http
Authorization: Bearer <ROUTER_ADMIN_API_KEY>
```

Available admin areas include model profiles, benchmarks, router statistics, cache analytics and invalidation, VRAM status, reprofiling, benchmark synchronization, routing explanations, audit logs, and dead-letter queue management. See [docs/api.md](docs/api.md) for the complete endpoint reference.

### Python client

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

More Python, JavaScript, curl, OpenWebUI, Continue, Cursor, and other examples are in [docs/examples/client-integration.md](docs/examples/client-integration.md).

## CLI

After installing dependencies, the `smarterrouter` command supports:

```bash
smarterrouter setup
smarterrouter check --config .env
smarterrouter generate-env --output .env --overwrite
```

`setup` detects Ollama and GPU hardware and interactively writes a configuration. `check` validates configuration, Ollama reachability, available models, and GPU detection; it exits non-zero when Ollama is unreachable. `generate-env` creates a suggested configuration without requiring the interactive wizard.

The equivalent module command is `python -m router.cli ...`. The older `python -m smarterrouter ...` form is not the package entry point in this repository.

## Operations and deployment

### Production checklist

1. Set a strong `ROUTER_ADMIN_API_KEY`.
2. Keep `.env` and provider keys out of version control.
3. Enable rate limiting and restrict CORS origins.
4. Put the service behind HTTPS termination and a reverse proxy.
5. Persist `./data` or the configured database volume.
6. Scrape `/metrics` with Prometheus and monitor `/health`.
7. Size timeouts for the largest model and expected cold-load time.
8. Restrict network access to the router and backend ports.

Admin endpoints are not a substitute for network security. Review [SECURITY.md](SECURITY.md) before exposing the service outside a trusted network.

### Docker persistence and logs

```bash
docker compose ps
docker compose logs -f smarterrouter
docker compose restart smarterrouter
docker compose down
```

Do not remove `./data` unless you intend to remove the SQLite database, profiles, cache state, and other persisted application data.

### Kubernetes

The repository includes a deployment guide with Helm-style and raw-manifest examples in [docs/kubernetes.md](docs/kubernetes.md). Validate image names, chart availability, ports, secrets, and GPU device plugins against your cluster before using those examples in production.

## Development and testing

Install development dependencies from `requirements.txt`, then run:

```bash
pytest
pytest --cov=. --cov-report=term-missing
ruff check .
ruff format --check .
mypy .
```

Run a focused test file or test:

```bash
pytest tests/test_backend_contract.py -v
pytest tests/test_security.py -v
pytest tests/test_router.py::test_select_model_coding_prompt -v
```

Tests use `pytest-asyncio`; `slow` and `integration` markers are registered in [tests/conftest.py](tests/conftest.py). The development server with automatic reload is:

```bash
python -m uvicorn main:app --reload --host 0.0.0.0 --port 11436
```

Before opening a pull request, add or update tests for behavior changes, run linting and type checking, and update documentation when configuration or API behavior changes. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Repository guide

| Path | Responsibility |
| --- | --- |
| [main.py](main.py) | FastAPI app creation and router registration |
| [router/config.py](router/config.py) | Pydantic Settings and environment configuration |
| [router/api](router/api) | Health, model, chat, demo, and admin HTTP endpoints |
| [router/router.py](router/router.py) | Prompt analysis, model scoring, routing, and semantic cache |
| [router/backends](router/backends) | Backend abstraction and provider implementations |
| [router/profiler.py](router/profiler.py) | Local model profiling |
| [router/benchmark_db.py](router/benchmark_db.py) | Local benchmark storage and synchronization |
| [router/provider_db.py](router/provider_db.py) | External provider benchmark database |
| [router/vram_monitor.py](router/vram_monitor.py) | GPU and VRAM detection/monitoring |
| [router/lifecycle.py](router/lifecycle.py) | Startup, background jobs, and shutdown |
| [router/entrypoint.py](router/entrypoint.py) | Docker setup, validation, and server startup |
| [tests](tests) | Unit, integration, API, backend, cache, and security tests |

## Documentation

- [Installation](docs/installation.md)
- [Configuration reference](docs/configuration.md)
- [API reference](docs/api.md)
- [Backend providers](docs/backends.md)
- [External providers](docs/external-providers.md)
- [Docker deployment](docs/DOCKER.md)
- [Production deployment](docs/examples/production-deployment.md)
- [Performance tuning](docs/performance.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Architecture](docs/architecture.md)
- [Deep dive and implementation notes](DEEPDIVE.md)
- [Changelog](CHANGELOG.md)
- [Contributing](CONTRIBUTING.md)
- [Security policy](SECURITY.md)

## License

SmarterRouter is released under the MIT License. See [LICENSE](LICENSE).

MIT License - see [LICENSE](LICENSE) for details.
