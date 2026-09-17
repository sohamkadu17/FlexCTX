# Documentation Guide

This directory documents the behavior of the current SmarterRouter source tree. Start with [installation.md](installation.md), then use [configuration.md](configuration.md) and [api.md](api.md) as the operational references.

## Source Of Truth

| Topic | Authoritative source |
| :--- | :--- |
| Package version and supported Python version | [pyproject.toml](../pyproject.toml) |
| Environment variable names and runtime defaults | [router/config.py](../router/config.py) |
| Annotated environment template | [ENV_DEFAULT](../ENV_DEFAULT) |
| HTTP routes and request handling | [router/api/](../router/api/) and [main.py](../main.py) |
| Container image, ports, and mounted data | [docker-compose.yml](../docker-compose.yml), [Dockerfile](../Dockerfile) |
| Supported behavior and edge cases | [tests/](../tests/) |

The package metadata is `2.1.5`, while the FastAPI application constant is `2.2.7`. This is an existing release-metadata inconsistency. Health responses and OpenAPI currently expose the application value.

## Documentation Map

- [installation.md](installation.md): Docker, local setup, CLI checks, and GPU prerequisites.
- [configuration.md](configuration.md): settings grouped by runtime concern.
- [api.md](api.md): public, demo, health, metrics, and authenticated admin endpoints.
- [backends.md](backends.md): Ollama, llama.cpp, and OpenAI-compatible upstreams.
- [external-providers.md](external-providers.md): upstream provider and benchmark notes.
- [architecture.md](architecture.md): request flow and component boundaries.
- [performance.md](performance.md): profiling, caching, compression, and VRAM tuning.
- [troubleshooting.md](troubleshooting.md): diagnosis by symptom and subsystem.
- [DOCKER.md](DOCKER.md) and [kubernetes.md](kubernetes.md): deployment-specific operations.
- [examples/](examples/): client, OpenWebUI, and production integration examples.

## Validation Commands

Run these from the repository root after changing implementation or documentation:

```powershell
python -m pytest
python -m ruff check .
python -m uvicorn main:app --host 127.0.0.1 --port 11436
```

The server command requires a reachable configured backend only for backend-dependent operations. The FastAPI OpenAPI document is available at `/openapi.json`, with interactive documentation at `/docs` and `/redoc`.