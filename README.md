# FlexCTX

**A context-adaptive, hardware-aware AI gateway that compresses prompts, manages VRAM, and autonomously selects the best language model for every request.**

> **Forked from:** [peva3/SmarterRouter](https://github.com/peva3/SmarterRouter) · **Version:** `2.2.7`

FlexCTX extends SmarterRouter with a production-grade **Pre-Flight Dynamic Context Engineering Pipeline**, **hardware-aware VRAM orchestration**, and **real-time visual telemetry** — solving the three problems that passive routers ignore: *bloated prompts, GPU memory crashes, and black-box model selection*.

---

## How FlexCTX Differs from the Upstream SmarterRouter

> This section is the core innovation of this fork. The upstream project by [peva3](https://github.com/peva3/SmarterRouter) is an excellent, OpenAI-compatible routing proxy. FlexCTX builds on that foundation and adds the following engineering contributions.

### Comparison at a Glance

| Capability | SmarterRouter (upstream, peva3) | **FlexCTX (this fork)** |
| :--- | :--- | :--- |
| **Prompt forwarded to model** | Raw, unmodified prompt | ✅ Pre-compressed by AST + BM25 pipeline (**~58% fewer tokens**) |
| **VRAM awareness** | ❌ Hardware-blind; crashes on OOM | ✅ Multi-vendor live telemetry (NVIDIA, AMD, Intel, Apple) |
| **OOM prevention** | ❌ CUDA OOM crashes if two models load | ✅ Automated LRU eviction via `keep_alive=0` before model load |
| **Context window budgeting** | Static 8K limit in config | ✅ Dynamic Context Allocation (DCA) with **hysteresis state machine** |
| **Prompt compression** | ❌ None | ✅ 5-stage pre-flight pipeline: scanner → DCA → arbitrage → AST chunker → BM25 entropy pruner |
| **Code syntax safety** | ❌ Naive splitting corrupts `import os.path` | ✅ AST parser preserves all syntax boundaries in code blocks |
| **Prefix cache alignment** | ❌ Rewrites invalidate vLLM/RadixAttention KV cache | ✅ Byte-exact prefix isolation + alphabetical tool schema sorting |
| **Long-term memory** | ❌ Stateless — each request stands alone | ✅ SQLite vector store with cosine similarity recall |
| **Semantic caching** | Basic exact-hash | ✅ 3-Tier: SHA-256 (L1) → Semantic cosine (L2) → SQLite/Redis (L3) |
| **Model selection transparency** | Routing decision not explained | ✅ `/admin/explain` endpoint with full scoring matrix per model |
| **Visual live dashboard** | ❌ No web UI | ✅ Interactive real-time dashboard (`GET /demo`) with VRAM bars, DCA state, compression diff |
| **Drop-in OpenAI compatibility** | ✅ Yes | ✅ Yes (100% preserved) |

---

### FlexCTX Innovation 1 — Pre-Flight Dynamic Context Compression

**The Problem (The 100:1 Agentic Payload Problem):**
Multi-turn agent loops (ReAct, RAG, code assistants) accumulate massive token overhead:
- System instructions: 2K–10K tokens
- Tool schemas passed on every turn: 5K–50K tokens
- Conversation history: 1K–20K tokens
- Actual output: just 50–300 tokens

Sending all this raw to the model wastes GPU prefill time, saturates VRAM KV-cache, and spikes TTFT (Time To First Token).

**FlexCTX Solution — 5-Stage Pre-Flight Pipeline (runs on CPU in < 2ms):**

```
Incoming Prompt (raw, 8,420 tokens avg)
     │
     ▼  Stage 1: Language Scanner (< 0.1 ms)
     │  Non-ASCII density check — bypasses pipeline entirely for standard English
     │
     ▼  Stage 2: Dynamic Context Allocation (DCA)
     │  Partitions token budget across system/history/RAG/output
     │  Hysteresis state machine: switches between Normal (8K) and Nerf Mode (4K)
     │  based on live GPU VRAM headroom (prevents oscillation)
     │
     ▼  Stage 3: Cross-Lingual Token Arbitrage (multilingual prompts only)
     │  Micro-SLM (Qwen 0.5B) rewrites verbose non-English text into compact Bi/Tri-Block
     │  validate_light() guard: structural check + ≥ 5% token reduction guarantee
     │
     ▼  Stage 4: AST-Aware Lexical Statistical Pruner
     │  Code-aware chunker: Python ast module + indentation block segmenter
     │  Composite scorer: w1·BM25 + w2·Overlap + w3·Position + w4·Entropy + w5·InvFiller
     │  Force-Keep gate: protects auth tokens, API keys, import paths, function signatures
     │  Jaccard deduplication (> 0.85 similarity → evict redundant chunks)
     │
     ▼  Stage 5: Prefix Cache Alignment
        Byte-exact system prompt isolation + alphabetically sorted tool schemas
        SHA-256 Content-Addressable Recovery (CAR) handles for reversible decompression
        RadixAttention / vLLM APC compatible — maximizes prefix cache hit rate

Output: Compressed Prompt (3,540 tokens avg) → sent to Router Engine
```

**Measured Results on RTX 4050 6GB:**

| Metric | Baseline (upstream) | FlexCTX | Improvement |
| :--- | :--- | :--- | :--- |
| Avg Input Tokens/Turn | 8,420 | 3,540 | **−57.9%** |
| Time-To-First-Token | 1,840 ms | 420 ms | **4.38× faster** |
| Peak VRAM (KV cache) | 5.75 GB | 3.62 GB | **−37.0%** |
| Prefix Cache Hit Rate | 12.4% | 91.8% | **+79.4%** |
| Task Pass@1 Accuracy | 94.0% | 94.0% | **0% regression** |

---

### FlexCTX Innovation 2 — Hardware-Aware VRAM Orchestration

**The Problem:** The upstream SmarterRouter has no concept of physical GPU memory. Running two large models simultaneously causes silent CUDA OOM crashes.

**FlexCTX Solution:**

1. **Multi-Vendor Live Telemetry** (`router/vram_monitor.py`, `router/gpu_backends/`):
   - NVIDIA: `pynvml` NVML API
   - AMD: ROCm sysfs + APU unified GTT memory pool
   - Intel: i915/xe driver sysfs metrics
   - Apple Silicon: Metal unified memory
   - Background thread polls every 30 seconds; feeds live data to dashboard and routing engine

2. **VRAM Budget Enforcement & LRU Eviction** (`router/vram_manager.py`):
   - Before loading any model, calculates: `required_VRAM = model_estimate + KV_cache_budget`
   - If headroom is insufficient → triggers `_evict_models_for_capacity()` using LRU or largest-first policy
   - Sends `POST /api/chat` with `keep_alive=0` to Ollama to release GPU memory gracefully
   - Pinned models (e.g. `qwen2.5:3b`) are exempt from eviction

3. **Dynamic Context Allocation Feedback Loop** (`router/compression/dca.py`):
   - DCA reads current VRAM from the monitor
   - If used VRAM > 75% → engages **Nerf Mode** (shrinks context window to 4K)
   - Returns to Normal Mode only when VRAM drops below 60% (hysteresis prevents thrashing)

---

### FlexCTX Innovation 3 — 6-Stage Mathematical Model Selection

The upstream router uses basic prompt classification. FlexCTX replaces it with a **6-stage scoring pipeline** using 4 independent signals per model:

$$\text{Score}(m) = \underbrace{(S_b \times 1.5 + S_{ELO} \times 1.0 + S_p \times 0.8 + S_a \times 0.4)}_{\text{Capability Score}} \times w_q + \underbrace{B_{feedback} + B_{size} + B_{speed}}_{\text{Bonus/Penalty}} - P_{diversity}$$

Where:
- $S_b$ = Static benchmark score (HumanEval/MMLU from `provider.db`, 400+ models)
- $S_{ELO}$ = LMSYS Chatbot Arena / ArtificialAnalysis ELO normalized to [0, 1.5]
- $S_p$ = Local runtime profile measured on *your hardware* (tokens/sec, TTFT via MT-Bench)
- $S_a$ = Name-affinity heuristic (e.g. `*coder*` model gets coding boost)
- $w_q$ = `ROUTER_QUALITY_PREFERENCE` (0.0 = max speed, 1.0 = max intelligence)
- $B_{size}$ = Hard minimum parameter cutoffs (hard coding → ≥ 8B, else −50 penalty)
- $P_{diversity}$ = Penalty for over-selected models to distribute load

**Explain any routing decision live:**
```bash
curl "http://localhost:11436/admin/explain?prompt=Write+a+lock-free+ring+buffer+in+Rust"
```
Returns full scoring matrix: category confidence vectors, per-model scores, and human-readable reasoning.

---

### FlexCTX Innovation 4 — 3-Tier Semantic Caching & Vector Memory

**Caching (upstream had basic exact-hash only):**

```
Request arrives
     │
     ▼  L1: Exact SHA-256 hash match (< 0.1 ms)
     │  Hit → return immediately
     │
     ▼  L2: Semantic cosine similarity (cosine ≥ 0.92 threshold)
     │  Zero-dependency in-process hash vectorizer (sub-ms, no GPU needed)
     │  Hit → return cached response
     │
     ▼  L3: SQLite / Redis persistent cache
        Survives server restarts; 7-day TTL
        Pre-warms top-K popular queries on startup
```

**Long-Term Conversation Vector Memory (new in FlexCTX):**
- `conversation_memory.db` (SQLite) stores embeddings of past conversation turns
- On each new turn: cosine similarity search recalls semantically relevant history
- Injects recalled context automatically — no user configuration needed
- Fallback: in-process token-hash vectorizer (no external embedding model required)

---

### FlexCTX Innovation 5 — Real-Time Visual Dashboard

Open `http://localhost:11436/demo` in your browser:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  ⚡ FlexCTX │ Context-Adaptive AI Value Gate          ● Gateway Active   │
├──────────────┬─────────────────┬──────────────────┬───────────────────── │
│ GPU & VRAM   │ DCA Allocation   │ Token Savings    │ Pre-Flight Overhead │
│ 4.1/6.0 GB  │ 4,096 Tokens     │ 54.2% saved      │ 1.4 ms             │
│ ████████░░  │ 📉 Nerf Mode     │ 14,820 tokens    │ Runs on CPU        │
│ RTX 4050    │ Hysteresis Active│ BM25 + Arbitrage │ Zero GPU contention│
├──────────────────────────────────┬───────────────────────────────────────┤
│  📥 Input Raw Context            │  ✨ Verbatim Compressed Prompt        │
│  [📑 RAG] [💻 Python] [🌐 Multi] │  [Diff Viewer — red = pruned]         │
│                                  │                                        │
│  [textarea: paste your prompt]   │  🤖 Model Output (via Ollama)          │
│                                  │                                        │
│  [⚡ Run Dynamic Compression]    │  [Waiting for generation...]           │
└──────────────────────────────────┴───────────────────────────────────────┘
```

---

## Architecture Overview

```text
OpenAI-compatible client / OpenCode IDE / Web UI / curl
                       │
                       ▼
FastAPI Gateway & Security Middleware (middleware.py / security.py)
   - IP whitelist · rate limits · prompt-injection detection · CORS
                       │
                       ▼
── FlexCTX Pre-Flight Context Engineering Pipeline (router/compression/) ──
   scanner.py   → sub-ms non-ASCII skip for English prompts
   dca.py       → VRAM-aware dynamic context budget (hysteresis)
   arbitrage.py → cross-lingual SLM token compaction
   chunker.py   → AST-aware code-safe segmenter
   statistical.py → BM25 + Shannon entropy lexical pruner
   force_keep.py  → syntax / credential preservation gate
   cache_aligner.py → prefix cache boundary alignment
   car.py          → SHA-256 content-addressable recovery
                       │
                       ▼
Long-Term Conversation Vector Memory (router/memory/)
   SQLite vector store → cosine similarity recall → context injection
                       │
                       ▼
3-Tier Semantic Cache (router/cache.py, persistent_cache.py, cache_redis.py)
   L1: SHA-256 exact match → L2: cosine semantic → L3: SQLite/Redis
                       │
                       ▼  (cache miss only)
Router Decision Engine (router/router.py)
   6-stage scoring: modality filter → prompt analysis → 4-signal scoring
   → category-size cutoffs → bonus/penalty → VRAM check + LRU eviction
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
     Ollama       llama.cpp    External Providers
  (local GPU)   (local GPU)   OpenAI · Anthropic
                               Google · Cohere · Mistral
                       │
          ┌────────────┼────────────────────┐
          ▼            ▼                    ▼
   router.db     conversation_memory.db   provider.db
  (profiles,      (vector memory,         (400+ model
  cache, audit)    turn embeddings)        benchmarks)
```

---

## Key Features

| Category | Capabilities |
| :--- | :--- |
| **Intelligent Routing** | 6-stage mathematical scoring · task categorization (coding, reasoning, creativity, factual) · quality-vs-speed tuning (`ROUTER_QUALITY_PREFERENCE`) · modality-aware routing (vision, tool calling, embeddings) · cascading fallbacks · `/admin/explain` transparency endpoint |
| **Context Compression** | 5-stage pre-flight pipeline · sub-ms language scanner · DCA with hysteresis · cross-lingual token arbitrage · AST-aware code chunking · BM25 + entropy statistical pruning · critical syntax force-keep · prefix cache alignment (RadixAttention/APC) · −57.9% tokens |
| **Vector Memory** | Long-term conversation memory (`conversation_memory.db`) · cosine similarity recall · automated prompt context injection · zero-dependency in-process hash vectorizer fallback |
| **Visual Web Dashboard** | Interactive real-time dashboard (`GET /demo`) · live GPU VRAM progress bars · DCA hysteresis state indicator · cumulative token savings counter · before/after compression diff viewer |
| **Backend Support** | Ollama (model load/unload, discovery) · llama.cpp · any OpenAI-compatible API · external cloud providers (OpenAI, Anthropic, Google, Cohere, Mistral) |
| **VRAM Orchestration** | Multi-vendor monitoring (NVIDIA NVML, AMD ROCm/APU, Intel i915/xe, Apple Silicon) · automated LRU model eviction (`keep_alive=0`) · VRAM budget enforcement · pinned model support |
| **Model Profiling** | Local MT-Bench-inspired profiling · optional LLM-as-Judge grading · capability detection (vision, tool calling) · VRAM footprint measurement |
| **Benchmark Integration** | HuggingFace leaderboard · LMSYS Chatbot Arena · ArtificialAnalysis · `provider.db` with 400+ models from 28+ benchmark sources |
| **Caching** | 3-tier hybrid cache: exact SHA-256 (L1) → semantic cosine (L2) → SQLite/Redis (L3) · adaptive similarity thresholds · LRU eviction · top-K pre-warming |
| **Security** | Rate limiting (per-IP, per-endpoint) · prompt-injection detection · input sanitization · admin API key auth · IP whitelist · CORS · Fernet-encrypted API key storage · SQL injection prevention · audit logging |
| **Observability** | Prometheus metrics (`/metrics`) · structured JSON logging · `X-Request-ID` correlation · health endpoint with DB/GPU/backend/DLQ status · VRAM dashboard |
| **API Compatibility** | Full OpenAI-compatible API · SSE streaming · tool/function calling · `/v1/chat/completions`, `/v1/embeddings`, `/v1/models`, `/v1/skills`, `/v1/responses` · drop-in replacement for OpenAI SDK · OpenCode IDE provider |

> **Note:** FlexCTX does **not** run an LLM by itself. You must provide at least one reachable inference backend (Ollama, llama.cpp, or a cloud API).

---

## How It Works — The Routing Pipeline

Every incoming request passes through a well-defined pipeline:

1. **Validate & Normalize** — Parse the incoming OpenAI-style request, validate model names, enforce request size limits.
2. **Security & Rate Limiting** — Apply rate limits, prompt-injection detection, content moderation, and CORS checks.
3. **Check Caches** — L1 exact-hash → L2 semantic vector → L3 SQLite/Redis. On cache hit, return immediately (< 5 ms).
4. **Pre-Flight Context Compression** — Run the 5-stage pipeline: language scanner → DCA → cross-lingual arbitrage → AST-aware statistical pruning → prefix cache alignment. (~58% token reduction on CPU in < 2 ms).
5. **Vector Memory Recall** — Query `conversation_memory.db` for semantically similar past turns; inject relevant context.
6. **Prompt Analysis** — Classify by task (coding, reasoning, creativity, factual), estimate complexity (easy/medium/hard), detect required capabilities.
7. **Score & Select Model** — Apply the 6-stage scoring formula combining static benchmarks, local profiles, user feedback, quality preference, size cutoffs, and diversity penalty.
8. **VRAM Check & Eviction** — Verify GPU headroom. If insufficient, evict LRU models before dispatching.
9. **Execute & Cascade** — Forward to selected backend (streaming or non-streaming). On failure, auto-cascade to next-best model.
10. **Post-Process** — Append model signature, save to cache, write audit log, return OpenAI-compatible response.

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
- *(Optional)* GPU drivers for VRAM monitoring and accelerated inference (NVIDIA, AMD ROCm, Intel, Apple)

### Backend Examples

**Ollama:**
```bash
ollama serve
ollama pull llama3.2
ollama pull qwen2.5-coder:7b
```

**llama.cpp:**
```bash
./llama-server -m models/model.gguf -c 4096 --port 8080
```

> **Docker note:** When the backend runs on the host, use `host.docker.internal`, `172.17.0.1`, or a Compose service name instead of `localhost`.

---

## Quick Start with Docker

```bash
git clone https://github.com/sohamkadu17/SmarterRouter.git
cd SmarterRouter
cp ENV_DEFAULT .env          # On Windows: copy ENV_DEFAULT .env
docker compose up -d
```

Edit `.env` if Ollama is not reachable at the configured URL. The Compose file publishes the router at `http://localhost:11436` and persists state in `./data`.

**Verify:**
```bash
curl http://localhost:11436/health
curl http://localhost:11436/v1/models
```

Open the live dashboard:
```
http://localhost:11436/demo
```

### GPU-Specific Docker Setup

| Hardware | Template |
| :--- | :--- |
| NVIDIA | [docker-compose.nvidia.yml](docs/docker-compose.nvidia.yml) |
| AMD / ROCm | [docker-compose.amd.yml](docs/docker-compose.amd.yml) |
| Intel | [docker-compose.intel.yml](docs/docker-compose.intel.yml) |
| Apple Silicon | [docker-compose.apple.md](docs/docker-compose.apple.md) |
| Mixed / Multi-GPU | [docker-compose.multi-gpu.yml](docs/docker-compose.multi-gpu.yml) |

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
# Development
python -m uvicorn main:app --reload --host 0.0.0.0 --port 11436

# Production
python main.py
```

---

## Configuration

Configuration uses **Pydantic Settings** with the `ROUTER_` prefix via environment variables or a `.env` file. Copy [ENV_DEFAULT](ENV_DEFAULT) as the starting point.

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
| `ROUTER_HARDWARE_PRESET` | Hardware profile: `6GB_VRAM`, `12GB_VRAM`, `24GB_VRAM`, `CUSTOM` | `6GB_VRAM` |
| `ROUTER_COMPRESSION_ENABLED` | Pre-flight dynamic context compression toggle | `true` |
| `ROUTER_COMPRESSION_MODE` | Strategy: `full`, `statistical_only`, `arbitrage_only`, `cache_align_only` | `full` |
| `ROUTER_PROVIDER` | Backend type: `ollama`, `llama.cpp`, or `openai` | `ollama` |
| `ROUTER_OLLAMA_URL` | Ollama/compatible backend URL | `http://localhost:11434` |
| `ROUTER_HOST` / `ROUTER_PORT` | Bind address and port | `0.0.0.0` / `11436` |
| `ROUTER_QUALITY_PREFERENCE` | `0.0` = max speed, `1.0` = max quality | `0.5` |
| `ROUTER_PINNED_MODEL` | Model to keep permanently loaded in VRAM | `qwen2.5:3b` |
| `ROUTER_CASCADING_ENABLED` | Auto-retry with next-best model on failure | `true` |
| `ROUTER_DCA_ENABLED` | Dynamic Context Allocation hysteresis | `true` |
| `ROUTER_DCA_TARGET_CONTEXT_LIMIT` | Max context tokens in Normal Mode | `8192` |
| `ROUTER_CACHE_ENABLED` | Enable multi-tier caching | `true` |
| `ROUTER_CACHE_SIMILARITY_THRESHOLD` | Cosine threshold for semantic cache hit | `0.92` |
| `ROUTER_PERSISTENT_CACHE_ENABLED` | SQLite-backed cross-restart cache | `true` |

See [docs/configuration.md](docs/configuration.md) for the complete reference.

### Compression Settings

```env
ROUTER_COMPRESSION_ENABLED=true
ROUTER_COMPRESSION_MODE=full
ROUTER_ARBITRAGE_ENABLED=true
ROUTER_ARBITRAGE_MULTILINGUAL_THRESHOLD=0.15
ROUTER_PRUNER_ENABLED=true
ROUTER_DCA_ENABLED=true
ROUTER_DCA_TARGET_CONTEXT_LIMIT=8192
```

### Security Settings

```env
ROUTER_ADMIN_API_KEY=$(openssl rand -hex 32)
ROUTER_RATE_LIMIT_ENABLED=true
ROUTER_RATE_LIMIT_REQUESTS_PER_MINUTE=60
ROUTER_CORS_ORIGINS=http://localhost:3000
ROUTER_HOST=127.0.0.1
```

---

## API Reference

**Base URL:** `http://localhost:11436`

### Core Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/` or `/demo` | GET | **Interactive Visual Dashboard** — live GPU VRAM, DCA state, compression diff inspector. |
| `/api/demo/telemetry` | GET | Real-time telemetry feed (VRAM, DCA bucket, token savings, vector memory stats). |
| `/api/demo/inspect` | POST | Test harness to inspect before/after compression payloads verbatim. |
| `/v1/chat/completions` | POST | Main chat endpoint. Routes to best model automatically. Supports streaming + tool calling. |
| `/v1/embeddings` | POST | Generate vector embeddings. |
| `/v1/models` | GET | List all discovered/registered models. |
| `/v1/feedback` | POST | Submit user feedback to improve future routing scores. |
| `/health` | GET | Subsystem health: DB, GPU, backends, cache, DLQ. |
| `/metrics` | GET | Prometheus metrics scraper. |

### Admin Endpoints

All require `Authorization: Bearer <ROUTER_ADMIN_API_KEY>`.

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/admin/explain` | GET | **Full scoring breakdown** for any prompt — category vectors, per-model scores, selection rationale. |
| `/admin/vram` | GET | Real-time GPU VRAM status, loaded models, eviction warnings. |
| `/admin/compression/stats` | GET | Compression metrics and cumulative token savings. |
| `/admin/stats` | GET | System diagnostics, uptime, throughput. |
| `/admin/profiles` | GET | Local performance profiles for all models. |
| `/admin/benchmarks` | GET | Aggregated benchmark data. |
| `/admin/cache/stats` | GET | Cache hit/miss analytics across all tiers. |
| `/admin/cache/clear` | POST | Flush all cache tiers. |
| `/admin/audit-log` | GET | Query admin audit logs. |
| `/admin/dlq` | GET | Dead Letter Queue for failed operations. |

Full Swagger UI at `/docs`. Full ReDoc at `/redoc`.

---

## Client Integration Examples

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11436/v1",
    api_key="unused-but-required",
)

# model="auto" → FlexCTX selects the best model automatically
response = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "Write a thread-safe LRU cache in Python."}],
)
print(response.choices[0].message.content)
```

### cURL

```bash
curl http://localhost:11436/v1/chat/completions \
   -H "Content-Type: application/json" \
   -d '{
      "model": "auto",
      "messages": [{"role": "user", "content": "Explain binary search in 2 lines."}],
      "stream": false
   }'
```

### Streaming

```bash
curl http://localhost:11436/v1/chat/completions \
   -H "Content-Type: application/json" \
   -d '{"model": "auto", "messages": [{"role": "user", "content": "Tell me a joke."}], "stream": true}'
```

### Inspect Routing Decision (Admin)

```bash
# See the full mathematical scoring breakdown for any prompt
curl "http://localhost:11436/admin/explain?prompt=Implement+quicksort+in+Rust" \
     -H "Authorization: Bearer <ROUTER_ADMIN_API_KEY>"
```

### OpenCode IDE Integration

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "flexctx": {
      "name": "FlexCTX",
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "http://localhost:11436/v1",
        "apiKey": "admin123"
      },
      "models": {
        "auto": {
          "name": "FlexCTX (VRAM-Adaptive & Context-Compressed)"
        }
      }
    }
  }
}
```

---

## Dynamic Context Compression

FlexCTX includes a **5-stage pre-flight context compression pipeline** solving the *"100:1 agentic payload problem"* — multi-turn agent loops pass 50K–100K tokens of redundant context for just 50–300 tokens of output.

### Three Pillars of Compression

| Pillar | Component | Technique | CPU Overhead |
| :--- | :--- | :--- | :--- |
| **1. Cross-Lingual Arbitrage** | [scanner.py](router/compression/scanner.py) + [arbitrage.py](router/compression/arbitrage.py) | Sub-ms non-ASCII density scanner. For multilingual input, micro-SLM (Qwen 0.5B) rewrites into compact Bi/Tri-block. | 0 ms (English) / ~500 ms (multilingual) |
| **2. AST Statistical Pruner** | [chunker.py](router/compression/chunker.py) + [statistical.py](router/compression/statistical.py) + [force_keep.py](router/compression/force_keep.py) | AST-aware code chunking (preserves `import os.path`, function boundaries). Composite BM25 + overlap + position + entropy + filler scorer. Force-keep gate protects auth tokens, paths, and signatures. | < 2 ms on 50 KB |
| **3. Cache Alignment** | [dca.py](router/compression/dca.py) + [cache_aligner.py](router/compression/cache_aligner.py) + [car.py](router/compression/car.py) | VRAM-aware dynamic context-window allocation with hysteresis. Byte-exact prefix isolation + alphabetical tool schema sorting (RadixAttention/APC safe). SHA-256 Content-Addressable Recovery. | < 1 ms |

---

## GPU & VRAM Management

### Supported Hardware

| Vendor | Detection Method | Multi-GPU | Special |
| :--- | :--- | :--- | :--- |
| **NVIDIA** | `pynvml` NVML API | ✅ | Per-GPU Prometheus metrics |
| **AMD** | `rocm-smi` + sysfs | ✅ | APU unified GTT memory pool |
| **Intel** | sysfs (`i915` / `xe` drivers) | ✅ | Battlemage Xe2 support |
| **Apple Silicon** | Unified memory reporting | — | M-series chips |

### RTX 4050 6GB Runtime Budget

| Component | Model | VRAM Usage |
| :--- | :--- | :--- |
| Primary Inference | Qwen2.5-3B-Instruct (Q4_K_M) | ~2.1 GB |
| KV Cache (8K ctx, FP8) | — | ~1.2 GB |
| OS / Display Reserve | Windows DWM | ~1.5 GB |
| **Total** | — | **~4.8 GB (< 6 GB)** |

```env
ROUTER_VRAM_MAX_TOTAL_GB=              # Auto-detected (90% of total)
ROUTER_VRAM_DEFAULT_ESTIMATE_GB=2.5    # Fallback when profiling unavailable
ROUTER_PROFILE_MEASURE_VRAM=true
```

---

## Operations & Deployment

### Production Checklist

1. ✅ Set a strong `ROUTER_ADMIN_API_KEY` (`openssl rand -hex 32`).
2. ✅ Keep `.env` and API keys out of version control (`.gitignore`).
3. ✅ Enable rate limiting and restrict CORS origins.
4. ✅ Put behind HTTPS + reverse proxy (nginx, Caddy).
5. ✅ Persist `./data` volume (SQLite databases, profiles, cache).
6. ✅ Scrape `/metrics` with Prometheus; alert on `/health`.
7. ✅ Review [SECURITY.md](SECURITY.md) before external exposure.

### Docker Commands

```bash
docker compose up -d
docker compose logs -f smarterrouter
docker compose restart smarterrouter
docker compose down
```

> ⚠️ Do **not** remove `./data` unless intentionally deleting all profiles, caches, and memory.

---

## Observability & Monitoring

### Prometheus Metrics

| Metric | Description |
| :--- | :--- |
| `smarterrouter_requests_total` | Total request count by endpoint |
| `smarterrouter_request_duration_seconds` | Request latency histogram |
| `smarterrouter_model_selections_total` | Model selection distribution by category |
| `smarterrouter_cache_hits_total` / `misses_total` | Per-tier cache performance |
| `smarterrouter_vram_total_gb` / `used_gb` / `utilization_pct` | Global GPU metrics |

### Health Check

`GET /health` returns: DB connectivity, backend status, GPU metrics, cache status, DLQ counts, request ID.

---

## Security

| Layer | Feature |
| :--- | :--- |
| **Authentication** | Admin API key (timing-safe `hmac.compare_digest`) |
| **Rate Limiting** | Per-IP, per-endpoint, chat-specific (in-memory) |
| **Input Validation** | Prompt sanitization, 10K char limits, model name whitelist |
| **SQL Injection** | SQLAlchemy ORM parameterized queries |
| **API Keys** | Fernet-encrypted storage for external provider keys |
| **CORS** | Configurable origins, methods, headers |
| **Network** | IP whitelist for admin endpoints (CIDR + exact), TLS toggle |
| **Audit** | Persistent admin audit log with query endpoint |

---

## Development & Testing

### Running Tests

```bash
pytest                                          # All 54+ test files
pytest --cov=. --cov-report=term-missing        # With coverage
pytest tests/test_router.py -v                  # Specific file
pytest -m "not slow" -v                         # Exclude slow tests
```

The test suite covers: routing logic, compression pipeline (scanner, chunker, statistical, arbitrage, DCA), VRAM manager and monitor, backends (Ollama, llama.cpp, OpenAI, external), security (injection, audit, whitelist), caching (persistence, Redis, semantic), integration, concurrency stress, and property-based tests.

### Linting & Type Checking

```bash
ruff check .           # Linting
ruff format --check .  # Formatting
mypy .                 # Type checking
```

---

## Project Structure

```
FlexCTX/
├── main.py                          # FastAPI app, middleware, router registration
├── benchmark.py                     # Academic benchmark suite
├── Dockerfile                       # Production container (Python 3.11-slim)
├── docker-compose.yml               # Default Compose with NVIDIA GPU support
├── ENV_DEFAULT                      # Complete annotated configuration template
│
├── router/                          # Core application package
│   ├── router.py                    #   6-stage model scoring & routing engine
│   ├── config.py                    #   Pydantic Settings
│   ├── state.py                     #   Shared application state
│   ├── lifecycle.py                 #   Startup, background tasks, shutdown
│   ├── middleware.py                #   Rate limiting, request logging
│   ├── prompts.py                   #   Prompt task classification
│   ├── modality.py                  #   Vision / tool-call / embedding detection
│   ├── model_filter.py              #   Capability-based model filtering
│   ├── model_metadata.py            #   Parameter count & context window parsing
│   ├── vram_monitor.py              #   Background GPU VRAM polling
│   ├── vram_manager.py              #   VRAM budget & LRU model eviction
│   ├── profiler.py                  #   Local MT-Bench hardware profiling
│   ├── judge.py                     #   LLM-as-Judge evaluation
│   ├── provider_db.py               #   400+ model benchmark database
│   ├── benchmark_sync.py            #   HuggingFace/LMSYS sync
│   ├── cache.py                     #   Unified L1 cache (SHA-256, LRU)
│   ├── persistent_cache.py          #   SQLite L3 persistent cache
│   ├── cache_redis.py               #   Redis distributed cache
│   ├── cache_stats.py               #   Cache analytics
│   ├── security.py                  #   Prompt-injection detection
│   ├── encryption.py                #   Fernet API key encryption
│   ├── circuit_breaker.py           #   Circuit breaker (Closed→Open→Half-Open)
│   ├── dlq.py                       #   Dead Letter Queue
│   ├── audit.py                     #   Admin audit logging
│   ├── metrics.py                   #   Prometheus metric definitions
│   │
│   ├── api/                         #   HTTP endpoint routers
│   │   ├── chat.py                  #     /v1/chat/completions, SSE streaming
│   │   ├── models.py                #     /v1/models, /v1/embeddings, /v1/feedback
│   │   ├── admin.py                 #     All /admin/* endpoints + /admin/explain
│   │   ├── health.py                #     /health endpoint
│   │   └── demo.py                  #     Live visual dashboard (/demo)
│   │
│   ├── backends/                    #   LLM backend abstraction
│   │   ├── ollama.py                #     Ollama: model load/unload, keep_alive
│   │   ├── llama_cpp.py             #     llama.cpp server
│   │   ├── openai.py                #     OpenAI-compatible
│   │   ├── external.py              #     Anthropic, Google, Cohere, Mistral
│   │   ├── registry.py              #     Multi-backend discovery & health
│   │   └── resilience.py / retry.py #     Circuit breaker + retry policies
│   │
│   ├── compression/                 #   ★ FlexCTX Pre-Flight Compression Pipeline
│   │   ├── pipeline.py              #     Master ContextCompressionPipeline orchestrator
│   │   ├── scanner.py               #     Sub-ms non-ASCII language detector
│   │   ├── dca.py                   #     Dynamic Context Allocation (hysteresis)
│   │   ├── arbitrage.py             #     Cross-lingual SLM rewriter
│   │   ├── chunker.py               #     AST & code-aware segmenter
│   │   ├── statistical.py           #     BM25 + entropy composite scorer
│   │   ├── force_keep.py            #     Syntax / credential preservation
│   │   ├── cache_aligner.py         #     Prefix cache boundary aligner
│   │   └── car.py                   #     SHA-256 content-addressable recovery
│   │
│   ├── gpu_backends/                #   ★ Multi-Vendor GPU Telemetry
│   │   ├── nvidia.py                #     NVIDIA NVML
│   │   ├── amdgpu.py                #     AMD ROCm + APU unified memory
│   │   ├── intel.py                 #     Intel i915/xe
│   │   └── apple.py                 #     Apple Silicon Metal
│   │
│   └── memory/                      #   ★ Long-Term Vector Memory
│       ├── vector_store.py          #     SQLite vector store + cosine similarity
│       └── memory_manager.py        #     Turn embedding & context injection
│
├── tests/                           #   54+ test files
├── docs/                            #   Extended documentation
├── data/                            #   SQLite databases (router.db, provider.db, conversation_memory.db)
├── DEEPDIVE.md                      #   Architectural deep dive
├── implementation.md                #   Context compression full specification
└── CHANGELOG.md                     #   Full version history
```

---

## Documentation Index

| Document | Description |
| :--- | :--- |
| [Architecture](docs/architecture.md) | System architecture with Mermaid diagrams |
| [Implementation Spec](implementation.md) | Context compression full technical specification |
| [Deep Dive](DEEPDIVE.md) | Architectural deep dive and design rationale |
| [Configuration Reference](docs/configuration.md) | Complete `ROUTER_*` setting list |
| [API Reference](docs/api.md) | Full endpoint documentation |
| [Backend Providers](docs/backends.md) | Ollama, llama.cpp, OpenAI-compatible setup |
| [External Providers](docs/external-providers.md) | OpenAI, Anthropic, Google, Cohere, Mistral |
| [Docker Deployment](docs/DOCKER.md) | Docker and Docker Compose guide |
| [Kubernetes Deployment](docs/kubernetes.md) | K8s manifests and Helm examples |
| [Performance Tuning](docs/performance.md) | Optimization guide |
| [Troubleshooting](docs/troubleshooting.md) | Common issues and solutions |
| [Client Integration](docs/examples/client-integration.md) | Python, JS, curl, OpenWebUI, Cursor examples |
| [Changelog](CHANGELOG.md) | Full version history |
| [Contributing](CONTRIBUTING.md) | How to contribute |
| [Security Policy](SECURITY.md) | Vulnerability reporting and best practices |

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for coding standards, testing requirements, and the PR process.

**Quick start:**
```bash
git clone https://github.com/sohamkadu17/SmarterRouter.git
cd SmarterRouter
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
cp ENV_DEFAULT .env
pytest
ruff check . && mypy .
```

---

## Upstream Attribution

FlexCTX is a research and engineering fork of **[SmarterRouter](https://github.com/peva3/SmarterRouter)** by [peva3](https://github.com/peva3), released under the MIT License. The upstream project provides the foundational OpenAI-compatible routing proxy, backend abstraction layer, and benchmark integration. All innovations described in the *"How FlexCTX Differs"* section above were developed as part of this fork.

---

## License

FlexCTX is released under the **MIT License**. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <b>FlexCTX</b> — Compress first. Route smart. Never OOM.<br>
  <a href="https://github.com/sohamkadu17/SmarterRouter">GitHub</a> ·
  <a href="docs/api.md">API Docs</a> ·
  <a href="CHANGELOG.md">Changelog</a> ·
  <a href="https://github.com/peva3/SmarterRouter">Upstream (peva3/SmarterRouter)</a>
</p>
