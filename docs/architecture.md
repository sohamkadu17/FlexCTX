# Architecture Overview

This document describes the SmarterRouter architecture, data flows, pre-flight context engineering, and execution layers.

## System Architecture

```mermaid
graph TB
    subgraph Client["Client Applications"]
        OWUI["OpenWebUI"]
        OPCODE["OpenCode IDE"]
        IDE["Cursor / Continue"]
        SDK["OpenAI SDKs (Py/JS)"]
        DEMO_UI["Built-in Web Demo Dashboard"]
    end

    subgraph Router["SmarterRouter Gateway"]
        subgraph API["API Layer"]
            DEMO_ROUTE[GET / & /demo]
            CHAT[/v1/chat/completions\]
            MODELS[/v1/models\]
            SKILLS[/v1/skills\]
            HEALTH[/health\]
            ADMIN[/admin/*\]
        end

        subgraph PreFlight["Pre-Flight Context Engineering Pipeline"]
            SCANNER["Language Density Scanner"]
            DCA["Dynamic Context Allocation (DCA)"]
            ARBITRAGE["Cross-Lingual Token Arbitrage (Micro-SLM)"]
            CHUNKER["AST & Code Chunker"]
            PRUNER["Statistical Lexical Pruner (BM25 + Entropy)"]
            FORCE_KEEP["Critical Syntax & Secret Force-Keep"]
            ALIGNER["Prefix Cache Aligner & APC"]
            CAR["Content-Addressable Recovery (CAR)"]
        end

        subgraph Memory["Long-Term Conversation Memory"]
            VEC_STORE[(SQLite Vector Store)]
            EMBED_ENGINE["Ollama / Fallback Deterministic Vectorizer"]
            RECALL["Context Recall & Formatter"]
        end

        subgraph Core["Core Engine & Resilience"]
            ROUTER_ENG["Intelligent Router Engine"]
            PROFILER["Automated Model Profiler"]
            JUDGE["LLM-as-Judge Evaluation"]
            CACHE["Multi-Tier Cache (Exact + Semantic)"]
            VRAM_MGR["VRAM Manager & Eviction Coordinator"]
            CIRCUIT["Circuit Breakers & Retries"]
            DLQ["Dead Letter Queue (DLQ)"]
            SECURITY["Security, Moderation & Audit"]
        end

        subgraph Data["Data & Storage Layer"]
            SQLITE[(SQLite router.db)]
            REDIS[(Redis Distributed Cache)]
            PROVIDER_DB[(provider.db - 400+ Models)]
            MEM_DB[(conversation_memory.db)]
        end

        subgraph Backends["Inference Backends (BackendRegistry)"]
            OLLAMA["Local Ollama"]
            LLAMACPP["Local llama.cpp"]
            OPENAI_COMPAT["OpenAI-Compatible Endpoints"]
            EXTERNAL["Cloud: OpenAI / Anthropic / Google / Cohere / Mistral"]
        end

        subgraph GPU["Cross-Vendor GPU / VRAM Subsystem"]
            NVIDIA["NVIDIA (nvidia-smi / NVML)"]
            AMD["AMD (rocm-smi + sysfs / APU GTT Pool)"]
            INTEL["Intel Arc (i915/xe sysfs)"]
            APPLE["Apple Silicon (Unified Memory)"]
        end
    end

    OWUI --> CHAT
    OPCODE --> CHAT
    IDE --> CHAT
    SDK --> CHAT
    DEMO_UI --> DEMO_ROUTE
    DEMO_ROUTE --> PreFlight

    CHAT --> SECURITY
    SECURITY --> PreFlight
    PreFlight --> Memory
    Memory --> ROUTER_ENG
    ROUTER_ENG --> CACHE
    ROUTER_ENG --> PROFILER
    ROUTER_ENG --> VRAM_MGR
    ROUTER_ENG --> CIRCUIT
    CIRCUIT --> Backends

    VRAM_MGR --> GPU
    PROFILER --> GPU
    PROFILER --> Backends
    JUDGE --> Backends

    CACHE --> REDIS
    CACHE --> SQLITE
    ROUTER_ENG --> SQLITE
    ROUTER_ENG --> PROVIDER_DB
    Memory --> MEM_DB

    ADMIN --> Core
    HEALTH --> Core

    style Router fill:#e1f5fe
    style PreFlight fill:#f3e5f5
    style Memory fill:#e0f2f1
    style Core fill:#fff3e0
    style Data fill:#e8f5e9
    style Backends fill:#fce4ec
    style GPU fill:#ede7f6
```

## Request Flow

```mermaid
sequenceDiagram
    participant Client as Client Application
    participant API as FastAPI / Middleware
    participant Security as Security & Rate Limiter
    participant Cache as Exact & Semantic Cache
    participant Pipeline as Context Compression Pipeline
    participant Memory as Conversation Vector Memory
    participant Router as Router Engine & VRAM Manager
    participant Backend as Selected LLM Backend

    Client->>API: POST /v1/chat/completions
    API->>Security: Enforce IP Whitelist, Rate Limits, Body Size
    Security->>Security: Check Prompt Injection & Content Moderation

    alt Exact / Semantic Cache Hit
        Security->>Cache: Lookup prompt hash & embedding similarity
        Cache-->>Client: Return cached response immediately
    else Cache Miss
        Security->>Pipeline: Pass payload to Pre-Flight Compression
        Pipeline->>Pipeline: 1. Non-ASCII density scan (<0.1ms)
        Pipeline->>Pipeline: 2. Dynamic Context Allocation (DCA limit)
        Pipeline->>Pipeline: 3. Cross-lingual arbitrage (if multilingual)
        Pipeline->>Pipeline: 4. AST code chunking & BM25/entropy pruning
        Pipeline->>Pipeline: 5. Force-keep rules (keys, paths, signatures)
        Pipeline->>Pipeline: 6. Cache prefix alignment & CAR handles

        Pipeline->>Memory: Query relevant historical turns
        Memory->>Memory: Semantic similarity search (SQLite vector store)
        Memory-->>Pipeline: Inject recalled conversation turns

        Pipeline->>Router: Forward optimized prompt & tool schemas
        Router->>Router: Analyze category, complexity & capabilities
        Router->>Router: Score models (benchmarks + profiles + feedback)
        Router->>Router: Check VRAM headroom & evict models if needed

        Router->>Backend: Dispatch inference request (with retry & circuit breaker)
        Backend-->>Router: Stream tokens or return complete generation

        Router->>Cache: Store decision & response in multi-tier cache
        Router->>Memory: Asynchronously record turn vector embedding
        Router-->>Client: Return OpenAI-compatible completion
    end
```

## Component Breakdown

### 1. API Layer

- **REST API**: High-concurrency FastAPI gateway supporting `/v1/chat/completions`, `/v1/models`, `/v1/embeddings`, `/v1/feedback`, and `/v1/skills`.
- **Interactive Web Demo UI**: Mounted at `GET /` and `GET /demo` providing real-time hardware telemetry and live before/after prompt compression inspections.
- **Admin Control Plane**: Protected `/admin/*` management suite for profiling, cache invalidation, DLQ retries, benchmark syncing, and audit logs.

### 2. Pre-Flight Dynamic Context Compression Pipeline

Operates before model dispatch to solve the "100:1 agentic payload problem":
- **Language Density Scanner (`scanner.py`)**: Sub-millisecond ASCII/Unicode ratio calculation. English prompts bypass translation SLMs entirely with zero latency.
- **Dynamic Context-Window Allocation (`dca.py`)**: Partitions token budgets with hysteresis dampening to prevent context thrashing.
- **Cross-Lingual Token Arbitrage (`arbitrage.py`)**: Compacts multilingual prompts via micro-SLMs into information-dense representations.
- **AST & Code Chunker (`chunker.py`)**: Lexical segmenter that respects programming language grammar, imports, and indentation.
- **Statistical Lexical Pruner (`statistical.py`)**: Composite scorer combining BM25, query overlap, position bias, entropy, and filler word filtering.
- **Critical Syntax & Secret Force-Keep (`force_keep.py`)**: Hard gate preserving API tokens, paths, URLs, and code signatures.
- **Prefix Cache Aligner & CAR (`cache_aligner.py`, `car.py`)**: Aligns static prompt prefixes and tool definitions for KV cache hit rates (RadixAttention/APC safe) with Content-Addressable Recovery handles.

### 3. Persistent Vector Long-Term Memory

- **`LocalVectorStore` (`router/memory/vector_store.py`)**: Embedded SQLite vector storage supporting cosine similarity matching and metadata filtering.
- **`ConversationMemoryManager` (`router/memory/memory_manager.py`)**: Ingests user and assistant turns, generates embeddings via Ollama or deterministic sub-millisecond hash vectorizer fallback, and formats relevant memory blocks into incoming prompt context.

### 4. Router Engine & VRAM Manager

- **Task Classification**: Analyzes prompts for task category (coding, reasoning, creativity, factual) and complexity.
- **Capability Detection**: Automatically identifies vision, function/tool calling, or embedding needs.
- **Multi-Factor Scoring**: Evaluates local runtime profiles, global benchmarks (provider.db), and historical user feedback tuned by `ROUTER_QUALITY_PREFERENCE`.
- **VRAM Headroom Allocator (`vram_manager.py`)**: Prevents out-of-memory errors by monitoring live GPU allocations and evicting inactive models (LRU or largest-first), respecting pinned models.

### 5. Backend Abstraction & Resilience

- **`BackendRegistry`**: Unified router dispatching across Ollama, llama.cpp, and external cloud APIs (OpenAI, Anthropic, Google Gemini, Cohere, Mistral).
- **Circuit Breaker Pattern (`circuit_breaker.py`)**: Fails fast during backend outages and probes recovery in half-open state.
- **Transient Retry Layer (`retry.py`)**: Exponential backoff for HTTP 429, 5xx, and network timeouts.
- **Dead Letter Queue (`dlq.py`)**: Persists failed asynchronous background tasks for inspection and manual/automatic retry.

### 6. GPU Monitoring Subsystem

Auto-detects active hardware on startup:
- **NVIDIA**: `nvidia-smi` and pyNVML per-GPU telemetry.
- **AMD / ROCm**: `rocm-smi` and sysfs memory tracking, including APU unified memory (GTT pool detection).
- **Intel**: sysfs monitoring for Arc / Xe GPUs (`lmem_total`).
- **Apple Silicon**: Unified memory detection (default: 75% of system RAM).

## Data Flow Detail

### Chat Completion Request

1. **Request Validation**
   - Parse JSON body, check request size limits (`ROUTER_MAX_REQUEST_BODY_BYTES`, `ROUTER_MAX_MESSAGE_CONTENT_LENGTH`).
   - Validate model alias and rate limits.

2. **Security & Moderation**
   - Prompt injection analysis (`log`, `warn`, or `block`).
   - Content moderation category screening.

3. **Multi-Tier Cache Inspection**
   - Check exact SHA-256 routing and response cache.
   - Check semantic similarity cache (cosine similarity via numpy).

4. **Pre-Flight Context Compression**
   - Run Language Scanner, DCA budget allocation, AST chunker, statistical pruner, and prefix cache alignment.

5. **Conversation Memory Retrieval**
   - Query `conversation_memory.db` for semantically relevant historical turns and inject context.

6. **Model Selection & VRAM Check**
   - Query model profiles and benchmarks.
   - Calculate capability, speed, and quality scores.
   - Verify VRAM budget and trigger automated model eviction if over threshold.

7. **Backend Dispatch & Streaming**
   - Dispatch through Circuit Breaker and Retry middleware.
   - Stream SSE chunks or return full JSON response.

8. **Post-Processing & Asynchronous Storage**
   - Append model signature (`ROUTER_SIGNATURE_ENABLED`).
   - Store response in cache.
   - Record turn in persistent vector memory.
   - Record admin audit trail.


### Background Tasks

```mermaid
graph LR
    subgraph Tasks["Background Tasks"]
        SYNC[Benchmark Sync]
        POLL[Model Polling]
        CACHE_CLEAN[Cache Cleanup]
        DLQ[Dead Letter Queue]
    end

    SYNC -->|Every 4h| PROVIDER[(Provider DB)]
    POLL -->|Every 5m| OLLAMA[Ollama Backend]
    CACHE_CLEAN -->|Hourly| SQLITE[(SQLite)]
    DLQ -->|Retry| SYNC

    style Tasks fill:#fff3e0
```

## Configuration Flow

```mermaid
graph TD
    ENV[Environment Variables] -->|ROUTER_*| CONFIG[config.py]
    CONFIG --> SETTINGS[Settings Object]
    SETTINGS -->|Injected| APP[FastAPI App]
    SETTINGS -->|Injected| ROUTER[Router Engine]
    SETTINGS -->|Injected| CACHE[Cache]
    SETTINGS -->|Injected| BACKENDS[Backends]

    style ENV fill:#e8f5e9
    style SETTINGS fill:#e1f5fe
```

## Scoring Algorithm

```mermaid
graph TD
    PROMPT[Prompt Analysis] -->|1.0| COMPLEXITY{Complexity Score}
    PROMPT -->|0.2| CATEGORY{Category}

    COMPLEXITY -->|Penalties| SIZE[Size Penalties]
    COMPLEXITY -->|Bonuses| SIZEB[Size Bonuses]
    CATEGORY -->|Boost| CATB[Category Boost]

    BENCHMARKS[Benchmarks] -->|0.5| WEIGHTED[Weighted Score]
    PROFILES[Profiles] -->|0.3| WEIGHTED
    FEEDBACK[User Feedback] -->|0.2| WEIGHTED

    WEIGHTED --> COMBINED[Combined Score]
    SIZE --> COMBINED
    SIZEB --> COMBINED
    CATB --> COMBINED

    COMBINED --> PROVIDER[Provider Bonus]
    PROVIDER --> DIVERSITY[Diversity Penalty]
    DIVERSITY --> FINAL[Final Score]

    FINAL --> SELECT[Select Best Model]

    style FINAL fill:#fff3e0
    style SELECT fill:#e8f5e9
```

## Scalability Considerations

### Horizontal Scaling

- Stateless design enables multiple instances
- Redis backend for distributed caching
- Database connection pooling
- Load balancer distributes requests

### Performance Optimizations

- Async/await throughout
- Connection pooling for HTTP clients
- Batched database operations
- Vectorized similarity calculations (numpy)
- LRU caching for profiles and benchmarks

### Resource Management

- Automatic VRAM monitoring
- Circuit breakers prevent cascading failures
- Rate limiting protects resources
- Background tasks don't block requests

## Security Architecture

```mermaid
graph TB
    subgraph Security["Security Layers"]
        IP[IP Whitelist]
        RATE[Rate Limiting]
        AUTH[API Key Auth]
        INPUT[Input Validation]
        PROMPT[Prompt Injection]
        CONTENT[Content Moderation]
    end

    CLIENT[Client] --> IP
    IP --> RATE
    RATE --> AUTH
    AUTH --> INPUT
    INPUT --> PROMPT
    PROMPT --> CONTENT
    CONTENT --> APP[Application]

    style Security fill:#ffebee
```

## Deployment Options

### Docker Compose (Single Node)

```mermaid
graph LR
    subgraph Docker["Docker Compose"]
        ROUTER[SmarterRouter]
        OLLAMA[Ollama]
        REDIS[Redis]
    end

    CLIENT --> ROUTER
    ROUTER --> OLLAMA
    ROUTER --> REDIS
```

### Kubernetes (Production)

```mermaid
graph TB
    subgraph K8s["Kubernetes Cluster"]
        INGRESS[Ingress]
        ROUTER1[SmarterRouter Pod 1]
        ROUTER2[SmarterRouter Pod 2]
        ROUTER3[SmarterRouter Pod 3]
        SVC[Service]
        PVC[PersistentVolume]
    end

    CLIENT --> INGRESS
    INGRESS --> SVC
    SVC --> ROUTER1
    SVC --> ROUTER2
    SVC --> ROUTER3
    ROUTER1 --> PVC
    ROUTER2 --> PVC
    ROUTER3 --> PVC
```

See [kubernetes.md](kubernetes.md) for detailed Kubernetes deployment instructions.

## Monitoring & Observability

### Metrics

- Request rate, latency, errors (RED metrics)
- Cache hit/miss rates
- GPU VRAM utilization
- Model selection distribution
- Backend health status

### Logging

- Structured JSON logging
- Request correlation IDs
- Sanitized user input
- Error context enrichment

### Health Checks

```
GET /health
```

Returns:
- Database connectivity
- Backend status
- GPU status
- Cache status
- Background task counts
- DLQ counts (if enabled)
