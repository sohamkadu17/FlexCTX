# Dynamic Context-Window Allocation & Semantic Compression Middleware
## Implementation Plan & Architecture Specification for SmarterRouter

---

## 1. Executive Summary & Problem Context

### 1.1 The Memory Wall & KV Cache Pathology
In local LLM inference engines (such as `Ollama`, `llama.cpp`, and `vLLM`), **VRAM is the ultimate physical constraint**. While model weights occupy static memory, the **Key-Value (KV) Cache** scales dynamically with context length ($L_{\text{context}}$) and batch size:

$$\text{Memory}_{\text{KV}} = 2 \times N_{\text{layers}} \times N_{\text{kv\_heads}} \times D_{\text{head}} \times L_{\text{context}} \times B_{\text{bytes}}$$

For an 8B–70B model with Grouped-Query Attention (GQA), running large context windows ($32\text{K} - 128\text{K}$ tokens) consumes $10\text{ GB} - 43\text{ GB}$ of VRAM solely for the KV Cache. On consumer and workstation GPUs (e.g., RTX 4050 6GB, RTX 4090 24GB, or RTX 6000 48GB), this leads directly to:
1. Out-of-Memory (OOM) crashes.
2. Layer offloading onto CPU/System RAM (reducing generation speed by $5\times - 10\times$).
3. Extreme Time-To-First-Token (TTFT) latency due to expensive quadratic/linear prefill passes.

### 1.2 The 100:1 Agentic Payload Problem
Multi-turn agent loops (e.g., ReAct, code reasoning, tool use) suffer from an asymmetric **100:1 to 267:1 input-to-output token ratio**:
- **System Instructions & Personas**: $2\text{K} - 10\text{K}$ tokens.
- **Repeated Tool Schemas (JSON/JSON-Schema)**: $5\text{K} - 50\text{K}$ tokens.
- **Growing Trajectory History**: $1\text{K} - 20\text{K}$ tokens.
- **RAG Documents & Code Workspace Chunks**: $5\text{K} - 100\text{K}$ tokens.
- **Generated Output**: Often only $50 - 300$ tokens per step.

### 1.3 The Solution: Edge Pre-Flight Middleware
Integrating a **Dynamic Context-Window Allocation (DCA) and Semantic Compression Middleware** directly into **SmarterRouter** transforms it into an edge-side **AI Value Gate**. Before dispatching prompts to local backends or cloud endpoints, the middleware:
1. **Arbitrages Multilingual Tokens**: Converts non-English inputs into token-dense English structured blocks via a lightweight local SLM (e.g., `qwen2.5:3b` or `llama3.2:3b`).
2. **Deterministically Prunes Context**: Employs an in-process, CPU-only lexical scorer (BM25 + Entropy + Position + Overlap) to compress RAG and code chunks in $<1\text{ms}$ with zero GPU overhead.
3. **Aligns Prefix Caches**: Enforces deterministic, byte-exact prefix structures and alphabetically sorted tool schemas to achieve $>90\%$ RadixAttention / APC cache hit rates.
4. **Dynamically Partitions Budgets**: Adjusts context limits between system prompt, chat history, and RAG based on query classification.
5. **Guarantees Execution Safety**: Protects authentication tokens and critical code syntax via regex force-keep rules and Content-Addressable Recovery (CAR).

---

## 2. Integrated System Architecture

```
                                  INCOMING REQUEST (OpenAI API Format)
                                                   │
                                                   ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   SMARTERROUTER FASTAPI GATEWAY                                 │
│                                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │                    PRE-FLIGHT CONTEXT ENGINEERING PIPELINE (router/compression/)            │  │
│  │                                                                                            │  │
│  │  1. Dynamic Context-Window Allocator (DCA)                                                  │  │
│  │     • Query Classifier (RAG-heavy vs Conversational vs Code Task)                           │  │
│  │     • Token Budget Partitioning: L_limit = T_sys + T_hist + T_rag + T_gen                  │  │
│  │                                                                                            │  │
│  │  2. Cross-Lingual Token Arbitrage Engine (SLM Rewriter)                                    │  │
│  │     • Detects Non-English / High-Subword Languages (Turkish, Spanish, Chinese, Arabic)     │  │
│  │     • Local SLM (Llama 3.2 3B / Qwen 2.5 3B) re-formats to Bi-Block / Tri-Block           │  │
│  │     • validate_light() Gate: Regex check & ≥5% Token Reduction Guarantee                   │  │
│  │                                                                                            │  │
│  │  3. Deterministic Statistical & Lexical Pruner (LeanCTX Engine)                            │  │
│  │     • In-process Sentence Tokenizer & Term Indexer (<1ms, CPU only)                        │  │
│  │     • Composite Scorer: w1*BM25 + w2*Overlap + w3*Position + w4*Entropy + w5*InvFiller     │  │
│  │     • Jaccard Deduplication + Regex Force-Keep Gate (Keys, Paths, Signatures)              │  │
│  │     • Budget-bounded Chronological Span Reassembler                                        │  │
│  │                                                                                            │  │
│  │  4. Cache-Aligned Assembly & Content-Addressable Recovery (CAR)                            │  │
│  │     • Byte-exact System Prompt Formatting & Alphabetically Sorted Tool Schemas             │  │
│  │     • SHA-256 Reversible Reference Handles ([Ref: <hash>]) for Pruned Data                 │  │
│  │                                                                                            │  │
│  └─────────────────────────────────────────────┬──────────────────────────────────────────────┘  │
│                                                │ (Optimized & Compressed Payload)                │
│                                                ▼                                                 │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │                      CORE ROUTING & VRAM MANAGEMENT ENGINE                                 │  │
│  │                                                                                            │  │
│  │  • RouterEngine: Hardware Profile Scoring & Benchmark Matching                             │  │
│  │  • VRAMManager: Dynamic VRAM Budget Checking & Layer Offload Coordination                  │  │
│  │  • Backend Handlers: Ollama / llama.cpp / OpenAI / Anthropic                               │  │
│  └─────────────────────────────────────────────┬──────────────────────────────────────────────┘  │
└────────────────────────────────────────────────┼─────────────────────────────────────────────────┘
                                                 ▼
                                   DOWNSTREAM INFERENCE BACKENDS
```

---

## 3. Core Technical Pillars & Implementation Details

### Pillar 1: Cross-Lingual Token Arbitrage & Structural Rewriter

#### 1. Mathematical Foundation
Non-English tokenization causes substantial subword bloat in standard tokenizers (`cl100k_base`, `o200k_base`, `llama3`):
- Spanish: $1.50\times$ token multiplier
- Turkish: $2.16\times$ token multiplier
- Simplified Chinese: $2.41\times$ token multiplier
- Arabic: $3.00\times$ token multiplier

Translating non-English input to structured English via a local 3B model compresses prompt tokens by $30\% - 65\%$ before hitting the primary model.

#### 2. Bi-Block & Tri-Block Formats
- **Bi-Block (Code & Automated Tasks)**:
  ```markdown
  [CONTEXT]
  Python coding task; tests are authoritative.
  
  [TASK]
  <Translated & distilled instruction with function signature & types>
  assert <Verbatim assert lines>
  ```
- **Tri-Block (Interactive IDE / General Tasks)**:
  ```markdown
  [CONTEXT]
  Repository workspace environment. Module: user_auth.
  
  [TASK]
  Rewrite the session validator to accept JWT tokens.
  
  [CONSTRAINTS]
  - Preserve existing DB Connection interface.
  - Execution time: <50ms.
  - Enforce camelCase naming.
  ```

#### 3. Validation & Fallback Gate (`validate_light`)
A strict programmatic gate verifies:
1. Output length $> 10$ chars.
2. No code leakage (` ```python ` or `<solution>` tags).
3. Required block markers (`[CONTEXT]`, `[TASK]`, etc.) are present.
4. **5% Token-Budget Fallback**: The rewritten prompt must achieve $\le 95\%$ of original estimated token count. If it fails after 2 repair retries, the router falls back safely to the original raw prompt.

---

### Pillar 2: Deterministic Statistical & Lexical Pruner (LeanCTX-Style)

For large RAG documents, code files, and conversation logs, generative SLM rewriting is too slow. The statistical pruner operates **in-process on CPU in $<1\text{ms}$** with byte-level reproducibility.

#### 1. Composite Scoring Equation
For each sentence $s_i$ against query $q$:

$$\text{Score}(s_i, q) = w_1 \cdot \text{BM25}(s_i, q) + w_2 \cdot \text{Overlap}(s_i, q) + w_3 \cdot \text{Position}(s_i) + w_4 \cdot \text{Entropy}(s_i) + w_5 \cdot \text{InvFiller}(s_i)$$

- **BM25 Relevance ($w_1 = 0.35$)**: Term-frequency inverse-document-frequency match.
- **Query Overlap ($w_2 = 0.25$)**: Intersection of content keywords.
- **Position Decay ($w_3 = 0.15$)**: Gives slight bias to head and tail of documents.
- **Token Entropy ($w_4 = 0.15$)**: $\text{Entropy}(s_i) = -\sum P(x)\log_2 P(x)$. Penalizes low-entropy repetitive boilerplate and logs.
- **Inverse Filler Score ($w_5 = 0.10$)**: Penalizes conversational fluff ("as mentioned before", "basically").

#### 2. Force-Keep Rules (F1 Failure Prevention)
Regex rules protect essential syntax from ever being pruned:
- Auth keys / secrets (`bearer`, `token`, `api_key`, `jwt`, `password`)
- System paths (`/`, `C:\`, URLs, file extensions `.py`, `.json`, `.ts`)
- Code signatures (`def `, `class `, `function `, `interface `, `import `)
- Explicit constraints (`MUST`, `NEVER`, `REQUIRED`)

#### 3. Deduplication & Span Selection
- **Jaccard Near-Duplicate Eviction**: Sentences with $>0.85$ Jaccard token similarity with an existing higher-scored sentence are dropped.
- **Budget Packing**: Highest-scoring and force-kept spans are packed into the allotted budget and sorted back into **chronological order**.

---

### Pillar 3: Dynamic Context-Window Allocation (DCA) & Cache Alignment

#### 1. Dynamic Token Budget Partitioning
The total allowed context limit $L_{\text{limit}}$ is dynamically partitioned:

$$L_{\text{limit}} = T_{\text{system}} + T_{\text{history}} + T_{\text{RAG}} + T_{\text{generation}}$$

- **Strategy A (RAG / Knowledge Heavy)**: Allocated $70\%$ to $T_{\text{RAG}}$, $15\%$ to $T_{\text{history}}$ (sliding window of last 3 turns), $15\%$ to $T_{\text{generation}}$.
- **Strategy B (Multi-turn Reasoning / Planning)**: Allocates $55\%$ to $T_{\text{history}}$ (with asynchronous JSON state-summarization for older turns), $25\%$ to $T_{\text{generation}}$, $20\%$ to $T_{\text{RAG}}$.

#### 2. Prefix Cache Alignment (RadixAttention & APC)
To maximize GPU prefix cache hits ($>90\%$ hit rate):
1. **Deterministic Static Prefix**: System instructions are formatted identically byte-for-byte.
2. **Alphabetically Sorted Tool Schemas**: Tool definitions are sorted alphabetically by function name.
3. **Left-to-Right Ordering**: Fixed static components always precede dynamic history and RAG context.

#### 3. Content-Addressable Recovery (CAR) (F3 Failure Prevention)
Pruned or summarized context chunks are stored in an in-memory / SQLite reversible index with a 16-character SHA-256 handle:
`[Ref: c3a492e8f1b20491]`
If the downstream model needs missing details, a lightweight helper function/tool (`ctx_expand(ref_id)`) retrieves the original raw chunk.

---

## 4. Module & Directory Structure

```
SmarterRouter/
├── router/
│   ├── compression/                    # [NEW] Compression & Context Engineering Package
│   │   ├── __init__.py                 # Package exports & pipeline entrypoint
│   │   ├── pipeline.py                 # Master ContextPipeline orchestrating DCA + Pruner + Arbitrage
│   │   ├── dca.py                      # Dynamic Context-Window Allocator & Query Classifier
│   │   ├── arbitrage.py                # Cross-Lingual Token Arbitrage & SLM Bi/Tri-Block Rewriter
│   │   ├── statistical.py              # In-process LeanCTX BM25 + Entropy + Jaccard Pruner
│   │   ├── force_keep.py               # Regex rule engine protecting credentials, paths & syntax
│   │   ├── cache_aligner.py            # RadixAttention / APC prefix-cache deterministic aligner
│   │   └── car.py                      # Content-Addressable Recovery index & reference manager
│   │
│   ├── api/
│   │   ├── chat.py                     # [MODIFY] Hook pre-flight compression into /v1/chat/completions
│   │   └── admin.py                    # [MODIFY] Add compression metrics & analytics endpoints
│   │
│   ├── config.py                       # [MODIFY] Add compression settings & threshold parameters
│   ├── state.py                        # [MODIFY] Store compression metrics and CAR index in app_state
│   └── models.py                       # [MODIFY] Add compression audit fields to DB logs
│
├── tests/
│   ├── test_compression_dca.py         # [NEW] Tests for DCA query classification & budget allocation
│   ├── test_compression_statistical.py # [NEW] Tests for BM25, entropy scoring & force-keep rules
│   ├── test_compression_arbitrage.py   # [NEW] Tests for multilingual translation & validate_light
│   └── test_compression_pipeline.py    # [NEW] End-to-end integration tests with chat completions
```

---

## 5. Configuration Settings (`.env` & `router/config.py`)

```env
# =============================================================================
# CONTEXT COMPRESSION & DCA SETTINGS
# =============================================================================

# Enable pre-flight context engineering pipeline
ROUTER_COMPRESSION_ENABLED=true

# Compression modes: "full", "statistical_only", "arbitrage_only", "cache_align_only"
ROUTER_COMPRESSION_MODE=full

# DCA Dynamic Budget Allocator
ROUTER_DCA_ENABLED=true
ROUTER_DCA_TARGET_CONTEXT_LIMIT=16384
ROUTER_DCA_RESERVE_GENERATION_TOKENS=2048

# Cross-Lingual Arbitrage (SLM Rewriter)
ROUTER_ARBITRAGE_ENABLED=true
ROUTER_ARBITRAGE_MODEL=qwen2.5:3b
ROUTER_ARBITRAGE_MAX_INPUT_CHARS=4000
ROUTER_ARBITRAGE_MIN_COMPRESSION_DELTA=0.05  # Revert if not >= 5% reduction

# Statistical Lexical Pruner (LeanCTX Engine)
ROUTER_PRUNER_ENABLED=true
ROUTER_PRUNER_BM25_WEIGHT=0.35
ROUTER_PRUNER_OVERLAP_WEIGHT=0.25
ROUTER_PRUNER_POSITION_WEIGHT=0.15
ROUTER_PRUNER_ENTROPY_WEIGHT=0.15
ROUTER_PRUNER_INV_FILLER_WEIGHT=0.10
ROUTER_PRUNER_JACCARD_THRESHOLD=0.85

# Prefix Cache Alignment
ROUTER_CACHE_ALIGN_SORT_TOOLS=true
ROUTER_CACHE_ALIGN_DETERMINISTIC_SYSTEM=true

# Content-Addressable Recovery (CAR)
ROUTER_CAR_ENABLED=true
ROUTER_CAR_MAX_ENTRIES=5000
ROUTER_CAR_TTL_SECONDS=86400
```

---

## 6. Phased Implementation Roadmap

| Phase | Focus Area | Deliverables | Verification Milestone |
| :--- | :--- | :--- | :--- |
| **Phase 1** | **Foundation & Statistical Engine** | 1. Implement `router/compression/statistical.py`<br>2. Implement `router/compression/force_keep.py`<br>3. Unit tests for BM25, Entropy, and Jaccard deduplication | Statistical pruner executes in $<2\text{ms}$ on 50KB text with 100% force-keep retention. |
| **Phase 2** | **Cache Alignment & DCA Allocation** | 1. Implement `router/compression/cache_aligner.py`<br>2. Implement `router/compression/dca.py`<br>3. Alphabetical tool schema sorting & query classifier | Static prefixes produce identical byte hashes; budgets partition correctly. |
| **Phase 3** | **Cross-Lingual Arbitrage & SLM Rewriter** | 1. Implement `router/compression/arbitrage.py`<br>2. Bi-Block & Tri-Block prompt builders<br>3. `validate_light()` gate with 5% delta budget fallback | Multilingual prompts (Spanish, Turkish, Chinese) compress by $\ge 30\%$ with zero syntax error. |
| **Phase 4** | **CAR & Gateway Pipeline Integration** | 1. Implement `router/compression/car.py`<br>2. Build `router/compression/pipeline.py`<br>3. Integrate with `router/api/chat.py` and `router/config.py` | Full `/v1/chat/completions` requests automatically compress and route seamlessly. |
| **Phase 5** | **Benchmarking & Validation Suite** | 1. Implement automated test suites in `tests/test_compression_*`<br>2. Add `/admin/compression/stats` metrics endpoint<br>3. Measure TTFT and token savings | End-to-end compression benchmarks verify $>40\%$ token savings and $2\times$ TTFT improvement. |

---

## 7. Verification & Benchmark Strategy

### 7.1 Automated Unit & Integration Tests
- **`test_compression_statistical.py`**:
  - Test BM25 relevance against synthetic queries.
  - Verify that low-entropy repetitive strings (e.g. repeated logs `===---===`) receive lower scores.
  - Verify that lines containing `Bearer <token>`, file paths, and `def main()` are never pruned (`force_keep=True`).
- **`test_compression_arbitrage.py`**:
  - Test `validate_light` against malformed rewrites and hallucinations.
  - Test the 5% fallback mechanism when rewrite fails to shrink the prompt.
- **`test_compression_dca.py`**:
  - Verify that RAG-heavy queries allocate $\ge 60\%$ context to documents.
  - Verify that conversational queries preserve recent history and trigger state summarization.

### 7.2 Performance & Quality Benchmarking Metrics
1. **Token Compression Ratio**: $\text{CR} = 1 - \frac{\text{Tokens}_{\text{compressed}}}{\text{Tokens}_{\text{original}}}$ (Target: $35\% - 60\%$).
2. **Execution Overhead**: Statistical pruning latency $<2\text{ms}$; full pipeline overhead $<20\text{ms}$.
3. **Prefix Cache Hit Rate**: Measure vLLM / SGLang / llama.cpp prompt cache hit rate (Target: $>85\%$).
4. **Task Accuracy Retention**: Zero regression on functional tests (e.g. HumanEval / Python assertions).
