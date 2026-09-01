"""Master Context Compression Pipeline orchestrating DCA, Statistical Pruning, Arbitrage, and Prefix Caching."""

import logging
import time
from typing import Any, NamedTuple

from router.compression.arbitrage import CrossLingualArbitrage
from router.compression.cache_aligner import CacheAligner
from router.compression.car import ContentAddressableRecovery
from router.compression.dca import DynamicContextAllocator, QueryCategory
from router.compression.statistical import StatisticalPruner

logger = logging.getLogger(__name__)


class CompressionResult(NamedTuple):
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None
    original_tokens: int
    compressed_tokens: int
    token_savings_pct: float
    latency_ms: float
    category: str
    arbitrage_triggered: bool


class ContextCompressionPipeline:
    """End-to-end Pre-Flight Context Engineering Pipeline for SmarterRouter."""

    def __init__(
        self,
        enabled: bool = True,
        mode: str = "full",  # "full", "statistical_only", "cache_align_only"
        target_context_limit: int = 8192,
        reserve_generation_tokens: int = 2048,
        arbitrage_slm_model: str = "qwen2.5:3b",
        multilingual_threshold: float = 0.15,
        bm25_weight: float = 0.35,
        overlap_weight: float = 0.25,
        position_weight: float = 0.15,
        entropy_weight: float = 0.15,
        inv_filler_weight: float = 0.10,
        jaccard_threshold: float = 0.85,
    ):
        self.enabled = enabled
        self.mode = mode

        self.dca = DynamicContextAllocator(
            default_target_limit=target_context_limit,
            default_generation_reserve=reserve_generation_tokens,
        )
        self.arbitrage = CrossLingualArbitrage(
            slm_model=arbitrage_slm_model,
            multilingual_threshold=multilingual_threshold,
        )
        self.pruner = StatisticalPruner(
            bm25_weight=bm25_weight,
            overlap_weight=overlap_weight,
            position_weight=position_weight,
            entropy_weight=entropy_weight,
            inv_filler_weight=inv_filler_weight,
            jaccard_threshold=jaccard_threshold,
        )
        self.cache_aligner = CacheAligner()
        self.car = ContentAddressableRecovery()

        # Cumulative Metrics
        self.total_requests = 0
        self.total_original_tokens = 0
        self.total_compressed_tokens = 0
        self.total_latency_ms = 0.0

    async def process_chat_payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        backend: Any = None,
        vram_monitor: Any = None,
    ) -> CompressionResult:
        """Execute the complete pre-flight context engineering and compression pipeline.

        Args:
            messages: List of OpenAI-compatible message objects.
            tools: Optional list of tool schema definitions.
            backend: Optional backend for executing the lightweight SLM rewriter if needed.
            vram_monitor: Optional live VRAMMonitor instance for real-time hardware telemetry.

        Returns:
            CompressionResult containing optimized messages and quantitative metrics.
        """
        start_time = time.perf_counter()

        if not self.enabled or not messages:
            est_tokens = sum(max(1, int(len(str(m.get("content", ""))) / 3.8)) for m in messages)
            return CompressionResult(
                messages=messages,
                tools=tools,
                original_tokens=est_tokens,
                compressed_tokens=est_tokens,
                token_savings_pct=0.0,
                latency_ms=0.0,
                category="passthrough",
                arbitrage_triggered=False,
            )

        # 1. Step 1: Prefix Cache Alignment (Canonical sorting & system prompt normalization)
        aligned_messages, sorted_tools = self.cache_aligner.format_cache_friendly_payload(messages, tools)

        # Separate system messages, history, and active user query
        system_msgs = [m for m in aligned_messages if m.get("role") == "system"]
        history_msgs = [m for m in aligned_messages if m.get("role") in ("user", "assistant")][:-1]
        active_user_msg = aligned_messages[-1] if aligned_messages else {"role": "user", "content": ""}

        user_prompt_raw = active_user_msg.get("content", "")
        if not isinstance(user_prompt_raw, str):
            user_prompt_raw = str(user_prompt_raw)

        # Estimate original tokens
        orig_tokens = sum(max(1, int(len(str(m.get("content", ""))) / 3.8)) for m in aligned_messages)

        # 2. Step 2: Dynamic Context-Window Allocation (DCA) coupled with Live VRAM Telemetry
        free_vram = None
        total_vram = None
        if vram_monitor and hasattr(vram_monitor, "get_current"):
            metrics = vram_monitor.get_current()
            if metrics:
                free_vram = metrics.free_gb
                total_vram = metrics.total_gb

        dynamic_limit = self.dca.calculate_dynamic_limit_from_vram(free_vram, total_vram)
        category = self.dca.classify_query(user_prompt_raw, history_len=len(history_msgs))
        budget = self.dca.allocate_budget(category, total_limit=dynamic_limit)

        # 3. Step 3: Selective Cross-Lingual Arbitrage (SLM Rewriter for multilingual queries)
        arbitrage_triggered = False
        optimized_prompt = user_prompt_raw

        if self.mode in ("full", "arbitrage_only"):
            is_code = (category == QueryCategory.CODE_DEVELOPMENT)
            optimized_prompt, arbitrage_triggered = await self.arbitrage.rewrite_if_needed(
                user_prompt_raw,
                backend=backend,
                is_code_task=is_code,
            )

        # 4. Step 4: AST & Lexical In-Process Statistical Pruner (LeanCTX)
        final_history_msgs = []
        if self.mode in ("full", "statistical_only"):
            # Prune user prompt / embedded RAG text if exceeding RAG budget
            if len(optimized_prompt) > 800:
                pruned_user_text, _, _ = self.pruner.prune_context(
                    context_text=optimized_prompt,
                    query=user_prompt_raw[:200],
                    target_token_budget=budget.rag_tokens,
                )
                optimized_prompt = pruned_user_text

            # Prune conversation history to fit history budget
            history_budget_remaining = budget.history_tokens
            for msg in reversed(history_msgs):
                content = str(msg.get("content", ""))
                msg_tokens = max(1, int(len(content) / 3.8))
                if history_budget_remaining >= msg_tokens:
                    final_history_msgs.insert(0, msg)
                    history_budget_remaining -= msg_tokens
                elif history_budget_remaining > 50:
                    pruned_content, _, pruned_toks = self.pruner.prune_context(
                        context_text=content,
                        query=user_prompt_raw[:200],
                        target_token_budget=history_budget_remaining,
                    )
                    final_history_msgs.insert(0, {**msg, "content": pruned_content})
                    history_budget_remaining = 0
                    break
        else:
            final_history_msgs = history_msgs

        # 5. Assemble final compressed message array
        final_messages = []
        final_messages.extend(system_msgs)
        final_messages.extend(final_history_msgs)
        final_messages.append({**active_user_msg, "content": optimized_prompt})

        comp_tokens = sum(max(1, int(len(str(m.get("content", ""))) / 3.8)) for m in final_messages)
        savings_pct = max(0.0, ((orig_tokens - comp_tokens) / orig_tokens) * 100) if orig_tokens > 0 else 0.0

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Update cumulative tracking
        self.total_requests += 1
        self.total_original_tokens += orig_tokens
        self.total_compressed_tokens += comp_tokens
        self.total_latency_ms += latency_ms

        logger.info(
            f"Context compression complete: {orig_tokens} -> {comp_tokens} tokens "
            f"(-{savings_pct:.1f}%) in {latency_ms:.2f}ms [Category: {category.value}]"
        )

        return CompressionResult(
            messages=final_messages,
            tools=sorted_tools,
            original_tokens=orig_tokens,
            compressed_tokens=comp_tokens,
            token_savings_pct=savings_pct,
            latency_ms=latency_ms,
            category=category.value,
            arbitrage_triggered=arbitrage_triggered,
        )

    def get_metrics_summary(self) -> dict[str, Any]:
        """Return cumulative compression performance metrics."""
        avg_savings = 0.0
        if self.total_original_tokens > 0:
            avg_savings = (
                (self.total_original_tokens - self.total_compressed_tokens) / self.total_original_tokens
            ) * 100

        avg_latency = (
            self.total_latency_ms / self.total_requests if self.total_requests > 0 else 0.0
        )

        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "total_requests_processed": self.total_requests,
            "total_original_tokens": self.total_original_tokens,
            "total_compressed_tokens": self.total_compressed_tokens,
            "total_tokens_saved": max(0, self.total_original_tokens - self.total_compressed_tokens),
            "average_token_savings_pct": round(avg_savings, 2),
            "average_latency_ms": round(avg_latency, 2),
        }
