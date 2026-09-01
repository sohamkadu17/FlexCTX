"""Dynamic Context-Window Allocation (DCA) and Query Complexity Classifier."""

import re
from enum import Enum
from typing import NamedTuple


class QueryCategory(str, Enum):
    RAG_KNOWLEDGE = "rag_knowledge"
    CONVERSATIONAL_REASONING = "conversational_reasoning"
    CODE_DEVELOPMENT = "code_development"
    GENERAL = "general"


class TokenBudget(NamedTuple):
    target_limit: int
    system_tokens: int
    history_tokens: int
    rag_tokens: int
    generation_tokens: int


class DynamicContextAllocator:
    """Dynamic Context-Window Allocation (DCA) partitioner with stateful Hysteresis Margins."""

    def __init__(
        self,
        default_target_limit: int = 8192,
        default_generation_reserve: int = 2048,
    ):
        self.target_limit = default_target_limit
        self.generation_reserve = default_generation_reserve
        self._current_bucket: int = 4096  # Conservative baseline bucket

    def calculate_dynamic_limit_from_vram(
        self,
        free_vram_gb: float | None,
        total_vram_gb: float | None = None,
    ) -> int:
        """Dynamically compute the optimal Nerf/Berf token context limit from real-time GPU VRAM headroom.

        Features:
        1. NVML Fault Tolerance: Gracefully falls back to conservative bucket (e.g. 4096) on driver failure/timeout.
        2. Hysteresis Margins: Asymmetric thresholds (step-up vs step-down delta buffer) prevent rapid oscillation.
        """
        # 1. Driver Fault Tolerance Fallback
        if free_vram_gb is None or not isinstance(free_vram_gb, (int, float)) or free_vram_gb <= 0:
            return self._current_bucket or 4096

        curr = self._current_bucket

        # 2. Stateful Hysteresis Boundary Transition Engine
        if curr == 2048:
            # Step up to 4k only if headroom exceeds 1.4 GB
            if free_vram_gb >= 1.4:
                curr = 4096
        elif curr == 4096:
            # Step down to 2k if free VRAM drops below 1.1 GB
            if free_vram_gb < 1.1:
                curr = 2048
            # Step up to 8k requires 2.7 GB free buffer
            elif free_vram_gb >= 2.7:
                curr = 8192
        elif curr == 8192:
            # Drop back to 4k if free VRAM dips below 2.3 GB
            if free_vram_gb < 2.3:
                curr = 4096
            # Step up to 16k requires 4.8 GB free buffer
            elif free_vram_gb >= 4.8:
                curr = 16384
        elif curr == 16384:
            # Drop back to 8k if free VRAM dips below 4.2 GB
            if free_vram_gb < 4.2:
                curr = 8192
            # Step up to 32k requires 8.5 GB free buffer
            elif free_vram_gb >= 8.5:
                curr = 32768
        elif curr == 32768:
            # Drop back to 16k if free VRAM dips below 7.5 GB
            if free_vram_gb < 7.5:
                curr = 16384
        else:
            # Fallback initialization based on absolute values
            if free_vram_gb < 1.2:
                curr = 2048
            elif free_vram_gb < 2.5:
                curr = 4096
            elif free_vram_gb < 4.5:
                curr = 8192
            elif free_vram_gb < 8.0:
                curr = 16384
            else:
                curr = 32768

        self._current_bucket = curr
        return curr

    def classify_query(self, prompt: str, history_len: int = 0) -> QueryCategory:
        """Classify incoming query complexity to determine the optimal budget partitioning strategy."""
        prompt_lower = prompt.lower()

        # Check coding markers
        code_markers = [
            "def ", "class ", "function", "import ", "const ", "var ",
            "```", "syntax", "refactor", "bug", "traceback", "exception",
            "assert", "unit test", "pytest", "return "
        ]
        if any(marker in prompt_lower for marker in code_markers):
            return QueryCategory.CODE_DEVELOPMENT

        # Check RAG / Document / Knowledge lookup markers
        rag_markers = [
            "according to", "documentation", "reference", "manual", "source code",
            "file:", "context:", "based on", "excerpt", "search result"
        ]
        if any(marker in prompt_lower for marker in rag_markers) or len(prompt) > 2500:
            return QueryCategory.RAG_KNOWLEDGE

        # Check Conversational Reasoning
        if history_len >= 3 or any(w in prompt_lower for w in ["earlier", "previously", "step 2", "why did you"]):
            return QueryCategory.CONVERSATIONAL_REASONING

        return QueryCategory.GENERAL

    def allocate_budget(
        self,
        category: QueryCategory,
        total_limit: int | None = None,
        reserve_generation: int | None = None,
    ) -> TokenBudget:
        """Dynamically distribute the total token budget based on query category.

        Formula: L_limit = T_system + T_history + T_rag + T_generation
        """
        limit = total_limit or self.target_limit
        gen_reserve = reserve_generation or self.generation_reserve
        usable_context = max(512, limit - gen_reserve)

        # Baseline system prompt allocation
        sys_tokens = 500

        available_for_context = max(256, usable_context - sys_tokens)

        if category == QueryCategory.RAG_KNOWLEDGE:
            # Strategy A: Maximize RAG / context documents (70%), tight history sliding window
            rag_tokens = int(available_for_context * 0.70)
            hist_tokens = available_for_context - rag_tokens
        elif category == QueryCategory.CONVERSATIONAL_REASONING:
            # Strategy B: Maximize chat history (65%), smaller RAG allowance
            hist_tokens = int(available_for_context * 0.65)
            rag_tokens = available_for_context - hist_tokens
        elif category == QueryCategory.CODE_DEVELOPMENT:
            # Balanced code context: 50% workspace RAG/code, 50% history/tests
            rag_tokens = int(available_for_context * 0.50)
            hist_tokens = available_for_context - rag_tokens
        else:
            # General distribution: 40% RAG, 60% history
            rag_tokens = int(available_for_context * 0.40)
            hist_tokens = available_for_context - rag_tokens

        return TokenBudget(
            target_limit=limit,
            system_tokens=sys_tokens,
            history_tokens=hist_tokens,
            rag_tokens=rag_tokens,
            generation_tokens=gen_reserve,
        )
